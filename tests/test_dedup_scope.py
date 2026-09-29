"""Dedup must stay inside one project and entity, and never revive a retired row.

Bug-team audit 2026-09-29:
  B4: store()'s hash dedup ignored status and scope, so re-stating a retired
      memory collapsed into the retired row, and identical text in another
      project collapsed into that project's row. auto_capture then ran its
      post-store phases on the existing node as if it were new, overwriting
      its metadata.
"""

import pytest

from omega.sqlite_store import SQLiteStore


@pytest.fixture
def plain_store(tmp_omega_dir):
    s = SQLiteStore(db_path=tmp_omega_dir / "dedup.db")
    yield s
    s.close()


def _put(store, content, project, entity_id=None, event_type="lesson_learned"):
    node_id = store.store(
        content=content,
        metadata={"event_type": event_type, "project": project},
        entity_id=entity_id,
        skip_inference=True,
    )
    return node_id, store.get_last_store_deduped()


TEXT = "Run the test suite with pytest -x before every commit."


class TestHashDedupScope:
    def test_same_project_still_dedups(self, plain_store):
        first, _ = _put(plain_store, TEXT, "/work/alpha")
        second, deduped = _put(plain_store, TEXT, "/work/alpha")
        assert (second, deduped) == (first, True)

    def test_other_project_gets_its_own_memory(self, plain_store):
        first, _ = _put(plain_store, TEXT, "/work/alpha")
        second, deduped = _put(plain_store, TEXT, "/work/beta")
        assert second != first
        assert deduped is False
        project = plain_store._conn.execute(
            "SELECT project FROM memories WHERE node_id = ?", (second,)
        ).fetchone()[0]
        assert project == "/work/beta"

    def test_other_entity_gets_its_own_memory(self, plain_store):
        first, _ = _put(plain_store, TEXT, "/p", entity_id="acme")
        second, deduped = _put(plain_store, TEXT, "/p", entity_id="globex")
        assert second != first
        assert deduped is False

    def test_reformatted_duplicate_in_other_project_is_kept(self, plain_store):
        """Canonical-hash dedup is scoped the same way as content-hash dedup."""
        first, _ = _put(plain_store, TEXT, "/work/alpha")
        second, deduped = _put(plain_store, "  " + TEXT.upper() + "  ", "/work/beta")
        assert second != first
        assert deduped is False

    def test_restating_a_retired_memory_stores_a_new_one(self, plain_store):
        first, _ = _put(plain_store, TEXT, "/work/alpha")
        replacement, _ = _put(plain_store, "Run pytest before pushing, not before committing.", "/work/alpha")
        plain_store.mark_superseded(first, replacement)

        restated, deduped = _put(plain_store, TEXT, "/work/alpha")

        assert restated != first
        assert deduped is False
        status = plain_store._conn.execute(
            "SELECT status FROM memories WHERE node_id = ?", (first,)
        ).fetchone()[0]
        assert status == "superseded"

    def test_metadata_only_retirement_is_also_excluded(self, plain_store):
        """Some paths (compaction, reminders) set only metadata.superseded."""
        first, _ = _put(plain_store, TEXT, "/work/alpha")
        node = plain_store.get_node(first, track_access=False)
        plain_store.update_node(first, metadata={**node.metadata, "superseded": True})

        restated, deduped = _put(plain_store, TEXT, "/work/alpha")

        assert restated != first
        assert deduped is False


# ---------------------------------------------------------------------------
# auto_capture after a dedup
# ---------------------------------------------------------------------------

# user_preference has no Jaccard dedup threshold, so a repeat reaches store()'s
# hash dedup; it is also long enough for Phase 3.5's observation summary.
PREFERENCE = (
    "User prefers small, focused pull requests that change one thing at a time. "
    "Large refactors should land as a series of reviewable steps, each passing CI "
    "on its own, rather than as one branch."
)


@pytest.mark.usefixtures("_reset_bridge")
class TestAutoCaptureAfterDedup:
    def test_dedup_leaves_the_existing_memory_untouched(self):
        import omega.bridge as bridge

        first = bridge.store(
            PREFERENCE, event_type="user_preference", project="/work/alpha",
            metadata={"source": "alpha-source", "tags": ["alpha-tag"]},
        )
        node_id = first.split()[1]
        store = bridge._get_store()
        before = store.get_node(node_id, track_access=False).metadata

        second = bridge.store(
            PREFERENCE, event_type="user_preference", project="/work/alpha",
            metadata={"source": "beta-source", "tags": ["beta-tag"]},
        )

        assert second == f"Deduped → {node_id}"
        after = store.get_node(node_id, track_access=False).metadata
        assert after["source"] == "alpha-source"
        assert after == before

    def test_post_store_phases_do_not_run_on_a_dedup(self, monkeypatch):
        """Enrichment belongs to a new memory; a dedup wrote nothing new."""
        import omega.bridge as bridge

        bridge.store(PREFERENCE, event_type="user_preference", project="/work/alpha")
        enriched = []
        monkeypatch.setattr(
            bridge, "_compress_to_observation",
            lambda content, event_type="": enriched.append("observation"),
        )
        monkeypatch.setattr(
            bridge, "_schedule_auto_relate",
            lambda store, node_id: enriched.append("auto_relate"),
        )

        result = bridge.store(PREFERENCE, event_type="user_preference", project="/work/alpha")

        assert result.startswith("Deduped")
        assert enriched == []


# ---------------------------------------------------------------------------
# B3: auto_capture's word-overlap dedup and evolution
# ---------------------------------------------------------------------------
#
# The Jaccard comparison ignores words shorter than 4 characters, so "100" vs
# "300", "not", and "on"/"off" were invisible: 4 of these 6 updates were
# dropped as duplicates of the memory they were updating. It also compared
# against memories in any project or entity.

UPDATES_BY_VALUE = [
    ("decision", "Rate limit for the public API is 100 requests per minute.",
     "Rate limit for the public API is 300 requests per minute."),
    ("decision", "We will deploy on Fridays.", "We will not deploy on Fridays."),
    ("user_fact", "User's monthly budget for cloud hosting is $200.",
     "User's monthly budget for cloud hosting is $500."),
    ("decision", "Feature flag checkout_v2 is on in production.",
     "Feature flag checkout_v2 is off in production."),
    ("memory", "The staging database password rotates every 30 days.",
     "The staging database password rotates every 90 days."),
    ("lesson_learned", "Set the worker timeout to 30s for large uploads.",
     "Set the worker timeout to 120s for large uploads."),
]


@pytest.mark.usefixtures("_reset_bridge")
class TestWordOverlapDedup:
    @pytest.mark.parametrize("event_type,older,newer", UPDATES_BY_VALUE)
    def test_update_differing_in_numbers_or_negation_is_stored(self, event_type, older, newer):
        import omega.bridge as bridge

        bridge.store(older, event_type=event_type, project="/work/alpha")
        result = bridge.store(newer, event_type=event_type, project="/work/alpha")

        assert result.startswith("Stored"), result
        contents = {
            row[0] for row in bridge._get_store()._conn.execute("SELECT content FROM memories")
        }
        assert {older, newer} <= contents

    def test_restatement_in_same_project_still_dedups(self):
        import omega.bridge as bridge

        text = "Rate limit for the public API is 100 requests per minute."
        first = bridge.store(text, event_type="decision", project="/work/alpha")
        again = bridge.store(text + " ", event_type="decision", project="/work/alpha")
        assert again == f"Deduped → {first.split()[1]}"

    def test_same_decision_in_another_project_is_stored_there(self):
        import omega.bridge as bridge

        text = "Rate limit for the public API is 100 requests per minute."
        bridge.store(text, event_type="decision", project="/work/client-a")
        result = bridge.store(text, event_type="decision", project="/work/client-b")

        assert result.startswith("Stored"), result
        projects = sorted(
            row[0] for row in bridge._get_store()._conn.execute("SELECT project FROM memories")
        )
        assert projects == ["/work/client-a", "/work/client-b"]

    def test_same_decision_for_another_entity_is_stored_for_it(self):
        import omega.bridge as bridge

        text = "Invoices are due within 30 days of issue."
        bridge.store(text, event_type="decision", project="/p", entity_id="acme")
        result = bridge.store(text, event_type="decision", project="/p", entity_id="globex")
        assert result.startswith("Stored"), result

    def test_evolution_never_rewrites_another_projects_memory(self):
        import omega.bridge as bridge

        base = (
            "Always run the database migrations before deploying the API service "
            "to production and verify the schema version."
        )
        first = bridge.store(base, event_type="lesson_learned", project="/work/client-a")
        old_id = first.split()[1]

        result = bridge.store(
            base + " Record the migration duration in the release notes.",
            event_type="lesson_learned", project="/work/client-b",
        )

        assert result.startswith("Stored"), result
        old = bridge._get_store().get_node(old_id, track_access=False)
        assert old.content == base

    def test_reconfirmation_needs_the_same_numbers(self):
        """Phase 2 used to answer 'Reconfirmed' when only a number changed."""
        import omega.bridge as bridge

        bridge.store(
            "Always set the worker timeout to 30s for large uploads, and retry twice on failure.",
            event_type="lesson_learned", project="/work/alpha",
        )
        result = bridge.store(
            "Set the worker timeout to 90s for large uploads, and retry on failure.",
            event_type="lesson_learned", project="/work/alpha",
        )
        assert result.startswith("Stored"), result

    def test_error_patterns_still_dedup_across_line_numbers(self):
        """error_pattern normalizes numbers on purpose: line numbers vary per run."""
        import omega.bridge as bridge

        first = bridge.store(
            "ValueError: invalid literal for int() at parser.py line 42 while reading config",
            event_type="error_pattern", project="/work/alpha",
        )
        again = bridge.store(
            "ValueError: invalid literal for int() at parser.py line 57 while reading config",
            event_type="error_pattern", project="/work/alpha",
        )
        assert again == f"Deduped → {first.split()[1]}"
