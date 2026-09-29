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
