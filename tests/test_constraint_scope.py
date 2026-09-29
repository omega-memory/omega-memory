"""Constraint memories reach only the project and entity they were stored for.

Bug-team audit 2026-09-29, finding B5: every "constraint" memory was injected
into every session, query and welcome briefing in every project, with no
length cap. That broke project separation, and anyone able to store a memory
could plant a rule in every other project.
"""

import os

import pytest

pytestmark = pytest.mark.usefixtures("_reset_bridge")

ALPHA_RULE = "Never deploy the alpha storefront on Fridays; releases wait until Monday."
BETA_RULE = "Never deploy the beta billing service without a second reviewer present."
ACME_RULE = "Never email acme invoices before the finance team approves the batch."


def _rules(ctx):
    return [item["text"] for item in ctx["context_items"] if item["tag"] == "RULE"]


@pytest.fixture
def rules():
    import omega.bridge as bridge

    bridge.store(ALPHA_RULE, event_type="constraint", project="/work/alpha")
    bridge.store(BETA_RULE, event_type="constraint", project="/work/beta")
    bridge.store(ACME_RULE, event_type="constraint", project="/work/alpha", entity_id="acme")
    return bridge


class TestSessionContext:
    def test_only_this_projects_rules(self, rules):
        texts = _rules(rules.get_session_context(project="/work/alpha"))
        assert ALPHA_RULE in texts
        assert BETA_RULE not in texts

    def test_other_entitys_rule_needs_that_entity(self, rules):
        assert ACME_RULE not in _rules(rules.get_session_context(project="/work/alpha"))
        assert ACME_RULE not in _rules(
            rules.get_session_context(project="/work/alpha", entity_id="globex")
        )
        acme = _rules(rules.get_session_context(project="/work/alpha", entity_id="acme"))
        assert ACME_RULE in acme
        assert ALPHA_RULE in acme  # the project's entity-free rules still apply

    def test_no_project_means_the_current_directory(self, rules):
        rules.store("Keep commit subjects under 72 characters in this repository.",
                    event_type="constraint")
        texts = _rules(rules.get_session_context())
        assert "Keep commit subjects under 72 characters in this repository." in texts
        assert ALPHA_RULE not in texts and BETA_RULE not in texts

    def test_rule_text_is_capped(self):
        import omega.bridge as bridge

        bridge.store(
            "Never rotate the signing key without notice.",
            event_type="constraint",
            project="/work/alpha",
            metadata={"observation": "x" * 5000},
        )
        [text] = _rules(bridge.get_session_context(project="/work/alpha"))
        assert len(text) <= 300


class TestQueryInjection:
    """Rules are injected next to results; filtering to decisions isolates them."""

    QUERY = "never deploy release email"

    def test_query_injects_only_this_projects_rules(self, rules):
        out = rules.query(self.QUERY, project="/work/beta", event_type="decision")
        assert BETA_RULE[:60] in out
        assert ALPHA_RULE[:60] not in out
        assert ACME_RULE[:60] not in out

    def test_query_structured_injects_only_this_projects_rules(self, rules):
        results = rules.query_structured(self.QUERY, project="/work/beta", event_type="decision")
        injected = [r["content"] for r in results if r.get("is_constraint")]
        assert injected == [BETA_RULE]

    def test_query_structured_caps_injected_rule(self):
        import omega.bridge as bridge

        long_rule = "Never deploy on Fridays. " + "Details follow. " * 400
        bridge.store(long_rule, event_type="constraint", project="/work/alpha")
        results = bridge.query_structured(
            "never deploy friday", project="/work/alpha", event_type="decision"
        )
        injected = [r["content"] for r in results if r.get("is_constraint")]
        assert injected and all(len(c) <= 300 for c in injected)


class TestWelcome:
    def test_welcome_shows_only_this_projects_rules(self, rules):
        rules._welcome_cache.clear()
        prefix = rules.welcome(project="/work/beta")["observation_prefix"]
        assert BETA_RULE in prefix
        assert ALPHA_RULE not in prefix
        assert ACME_RULE not in prefix


def test_scoped_lookup_defaults_to_the_store_default(rules):
    """The store writes os.getcwd() when no project is given; lookups match it."""
    store = rules._get_store()
    here = store.get_by_type_in_scope("constraint", project=os.getcwd())
    assert all(r.metadata.get("event_type") == "constraint" for r in here)
    assert ALPHA_RULE not in [r.content for r in here]


def test_welcome_fallback_never_shows_another_projects_rule():
    """With no high-value memories, welcome falls back to the most recent ones."""
    import omega.bridge as bridge

    bridge.store(ALPHA_RULE, event_type="constraint", project="/work/alpha")
    bridge._welcome_cache.clear()
    prefix = bridge.welcome(project="/work/beta")["observation_prefix"]
    assert ALPHA_RULE not in prefix
