"""Implicit "helpful" feedback comes from an agent's retrieval, not the store's own lookup.

Bug-team audit 2026-09-29, finding B6: auto_capture searches for similar
memories before every store (dedup and evolution). That search was recorded
as a retrieval, so Phase 5 ("retrieval-then-store") then credited the very
memories it had just looked up as helpful: each store rewarded its nearest
neighbours, whatever they were.
"""

import pytest

pytestmark = pytest.mark.usefixtures("_reset_bridge")

IMPLICIT = "implicit: retrieval-then-store"

UNRELATED_DECISIONS = [
    "Use PostgreSQL for the orders service database.",
    "Adopt Tailwind for styling the dashboard frontend.",
    "Pin numpy to 1.26 in the ML service requirements.",
    "Deploy the background worker on Fly.io.",
    "Use JWT access tokens that expire after 15 minutes.",
    "Log errors to Sentry from the Python backend.",
]


def _implicit_signals(store, node_id):
    meta = store.get_node(node_id, track_access=False).metadata or {}
    return [s for s in meta.get("feedback_signals", []) if s.get("reason") == IMPLICIT]


def test_storing_does_not_reward_its_own_neighbours():
    import omega.bridge as bridge

    ids = [
        bridge.store(text, event_type="decision", project="/work/alpha").split()[1]
        for text in UNRELATED_DECISIONS
    ]

    store = bridge._get_store()
    assert {i: _implicit_signals(store, i) for i in ids} == {i: [] for i in ids}


def test_an_agent_retrieval_followed_by_a_store_still_counts():
    import omega.bridge as bridge

    target = bridge.store(
        "The payments webhook must verify the Stripe signature before parsing JSON.",
        event_type="lesson_learned", project="/work/alpha",
    ).split()[1]
    hits = bridge.query_structured("payments webhook Stripe signature", limit=5)
    assert target in {h["id"] for h in hits}

    bridge.store(
        "Decided the payments webhook handler rejects requests whose Stripe signature fails.",
        event_type="decision", project="/work/alpha",
    )

    assert _implicit_signals(bridge._get_store(), target)


def test_untracked_lookup_is_not_recorded(store):
    store.store(content="Cache product listings for 5 minutes at the CDN.",
                metadata={"event_type": "decision"})

    with store.untracked_lookup():
        results = store.query("product listings CDN cache", limit=5, use_cache=False)

    assert results
    assert store.get_retrieval_context() == []
    store.query("product listings CDN cache", limit=5, use_cache=False)
    assert store.get_retrieval_context()
