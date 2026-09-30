"""store() finds similar memories without the cross-encoder; searches still use it.

The cross-encoder is for ranking a person's search results. store() used to
run it twice per write: once in the lookup for duplicates to dedup or evolve
against, and once to score contradiction candidates that could not become
contradictions at any score.
"""

import pytest

import omega.reranker as reranker
from omega.contradictions import detect_contradictions

MEMORIES = [
    "Decided to keep the audit log in sqlite with a nightly vacuum job.",
    "Decided the audit log export runs through the sqlite backup API.",
    "Decided to rotate the audit log weekly and archive it to cold storage.",
    "Decided the audit log keeps request ids so support can trace a call.",
]


@pytest.fixture
def full_pipeline(monkeypatch):
    """Turn off the shortcut that skips reranking when one text match stands out."""
    import omega.sqlite_store._query as query_module

    monkeypatch.setattr(query_module, "STRONG_SIGNAL_THRESHOLD", 2.0)


@pytest.fixture
def cross_encoder_calls(monkeypatch):
    calls = []

    def fake_cross_encoder_score(query, passages, temporal_metadata=None):
        calls.append(query)
        return [float(len(p) % 7) for p in passages]

    monkeypatch.setattr(reranker, "cross_encoder_score", fake_cross_encoder_score)
    return calls


def test_store_dedup_lookup_does_not_rerank(_reset_bridge, cross_encoder_calls):
    from omega import bridge

    for text in MEMORIES:
        bridge.auto_capture(content=text, event_type="decision", project="/work/app")
    cross_encoder_calls.clear()

    result = bridge.auto_capture(
        content="Decided to keep the audit log in sqlite with a nightly vacuum job and checksums.",
        event_type="decision",
        project="/work/app",
    )

    assert result.split()[0] in ("Stored", "Evolved", "Deduped", "Reconfirmed")
    assert cross_encoder_calls == []


def test_search_still_reranks(store, cross_encoder_calls, full_pipeline):
    for text in MEMORIES:
        store.store(content=text, metadata={"event_type": "decision"})
    cross_encoder_calls.clear()

    results = store.query("audit log sqlite storage decisions", use_cache=False)

    assert results
    assert cross_encoder_calls, "a search must still go through the cross-encoder"


def test_unreranked_lookup_is_not_served_to_a_search_from_cache(
    store, cross_encoder_calls, full_pipeline
):
    for text in MEMORIES:
        store.store(content=text, metadata={"event_type": "decision"})
    cross_encoder_calls.clear()

    store.query("audit log sqlite storage decisions", rerank=False)
    assert cross_encoder_calls == []

    store.query("audit log sqlite storage decisions")
    assert cross_encoder_calls, "the cached unreranked result must not answer a search"


def test_contradiction_check_skips_similarity_when_nothing_can_pass(monkeypatch):
    calls = []
    monkeypatch.setattr(
        "omega.contradictions._get_similarity_scores",
        lambda query, passages: calls.append(query) or [1.0] * len(passages),
    )

    results = detect_contradictions(
        "The build uses pytest for the unit tests.",
        ["The build uses ruff for linting.", "Unit tests live under tests/."],
    )

    assert results == []
    assert calls == []


def test_contradiction_check_scores_similarity_when_a_candidate_can_pass(monkeypatch):
    calls = []
    monkeypatch.setattr(
        "omega.contradictions._get_similarity_scores",
        lambda query, passages: calls.append(query) or [2.0, 0.0][: len(passages)],
    )

    results = detect_contradictions(
        "We never deploy on Fridays, the release is now on Tuesday.",
        ["We always deploy on Fridays after the release review.", "Lunch is at noon."],
    )

    assert calls, "a candidate with strong signals needs the similarity score"
    assert [r.candidate_index for r in results] == [0]
