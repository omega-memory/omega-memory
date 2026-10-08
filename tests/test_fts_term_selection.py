"""Full-text search matches on a query's rare terms, not on every term.

OR-matching every query word scored nearly every memory, a cost that grew
with the store. The match expression keeps the rarest terms within a fixed
budget, plus the cheap "all terms together" and adjacent-phrase matches.
"""

import pytest

from omega.sqlite_store import SQLiteStore


def _fill(store, texts):
    return [store.store(content=text, metadata={"event_type": "memory"}) for text in texts]


@pytest.fixture
def tight_budget(monkeypatch):
    """Make a 40-memory store behave like a large one."""
    monkeypatch.setattr(SQLiteStore, "_FTS_MAX_POSTINGS", 5)


def test_small_store_keeps_every_topic_term(store):
    _fill(store, ["The deploy pipeline uses canary releases", "Canary releases need a rollback plan"])

    expression = store._fts_match_expression("canary releases rollback pipeline")

    for term in ("canary", "releases", "rollback", "pipeline"):
        assert f'"{term}"' in expression.split(" OR ")


def test_stopwords_and_absent_terms_are_not_matched(store):
    _fill(store, ["The deploy pipeline uses canary releases"])

    expression = store._fts_match_expression("what about those canary zeppelins")

    assert expression == '"canary"'


def test_no_matching_term_means_no_expression(store):
    _fill(store, ["The deploy pipeline uses canary releases"])

    assert store._fts_match_expression("zeppelin moorings") is None
    assert store._text_search("zeppelin moorings") == []


def test_common_terms_leave_the_or_match_in_a_large_store(store, tight_budget):
    _fill(store, [f"service note {i} about the gateway timeout" for i in range(40)])
    rare = _fill(store, ["service note about the kerberos gateway handshake"])[0]

    expression = store._fts_match_expression("gateway kerberos service")
    alternatives = expression.split(" OR ")

    assert '"kerberos"' in alternatives
    assert '"gateway"' not in alternatives
    assert '"service"' not in alternatives
    assert store._text_search("gateway kerberos service")[0].id == rare


def test_all_common_terms_together_still_find_their_memory(store, tight_budget):
    _fill(store, [f"alpha beta filler {i}" for i in range(15)])
    _fill(store, [f"beta gamma filler {i}" for i in range(15)])
    _fill(store, [f"alpha gamma filler {i}" for i in range(15)])
    target = _fill(store, ["alpha beta gamma together in a single memory"])[0]

    expression = store._fts_match_expression("alpha beta gamma")
    results = store._text_search("alpha beta gamma", limit=5)

    assert "AND" in expression
    assert results[0].id == target


def test_adjacent_terms_are_matched_as_a_phrase(store):
    _fill(store, ["rotate the signing key every quarter", "the quarter ends with a signing ceremony"])

    expression = store._fts_match_expression("signing key rotation quarter")

    assert '"signing key"' in expression.split(" OR ")


def test_search_works_without_term_frequencies(store, monkeypatch):
    target = _fill(store, ["rotate the signing key every quarter", "lunch is at noon"])[0]
    monkeypatch.setattr(SQLiteStore, "_fts_doc_frequencies", lambda self, terms: None)

    results = store._text_search("signing key rotation")

    assert [r.id for r in results] == [target]


def test_term_frequencies_add_nothing_to_the_database_file(store):
    _fill(store, ["rotate the signing key every quarter"])

    assert store._fts_doc_frequencies(["signing", "zeppelin"]) == {"signing": 1}
    in_file = store._conn.execute(
        "SELECT name FROM sqlite_master WHERE name LIKE '%vocab%'"
    ).fetchall()
    assert in_file == []
