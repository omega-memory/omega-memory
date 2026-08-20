"""Hard query constraints are monotonic.

Once a candidate fails a hard constraint, no later ranking, expansion, or
graph phase may resurrect it. These tests assert the invariant itself —
"graph connectivity cannot cross a hard query constraint" — rather than
asserting that any single filter works.

Regression origin: Phase 6 (graph expansion) wrote neighbours straight into
the candidate set after Phases 4/5 had filtered, so connectivity silently
overrode the caller's constraints. The bug was invisible whenever the Phase
2.5 strong-signal short-circuit fired, because that path returns before
Phase 6 runs. Every test here therefore pins the short-circuit explicitly
instead of letting corpus shape decide which path executes.
"""

import pytest

from omega.sqlite_store import SQLiteStore


@pytest.fixture
def store(tmp_path):
    return SQLiteStore(db_path=tmp_path / "invariant.db")


@pytest.fixture
def force_graph_phase(monkeypatch):
    """Make the strong-signal short-circuit unreachable so Phase 6 always runs.

    Without this, a slam-dunk FTS match returns before graph expansion and a
    broken pipeline looks healthy.
    """
    monkeypatch.setattr("omega.sqlite_store._query.STRONG_SIGNAL_THRESHOLD", 99.0)


@pytest.fixture
def force_short_circuit(monkeypatch):
    """Force the Phase 2.5 early-return path instead."""
    monkeypatch.setattr("omega.sqlite_store._query.STRONG_SIGNAL_THRESHOLD", 0.0)
    monkeypatch.setattr("omega.sqlite_store._query.STRONG_SIGNAL_GAP", 0.0)


QUERY = "zebrafish record"


def seed(store, content, event_type=None, project=None, **kwargs):
    """Store one memory and return its node id.

    event_type/project live in metadata; entity_id/agent_type/session_id are
    first-class store() arguments. skip_inference keeps auto-association from
    adding edges the test did not ask for.
    """
    metadata = {}
    if event_type is not None:
        metadata["event_type"] = event_type
    if project is not None:
        metadata["project"] = project
    return store.store(
        content=content, metadata=metadata or None, skip_inference=True, **kwargs
    )


def linked_pair(store, anchor_kwargs, neighbour_kwargs, hops=1, graph_only=False):
    """Create an anchor the query matches, plus a graph-linked neighbour.

    By default the neighbour also matches the query. That is the shape of the
    original defect: the neighbour earns a score, a hard filter removes it from
    node_scores, and Phase 6 then re-adds it — because the traversal's
    "already a candidate" set is taken from the *filtered* scores, so a
    just-filtered node no longer looks present. A neighbour that never scored
    at all does not reproduce it.

    graph_only=True instead gives a neighbour with no lexical overlap, whose
    sole route into the results is graph expansion. Used by the controls that
    prove legitimate expansion still works.
    """
    anchor = seed(store, "MARKERANCHOR zebrafish payroll ledger Helsinki", **anchor_kwargs)
    neighbour_text = (
        "MARKERNEIGHBOUR quarterly narwhal telemetry"
        if graph_only
        else "MARKERNEIGHBOUR zebrafish sonar calibration knots"
    )
    neighbour = seed(store, neighbour_text, **neighbour_kwargs)
    if hops == 1:
        store.add_edge(anchor, neighbour, edge_type="related", weight=1.0)
    else:
        middle = seed(store, "MARKERMIDDLE zebrafish intermediate hop record")
        store.add_edge(anchor, middle, edge_type="related", weight=1.0)
        store.add_edge(middle, neighbour, edge_type="related", weight=1.0)
    return anchor, neighbour


def ids(results):
    return {r.id for r in results}


class TestGraphCannotCrossHardConstraints:
    """One test per hard predicate that a graph edge can straddle."""

    def test_entity_id(self, store, force_graph_phase):
        anchor, neighbour = linked_pair(
            store, {"entity_id": "roota"}, {"entity_id": "rootb"}
        )
        got = ids(store.query(QUERY, limit=10, entity_id="roota", use_cache=False))
        assert anchor in got
        assert neighbour not in got, "graph hop crossed an entity_id boundary"

    def test_nonexistent_entity_reaches_no_scoped_record(self, store, force_graph_phase):
        """A supplied-but-unknown entity yields unscoped records at most.

        The unscoped record must survive the filter and seed graph expansion,
        otherwise Phase 6's `if node_scores` gate skips expansion and the test
        passes for the wrong reason.
        """
        unscoped = seed(store, "MARKERUNSCOPED zebrafish overcast forecast Tuesday")
        scoped_a = seed(store, "MARKERANCHOR zebrafish payroll ledger Helsinki",
                        entity_id="roota")
        scoped_b = seed(store, "MARKERNEIGHBOUR zebrafish sonar calibration knots",
                        entity_id="rootb")
        store.add_edge(unscoped, scoped_a, edge_type="related", weight=1.0)
        store.add_edge(unscoped, scoped_b, edge_type="related", weight=1.0)

        got = ids(store.query(QUERY, limit=10, entity_id="ghost", use_cache=False))
        assert unscoped in got, "unscoped records must remain visible"
        assert scoped_a not in got and scoped_b not in got, (
            "a nonexistent entity must never fall back to other entities"
        )

    def test_exclude_types(self, store, force_graph_phase):
        anchor, neighbour = linked_pair(store, {}, {"event_type": "decision"})
        got = ids(store.query(
            QUERY, limit=10, exclude_types=["decision"], use_cache=False,
        ))
        assert anchor in got
        assert neighbour not in got, "graph hop crossed an exclude_types boundary"

    def test_infrastructure_types_excluded_by_default(self, store, force_graph_phase):
        anchor, neighbour = linked_pair(store, {}, {"event_type": "code_chunk"})
        got = ids(store.query(QUERY, limit=10, use_cache=False))
        assert anchor in got
        assert neighbour not in got, "graph hop reintroduced an infrastructure type"

    def test_agent_type(self, store, force_graph_phase):
        anchor, neighbour = linked_pair(
            store, {"agent_type": "reviewer"}, {"agent_type": "planner"}
        )
        got = ids(store.query(
            QUERY, limit=10, agent_type="reviewer", use_cache=False,
        ))
        assert anchor in got
        assert neighbour not in got, "graph hop crossed an agent_type boundary"

    def test_session_scope(self, store, force_graph_phase):
        anchor, neighbour = linked_pair(
            store, {"session_id": "sess-a"}, {"session_id": "sess-b"}
        )
        got = ids(store.query(
            QUERY, limit=10, session_id="sess-a", scope="session", use_cache=False,
        ))
        assert anchor in got
        assert neighbour not in got, "graph hop crossed a session boundary"

    def test_project_scope(self, store, force_graph_phase):
        anchor, neighbour = linked_pair(
            store, {"project": "/proj/a"}, {"project": "/proj/b"}
        )
        got = ids(store.query(
            QUERY, limit=10, project_path="/proj/a", scope="project", use_cache=False,
        ))
        assert anchor in got
        assert neighbour not in got, "graph hop crossed a project boundary"

    def test_two_hop_expansion_is_also_constrained(self, store, force_graph_phase):
        anchor, neighbour = linked_pair(
            store, {"entity_id": "roota"}, {"entity_id": "rootb"}, hops=2,
        )
        got = ids(store.query(QUERY, limit=10, entity_id="roota", use_cache=False))
        assert anchor in got
        assert neighbour not in got, "two-hop graph path crossed an entity boundary"

    def test_short_circuit_path_also_holds(self, store, force_short_circuit):
        """The early-return path must enforce the same invariant."""
        _, neighbour = linked_pair(
            store, {"entity_id": "roota"}, {"entity_id": "rootb"}
        )
        got = ids(store.query(QUERY, limit=10, entity_id="roota", use_cache=False))
        assert neighbour not in got


class TestLegitimateExpansionStillWorks:
    """The invariant must not be satisfied by disabling graph expansion."""

    def test_unconstrained_query_still_receives_graph_neighbours(
        self, store, force_graph_phase,
    ):
        anchor, neighbour = linked_pair(store, {}, {}, graph_only=True)
        got = ids(store.query(QUERY, limit=10, use_cache=False))
        assert anchor in got
        assert neighbour in got, (
            "graph expansion stopped working; the invariant must constrain "
            "expansion, not disable it"
        )

    def test_allowed_neighbour_keeps_its_decayed_score(
        self, store, force_graph_phase,
    ):
        """Ranking control: an admissible neighbour is scored as before."""
        anchor, neighbour = linked_pair(
            store, {"entity_id": "roota"}, {"entity_id": "roota"}, graph_only=True,
        )
        results = store.query(QUERY, limit=10, entity_id="roota", use_cache=False)
        by_id = {r.id: r for r in results}
        assert neighbour in by_id, "same-entity neighbour was wrongly blocked"
        # Hop decay must still place the neighbour below the anchor.
        assert by_id[anchor].relevance > by_id[neighbour].relevance

    def test_unscoped_neighbour_remains_visible_under_a_scoped_query(
        self, store, force_graph_phase,
    ):
        """entity_id admits unscoped records; agent_type does not. Pin it."""
        anchor, neighbour = linked_pair(store, {"entity_id": "roota"}, {})
        got = ids(store.query(QUERY, limit=10, entity_id="roota", use_cache=False))
        assert neighbour in got, "unscoped records must stay globally visible"


class TestInvariantBackstop:
    """The final pre-assembly check is the safety net for future phases."""

    def test_backstop_removes_and_reports_an_injected_violation(
        self, store, force_graph_phase, caplog,
    ):
        import logging

        anchor = seed(store, "MARKERANCHOR zebrafish payroll ledger Helsinki",
                      entity_id="roota")
        intruder = seed(store, "MARKERINTRUDER unrelated content", entity_id="rootb")

        # Simulate a future phase that admits a candidate bypassing the filters.
        original = SQLiteStore._query_phase_rerank

        def leaky(self, query_text, all_results, node_scores, limit, pw_graph,
                  constraints=None):
            original(self, query_text, all_results, node_scores, limit, pw_graph, None)
            row = self.get_node(intruder)
            all_results[intruder] = row
            node_scores[intruder] = 99.0

        SQLiteStore._query_phase_rerank = leaky
        try:
            with caplog.at_level(logging.WARNING, logger="omega.sqlite_store"):
                got = ids(store.query(QUERY, limit=10, entity_id="roota",
                                      use_cache=False))
        finally:
            SQLiteStore._query_phase_rerank = original

        assert anchor in got
        assert intruder not in got, "backstop failed to remove an injected violation"
        assert "Hard-constraint invariant violated" in caplog.text
        assert "entity_id" in caplog.text
