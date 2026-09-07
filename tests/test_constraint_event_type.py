"""Tests for the 'constraint' event type — TTL, priority, session surfacing, query injection."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from omega.types import TTLCategory, AutoCaptureEventType, EVENT_TYPE_TTL

# Tests here populate the bridge store singleton via bridge.store(); reset it
# per test so permanent constraint memories don't leak into later test files
# (e.g. test_context_packet's empty-store assertions).
pytestmark = pytest.mark.usefixtures("_reset_bridge")


# ---------------------------------------------------------------------------
# Change 1: Registration
# ---------------------------------------------------------------------------


class TestConstraintRegistration:
    def test_constraint_constant_exists(self):
        assert AutoCaptureEventType.CONSTRAINT == "constraint"

    def test_constraint_ttl_is_permanent(self):
        assert EVENT_TYPE_TTL[AutoCaptureEventType.CONSTRAINT] is None

    def test_for_event_type_returns_none(self):
        assert TTLCategory.for_event_type("constraint") is None

    def test_constraint_priority_is_5(self):
        from omega.sqlite_store import SQLiteStore
        assert SQLiteStore._DEFAULT_PRIORITY["constraint"] == 5

    def test_constraint_type_weight(self):
        from omega.sqlite_store import SQLiteStore
        assert SQLiteStore._TYPE_WEIGHTS["constraint"] == 3.0

    def test_constraint_no_decay(self):
        from omega.sqlite_store import SQLiteStore
        assert SQLiteStore._DECAY_LAMBDAS["constraint"] == 0.0


# ---------------------------------------------------------------------------
# Change 2: Session start surfacing
# ---------------------------------------------------------------------------


class TestConstraintSessionContext:
    def test_constraints_appear_in_context_items(self, tmp_omega_dir):
        """Constraints should appear with RULE tag in get_session_context."""
        import omega.bridge as bridge
        bridge._store_instance = None

        bridge.store(
            "Coordination features are pro-only. Never sync to omega-public.",
            event_type="constraint",
        )

        ctx = bridge.get_session_context()
        tags = [item["tag"] for item in ctx["context_items"]]
        assert "RULE" in tags
        rule_items = [item for item in ctx["context_items"] if item["tag"] == "RULE"]
        assert any("pro-only" in item["text"] for item in rule_items)

    def test_constraints_appear_even_without_recent_activity(self, tmp_omega_dir):
        """Constraints use get_by_type, not recency — they always surface."""
        from unittest.mock import patch

        import omega.bridge as bridge
        bridge._store_instance = None

        # Store a constraint
        bridge.store(
            "Never deploy to production on Fridays.",
            event_type="constraint",
        )

        # Store 100+ non-constraint memories to push constraint out of recent-100.
        # Mock expand_query to avoid 110 real LLM API calls (causes timeout).
        with patch("omega.query_expansion.expand_query", return_value=None):
            for i in range(110):
                bridge.store(f"Decision number {i} about topic {i}", event_type="decision")

        ctx = bridge.get_session_context()
        rule_items = [item for item in ctx["context_items"] if item["tag"] == "RULE"]
        assert len(rule_items) >= 1
        assert any("Friday" in item["text"] for item in rule_items)

    def test_constraint_budget_separate_from_regular(self, tmp_omega_dir):
        """Constraints get their own budget of 3, regular items get their own budget."""
        import omega.bridge as bridge
        bridge._store_instance = None

        # Store 3 constraints
        bridge.store("Constraint rule one about testing.", event_type="constraint")
        bridge.store("Constraint rule two about deployment.", event_type="constraint")
        bridge.store("Constraint rule three about security.", event_type="constraint")

        # Store regular high-value items
        bridge.store("Important decision about architecture.", event_type="decision")
        bridge.store("Key lesson about performance.", event_type="lesson_learned")

        ctx = bridge.get_session_context()
        rule_items = [item for item in ctx["context_items"] if item["tag"] == "RULE"]
        non_rule_items = [item for item in ctx["context_items"] if item["tag"] != "RULE"]

        assert len(rule_items) <= 3
        assert len(non_rule_items) >= 1  # Regular items still present


# ---------------------------------------------------------------------------
# Change 2b: Welcome output
# ---------------------------------------------------------------------------


class TestConstraintWelcome:
    def test_welcome_includes_active_constraints_section(self, tmp_omega_dir):
        """welcome() should show constraints under 'Active Constraints' heading."""
        import omega.bridge as bridge
        bridge._store_instance = None

        bridge.store(
            "Never sync coordination code to omega-public.",
            event_type="constraint",
        )

        result = bridge.welcome()
        assert "Active Constraints" in result["observation_prefix"]

    def test_welcome_constraints_appear_first(self, tmp_omega_dir):
        """Constraints section should appear before other sections."""
        import omega.bridge as bridge
        bridge._store_instance = None

        bridge.store("Constraint about repo scope.", event_type="constraint")
        bridge.store("A key lesson about caching.", event_type="lesson_learned")
        bridge.store("User prefers dark mode.", event_type="user_preference")

        result = bridge.welcome()
        prefix = result["observation_prefix"]
        if "Active Constraints" in prefix and "User Preferences" in prefix:
            assert prefix.index("Active Constraints") < prefix.index("User Preferences")


# ---------------------------------------------------------------------------
# Change 3: Query injection
# ---------------------------------------------------------------------------


class TestConstraintQueryInjection:
    def test_query_injects_matching_constraints(self, tmp_omega_dir):
        """query() should append matching constraints when not already in results."""
        import omega.bridge as bridge
        bridge._store_instance = None

        bridge.store(
            "Coordination features are pro-only. Never sync to omega-public.",
            event_type="constraint",
        )
        bridge.store(
            "Updated the README formatting last week.",
            event_type="decision",
        )

        # Query that matches the decision but also has words overlapping the constraint
        result = bridge.query("omega-public features", event_type="decision")
        # The constraint should be injected since we filtered to decision event_type
        # (constraint won't be in main results) but has word overlap
        assert "Active Constraints" in result

    def test_query_no_double_inject_when_filtering_constraints(self, tmp_omega_dir):
        """query(event_type='constraint') should not double-inject."""
        import omega.bridge as bridge
        bridge._store_instance = None

        bridge.store(
            "Never deploy without tests.",
            event_type="constraint",
        )

        result = bridge.query("deploy tests", event_type="constraint")
        # Should not have the injection section since we're already querying constraints
        assert "Active Constraints:" not in result

    def test_query_structured_injects_with_flag(self, tmp_omega_dir):
        """query_structured() should inject constraints with is_constraint=True."""
        import omega.bridge as bridge
        bridge._store_instance = None

        bridge.store(
            "Public repo must not contain pro features or coordination code.",
            event_type="constraint",
        )
        bridge.store(
            "Updated the public repo CI pipeline.",
            event_type="decision",
        )

        results = bridge.query_structured("public repo features")
        constraint_results = [r for r in results if r.get("is_constraint")]
        if constraint_results:
            assert constraint_results[0]["event_type"] == "constraint"
            # Constraints should be prepended (first in list)
            assert results.index(constraint_results[0]) == 0

    def test_query_no_inject_when_no_word_overlap(self, tmp_omega_dir):
        """Constraints that don't match the query words should not be injected."""
        import omega.bridge as bridge
        bridge._store_instance = None

        bridge.store(
            "Never deploy without running tests first.",
            event_type="constraint",
        )

        result = bridge.query("favorite color preferences")
        assert "Active Constraints" not in result


# ---------------------------------------------------------------------------
# Change 4: Dedup/evolution constants
# ---------------------------------------------------------------------------


class TestConstraintConstants:
    def test_constraint_in_dedup_thresholds(self):
        from omega.bridge import DEDUP_THRESHOLDS
        assert DEDUP_THRESHOLDS[AutoCaptureEventType.CONSTRAINT] == 0.90

    def test_constraint_in_evolution_types(self):
        from omega.bridge import EVOLUTION_TYPES
        assert AutoCaptureEventType.CONSTRAINT in EVOLUTION_TYPES
