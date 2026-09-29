"""Tests for LLM usage tracking."""


def test_log_call_and_query(tmp_path):
    from omega.usage_tracker import UsageTracker

    tracker = UsageTracker(db_path=str(tmp_path / "usage.db"))
    tracker.log_call(
        session_id="sess-123",
        tool_name="omega_store",
        model="claude-opus-4-6",
        input_tokens=1000,
        output_tokens=500,
        project="test",
    )
    usage = tracker.get_usage(days=1, group_by="model")
    assert len(usage) == 1
    assert usage[0]["model"] == "claude-opus-4-6"
    assert usage[0]["total_input_tokens"] == 1000
    assert usage[0]["total_output_tokens"] == 500
    tracker.close()


def test_cost_estimation(tmp_path):
    from omega.usage_tracker import UsageTracker

    tracker = UsageTracker(db_path=str(tmp_path / "usage.db"))
    tracker.log_call(
        session_id="sess-123",
        tool_name="omega_query",
        model="claude-opus-4-6",
        input_tokens=1_000_000,
        output_tokens=100_000,
    )
    cost = tracker.get_cost_estimate(days=30)
    # Opus: 15/M input + 75/M output = $15 + $7.50 = $22.50
    assert cost["total_usd"] > 20
    assert cost["total_usd"] < 25
    tracker.close()


def test_top_tools(tmp_path):
    from omega.usage_tracker import UsageTracker

    tracker = UsageTracker(db_path=str(tmp_path / "usage.db"))
    for i in range(5):
        tracker.log_call("s1", "omega_store", "claude-sonnet-4-6", 100, 50)
    for i in range(2):
        tracker.log_call("s1", "omega_query", "claude-sonnet-4-6", 200, 100)

    top = tracker.get_top_tools(days=1, limit=5)
    assert top[0]["tool_name"] == "omega_store"
    assert top[0]["call_count"] == 5
    tracker.close()


def test_local_embedding_zero_cost(tmp_path):
    from omega.usage_tracker import UsageTracker

    tracker = UsageTracker(db_path=str(tmp_path / "usage.db"))
    tracker.log_call("s1", "embed", "nomic-embed-text", 5000, 0)
    cost = tracker.get_cost_estimate(days=1)
    assert cost["total_usd"] == 0.0
    tracker.close()


def test_usage_windows_exclude_calls_older_than_the_window(tmp_path):
    """Windows compare stored isoformat text with a cutoff in the same format.

    Comparing it with SQLite datetime() text sorted 'T' after ' ', so every
    call from the cutoff's calendar day counted as inside the window.
    """
    from datetime import datetime, timedelta, timezone

    from omega.usage_tracker import UsageTracker

    tracker = UsageTracker(db_path=str(tmp_path / "usage.db"))
    tracker.log_call("s1", "omega_store", "claude-sonnet-4-6", 100, 50)
    tracker.log_call("s1", "omega_store", "claude-sonnet-4-6", 100, 50)
    window_start = datetime.now(timezone.utc) - timedelta(days=7)
    tracker._conn.executemany(
        "UPDATE llm_usage SET created_at = ? WHERE id = ?",
        [
            ((window_start - timedelta(seconds=1)).isoformat(), 1),
            ((window_start + timedelta(hours=1)).isoformat(), 2),
        ],
    )
    tracker._conn.commit()

    assert tracker.get_usage(days=7)[0]["call_count"] == 1
    assert tracker.get_cost_estimate(days=7)["total_calls"] == 1
    assert tracker.get_top_tools(days=7)[0]["call_count"] == 1
    tracker.close()
