"""Test session stop utilization report."""
import pytest


def test_utilization_report_flags_missing_tools():
    """When agent never called omega_reflect or omega_decision_query,
    the report should flag them as unused."""
    from omega.hooks.session_stop import _build_utilization_report

    # Simulate a session that called some tools but skipped critical ones
    tool_calls = [
        "omega_welcome", "omega_protocol", "omega_query", "omega_store",
        "Read", "Edit", "Bash", "omega_query", "omega_store",
    ]
    report = _build_utilization_report(tool_calls)

    assert "omega_reflect" in report["missed"]
    assert "omega_decision_query" in report["missed"]
    assert report["score"] < 100  # Not a perfect score


def test_utilization_report_perfect_score():
    """When all critical tools were called, score is 100."""
    from omega.hooks.session_stop import _build_utilization_report

    tool_calls = [
        "omega_welcome", "omega_protocol", "omega_query", "omega_store",
        "omega_reflect", "omega_decision_query", "omega_file_check",
        "omega_checkpoint", "omega_coord_status",
    ]
    report = _build_utilization_report(tool_calls)

    assert len(report["missed"]) == 0
    assert report["score"] == 100


def test_utilization_report_empty_session():
    """An empty session should flag all critical tools."""
    from omega.hooks.session_stop import _build_utilization_report

    report = _build_utilization_report([])
    assert report["score"] == 0
    assert len(report["missed"]) > 0
