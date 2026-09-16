"""Behavioral analysis must be a quiet no-op on a Core install (#80)."""
import logging


def test_behavioral_analysis_is_a_quiet_no_op_without_a_coordination_db(monkeypatch, caplog):
    from omega.behavioral import BehavioralAnalyzer

    monkeypatch.setattr(BehavioralAnalyzer, "_get_conn", lambda self: None)

    with caplog.at_level(logging.WARNING, logger="omega.behavioral"):
        result = BehavioralAnalyzer().analyze_and_store()

    assert result["stored"] == 0
    assert [r for r in caplog.records if r.levelno >= logging.WARNING] == []
