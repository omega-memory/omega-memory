"""Test mid-session utilization nudges."""


def test_nudge_after_many_edits_without_file_check():
    from omega.hooks.surface_memories import _check_nudge

    # 15 edits, 0 file_checks => should nudge
    nudge = _check_nudge(edit_count=15, tool_calls=["Edit"] * 15)
    assert nudge is not None
    assert "omega_file_check" in nudge


def test_no_nudge_when_file_check_called():
    from omega.hooks.surface_memories import _check_nudge

    calls = ["Edit"] * 15 + ["mcp__omega-memory__omega_file_check"]
    nudge = _check_nudge(edit_count=15, tool_calls=calls)
    assert nudge is None


def test_nudge_reflect_after_30_tool_calls():
    from omega.hooks.surface_memories import _check_nudge

    calls = ["Bash"] * 35
    nudge = _check_nudge(edit_count=0, tool_calls=calls)
    assert nudge is not None
    assert "omega_reflect" in nudge
