"""Core hook handlers: the five hooks every omega-memory install registers.

Each handler runs the corresponding ``omega.hooks`` module in-process, with
its output captured instead of printed, and returns the daemon's response
shape ``{"output": str, "error": str | None}``. The hook modules are the
single implementation; the standalone fallback path runs the same code in a
fresh interpreter.
"""

from __future__ import annotations

import logging
from collections.abc import Callable

from omega.hooks import assistant_capture, auto_capture, session_start, session_stop, surface_memories
from omega.hooks._output import capture, captured_text

from . import _debounce_state, _last_surface, SURFACE_DEBOUNCE_S, _MAX_SURFACE_ENTRIES
from .utils import _debounce_check, _get_file_path_from_input, _log_hook_error, _parse_tool_input

logger = logging.getLogger("omega.hook_server")

_FILE_TOOLS = frozenset({"Edit", "Write", "NotebookEdit", "Read"})


def _run_hook(hook_name: str, run: Callable[[dict], None], payload: dict) -> dict:
    """Run one hook module in-process and package its output for the client.

    A failing hook never fails the client: whatever it emitted before the error
    is still returned, and the error is logged with its session for diagnosis.
    """
    with capture() as lines:
        try:
            run(payload)
        except Exception as error:
            _log_hook_error(hook_name, error, session_id=payload.get("session_id", ""))
            return {"output": captured_text(lines), "error": str(error)}
    return {"output": captured_text(lines), "error": None}


def handle_session_start(payload: dict) -> dict:
    """SessionStart: periodic maintenance plus the welcome briefing."""
    return _run_hook("session_start", session_start.run, payload)


def handle_session_stop(payload: dict) -> dict:
    """Stop: activity report and session summary, then release per-session state."""
    response = _run_hook("session_stop", session_stop.run, payload)
    session_id = payload.get("session_id", "")
    if session_id:
        _debounce_state.cleanup(session_id)
    return response


def handle_surface_memories(payload: dict) -> dict:
    """PostToolUse: surface memories for the touched file, capture Bash errors.

    The same file is often touched several times in a row (Read then Edit,
    repeated Edits). Within ``SURFACE_DEBOUNCE_S`` the daemon answers from the
    debounce cache instead of re-querying the store.
    """
    if payload.get("tool_name") in _FILE_TOOLS:
        file_path = _get_file_path_from_input(_parse_tool_input(payload))
        if file_path and not _debounce_check(_last_surface, file_path, SURFACE_DEBOUNCE_S, _MAX_SURFACE_ENTRIES):
            return {"output": "", "error": None}
    return _run_hook("surface_memories", surface_memories.run, payload)


def handle_auto_capture(payload: dict) -> dict:
    """UserPromptSubmit: store decisions and lessons stated in the prompt."""
    return _run_hook("auto_capture", auto_capture.run, payload)


def handle_assistant_capture(payload: dict) -> dict:
    """Stop: store fixes, decisions, and lessons stated in the assistant's reply."""
    return _run_hook("assistant_capture", assistant_capture.run, payload)
