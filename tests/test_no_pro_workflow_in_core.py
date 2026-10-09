"""Ratchet Core's remaining explicit Pro imports toward zero."""

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

# This allowlist must only shrink; do not add new Core importers of omega_platform.
ALLOWLIST = frozenset({
    "src/omega/behavioral.py",
    "src/omega/bridge.py",
    "src/omega/cli.py",
    "src/omega/council.py",
    "src/omega/hooks/pre_add_guard.py",
    "src/omega/hooks/pre_alignment_gate.py",
    "src/omega/hooks/pre_commit_guard.py",
    "src/omega/hooks/pre_edit_surface.py",
    "src/omega/hooks/pre_file_guard.py",
    "src/omega/hooks/pre_push_guard.py",
    "src/omega/hooks/pre_task_guard.py",
    "src/omega/hooks/session_stop.py",
    "src/omega/hooks/surface_memories.py",
    "src/omega/server/handlers.py",
    "src/omega/server/mcp_server.py",
    "src/omega/sqlite_store/_maintenance.py",
    "src/omega/sqlite_store/_query.py",
})


def _imports_platform(source: str) -> bool:
    return any(
        isinstance(node, ast.Import) and any(alias.name.startswith("omega_platform") for alias in node.names)
        or isinstance(node, ast.ImportFrom) and (node.module or "").startswith("omega_platform")
        for node in ast.walk(ast.parse(source))
    )


# Hook scripts Core ships. This list must only shrink: a session workflow that
# belongs to an extension is served through the plugin HOOK_HANDLERS contract.
SHIPPED_HOOK_SCRIPTS = frozenset({
    "__init__.py",
    "_output.py",
    "assistant_capture.py",
    "auto_capture.py",
    "auto_claim_file.py",
    "coord_heartbeat.py",
    "fast_hook.py",
    "post_edit_test.py",
    "pre_add_guard.py",
    "pre_alignment_gate.py",
    "pre_commit_guard.py",
    "pre_deploy_guard.py",
    "pre_edit_surface.py",
    "pre_file_guard.py",
    "pre_protocol_gate.py",
    "pre_push_guard.py",
    "pre_task_guard.py",
    "session_start.py",
    "session_stop.py",
    "surface_memories.py",
    "trace_capture.py",
    "track_file_read.py",
})


def test_core_ships_no_hook_script_outside_the_shrinking_list():
    shipped = {path.name for path in (ROOT / "src/omega/hooks").glob("*.py")}
    assert shipped <= SHIPPED_HOOK_SCRIPTS


def test_fast_hook_falls_back_only_to_scripts_core_ships():
    from omega.hooks import fast_hook

    scripts = {f"{name}.py" for name in fast_hook._FALLBACK_SCRIPTS.values()}
    assert scripts <= SHIPPED_HOOK_SCRIPTS


def test_large_core_platform_import_allowlist_only_shrinks():
    importers = {
        path.relative_to(ROOT).as_posix()
        for path in (ROOT / "src/omega").rglob("*.py")
        if len((source := path.read_text()).splitlines()) > 80 and _imports_platform(source)
    }
    assert importers <= ALLOWLIST
