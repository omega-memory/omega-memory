#!/usr/bin/env python3
"""OMEGA fast hook client — routes to daemon via UDS, falls back to direct.

This is a thin client that connects to the hook server running inside the
MCP process. It avoids importing any OMEGA modules on the fast path, keeping
startup to ~50ms (Python interpreter only).

If the daemon socket is unavailable (MCP not started yet, or crashed),
falls back to running the original hook script directly.
"""
import json
import os
import socket
import sys
import time

# Windows uses TCP loopback; Unix uses domain socket
if sys.platform == "win32":
    SOCK_PATH = None
    HOOK_HOST = "127.0.0.1"
    HOOK_PORT = 19876
else:
    SOCK_PATH = os.path.expanduser("~/.omega/hook.sock")
    HOOK_HOST = None
    HOOK_PORT = None

# Written by the hook server that owns the socket (hook_server/owner_state.py).
OWNER_STATE_PATH = os.path.expanduser("~/.omega/hook.sock.owner.json")

# Map hook names to their original script modules for fallback
_FALLBACK_SCRIPTS = {
    "session_start": "session_start",
    "session_stop": "session_stop",
    "surface_memories": "surface_memories",
    "auto_capture": "auto_capture",
    "assistant_capture": "assistant_capture",
    "coord_session_start": "coord_session_start",
    "coord_session_stop": "coord_session_stop",
    "coord_heartbeat": "coord_heartbeat",
    "auto_claim_file": "auto_claim_file",
    "pre_add_guard": "pre_add_guard",
    "pre_file_guard": "pre_file_guard",
    "pre_task_guard": "pre_task_guard",
    "pre_push_guard": "pre_push_guard",
    "pre_deploy_guard": "pre_deploy_guard",
    "pre_commit_guard": "pre_commit_guard",
    "pre_protocol_gate": "pre_protocol_gate",
    "pre_alignment_gate": "pre_alignment_gate",
    "trace_capture": "trace_capture",
}

# Hooks that require longer timeouts (e.g., git network operations)
_SLOW_HOOKS = {"pre_push_guard"}

# Safety-critical hooks that MUST run even in fallback mode.
# These are pre-action guards that can block dangerous operations (exit code 2).
# All other hooks are informational and safe to skip if daemon is unavailable.
_BLOCKING_HOOKS = {
    "pre_add_guard", "pre_file_guard", "pre_task_guard", "pre_push_guard",
    "pre_deploy_guard", "pre_commit_guard", "pre_alignment_gate",
    "pre_protocol_gate",
}

# Best-effort hooks run in fallback mode but never block the session.
# These capture high-value content that would otherwise be silently dropped.
_BEST_EFFORT_HOOKS = {
    "assistant_capture",
    "coord_session_stop",
    "coord_heartbeat",       # heartbeat must reach Supabase even without daemon
    "coord_session_start",   # belt-and-suspenders: omega_welcome also registers,
                             # but fallback covers agents that skip omega_welcome
    "trace_capture",         # captures content that would otherwise be lost
}

# How long to wait on the daemon before giving up. hooks-core.json's timeouts
# (seconds, read by Claude Code) sit above this plus the retry window, so
# Claude Code never cancels a hook this client would still wait for.
_DAEMON_TIMEOUT_S = 5.0
_SLOW_DAEMON_TIMEOUT_S = 20.0

# Retry settings for startup race (hook fires before MCP server opens socket)
_CONNECT_RETRIES = 4
_CONNECT_RETRY_DELAY = 0.5  # seconds between retries


def _detect_client() -> str:
    """Detect which AI coding client invoked this hook.

    Priority: explicit env var > heuristic detection > "unknown".
    The env var OMEGA_CLIENT is set by `omega setup --client <name>`.
    """
    client = os.environ.get("OMEGA_CLIENT", "")
    if client:
        return client
    # Heuristic: check for known client config paths
    if os.path.exists(os.path.expanduser("~/.claude/settings.json")):
        return "claude-code"
    if os.path.exists(os.path.expanduser("~/.cursor/settings.json")):
        return "cursor"
    return "unknown"


def delegate(hook_names, payload, timeout=5.0):
    """Connect to daemon, send request, return parsed response.

    Accepts a single hook name (str) or multiple (list) for batching.
    Batch requests use {"hooks": [...]} and return {"results": [...]}.
    """
    if sys.platform == "win32":
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(timeout)
        s.connect((HOOK_HOST, HOOK_PORT))
    else:
        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        s.settimeout(timeout)
        s.connect(SOCK_PATH)
    try:
        if isinstance(hook_names, list):
            request = json.dumps({"hooks": hook_names, **payload}).encode("utf-8")
        else:
            request = json.dumps({"hook": hook_names, **payload}).encode("utf-8")
        s.sendall(request)
        s.shutdown(socket.SHUT_WR)

        response = b""
        while True:
            chunk = s.recv(8192)
            if not chunk:
                break
            response += chunk
        return json.loads(response.decode("utf-8"))
    finally:
        s.close()


def _socket_owner_alive():
    """True while the process that owns the hook socket is still running.

    A refused connection means nobody is listening on the socket: either a
    server is still starting (retrying helps) or the server that created the
    file died without removing it (retrying only adds seconds to every hook).
    The owner record tells the two apart. No record means no owner.
    """
    try:
        with open(OWNER_STATE_PATH, encoding="utf-8") as f:
            pid = int(json.load(f)["pid"])
    except (OSError, ValueError, KeyError, TypeError):
        return False
    if sys.platform == "win32":
        return True  # os.kill(pid, 0) terminates the process on Windows
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # alive, owned by another user
    return True


def _delegate_with_retries(hook_names, payload, timeout):
    """Send the request to the daemon; return its response, or None if none is reachable.

    Retries a refused connection only while the socket's owner is alive (the
    startup race). A missing socket, a slow daemon, or an owner that has
    exited returns None at once.
    """
    for attempt in range(_CONNECT_RETRIES + 1):
        try:
            return delegate(hook_names, payload, timeout=timeout)
        except socket.timeout:
            return None  # daemon exists but slow — don't retry, fall through
        except FileNotFoundError:
            return None  # socket file missing — daemon not started
        except OSError:
            # ConnectionRefusedError and friends: nobody is accepting.
            if attempt == _CONNECT_RETRIES or not _socket_owner_alive():
                return None
            time.sleep(_CONNECT_RETRY_DELAY)
    return None


def _fallback(hook_name, payload):
    """Run the original hook script directly (cold path).

    Sets env vars from *payload* before calling mod.main() so that
    individual hook scripts (which read os.environ) see the values
    that Claude Code passed via stdin JSON.
    """
    hooks_dir = os.path.dirname(os.path.abspath(__file__))
    script_name = _FALLBACK_SCRIPTS.get(hook_name)
    if not script_name:
        if hook_name in _BLOCKING_HOOKS:
            print(f"OMEGA: blocking hook '{hook_name}' has no fallback script — daemon-only", file=sys.stderr)
        return

    script_path = os.path.join(hooks_dir, f"{script_name}.py")
    if not os.path.exists(script_path):
        return

    # Bridge: set env vars from payload so hook scripts can read them.
    # Claude Code stdin JSON uses different field names than env vars:
    #   stdin: session_id, tool_name, tool_input (dict), tool_response, cwd
    #   env:   SESSION_ID, TOOL_NAME, TOOL_INPUT (str),  TOOL_OUTPUT,  PROJECT_DIR
    _ENV_MAP = {
        "session_id": "SESSION_ID",
        "tool_name": "TOOL_NAME",
        "tool_input": "TOOL_INPUT",
        "tool_response": "TOOL_OUTPUT",  # Claude Code calls it tool_response
        "tool_output": "TOOL_OUTPUT",    # legacy/internal name
        "cwd": "PROJECT_DIR",
        "project": "PROJECT_DIR",        # internal name used by some hooks
    }
    for payload_key, env_key in _ENV_MAP.items():
        val = payload.get(payload_key)
        if val:
            # tool_input/tool_response may be dicts from JSON parse — serialize
            if isinstance(val, (dict, list)):
                os.environ[env_key] = json.dumps(val)
            else:
                os.environ[env_key] = str(val)

    # Add hooks dir to path so the script can be imported
    if hooks_dir not in sys.path:
        sys.path.insert(0, hooks_dir)

    import importlib
    try:
        mod = importlib.import_module(script_name)
        if hasattr(mod, "main"):
            import inspect
            sig = inspect.signature(mod.main)
            if sig.parameters:
                mod.main(payload)
            else:
                mod.main()
    except Exception as e:
        print(f"OMEGA hook fallback error ({hook_name}): {e}", file=sys.stderr)


def _log_timing(hook_name, elapsed_ms, mode):
    """Log hook timing to ~/.omega/hooks.log."""
    try:
        from datetime import datetime
        from pathlib import Path
        log_path = Path.home() / ".omega" / "hooks.log"
        log_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        timestamp = datetime.now().isoformat(timespec="seconds")
        data = f"[{timestamp}] fast_hook/{hook_name}: OK ({elapsed_ms:.0f}ms, {mode})\n"
        fd = os.open(str(log_path), os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        try:
            os.write(fd, data.encode("utf-8"))
        finally:
            os.close(fd)
    except Exception:
        pass


def _parse_payload():
    """Build payload from env vars + stdin JSON."""
    payload = {
        "tool_name": os.environ.get("TOOL_NAME", ""),
        "tool_input": os.environ.get("TOOL_INPUT", "{}"),
        "tool_output": os.environ.get("TOOL_OUTPUT", ""),
        "session_id": os.environ.get("SESSION_ID", ""),
        "project": os.environ.get("PROJECT_DIR", os.getcwd()),
        "client": _detect_client(),
    }

    # Claude Code sends hook data as JSON on stdin for ALL hook types.
    if not sys.stdin.isatty():
        try:
            raw = sys.stdin.read()
            if raw.strip():
                try:
                    stdin_data = json.loads(raw)
                    if isinstance(stdin_data, dict):
                        if "tool_response" in stdin_data and "tool_output" not in stdin_data:
                            stdin_data["tool_output"] = stdin_data["tool_response"]
                        if "cwd" in stdin_data and "project" not in stdin_data:
                            stdin_data["project"] = stdin_data["cwd"]
                        if isinstance(stdin_data.get("tool_input"), (dict, list)):
                            stdin_data["tool_input"] = json.dumps(stdin_data["tool_input"])
                        if isinstance(stdin_data.get("tool_output"), (dict, list)):
                            stdin_data["tool_output"] = json.dumps(stdin_data["tool_output"])
                        for key, val in stdin_data.items():
                            if val or not payload.get(key):
                                payload[key] = val
                except json.JSONDecodeError:
                    payload["stdin"] = raw
        except Exception:
            pass

    return payload


def main():
    if len(sys.argv) < 2:
        print("Usage: fast_hook.py <hook_name[+hook_name...]>", file=sys.stderr)
        sys.exit(1)

    t0 = time.monotonic()
    hook_names = sys.argv[1].split("+")
    payload = _parse_payload()
    is_batch = len(hook_names) > 1

    # Use longer timeout for hooks with network operations (e.g., git fetch)
    timeout = _SLOW_DAEMON_TIMEOUT_S if _SLOW_HOOKS.intersection(hook_names) else _DAEMON_TIMEOUT_S

    result = _delegate_with_retries(hook_names if is_batch else hook_names[0], payload, timeout)

    elapsed_ms = (time.monotonic() - t0) * 1000

    if result is not None:
        # Daemon responded — process result
        if is_batch:
            outputs = []
            blocking_outputs = []
            exit_code = 0
            for r in result.get("results", []):
                if r.get("output"):
                    outputs.append(r["output"])
                    if r.get("exit_code"):
                        blocking_outputs.append(r["output"])
                if r.get("exit_code") and not exit_code:
                    exit_code = r["exit_code"]
            if outputs:
                print("\n".join(outputs))
            if exit_code and blocking_outputs:
                print("\n".join(blocking_outputs), file=sys.stderr)
            _log_timing("+".join(hook_names), elapsed_ms, "daemon")
            if exit_code:
                sys.exit(exit_code)
        else:
            if result.get("output"):
                print(result["output"])
                if result.get("exit_code"):
                    print(result["output"], file=sys.stderr)
            _log_timing(hook_names[0], elapsed_ms, "daemon")
            exit_code = result.get("exit_code")
            if exit_code:
                sys.exit(exit_code)
    else:
        # Daemon unavailable after retries.
        # Run fallback for safety-critical blocking hooks (pre_* guards)
        # and best-effort hooks (high-value captures that shouldn't be dropped).
        # Skip purely informational hooks to prevent the fallback stampede where
        # concurrent Python processes starve each other on CPU + SQLite locks.
        blocking = [h for h in hook_names if h in _BLOCKING_HOOKS]
        best_effort = [h for h in hook_names if h in _BEST_EFFORT_HOOKS]
        if blocking:
            for name in blocking:
                _fallback(name, payload)
        if best_effort:
            for name in best_effort:
                try:
                    _fallback(name, payload)
                except Exception:
                    pass  # Never block session for best-effort hooks
        if blocking or best_effort:
            elapsed_ms = (time.monotonic() - t0) * 1000
            _log_timing("+".join(hook_names), elapsed_ms, "fallback")
        else:
            _log_timing("+".join(hook_names), elapsed_ms, "skipped")


if __name__ == "__main__":
    main()
