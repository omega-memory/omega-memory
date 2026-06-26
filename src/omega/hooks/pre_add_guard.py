#!/usr/bin/env python3
"""OMEGA PreToolUse hook — Git add guard for multi-agent coordination.

BLOCKS `git add .`, `git add -A`, and `git commit -a` unconditionally (exit code 2).
BLOCKS staging files the agent didn't edit (not in own claim list) when coordination is active.
WARNS in solo mode when staging unclaimed files.

Prevents the root cause of mixed commits: broad staging captures pre-existing
dirty worktree changes from other agents or prior sessions.
"""
import json
import os
import re
import shlex
import time
import traceback
from datetime import datetime
from pathlib import Path


def _log_hook_error(hook_name, error):
    try:
        log_path = Path.home() / ".omega" / "hooks.log"
        log_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        timestamp = datetime.now().isoformat(timespec="seconds")
        tb = traceback.format_exc()
        data = f"[{timestamp}] {hook_name}: {error}\n{tb}\n"
        fd = os.open(str(log_path), os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        try:
            os.write(fd, data.encode("utf-8"))
        finally:
            os.close(fd)
    except Exception:
        pass


def _log_timing(hook_name, elapsed_ms):
    try:
        log_path = Path.home() / ".omega" / "hooks.log"
        log_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        timestamp = datetime.now().isoformat(timespec="seconds")
        data = f"[{timestamp}] {hook_name}: OK ({elapsed_ms:.0f}ms)\n"
        fd = os.open(str(log_path), os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        try:
            os.write(fd, data.encode("utf-8"))
        finally:
            os.close(fd)
    except Exception:
        pass


# Options to `git commit` whose following token is a value that may itself
# contain dashes (messages, paths, authors, dates, refs). Their value token
# must be skipped so it is never mistaken for a `-a` flag.
_COMMIT_VALUE_OPTS = {
    "-m", "--message", "-F", "--file", "-C", "--reuse-message",
    "-c", "--reedit-message", "-t", "--template",
    "--author", "--date", "--fixup", "--squash",
}


def _commit_has_all_flag(command):
    """True if *command* contains a `git commit` carrying an -a/--all flag.

    Tokenizes with shlex so dash-text inside quoted messages or paths
    (e.g. -m "fix the-hard-way", -F some/path) is never mistaken for a
    `-a` flag — that false positive is why ordinary commits got blocked.
    Falls back to a flag-anchored regex if quotes are unbalanced.
    """
    try:
        tokens = shlex.split(command, posix=True)
    except ValueError:
        # Unbalanced quotes: match -a/--all only as a flag token, never as
        # arbitrary -...a text embedded inside a word.
        return bool(re.search(
            r"\bgit\s+commit\b(?:\s+-(?!-)[a-zA-Z]*a[a-zA-Z]*|\s+--all)(?=\s|$)",
            command,
        ))

    i, n = 0, len(tokens)
    while i < n:
        if tokens[i] == "git" and i + 1 < n and tokens[i + 1] == "commit":
            j = i + 2
            while j < n:
                tok = tokens[j]
                if tok in (";", "&", "&&", "|", "||"):
                    break  # start of a chained command
                if tok == "--all":
                    return True
                if tok == "--":
                    break  # end of options; remainder are pathspecs
                if tok.startswith("--"):
                    if tok.split("=", 1)[0] in _COMMIT_VALUE_OPTS and "=" not in tok:
                        j += 1  # consume value token
                elif tok.startswith("-") and len(tok) > 1:
                    if "a" in tok[1:]:
                        return True  # -a / -am / -va etc.
                    if tok in _COMMIT_VALUE_OPTS:
                        j += 1  # consume value token
                # else: non-option token (subject / pathspec) — ignore
                j += 1
            i = j
            continue
        i += 1
    return False


def _is_broad_add(command):
    """Detect git add . / git add -A / git add --all / git commit -a patterns."""
    # git add . or git add -A or git add --all
    if re.search(r"\bgit\s+add\s+(\.\s*$|-A\b|--all\b)", command):
        return True
    # git commit -a / git commit -am
    if _commit_has_all_flag(command):
        return True
    return False


def _extract_add_paths(command):
    """Extract file paths from a git add command. Returns list of paths or None if not a git add."""
    m = re.search(r"\bgit\s+add\s+(.*)", command)
    if not m:
        return None
    args_str = m.group(1).strip()
    # Filter out flags
    paths = []
    for token in args_str.split():
        if token.startswith("-"):
            continue
        paths.append(token)
    return paths


def main():
    tool_name = os.environ.get("TOOL_NAME", "")
    tool_input = os.environ.get("TOOL_INPUT", "{}")

    if tool_name != "Bash":
        return

    try:
        input_data = json.loads(tool_input)
    except (json.JSONDecodeError, TypeError):
        return

    command = input_data.get("command", "")

    # Only trigger on git add or git commit -a
    is_git_add = re.search(r"\bgit\s+add\b", command)
    is_commit_a = _commit_has_all_flag(command)
    if not is_git_add and not is_commit_a:
        return

    session_id = os.environ.get("SESSION_ID", "")
    project = os.environ.get("PROJECT_DIR", os.getcwd())

    # --- BLOCK broad staging unconditionally ---
    if _is_broad_add(command):
        lines = [
            "[ADD-GUARD] BLOCKED: broad staging command detected.",
            f"  Command: {command[:120]}",
            "",
            "Never use `git add .`, `git add -A`, or `git commit -a`.",
            "Stage specific files by name: git add <file1> <file2> ...",
        ]

        # Try to suggest the agent's own claimed files
        if session_id:
            try:
                from omega.coordination import get_manager
                mgr = get_manager()
                own_claims = mgr.get_session_claims(session_id)
                own_files = own_claims.get("file_claims", [])
                if own_files:
                    # Convert absolute paths to relative
                    rel_files = []
                    for f in own_files:
                        if project and f.startswith(project):
                            rel_files.append(os.path.relpath(f, project))
                        else:
                            rel_files.append(f)
                    lines.append("")
                    lines.append(f"Your claimed files ({len(rel_files)}):")
                    for rf in sorted(rel_files)[:20]:
                        lines.append(f"  {rf}")
                    if len(rel_files) > 20:
                        lines.append(f"  +{len(rel_files) - 20} more")
            except Exception:
                pass

        print("\n".join(lines))
        exit(2)

    # --- For specific git add <files>, check against claims ---
    add_paths = _extract_add_paths(command)
    if not add_paths:
        return  # Not a parseable git add (maybe piped or complex)

    if not session_id:
        return  # No session tracking, can't check claims

    try:
        from omega.coordination import get_manager
        mgr = get_manager()

        own_claims = mgr.get_session_claims(session_id)
        own_files = own_claims.get("file_claims", [])
        if not own_files:
            return  # No claims tracked, can't validate

        sessions = mgr.list_sessions(auto_clean=True)
        peers = [s for s in sessions if s.get("session_id") != session_id]
        has_peers = len(peers) > 0

        # Resolve add paths to check against claims
        unclaimed = []
        for path in add_paths:
            # Expand globs/directories via git ls-files if it's a directory
            import subprocess
            try:
                result = subprocess.run(
                    ["git", "diff", "--name-only", "--", path],
                    capture_output=True, text=True, timeout=5, cwd=project,
                )
                resolved_files = [f.strip() for f in result.stdout.strip().split("\n") if f.strip()]
            except Exception:
                resolved_files = [path]

            for rf in resolved_files:
                abs_path = os.path.join(project, rf) if not os.path.isabs(rf) else rf
                if abs_path not in own_files and rf not in own_files:
                    unclaimed.append(rf)

        if unclaimed:
            if has_peers:
                # BLOCK in multi-agent mode
                lines = [
                    f"[ADD-GUARD] BLOCKED: staging {len(unclaimed)} file(s) you didn't edit:",
                ]
                for f in unclaimed[:10]:
                    lines.append(f"  {f}")
                if len(unclaimed) > 10:
                    lines.append(f"  +{len(unclaimed) - 10} more")
                lines.append("")
                lines.append("These files have pre-existing changes from another agent or prior session.")
                lines.append("Stage only files you modified. Edit/Write auto-claims files for you.")

                mgr.record_metric(
                    "gate_blocked",
                    session_id=session_id,
                    metadata={"action": "add_unclaimed", "unclaimed_count": len(unclaimed)},
                )
                print("\n".join(lines))
                exit(2)
            else:
                # WARN in solo mode
                lines = [
                    f"[ADD-GUARD] WARNING: staging {len(unclaimed)} file(s) not in your claim list:",
                ]
                for f in unclaimed[:10]:
                    lines.append(f"  {f}")
                if len(unclaimed) > 10:
                    lines.append(f"  +{len(unclaimed) - 10} more")
                lines.append("")
                lines.append("Did you author these changes? If not, unstage with: git reset HEAD <file>")
                print("\n".join(lines))
                # Don't block in solo mode, just warn

    except ImportError:
        pass  # Coordination module not available
    except Exception as e:
        _log_hook_error("pre_add_guard", e)


if __name__ == "__main__":
    _t0 = time.monotonic()
    main()
    _log_timing("pre_add_guard", (time.monotonic() - _t0) * 1000)
