"""OMEGA CLI — Memory commands, setup, status, migration, and server management."""

import argparse
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

logger = logging.getLogger("omega.cli")


def _use_json(args) -> bool:
    """Check if JSON output requested via --json flag or OMEGA_JSON=1 env var."""
    return getattr(args, "json", False) or os.environ.get("OMEGA_JSON") == "1"


OMEGA_DIR = Path.home() / ".omega"
OMEGA_CACHE = Path.home() / ".cache" / "omega"
MAGMA_DIR = Path.home() / ".magma"
MAGMA_GRAPHS = Path.home() / ".claude" / "magma"
BGE_MODEL_DIR = OMEGA_CACHE / "models" / "bge-small-en-v1.5-onnx"
MINILM_MODEL_DIR = OMEGA_CACHE / "models" / "all-MiniLM-L6-v2-onnx"
# Primary model dir — bge-small-en-v1.5, falls back to all-MiniLM-L6-v2
ONNX_MODEL_DIR = BGE_MODEL_DIR


CLAUDE_MD_PATH = Path.home() / ".claude" / "CLAUDE.md"
SETTINGS_JSON_PATH = Path.home() / ".claude" / "settings.json"
DATA_DIR = Path(__file__).parent / "data"

OMEGA_BEGIN = "<!-- OMEGA:BEGIN"
OMEGA_END = "<!-- OMEGA:END -->"


def _python_has_omega(python_path: str) -> bool:
    """Check if a Python interpreter has omega installed."""
    try:
        result = subprocess.run(
            [python_path, "-c", "import omega; import mcp"],
            capture_output=True, timeout=5,
        )
        return result.returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


def _resolve_python_path() -> str:
    """Resolve the best Python interpreter path for hooks and MCP configs.

    Priority: first interpreter that can import omega wins.
    1. sys.executable (even if inside a venv -- that's where omega lives)
    2. 'python3' from PATH
    3. /opt/homebrew/bin/python3 (macOS Homebrew fallback)
    4. sys.executable as-is (best effort)
    """
    candidates = []

    exe = sys.executable
    if exe and Path(exe).exists():
        candidates.append(exe)

    which_py = shutil.which("python3")
    if which_py and which_py not in candidates:
        candidates.append(which_py)

    fallback = "/opt/homebrew/bin/python3"
    if Path(fallback).exists() and fallback not in candidates:
        candidates.append(fallback)

    for candidate in candidates:
        if _python_has_omega(candidate):
            return candidate

    # No candidate has omega -- return sys.executable as best effort
    return exe or "python3"


def _inject_claude_md(*, dry_run: bool = False):
    """Inject or update the OMEGA block in ~/.claude/CLAUDE.md (idempotent).

    Selects Pro or Core fragment based on available modules. Never overwrites
    user content — only touches the managed OMEGA block between markers.

    Args:
        dry_run: If True, print what would change without writing.
    """
    # Select tier-appropriate fragment
    if _has_commercial_modules():
        fragment_file = DATA_DIR / "claude-md-fragment-pro.md"
    else:
        fragment_file = DATA_DIR / "claude-md-fragment.md"
    fragment = fragment_file.read_text()

    if CLAUDE_MD_PATH.exists():
        content = CLAUDE_MD_PATH.read_text()
    else:
        CLAUDE_MD_PATH.parent.mkdir(parents=True, exist_ok=True)
        content = ""

    if OMEGA_BEGIN in content:
        # Replace existing block (upgrade path)
        pattern = re.compile(
            r"<!-- OMEGA:BEGIN[^\n]*-->.*?<!-- OMEGA:END -->",
            re.DOTALL,
        )
        new_content = pattern.sub(fragment.rstrip(), content)
        if new_content == content:
            print("  CLAUDE.md: OMEGA block already up to date")
            return
        if dry_run:
            print("  CLAUDE.md: would update OMEGA block (dry-run)")
            return
        CLAUDE_MD_PATH.write_text(new_content)
        print("  CLAUDE.md: OMEGA block updated")
    else:
        # First time — back up existing file if it has content
        if content.strip():
            backup_path = CLAUDE_MD_PATH.with_suffix(".md.pre-omega")
            if not backup_path.exists():
                if dry_run:
                    print(f"  CLAUDE.md: would back up to {backup_path.name} (dry-run)")
                    print("  CLAUDE.md: would append OMEGA block (dry-run)")
                    return
                backup_path.write_text(content)
                print(f"  CLAUDE.md: backed up existing file to {backup_path.name}")
        elif dry_run:
            print("  CLAUDE.md: would create with OMEGA block (dry-run)")
            return
        separator = "\n" if content and not content.endswith("\n") else ""
        CLAUDE_MD_PATH.write_text(content + separator + fragment)
        print("  CLAUDE.md: OMEGA block appended")


def _has_commercial_modules() -> bool:
    """Check if commercial/coordination modules are available."""
    try:
        import omega.coordination  # noqa: F401

        return True
    except ImportError:
        pass
    try:
        from omega.plugins import discover_plugins

        for plugin in discover_plugins():
            if plugin.HOOKS_JSON:
                return True
    except Exception as e:
        logger.debug("Plugin hooks check failed: %s", e)
    return False


def _inject_settings_hooks(hooks_src: Path):
    """Inject OMEGA hook entries into ~/.claude/settings.json (idempotent).

    Uses hooks-core.json for core-only installs, or hooks.json (full) when
    commercial modules are available. Supports both old format (single dict
    per event) and new format (list of dicts per event) in hooks.json manifest.
    """
    if _has_commercial_modules():
        hooks_file = "hooks.json"
    else:
        hooks_file = "hooks-core.json"
    manifest = json.loads((DATA_DIR / hooks_file).read_text())

    # Determine the python path: prefer the running interpreter
    python_path = _resolve_python_path()

    if SETTINGS_JSON_PATH.exists():
        try:
            settings = json.loads(SETTINGS_JSON_PATH.read_text())
        except json.JSONDecodeError:
            print("  WARNING: settings.json is malformed, skipping hook injection")
            return
    else:
        SETTINGS_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
        settings = {}

    if "hooks" not in settings:
        settings["hooks"] = {}

    configured = 0
    skipped = 0
    repaired = 0

    for event, hook_defs in manifest.items():
        # Normalize: old format is a single dict, new format is a list of dicts
        if isinstance(hook_defs, dict):
            hook_defs = [hook_defs]

        for hook_def in hook_defs:
            script = hook_def["script"]
            command = f"{python_path} {hooks_src / script}"

            # Build a unique identifier for this hook (handles "fast_hook.py session_start" etc.)
            # Strip .py and use the full script string for matching
            script_key = script.replace(".py", "").replace(" ", "_")

            # Check if this OMEGA hook is already wired (match by script_key in command)
            existing_idx = None
            existing_hook_idx = None
            if event in settings["hooks"]:
                for i, entry in enumerate(settings["hooks"][event]):
                    for j, h in enumerate(entry.get("hooks", [])):
                        cmd = h.get("command", "")
                        if script_key in cmd.replace(".py", "").replace(" ", "_"):
                            existing_idx = i
                            existing_hook_idx = j
                            break
                    if existing_idx is not None:
                        break

            if existing_idx is not None:
                # Hook exists — check if the path is correct
                existing_cmd = settings["hooks"][event][existing_idx]["hooks"][existing_hook_idx]["command"]
                if existing_cmd == command:
                    skipped += 1
                    continue
                # Path changed (broken or outdated) — replace it
                settings["hooks"][event][existing_idx]["hooks"][existing_hook_idx]["command"] = command
                repaired += 1
                continue

            # Build the hook entry
            entry = {
                "hooks": [
                    {
                        "command": command,
                        "timeout": hook_def["timeout"],
                        "type": "command",
                    }
                ],
                "matcher": hook_def.get("matcher", ""),
            }

            if event not in settings["hooks"]:
                settings["hooks"][event] = []
            settings["hooks"][event].append(entry)
            configured += 1

    SETTINGS_JSON_PATH.write_text(json.dumps(settings, indent=2) + "\n")

    if configured > 0:
        print(f"  settings.json: {configured} hook(s) configured")
    if repaired > 0:
        print(f"  settings.json: {repaired} hook(s) repaired (paths updated)")
    if skipped > 0:
        print(f"  settings.json: {skipped} hook(s) already configured")
    if configured == 0 and skipped == 0:
        print("  settings.json: hooks configured")


def _download_file(url: str, target: Path) -> None:
    """Download a file with a progress bar showing bytes and percentage."""
    import urllib.request

    req = urllib.request.Request(url, headers={"User-Agent": "omega-memory/1.0"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        total = int(resp.headers.get("Content-Length", 0))
        downloaded = 0
        chunk_size = 64 * 1024  # 64 KB chunks

        # Write to a temp file, rename on success (no partial files left behind)
        tmp = target.with_suffix(target.suffix + ".tmp")
        try:
            with open(tmp, "wb") as f:
                while True:
                    chunk = resp.read(chunk_size)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total > 0:
                        pct = downloaded * 100 // total
                        mb_done = downloaded / (1024 * 1024)
                        mb_total = total / (1024 * 1024)
                        print(f"\r    {target.name}: {mb_done:.1f}/{mb_total:.1f} MB ({pct}%)", end="", flush=True)
                    else:
                        mb_done = downloaded / (1024 * 1024)
                        print(f"\r    {target.name}: {mb_done:.1f} MB", end="", flush=True)
            tmp.rename(target)
            print()  # newline after progress
        except BaseException:
            tmp.unlink(missing_ok=True)
            raise


def _download_bge_model(target_dir: Path, errors_ref: list) -> bool:
    """Download bge-small-en-v1.5 ONNX model from HuggingFace. Returns True on success."""
    target_dir.mkdir(parents=True, exist_ok=True)
    required = ["model.onnx", "tokenizer.json", "config.json"]
    if all((target_dir / f).exists() for f in required):
        print(f"  bge-small-en-v1.5 model already present at {target_dir}")
        return True

    print("  Downloading bge-small-en-v1.5 ONNX model (~130MB)...")
    try:
        hf_repo = "https://huggingface.co/BAAI/bge-small-en-v1.5/resolve/main"
        # model.onnx lives in onnx/ subdir, tokenizer files at repo root
        files = {
            "model.onnx": f"{hf_repo}/onnx/model.onnx",
            "tokenizer.json": f"{hf_repo}/tokenizer.json",
            "config.json": f"{hf_repo}/config.json",
            "tokenizer_config.json": f"{hf_repo}/tokenizer_config.json",
        }
        for fname, url in files.items():
            target = target_dir / fname
            if not target.exists():
                _download_file(url, target)
    except Exception as e:
        errors_ref.append(e)
        print(f"  ERROR: bge model download failed: {e}")
        print(f"  Manually place model files in {target_dir}")
        return False

    if not (target_dir / "model.onnx").exists():
        errors_ref.append("model.onnx not present after download")
        print("  ERROR: model.onnx still not present after download attempt")
        return False
    print(f"  bge-small-en-v1.5 model downloaded to {target_dir}")
    return True


# ---------------------------------------------------------------------------
# CLI Memory Commands — direct terminal access to OMEGA
# ---------------------------------------------------------------------------


def _format_age(created_at) -> str:
    """Format a datetime as relative age string (e.g. '2d ago', '1w ago')."""
    if not created_at:
        return ""
    now = datetime.now(timezone.utc)
    if created_at.tzinfo is None:
        # Naive datetime — assume UTC
        created_at = created_at.replace(tzinfo=timezone.utc)
    delta = now - created_at
    seconds = int(delta.total_seconds())
    if seconds < 60:
        return "just now"
    if seconds < 3600:
        return f"{seconds // 60}m ago"
    if seconds < 86400:
        return f"{seconds // 3600}h ago"
    days = seconds // 86400
    if days < 7:
        return f"{days}d ago"
    if days < 30:
        return f"{days // 7}w ago"
    return f"{days // 30}mo ago"


def cmd_query(args):
    """Search memories by semantic similarity or exact phrase."""
    query_text = " ".join(args.query_text)
    if not query_text.strip():
        print("Usage: omega query <search text>", file=sys.stderr)
        sys.exit(1)

    limit = getattr(args, "limit", 10)
    use_json = _use_json(args)
    exact = getattr(args, "exact", False)

    start = time.monotonic()

    if exact:
        # For --json, use the store directly
        if use_json:
            from omega.bridge import _get_store

            db = _get_store()
            results = db.phrase_search(phrase=query_text, limit=limit)
            elapsed = time.monotonic() - start
            out = []
            for node in results:
                out.append(
                    {
                        "id": node.id,
                        "content": node.content,
                        "event_type": (node.metadata or {}).get("event_type", "memory"),
                        "created_at": node.created_at.isoformat() if node.created_at else "",
                        "tags": (node.metadata or {}).get("tags", []),
                    }
                )
            print(json.dumps({"results": out, "count": len(out), "elapsed_s": round(elapsed, 3)}, indent=2))
        else:
            from omega.bridge import _get_store

            db = _get_store()
            results = db.phrase_search(phrase=query_text, limit=limit)
            elapsed = time.monotonic() - start
            if results:
                from omega.cli_ui import print_table

                rows = []
                for node in results:
                    etype = (node.metadata or {}).get("event_type", "memory")
                    preview = node.content[:120].replace("\n", " ")
                    age = _format_age(node.created_at)
                    mid = node.id[:12] if node.id else ""
                    rows.append(("--", etype, preview, age, mid))
                print_table(
                    None, ["Score", "Type", "Preview", "Age", "ID"], rows, styles=["dim", "bold", None, "dim", "dim"]
                )
                print(f"\n{len(results)} result(s) ({elapsed:.2f}s)")
            else:
                print(f'No results for "{query_text}" ({elapsed:.2f}s)')
    else:
        from omega.bridge import query_structured

        results = query_structured(query_text, limit=limit)
        elapsed = time.monotonic() - start

        if use_json:
            print(json.dumps({"results": results, "count": len(results), "elapsed_s": round(elapsed, 3)}, indent=2))
        else:
            if results:
                from omega.cli_ui import print_table

                rows = []
                for r in results:
                    relevance = f"{int(r.get('relevance', 0) * 100)}%"
                    etype = r.get("event_type", "memory")
                    preview = r.get("content", "")[:120].replace("\n", " ")
                    age = ""
                    if r.get("created_at"):
                        try:
                            dt = datetime.fromisoformat(r["created_at"])
                            age = _format_age(dt)
                        except (ValueError, TypeError):
                            pass
                    mid = r.get("id", "")[:12]
                    rows.append((relevance, etype, preview, age, mid))
                print_table(
                    None, ["Score", "Type", "Preview", "Age", "ID"], rows, styles=["cyan", "bold", None, "dim", "dim"]
                )
                print(f"\n{len(results)} result(s) ({elapsed:.2f}s)")
            else:
                print(f'No results for "{query_text}" ({elapsed:.2f}s)')


_CLI_TYPE_MAP = {
    "memory": "memory",
    "lesson": "lesson_learned",
    "decision": "decision",
    "error": "error_pattern",
    "task": "task_completion",
    "preference": "user_preference",
}


def cmd_store(args):
    """Store a memory with a specified type."""
    content = " ".join(args.content)
    if not content.strip():
        print("Usage: omega store <text> [-t TYPE]", file=sys.stderr)
        sys.exit(1)

    cli_type = getattr(args, "type", "memory")
    event_type = _CLI_TYPE_MAP.get(cli_type, cli_type)

    from omega.bridge import store

    store(content=content, event_type=event_type)

    if _use_json(args):
        print(json.dumps({"status": "ok", "content": content[:200], "type": cli_type}, indent=2))
    else:
        print(f"Stored [{cli_type}]: {content[:80]}")


def cmd_remember(args):
    """Store a permanent user preference."""
    text = " ".join(args.text)
    if not text.strip():
        print("Usage: omega remember <text>", file=sys.stderr)
        sys.exit(1)

    from omega.bridge import remember

    remember(text=text)

    if _use_json(args):
        print(json.dumps({"status": "ok", "content": text[:200]}, indent=2))
    else:
        print(f"Remembered: {text[:120]}")


def cmd_timeline(args):
    """Show memory timeline grouped by day."""
    days = getattr(args, "days", 7)
    use_json = _use_json(args)

    if use_json:
        from omega.bridge import _get_store

        db = _get_store()
        data = db.get_timeline(days=days, limit_per_day=20)
        out = {}
        for day, memories in (data or {}).items():
            out[day] = []
            for m in memories:
                out[day].append(
                    {
                        "id": m.id,
                        "content": m.content[:200],
                        "event_type": (m.metadata or {}).get("event_type", "memory"),
                        "created_at": m.created_at.isoformat() if m.created_at else "",
                    }
                )
        print(json.dumps(out, indent=2))
    else:
        from omega.bridge import _get_store
        from omega.cli_ui import print_header, print_table

        db = _get_store()
        data = db.get_timeline(days=days, limit_per_day=20)
        if not data:
            print(f"No memories in the last {days} days.")
            return

        total = sum(len(v) for v in data.values())
        print_header(f"Memory Timeline ({total} memories, last {days} days)")

        for day in sorted(data.keys(), reverse=True):
            memories = data[day]
            rows = []
            for m in memories:
                etype = (m.metadata or {}).get("event_type", "memory")
                preview = m.content[:100].replace("\n", " ")
                time_str = m.created_at.strftime("%H:%M") if m.created_at else ""
                mid = m.id[:12] if m.id else ""
                rows.append((time_str, etype, preview, mid))
            print_table(
                f"{day} ({len(memories)})",
                ["Time", "Type", "Preview", "ID"],
                rows,
                styles=["dim", "bold", None, "dim"],
            )


# ---------------------------------------------------------------------------
# Setup & Doctor
# ---------------------------------------------------------------------------


def _setup_claude_code(errors_ref: list, hooks_src: Path, hooks_only: bool = False, dry_run: bool = False):
    """Claude Code-specific setup: MCP registration, hooks, CLAUDE.md.

    If hooks_only=True, skips MCP server registration entirely. Hooks call
    bridge.py directly (no MCP process needed), saving ~600MB RAM per session.
    """
    if not hooks_only:
        # Register MCP server with Claude Code
        print("  Registering MCP server with Claude Code...")
        python_path = _resolve_python_path()
        try:
            result = subprocess.run(
                ["claude", "mcp", "add", "-s", "user", "omega-memory", "--", python_path, "-m", "omega.server.mcp_server"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode == 0:
                print("  MCP server registered successfully")
            else:
                errors_ref.append(1)
                print(f"  ERROR: MCP registration returned code {result.returncode}")
                if result.stderr:
                    print(f"  {result.stderr.strip()}")
                print(f"  Register manually: claude mcp add -s user omega-memory -- {python_path} -m omega.server.mcp_server")
        except FileNotFoundError:
            errors_ref.append(1)
            print("  ERROR: 'claude' command not found in PATH.")
            print("  Install Claude Code: https://docs.anthropic.com/en/docs/claude-code")
            print(f"  Or register manually: claude mcp add -s user omega-memory -- {python_path} -m omega.server.mcp_server")
        except Exception as e:
            errors_ref.append(1)
            print(f"  ERROR: MCP registration failed: {e}")
            print(f"  Register manually: claude mcp add -s user omega-memory -- {python_path} -m omega.server.mcp_server")
    else:
        print("  Skipping MCP server registration (--hooks-only mode)")
        print("  Hooks will call bridge.py directly (~600MB RAM saved per session)")
        print("  Note: omega_store, omega_query etc. won't be available as Claude tools")
        print("  To add MCP later: omega setup --client claude-code")

    # Install hooks
    hooks_dst = Path.home() / ".claude" / "scripts"
    hooks_dst.mkdir(parents=True, exist_ok=True)

    hook_files = ["session_start.py", "session_stop.py", "surface_memories.py", "auto_capture.py"]
    for hook in hook_files:
        src = hooks_src / hook
        dst = hooks_dst / f"omega-{hook}"
        if src.exists():
            shutil.copy2(src, dst)
            if sys.platform != "win32":
                dst.chmod(0o755)
            print(f"  Installed hook: {dst.name}")
        else:
            print(f"  WARNING: Hook source not found: {src}")

    # Wire hooks into settings.json
    try:
        _inject_settings_hooks(hooks_src)
    except Exception as e:
        errors_ref.append(1)
        print(f"  ERROR: Failed to configure settings.json hooks: {e}")

    # Inject OMEGA block into CLAUDE.md
    try:
        _inject_claude_md(dry_run=dry_run)
    except Exception as e:
        print(f"  WARNING: Failed to update CLAUDE.md: {e}")


def _mcp_server_json_snippet() -> str:
    """Return the MCP server JSON config snippet for manual copy-paste."""
    python_path = _resolve_python_path()
    return json.dumps({
        "omega-memory": {
            "command": python_path,
            "args": ["-m", "omega.server.mcp_server"],
            "env": {"OMEGA_CLIENT": "{{CLIENT}}"},
        }
    }, indent=2)


def _setup_generic_mcp_client(client_name: str):
    """Print MCP server config for clients that lack a `mcp add` command."""
    snippet = _mcp_server_json_snippet().replace("{{CLIENT}}", client_name)
    print(f"\n  === {client_name.title()} MCP Configuration ===")
    print(f"  Add this to your {client_name} MCP settings:\n")
    print(snippet)
    print("\n  Set the environment variable for client detection:")
    print(f"    export OMEGA_CLIENT={client_name}")
    print(f"\n  NOTE: Hooks are not available for {client_name}.")
    print("  Memory capture requires manual omega_store calls or MCP tool usage.")
    print("  Session start/stop hooks will not fire automatically.\n")


def _resolve_hooks_src() -> Path:
    """Resolve the hooks source directory.

    Priority:
    1. src/omega/hooks/ inside the installed package (pip install)
    2. hooks/ at repo root (development checkout)
    """
    pkg_hooks = Path(__file__).parent / "hooks"
    if pkg_hooks.exists() and (pkg_hooks / "fast_hook.py").exists():
        return pkg_hooks
    repo_hooks = Path(__file__).parent.parent.parent / "hooks"
    if repo_hooks.exists() and (repo_hooks / "fast_hook.py").exists():
        return repo_hooks
    return pkg_hooks  # will fail gracefully downstream


def cmd_hooks(args):
    """Manage Claude Code hooks: setup, path, doctor."""
    sub = getattr(args, "hooks_command", None)

    hooks_src = _resolve_hooks_src()
    python_path = _resolve_python_path()

    if sub == "setup":
        print("OMEGA hooks setup")
        print(f"  Python:  {python_path}")
        print(f"  Hooks:   {hooks_src}")

        if not (hooks_src / "fast_hook.py").exists():
            print("\n  ERROR: fast_hook.py not found at expected location.")
            print("  Try reinstalling: pip install omega-memory[server]")
            sys.exit(1)

        try:
            _inject_settings_hooks(hooks_src)
            print("\n  Hooks configured in ~/.claude/settings.json")
        except Exception as e:
            print(f"\n  ERROR: Failed to configure hooks: {e}")
            sys.exit(1)

        try:
            _inject_claude_md()
        except Exception as e:
            print(f"  WARNING: Failed to update CLAUDE.md: {e}")

        print("\n  Done! Restart Claude Code for changes to take effect.")

    elif sub == "path":
        # Machine-readable: just print the path
        print(hooks_src)

    elif sub == "doctor":
        print("OMEGA hooks doctor")
        print(f"  Python:     {python_path}")
        print(f"  Hooks dir:  {hooks_src}")

        # Check fast_hook.py exists
        fh = hooks_src / "fast_hook.py"
        if fh.exists():
            print(f"  fast_hook:  OK ({fh})")
        else:
            print(f"  fast_hook:  MISSING ({fh})")

        # Check settings.json has hooks
        if SETTINGS_JSON_PATH.exists():
            try:
                settings = json.loads(SETTINGS_JSON_PATH.read_text())
                hooks = settings.get("hooks", {})
                events_with_omega = 0
                broken_paths = []
                for event, entries in hooks.items():
                    for entry in entries:
                        for h in entry.get("hooks", []):
                            cmd = h.get("command", "")
                            if "omega" in cmd.lower() or "fast_hook" in cmd:
                                events_with_omega += 1
                                # Check if the path in the command exists
                                parts = cmd.split()
                                if len(parts) >= 2:
                                    py_path = parts[0]
                                    script_path = parts[1]
                                    if not Path(py_path).exists():
                                        broken_paths.append(f"{event}: Python not found: {py_path}")
                                    if not Path(script_path).exists():
                                        broken_paths.append(f"{event}: Script not found: {script_path}")

                print(f"  settings:   {events_with_omega} OMEGA hook events configured")
                if broken_paths:
                    print(f"  BROKEN:     {len(broken_paths)} path issue(s)")
                    for bp in broken_paths:
                        print(f"    - {bp}")
                    print("\n  Fix with: omega hooks setup")
                else:
                    print("  paths:      All OK")
            except json.JSONDecodeError:
                print("  settings:   MALFORMED (~/.claude/settings.json)")
        else:
            print("  settings:   NOT FOUND (~/.claude/settings.json)")
            print("\n  Fix with: omega hooks setup")

    else:
        print("Usage: omega hooks {setup|path|doctor}")
        print()
        print("  setup   Configure hooks in ~/.claude/settings.json")
        print("  path    Print the hooks directory path")
        print("  doctor  Check hook configuration health")


def _setup_cursor(errors_ref: list, hooks_src: Path):
    """Cursor-specific setup: print MCP config for manual paste."""
    _setup_generic_mcp_client("cursor")


def _setup_windsurf(errors_ref: list, hooks_src: Path):
    """Windsurf-specific setup: print MCP config for manual paste."""
    _setup_generic_mcp_client("windsurf")


def _setup_cline(errors_ref: list, hooks_src: Path):
    """Cline-specific setup: print MCP config for manual paste."""
    _setup_generic_mcp_client("cline")


def _setup_codex(errors_ref: list, hooks_src: Path):
    """OpenAI Codex CLI setup: merge MCP server into ~/.codex/config.toml."""
    print("  Configuring OpenAI Codex CLI...")
    config_path = Path.home() / ".codex" / "config.toml"
    config_path.parent.mkdir(parents=True, exist_ok=True)

    python_path = _resolve_python_path()

    # Read existing TOML content (preserve manually since tomllib is read-only)
    lines = []
    if config_path.exists():
        try:
            lines = config_path.read_text().splitlines(keepends=True)
        except OSError as e:
            errors_ref.append(e)
            print(f"  ERROR: Could not read {config_path}: {e}")
            return

    # Check if omega-memory is already configured
    content = "".join(lines)
    if "mcp_servers.omega-memory" in content:
        print(f"  omega-memory already configured in {config_path}")
        return

    # Build the TOML block to insert
    toml_block = (
        '\n[mcp_servers.omega-memory]\n'
        f'command = "{python_path}"\n'
        'args = ["-m", "omega.server.mcp_server"]\n'
    )

    # Insert before the first [projects.*] section if present, otherwise append
    insert_idx = None
    for i, line in enumerate(lines):
        if line.strip().startswith("[projects."):
            insert_idx = i
            break

    if insert_idx is not None:
        lines.insert(insert_idx, toml_block + "\n")
    else:
        lines.append(toml_block)

    try:
        config_path.write_text("".join(lines))
        print(f"  Wrote MCP config to {config_path}")
        print("  Restart Codex CLI to activate OMEGA.")
        print("  NOTE: Hooks (auto-capture, memory surfacing) are only available with Claude Code.")
    except OSError as e:
        errors_ref.append(e)
        print(f"  ERROR: Could not write {config_path}: {e}")


def _setup_antigravity(errors_ref: list, hooks_src: Path):
    """Antigravity IDE setup: write MCP config to ~/.gemini/antigravity/mcp_config.json."""
    print("  Configuring Antigravity IDE...")
    config_path = Path.home() / ".gemini" / "antigravity" / "mcp_config.json"
    config_path.parent.mkdir(parents=True, exist_ok=True)

    python_path = _resolve_python_path()
    mcp_entry = {
        "mcpServers": {
            "omega-memory": {
                "command": python_path,
                "args": ["-m", "omega.server.mcp_server"],
            }
        }
    }

    # Read or create config
    if config_path.exists():
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            config = {}
    else:
        config = {}

    if "mcpServers" not in config:
        config["mcpServers"] = {}

    if "omega-memory" in config.get("mcpServers", {}):
        print(f"  omega-memory already configured in {config_path}")
        return

    config["mcpServers"]["omega-memory"] = {
        "command": python_path,
        "args": ["-m", "omega.server.mcp_server"],
    }

    try:
        config_path.write_text(json.dumps(config, indent=2) + "\n")
        print(f"  Wrote MCP config to {config_path}")
        print("  Restart Antigravity to activate OMEGA.")
        print("  NOTE: Hooks (auto-capture, memory surfacing) are only available with Claude Code.")
    except OSError as e:
        errors_ref.append(e)
        print(f"  ERROR: Could not write {config_path}: {e}")


def _setup_venv(errors_ref: list, hooks_src: Path):
    """Venv setup: print MCP and CLI paths for manual client configuration."""
    python_path = _resolve_python_path()
    omega_bin = shutil.which("omega") or str(Path(python_path).parent / "omega")

    print("\n  OMEGA venv configuration:")
    print(f"  Python:  {python_path}")
    print(f"  CLI:     {omega_bin}")
    print("\n  MCP server (stdio):")
    print(f"    command: {python_path}")
    print('    args:    ["-m", "omega.server.mcp_server"]')
    print("\n  JSON config block (copy into your client):")
    config = json.dumps({
        "omega-memory": {
            "command": python_path,
            "args": ["-m", "omega.server.mcp_server"],
        }
    }, indent=2)
    for line in config.splitlines():
        print(f"    {line}")


def _setup_claude_desktop(errors_ref: list, hooks_src: Path, dry_run: bool = False):
    """Claude Desktop setup: inject MCP entry into claude_desktop_config.json."""
    # Determine config path
    if sys.platform == "darwin":
        config_path = Path.home() / "Library" / "Application Support" / "Claude" / "claude_desktop_config.json"
    else:
        appdata = os.environ.get("APPDATA", "")
        if not appdata:
            errors_ref.append(1)
            print("  ERROR: APPDATA not set, cannot find Claude Desktop config")
            return
        config_path = Path(appdata) / "Claude" / "claude_desktop_config.json"

    python_path = _resolve_python_path()
    mcp_entry = {
        "command": python_path,
        "args": ["-m", "omega.server.mcp_server"],
    }

    # Read or create config
    if config_path.exists():
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as e:
            print(f"  WARNING: Could not parse existing config ({e}), creating new")
            config = {}
    else:
        config = {}

    if "mcpServers" not in config:
        config["mcpServers"] = {}

    # Check if already configured and up to date
    existing = config["mcpServers"].get("omega-memory")
    if existing and existing.get("command") == python_path:
        print("  Claude Desktop: omega-memory already configured")
    else:
        config["mcpServers"]["omega-memory"] = mcp_entry
        if dry_run:
            print(f"  Claude Desktop: would write MCP entry to {config_path} (dry-run)")
        else:
            # Back up existing config
            if config_path.exists():
                backup = config_path.with_suffix(".json.bak")
                if not backup.exists():
                    shutil.copy2(config_path, backup)
                    print(f"  Backed up config to {backup.name}")
            config_path.parent.mkdir(parents=True, exist_ok=True)
            config_path.write_text(json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            print(f"  Claude Desktop: MCP server registered in {config_path}")

    # Inject CLAUDE.md (reuse existing function)
    try:
        _inject_claude_md(dry_run=dry_run)
    except Exception as e:
        print(f"  WARNING: Failed to update CLAUDE.md: {e}")


def cmd_setup(args):
    """Set up OMEGA: create dirs, download model, initialize DB. Optionally configure a client."""
    # ── Python version check ──────────────────────────────────────────
    if sys.version_info < (3, 11):
        print(f"ERROR: OMEGA requires Python 3.11 or higher (you have {sys.version_info.major}.{sys.version_info.minor}).")
        print("Install Python 3.11+: https://www.python.org/downloads/")
        sys.exit(1)

    client = getattr(args, "client", None)
    hooks_only = getattr(args, "hooks_only", False)
    dry_run = getattr(args, "dry_run", False)
    errors = []
    download_model = getattr(args, "download_model", False)

    # --hooks-only implies claude-code client
    if hooks_only and client is None:
        client = "claude-code"

    # ── Auto-detect Claude Code if --client not specified ─────────────
    if client is None and shutil.which("claude"):
        client = "claude-code"
        print("Setting up OMEGA (Claude Code detected)...")
    elif client is None:
        print("Setting up OMEGA...")
        print("  NOTE: Claude Code CLI not found in PATH.")
        print("  Skipping MCP registration and hooks. To add them later:")
        print("    omega setup --client claude-code")
        print()
    else:
        print("Setting up OMEGA...")

    # Track what we did for the summary
    steps_done = []
    steps_skipped = []
    files_modified = []

    # 1. Create directories with restricted permissions
    OMEGA_DIR.mkdir(parents=True, exist_ok=True, mode=0o700)
    (OMEGA_DIR / "graphs").mkdir(exist_ok=True, mode=0o700)
    print(f"  Created {OMEGA_DIR}")
    steps_done.append("Storage directory")

    # 2. Download ONNX model
    if download_model:
        _download_bge_model(BGE_MODEL_DIR, errors)
        steps_done.append("Embedding model (bge-small-en-v1.5)")
    else:
        bge_model = BGE_MODEL_DIR / "model.onnx"
        minilm_model = MINILM_MODEL_DIR / "model.onnx"
        if bge_model.exists():
            print(f"  ONNX model: bge-small-en-v1.5 at {BGE_MODEL_DIR}")
            steps_done.append("Embedding model (already present)")
        elif minilm_model.exists():
            print(f"  ONNX model: all-MiniLM-L6-v2 at {MINILM_MODEL_DIR}")
            print("  TIP: Run 'omega setup --download-model' to upgrade to bge-small-en-v1.5")
            steps_done.append("Embedding model (already present)")
        else:
            MINILM_MODEL_DIR.mkdir(parents=True, exist_ok=True)
            model_path = MINILM_MODEL_DIR / "model.onnx"
            print("  Downloading ONNX embedding model (all-MiniLM-L6-v2, ~90MB)...")
            script = Path(__file__).parent.parent.parent / "scripts" / "download_model.py"
            if script.exists():
                subprocess.run([sys.executable, str(script), str(MINILM_MODEL_DIR)], check=True)
            else:
                try:
                    hf_base = "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/onnx"
                    for fname in ["model.onnx", "tokenizer.json", "config.json", "tokenizer_config.json", "vocab.txt"]:
                        target = MINILM_MODEL_DIR / fname
                        if not target.exists():
                            _download_file(f"{hf_base}/{fname}", target)
                except Exception as e:
                    errors.append(e)
                    print(f"  ERROR: Model download failed: {e}")
                    print(f"  Manually place model files in {MINILM_MODEL_DIR}")
            if not model_path.exists():
                errors.append("model.onnx not present")
                print("  ERROR: model.onnx still not present after download attempt")
            else:
                print("  TIP: Run 'omega setup --download-model' to upgrade to bge-small-en-v1.5")
                steps_done.append("Embedding model (downloaded)")

    # 3. Check for existing MAGMA model and symlink
    gnosis_model = Path.home() / ".cache" / "gnosis" / "models" / "all-MiniLM-L6-v2-onnx"
    minilm_model_path = MINILM_MODEL_DIR / "model.onnx"
    if gnosis_model.exists() and not minilm_model_path.exists() and not (BGE_MODEL_DIR / "model.onnx").exists():
        print(f"  Found existing model at {gnosis_model}, creating symlink...")
        if MINILM_MODEL_DIR.exists():
            shutil.rmtree(MINILM_MODEL_DIR)
        MINILM_MODEL_DIR.symlink_to(gnosis_model)
        print("  Symlinked to existing model")

    # 4. Create default config
    config_path = OMEGA_DIR / "config.json"
    if not config_path.exists():
        config = {
            "storage_path": str(OMEGA_DIR),
            "model_dir": str(ONNX_MODEL_DIR),
            "version": "0.1.0",
            "entity_scoping": {"enabled": False},
        }
        config_path.write_text(json.dumps(config, indent=2))
        config_path.chmod(0o600)
        print(f"  Created config at {config_path}")
    steps_done.append("Config file")

    # 5. Client-specific setup
    hooks_src = _resolve_hooks_src()
    _CLIENT_SETUP = {
        "cursor": _setup_cursor,
        "windsurf": _setup_windsurf,
        "cline": _setup_cline,
        "codex": _setup_codex,
        "antigravity": _setup_antigravity,
        "venv": _setup_venv,
    }
    if client == "claude-code":
        _setup_claude_code(errors, hooks_src, hooks_only=hooks_only, dry_run=dry_run)
        if hooks_only:
            steps_done.append("MCP server registration (skipped — hooks-only)")
        else:
            steps_done.append("MCP server registration")
        steps_done.append("Hooks (settings.json)")
        steps_done.append("CLAUDE.md instructions")
        files_modified.append("~/.claude/settings.json (hook entries)")
        files_modified.append("~/.claude/CLAUDE.md (OMEGA instruction block)")
        if not hooks_only:
            files_modified.append("~/.claude.json (MCP server entry)")
    elif client == "claude-desktop":
        _setup_claude_desktop(errors, hooks_src, dry_run=dry_run)
        steps_done.append("Claude Desktop MCP registration")
        steps_done.append("CLAUDE.md instructions")
        if sys.platform == "darwin":
            config_display = "~/Library/Application Support/Claude/claude_desktop_config.json"
        else:
            config_display = "%APPDATA%/Claude/claude_desktop_config.json"
        files_modified.append(f"{config_display} (MCP server entry)")
        files_modified.append("~/.claude/CLAUDE.md (OMEGA instruction block)")
    elif client in _CLIENT_SETUP:
        _CLIENT_SETUP[client](errors, hooks_src)
        steps_done.append(f"MCP config snippet ({client})")
        steps_skipped.append(f"Hooks (not available for {client})")
    else:
        steps_skipped.append("MCP server registration (no client specified)")
        steps_skipped.append("Hooks (no client specified)")
        python_path = _resolve_python_path()
        print("\n  MCP server ready. Add to your client:")
        print(f"    Command: {python_path} -m omega.server.mcp_server")
        print("    Transport: stdio")

    # ── Summary ───────────────────────────────────────────────────────
    print()
    if errors:
        print(f"OMEGA setup completed with {len(errors)} error(s).")
        for step in steps_done:
            print(f"  [OK] {step}")
        for err in errors:
            print(f"  [FAIL] {err}")
        for step in steps_skipped:
            print(f"  [SKIP] {step}")
        print("\nRun 'omega doctor' to diagnose issues.")
        sys.exit(1)
    else:
        print("OMEGA setup complete!")
        try:
            from omega.telemetry import track_event
            track_event("setup_complete", {"client": client or "none"})
        except Exception:
            pass
        for step in steps_done:
            print(f"  [OK] {step}")
        for step in steps_skipped:
            print(f"  [SKIP] {step}")
        if files_modified:
            print("\n  Files modified outside ~/.omega/:")
            for f in files_modified:
                print(f"    {f}")
        print(f"\n  Storage: {OMEGA_DIR}")
        print("  Run 'omega doctor' to verify.")

        # Pro upgrade CTA on first install (skip if already licensed)
        _show_pro = True
        try:
            from omega_platform.license import is_pro
            if is_pro():
                _show_pro = False
        except Exception:
            pass
        # GitHub star ask -- always show on first setup
        print()
        print("  If OMEGA is useful, please star us on GitHub:")
        print("    https://github.com/omega-memory/omega-memory")

        if _show_pro:
            print()
            print("  ┌─────────────────────────────────────────────────────┐")
            print("  │  Unlock the full platform with OMEGA Pro            │")
            print("  │                                                     │")
            print("  │  + 98 Pro tools: coordination, LLM routing,         │")
            print("  │    knowledge base, entity management, oracle        │")
            print("  │  + Multi-agent coordination (53 tools)              │")
            print("  │  + Cloud sync via your own Supabase                 │")
            print("  │                                                     │")
            print("  │  $19/mo  ·  14-day money-back guarantee             │")
            print("  │  Run: omega upgrade                                 │")
            print("  │  Or visit: https://omegamax.co/pro?ref=cli-setup    │")
            print("  └─────────────────────────────────────────────────────┘")


def cmd_status(args):
    """Show OMEGA status: memory count, store size, model status."""
    use_json = _use_json(args)
    data = {}

    # Resolve Pro vs Free once. is_pro() is TTL-cached so the call is
    # cheap, and we use the result for both the tier badge and the cap
    # values. Missing omega_platform (standalone Free install) drops to
    # Free tier silently.
    is_pro_user = False
    try:
        from omega_platform.license import is_pro
        is_pro_user = is_pro()
    except ImportError:
        pass
    except Exception as e:
        logger.debug("is_pro() check failed in status: %s", e)
    data["tier"] = "pro" if is_pro_user else "free"
    if not is_pro_user:
        data["soft_cap"] = 2000
        data["hard_cap"] = 5000

    # SQLite database (primary backend)
    db_path = OMEGA_DIR / "omega.db"
    if db_path.exists():
        import sqlite3

        size_mb = db_path.stat().st_size / (1024 * 1024)
        data["backend"] = "sqlite"
        data["database"] = str(db_path)
        data["size_mb"] = round(size_mb, 2)
        try:
            conn = sqlite3.connect(str(db_path), timeout=30)
            count = conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
            data["memories"] = count
            try:
                import sqlite_vec

                conn.enable_load_extension(True)
                sqlite_vec.load(conn)
                conn.enable_load_extension(False)
                data["vector_search"] = True
            except Exception:
                data["vector_search"] = False
            conn.close()
        except Exception as e:
            data["error"] = str(e)
    else:
        store_path = OMEGA_DIR / "store.jsonl"
        if store_path.exists():
            size_mb = store_path.stat().st_size / (1024 * 1024)
            with open(store_path) as f:
                line_count = sum(1 for _ in f)
            data["backend"] = "jsonl"
            data["store"] = str(store_path)
            data["memories"] = line_count
            data["size_mb"] = round(size_mb, 2)
        else:
            data["backend"] = None
            data["memories"] = 0

    # Model
    bge_path = BGE_MODEL_DIR / "model.onnx"
    minilm_path = MINILM_MODEL_DIR / "model.onnx"
    if bge_path.exists():
        data["model"] = "bge-small-en-v1.5"
        data["model_size_mb"] = round(bge_path.stat().st_size / (1024 * 1024), 0)
    elif minilm_path.exists():
        data["model"] = "all-MiniLM-L6-v2"
        data["model_size_mb"] = round(minilm_path.stat().st_size / (1024 * 1024), 0)
    else:
        data["model"] = None

    # Profile
    profile_path = OMEGA_DIR / "profile.json"
    data["has_profile"] = profile_path.exists()

    # Config version
    config_path = OMEGA_DIR / "config.json"
    if config_path.exists():
        try:
            config = json.loads(config_path.read_text())
            data["version"] = config.get("version", "unknown")
        except Exception:
            pass

    # Cloud
    secrets_path = OMEGA_DIR / "secrets.json"
    cloud = {"configured": secrets_path.exists()}
    if cloud["configured"]:
        for marker_name, key in [("last-cloud-pull", "last_pull"), ("last-cloud-push", "last_push")]:
            marker = OMEGA_DIR / marker_name
            if marker.exists():
                try:
                    cloud[key] = marker.read_text().strip()
                except Exception:
                    pass
    data["cloud"] = cloud

    if use_json:
        print(json.dumps(data, indent=2, default=str))
        return

    # Rich/plain output (existing behavior preserved)
    from omega.cli_ui import print_header, print_kv

    print_header("OMEGA Status")
    kv: list[tuple[str, str]] = []

    if data.get("backend") == "sqlite":
        kv.append(("Backend", "SQLite"))
        kv.append(("Database", data.get("database", "")))
        kv.append(("Size", f"{data.get('size_mb', 0):.2f} MB"))
        kv.append(("Memories", str(data.get("memories", 0))))
        if data.get("vector_search"):
            kv.append(("Vector search", "enabled (sqlite-vec)"))
        else:
            kv.append(("Vector search", "text-only fallback"))
        if "error" in data:
            kv.append(("Error", data["error"]))
    elif data.get("backend") == "jsonl":
        kv.append(("Backend", "JSONL (legacy)"))
        kv.append(("Store", data.get("store", "")))
        kv.append(("Memories", str(data.get("memories", 0))))
        kv.append(("Size", f"{data.get('size_mb', 0):.2f} MB"))
        kv.append(("Tip", "Run 'omega migrate-db' to upgrade to SQLite"))
    else:
        kv.append(("Store", "not initialized"))
        kv.append(("Memories", "0"))

    if data.get("model"):
        model_label = data["model"]
        if data.get("model_size_mb"):
            model_label += f" ONNX ({data['model_size_mb']:.0f} MB)"
        kv.append(("Model", model_label))
        if data["model"] == "all-MiniLM-L6-v2":
            kv.append(("Tip", "Run 'omega setup --download-model' to upgrade to bge-small-en-v1.5"))
    else:
        kv.append(("Model", "not downloaded"))
        kv.append(("Tip", "Run 'omega setup' to download"))

    # Legacy graphs
    graphs_dir = OMEGA_DIR / "graphs"
    if graphs_dir.exists():
        graph_files = list(graphs_dir.glob("*.json"))
        if graph_files:
            kv.append(("Legacy graphs", f"{len(graph_files)} files (run 'omega migrate-db' to convert)"))

    if data.get("has_profile"):
        kv.append(("Profile", str(OMEGA_DIR / "profile.json")))

    if data.get("version"):
        kv.append(("Version", data["version"]))

    print_kv(kv)

    cloud = data.get("cloud", {})
    if cloud.get("configured"):
        cloud_kv = [("Cloud", "configured")]
        if cloud.get("last_pull"):
            cloud_kv.append(("Last pull", cloud["last_pull"]))
        if cloud.get("last_push"):
            cloud_kv.append(("Last push", cloud["last_push"]))
        print_kv(cloud_kv)
    else:
        print_kv([("Cloud", "not configured")])

    # Pro upgrade nudge for free users
    if not use_json:
        pro_licensed = False
        try:
            from omega_platform.license import is_pro
            pro_licensed = is_pro()
        except Exception:
            pass
        if not pro_licensed:
            mem_count = data.get("memories", 0)
            if mem_count >= 1500:
                print(f"\n  You have {mem_count:,} memories (limit: 2,000). Pro removes limits.")
            print("  Pro: coordination, routing, knowledge base, and 95 more tools. $19/mo")
            print("  Run 'omega upgrade' or visit https://omegamax.co/pro?ref=cli-status")

    if not use_json:
        _offer_email_capture()

    print()


def cmd_migrate(args):
    """Migrate data from MAGMA (~/.magma/) to OMEGA (~/.omega/). Non-destructive copy."""
    print("Migrating MAGMA data to OMEGA...")

    OMEGA_DIR.mkdir(parents=True, exist_ok=True, mode=0o700)
    (OMEGA_DIR / "graphs").mkdir(exist_ok=True, mode=0o700)

    copied = 0

    # Copy store.jsonl
    src = MAGMA_DIR / "store.jsonl"
    dst = OMEGA_DIR / "store.jsonl"
    if src.exists() and not dst.exists():
        shutil.copy2(src, dst)
        print(f"  Copied store.jsonl ({src.stat().st_size / 1024:.1f} KB)")
        copied += 1
    elif src.exists() and dst.exists():
        print(f"  Skipping store.jsonl (already exists in {OMEGA_DIR})")
    else:
        print(f"  No store.jsonl found at {src}")

    # Copy facts.jsonl
    src = MAGMA_DIR / "facts.jsonl"
    dst = OMEGA_DIR / "facts.jsonl"
    if src.exists() and not dst.exists():
        shutil.copy2(src, dst)
        print("  Copied facts.jsonl")
        copied += 1

    # Copy profile.json
    src = MAGMA_DIR / "profile.json"
    dst = OMEGA_DIR / "profile.json"
    if src.exists() and not dst.exists():
        shutil.copy2(src, dst)
        print("  Copied profile.json")
        copied += 1

    # Copy config.json (update storage_path)
    src = MAGMA_DIR / "config.json"
    dst = OMEGA_DIR / "config.json"
    if src.exists() and not dst.exists():
        config = json.loads(src.read_text())
        # Update paths
        for key in list(config.keys()):
            if isinstance(config[key], str):
                config[key] = config[key].replace(".magma", ".omega").replace("gnosis", "omega")
        dst.write_text(json.dumps(config, indent=2))
        print("  Copied config.json (paths updated)")
        copied += 1

    # Copy graph state files
    if MAGMA_GRAPHS.exists():
        for graph_file in MAGMA_GRAPHS.glob("*.json"):
            dst = OMEGA_DIR / "graphs" / graph_file.name
            if not dst.exists():
                shutil.copy2(graph_file, dst)
                print(f"  Copied graph: {graph_file.name}")
                copied += 1

    # Symlink ONNX model if available from gnosis
    gnosis_model = Path.home() / ".cache" / "gnosis" / "models" / "all-MiniLM-L6-v2-onnx"
    omega_model = OMEGA_CACHE / "models" / "all-MiniLM-L6-v2-onnx"
    if gnosis_model.exists() and not omega_model.exists():
        omega_model.parent.mkdir(parents=True, exist_ok=True)
        omega_model.symlink_to(gnosis_model)
        print(f"  Symlinked ONNX model from {gnosis_model}")
        copied += 1

    if copied > 0:
        print(f"\nMigration complete! Copied {copied} files.")
    else:
        print("\nNothing to migrate (all files already exist or no MAGMA data found).")
    print("Original MAGMA data is untouched.")

    # Auto-reingest into graph system
    store_path = OMEGA_DIR / "store.jsonl"
    if store_path.exists():
        print("\nIngesting store.jsonl into graph system...")
        cmd_reingest(args)


def cmd_reingest(args):
    """Reingest JSONL entries into the SQLite database."""
    store_path = OMEGA_DIR / "store.jsonl"
    pre_sqlite = OMEGA_DIR / "store.jsonl.pre-sqlite"
    # Check both current and backed-up JSONL
    if pre_sqlite.exists() and not store_path.exists():
        store_path = pre_sqlite
    if not store_path.exists():
        print(f"No JSONL store found at {OMEGA_DIR}")
        print("  Nothing to reingest (SQLite is the primary store now)")
        return

    from omega.bridge import reingest

    result = reingest(store_path=store_path)

    print("\nReingest complete:")
    print(f"  Ingested:   {result.get('ingested', 0)}")
    print(f"  Duplicates: {result.get('duplicates', 0)}")
    print(f"  Skipped:    {result.get('skipped', 0)}")
    print(f"  Errors:     {result.get('errors', 0)}")
    print(f"  Total:      {result.get('total', 0)}")

    from omega.bridge import status as omega_status

    s = omega_status()
    print(f"\nNode count: {s.get('node_count', 0)}")


def cmd_consolidate(args):
    """Run memory consolidation: deduplicate and prune old entries."""
    prune_days = getattr(args, "prune_days", 30)
    print(f"Running OMEGA consolidation (prune_days={prune_days})...")

    from omega.bridge import _get_store, deduplicate

    db = _get_store()
    node_count_before = db.node_count()
    print(f"  Nodes before: {node_count_before}")

    # Run deduplication via bridge
    result = deduplicate()
    merged = result.get("merged", 0) if isinstance(result, dict) else 0

    # Prune expired
    expired = db.cleanup_expired()

    # Evict old low-access entries if requested
    evicted = 0
    if prune_days > 0:
        evicted = db.evict_lru(count=0)  # 0 = only expired

    node_count_after = db.node_count()

    print("\nConsolidation complete:")
    print(f"  Duplicates merged: {merged}")
    print(f"  Expired pruned:    {expired}")
    print(f"  Evicted:           {evicted}")
    print(f"  Nodes after:       {node_count_after}")


def cmd_migrate_db(args):
    """Migrate from JSON graphs + JSONL to SQLite backend."""
    force = getattr(args, "force", False)
    from omega.migrate_to_sqlite import migrate

    report = migrate(force=force)
    if report.get("warnings"):
        for w in report["warnings"]:
            print(f"  WARNING: {w}")


def cmd_backup(args):
    """Back up omega.db to ~/.omega/backups/ with timestamp."""
    db_path = OMEGA_DIR / "omega.db"
    if not db_path.exists():
        print("No omega.db found — nothing to back up.")
        return

    backups_dir = OMEGA_DIR / "backups"
    backups_dir.mkdir(parents=True, exist_ok=True, mode=0o700)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    backup_path = backups_dir / f"omega-{timestamp}.db"

    import sqlite3
    from omega.crypto import secure_connect

    src = sqlite3.connect(str(db_path), timeout=30)
    dst = secure_connect(backup_path)
    src.backup(dst)
    dst.close()
    src.close()

    size_mb = backup_path.stat().st_size / (1024 * 1024)
    print(f"Backup saved: {backup_path} ({size_mb:.2f} MB)")

    # Rotate — keep only the 5 most recent backups
    backups = sorted(backups_dir.glob("omega-*.db"), key=lambda p: p.stat().st_mtime, reverse=True)
    for old in backups[5:]:
        old.unlink()
        print(f"  Rotated old backup: {old.name}")


def cmd_compact(args):
    """Cluster and summarize related memories to reduce noise."""
    event_type = getattr(args, "type", "lesson_learned")
    threshold = getattr(args, "threshold", 0.60)
    dry_run = getattr(args, "dry_run", False)

    print(f"Compacting {event_type} (threshold={threshold}, dry_run={dry_run})...")

    from omega.bridge import compact

    result = compact(
        event_type=event_type,
        similarity_threshold=threshold,
        dry_run=dry_run,
    )
    print(result)


def cmd_stats(args):
    """Show memory type distribution and health summary."""
    use_json = _use_json(args)
    use_card = getattr(args, "card", False)

    if use_card:
        from omega.bridge import stats_card_data
        from omega.cli_ui import print_stats_card

        data = stats_card_data()
        if use_json:
            print(json.dumps(data, indent=2, default=str))
        else:
            print_stats_card(data)
        return

    from omega.bridge import type_stats, status as omega_status

    stats = type_stats()
    health = omega_status()

    if use_json:
        print(json.dumps({"types": stats, "health": health}, indent=2, default=str))
        return

    from omega.cli_ui import print_bar_chart, print_header, print_kv

    total = sum(stats.values())
    print_header("OMEGA Stats")
    print_kv(
        [
            ("Memories", str(total)),
            ("DB size", f"{health.get('db_size_mb', 0):.2f} MB"),
            ("Edges", str(health.get("edge_count", 0))),
            ("Backend", health.get("backend", "unknown")),
        ]
    )
    print()
    items = sorted(stats.items(), key=lambda x: -x[1])
    print_bar_chart(items, title="Type Distribution", total=total)


def cmd_activity(args):
    """Show recent session activity: sessions, tasks, insights, claims."""
    days = getattr(args, "days", 7)
    use_json = _use_json(args)

    from omega.bridge import get_activity_summary

    data = get_activity_summary(days=days)

    if use_json:
        print(json.dumps(data, indent=2, default=str))
        return

    from omega.cli_ui import print_header, print_section, print_table

    print_header(f"OMEGA Activity (last {days} days)")

    # Sessions
    print_section("Active Sessions")
    if data["sessions"]:
        rows = []
        for s in data["sessions"]:
            project = s.get("project") or ""
            rows.append(
                (
                    s.get("session_id") or "",
                    project.split("/")[-1] or project,
                    (s.get("task") or "")[:50],
                    (s.get("started_at") or "")[:19],
                    s.get("status") or "",
                )
            )
        print_table(
            None,
            ["Session", "Project", "Task", "Started", "Status"],
            rows,
            styles=["cyan", "bold", None, "dim", "green"],
        )
    else:
        print("  No active sessions")

    # Tasks
    print_section("Open Tasks")
    if data["tasks"]:
        rows = []
        for t in data["tasks"]:
            progress = f"{t.get('progress', 0)}%" if t.get("status") == "in_progress" else ""
            rows.append(
                (
                    str(t.get("id", "")),
                    t.get("title", "")[:50],
                    t.get("status", ""),
                    progress,
                    t.get("created_at", "")[:19],
                )
            )
        print_table(
            None,
            ["ID", "Title", "Status", "Progress", "Created"],
            rows,
            styles=["dim", "bold", "yellow", "cyan", "dim"],
        )
    else:
        print("  No open tasks")

    # Recent Insights
    print_section("Recent Insights")
    if data["insights"]:
        rows = []
        for i in data["insights"]:
            rows.append(
                (
                    i.get("type", ""),
                    i.get("preview", "")[:80],
                    i.get("created_at", "")[:19],
                    i.get("id", ""),
                )
            )
        print_table(None, ["Type", "Preview", "Created", "ID"], rows, styles=["bold", None, "dim", "dim"])
    else:
        print("  No recent insights")

    # Claims
    print_section("Active Claims")
    if data["claims"]:
        rows = []
        for c in data["claims"]:
            rows.append(
                (
                    c.get("type", ""),
                    c.get("path", ""),
                    c.get("session", ""),
                )
            )
        print_table(None, ["Type", "Path/Branch", "Session"], rows, styles=["bold", None, "dim"])
    else:
        print("  No active claims")


def _send_notification(text: str, context: str = None):
    """Send a macOS notification via osascript. Best-effort."""
    try:
        text_escaped = text.replace('"', '\\"')
        subtitle = ""
        if context:
            ctx_escaped = context[:80].replace('"', '\\"')
            subtitle = f' subtitle "{ctx_escaped}"'
        script = f'display notification "{text_escaped}" with title "OMEGA Reminder"{subtitle} sound name "Glass"'
        subprocess.run(
            ["osascript", "-e", script],
            capture_output=True,
            timeout=5,
        )
    except Exception as e:
        logger.debug("macOS notification failed: %s", e)


def cmd_remind(args):
    """Manage reminders: set, list, check, dismiss."""
    sub = getattr(args, "remind_command", None)

    if sub == "set":
        text = " ".join(args.text)
        duration = args.duration
        context = getattr(args, "context", None)
        if not text.strip():
            print("Usage: omega remind set <text> -d <duration>", file=sys.stderr)
            sys.exit(1)

        from omega.bridge import create_reminder

        try:
            result = create_reminder(text=text, duration=duration, context=context)
            print(f"Reminder set: {result['text']}")
            print(f"  Due at: {result['remind_at_local']}")
            print(f"  ID: {result['reminder_id']}")
        except ValueError as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)

    elif sub == "list":
        from omega.bridge import list_reminders

        status = getattr(args, "status", None)
        include_dismissed = status in ("dismissed", "all")
        reminders = list_reminders(status=status, include_dismissed=include_dismissed)

        if not reminders:
            print("No reminders found.")
            return

        print(f"Reminders ({len(reminders)} found):\n")
        for r in reminders:
            overdue = " [OVERDUE]" if r.get("is_overdue") else ""
            print(f"  [{r['status']}]{overdue} {r['text']}")
            print(f"    Due: {r['remind_at_local']} | Time: {r['time_until']}")
            if r.get("context"):
                print(f"    Context: {r['context'][:120]}")
            print(f"    ID: {r['id']}")

    elif sub == "check":
        from omega.bridge import get_due_reminders

        notify = getattr(args, "notify", False)
        due = get_due_reminders(mark_fired=True)

        if not due:
            print("No reminders due.")
            return

        for r in due:
            overdue = " [OVERDUE]" if r.get("is_overdue") else ""
            print(f"[REMINDER]{overdue} {r['text']}")
            if r.get("context"):
                print(f"  Context: {r['context'][:120]}")
            print(f"  ID: {r['id']}")

            if notify:
                _send_notification(r["text"], r.get("context"))

    elif sub == "dismiss":
        reminder_id = args.reminder_id
        from omega.bridge import dismiss_reminder

        result = dismiss_reminder(reminder_id)
        if result.get("success"):
            print(f"Dismissed: {result.get('text', reminder_id)}")
        else:
            print(f"Error: {result.get('error')}", file=sys.stderr)
            sys.exit(1)

    else:
        print("Usage: omega remind {set,list,check,dismiss}", file=sys.stderr)
        sys.exit(1)


def cmd_logs(args):
    """Show recent entries from ~/.omega/hooks.log."""
    hooks_log = OMEGA_DIR / "hooks.log"
    if not hooks_log.exists():
        print("No hooks.log found — no hook errors recorded.")
        return

    n = getattr(args, "lines", 50)
    lines = hooks_log.read_text().strip().split("\n")
    recent = lines[-n:] if len(lines) > n else lines
    print(f"--- Last {len(recent)} lines from {hooks_log} ---\n")
    for line in recent:
        print(line)


def cmd_validate(args):
    """Validate omega.db integrity: SQLite PRAGMA + FTS5 checks."""
    from omega.cli_ui import print_header, print_section, print_status_line, print_summary, print_table

    db_path = OMEGA_DIR / "omega.db"
    if not db_path.exists():
        print("No omega.db found.")
        return

    import sqlite3

    conn = sqlite3.connect(str(db_path), timeout=30)
    errors = 0

    print_header("OMEGA Validate")

    # SQLite integrity check
    print_section("SQLite Integrity")
    result = conn.execute("PRAGMA integrity_check").fetchone()[0]
    if result == "ok":
        print_status_line("ok", "PRAGMA integrity_check passed")
    else:
        errors += 1
        print_status_line("fail", result)

    # FTS5 integrity
    print_section("FTS5 Index")
    try:
        conn.execute("INSERT INTO memories_fts(memories_fts) VALUES('integrity-check')")
        print_status_line("ok", "FTS5 integrity check passed")
    except Exception as e:
        errors += 1
        print_status_line("fail", f"FTS5 integrity: {e}")
        if getattr(args, "repair", False):
            print("  Attempting rebuild...")
            try:
                conn.execute("INSERT INTO memories_fts(memories_fts) VALUES('rebuild')")
                conn.commit()
                print_status_line("ok", "FTS5 index rebuilt")
                errors -= 1
            except Exception as rebuild_err:
                print_status_line("fail", f"Rebuild failed: {rebuild_err}")

    # Row counts (allowlist — these names are used in f-string SQL)
    print_section("Table Counts")
    _VALID_TABLES = frozenset(
        [
            "memories",
            "edges",
            "entity_index",
            "coord_sessions",
            "coord_file_claims",
            "coord_branch_claims",
            "coord_intents",
            "coord_snapshots",
            "coord_tasks",
            "coord_audit",
        ]
    )
    table_rows = []
    for tbl in sorted(_VALID_TABLES):
        try:
            # SECURITY: tbl from _VALID_TABLES hardcoded frozenset, not user input
            count = conn.execute(f"SELECT COUNT(*) FROM {tbl}").fetchone()[0]
            table_rows.append((tbl, str(count)))
        except Exception as e:
            logger.debug("Table count failed for %s: %s", tbl, e)
    print_table(None, ["Table", "Count"], table_rows)

    conn.close()
    print()
    print_summary(errors, 0)
    sys.exit(1 if errors > 0 else 0)


_PLIST_LABEL = "com.omega.mcp-daemon"
_PLIST_DEST = Path.home() / "Library" / "LaunchAgents" / f"{_PLIST_LABEL}.plist"
_DEFAULT_HTTP_PORT = 8377
_DEFAULT_HTTP_HOST = "127.0.0.1"

_JP_PLIST_LABEL = "com.omega.jit-proxy-daemon"
_JP_PLIST_DEST = Path.home() / "Library" / "LaunchAgents" / f"{_JP_PLIST_LABEL}.plist"
_JP_HTTP_PORT = 8378
_JP_HTTP_HOST = "127.0.0.1"


def cmd_serve(args):
    """Run the OMEGA MCP server. Supports stdio (default) and HTTP daemon mode."""
    import asyncio

    subcmd = getattr(args, "serve_command", None)

    if subcmd == "install":
        _serve_install(args)
        return
    elif subcmd == "uninstall":
        _serve_uninstall(args)
        return
    elif subcmd == "status":
        _serve_status(args)
        return
    elif subcmd == "migrate-config":
        _serve_migrate_config(args)
        return
    elif subcmd == "restore-config":
        _serve_restore_config(args)
        return

    # Default: run the MCP server
    if getattr(args, "no_condensed", False):
        os.environ["OMEGA_CONDENSED"] = "0"

    if getattr(args, "daemon", False):
        os.environ["OMEGA_TRANSPORT"] = "http"

    try:
        from omega.server.mcp_server import main
    except SystemExit:
        return

    asyncio.run(main())


def _serve_install(args):
    """Generate launchd plist and load the daemon."""
    plist_template = (DATA_DIR / "com.omega.mcp-daemon.plist").read_text()

    python_path = _resolve_python_path()
    omega_home = str(OMEGA_DIR)

    # Resolve PYTHONPATH so omega package is importable
    try:
        import omega
        pythonpath = str(Path(omega.__file__).parent.parent)
    except Exception:
        pythonpath = ""

    # Ensure log directory exists
    log_dir = OMEGA_DIR / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    plist_content = (
        plist_template
        .replace("__PYTHON_PATH__", python_path)
        .replace("__OMEGA_HOME__", omega_home)
        .replace("__PYTHONPATH__", pythonpath)
    )

    _PLIST_DEST.parent.mkdir(parents=True, exist_ok=True)
    _PLIST_DEST.write_text(plist_content)
    print(f"Plist written to {_PLIST_DEST}")

    # Load the daemon
    result = subprocess.run(
        ["launchctl", "load", str(_PLIST_DEST)],
        capture_output=True, text=True,
    )
    if result.returncode == 0:
        print("Daemon loaded. It will start automatically on login.")
        print(f"\nVerify: curl http://{_DEFAULT_HTTP_HOST}:{_DEFAULT_HTTP_PORT}/health")
        print("\nTo use with Claude Code, run: omega serve migrate-config")
    else:
        print(f"launchctl load failed: {result.stderr.strip()}")
        sys.exit(1)


def _serve_uninstall(args):
    """Unload and remove the daemon plist."""
    if _PLIST_DEST.exists():
        subprocess.run(
            ["launchctl", "unload", str(_PLIST_DEST)],
            capture_output=True, text=True,
        )
        _PLIST_DEST.unlink()
        print("Daemon unloaded and plist removed.")
        print("\nTo restore stdio config, run: omega serve restore-config")
    else:
        print("No daemon plist found. Nothing to uninstall.")


def _serve_status(args):
    """Check daemon status via launchd and health endpoint."""
    import urllib.request
    import urllib.error

    # Check launchd
    result = subprocess.run(
        ["launchctl", "list", _PLIST_LABEL],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        print("Daemon: not loaded (launchd)")
    else:
        lines = result.stdout.strip().split("\n")
        print("Daemon: loaded (launchd)")
        for line in lines:
            if "PID" in line or '"PID"' in line:
                print(f"  {line.strip()}")

    # Check health endpoint
    url = f"http://{_DEFAULT_HTTP_HOST}:{_DEFAULT_HTTP_PORT}/health"
    try:
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=3) as resp:
            data = json.loads(resp.read())
            print(f"\nHealth: {data.get('status', 'unknown')}")
            print(f"  PID: {data.get('pid')}")
            print(f"  RSS: {data.get('rss_mb')} MB")
            print(f"  Uptime: {data.get('uptime_s')}s")
            print(f"  Tools: {data.get('tool_count')}")
    except (urllib.error.URLError, OSError):
        print(f"\nHealth: unreachable ({url})")


def _serve_migrate_config(args):
    """Migrate ~/.claude.json omega-memory entries from stdio to http."""
    claude_json = Path.home() / ".claude.json"
    if not claude_json.exists():
        print("No ~/.claude.json found.")
        return

    content = claude_json.read_text()
    config = json.loads(content)

    # Create backup
    backup = claude_json.with_suffix(".json.bak")
    backup.write_text(content)
    print(f"Backup saved to {backup}")

    url = f"http://{_DEFAULT_HTTP_HOST}:{_DEFAULT_HTTP_PORT}/mcp"
    changed = 0

    projects = config.get("projects", {})
    for proj_path, proj_config in projects.items():
        servers = proj_config.get("mcpServers", {})
        if "omega-memory" in servers:
            entry = servers["omega-memory"]
            if entry.get("type") == "stdio":
                servers["omega-memory"] = {
                    "type": "http",
                    "url": url,
                }
                changed += 1

    if changed > 0:
        claude_json.write_text(json.dumps(config, indent=2) + "\n")
        print(f"Migrated {changed} project(s) from stdio to http.")
        print(f"MCP endpoint: {url}")
        print("\nRestart Claude Code terminals to use the daemon.")
    else:
        print("No stdio omega-memory entries found to migrate.")


def _serve_restore_config(args):
    """Restore ~/.claude.json from backup."""
    claude_json = Path.home() / ".claude.json"
    backup = claude_json.with_suffix(".json.bak")

    if not backup.exists():
        print("No backup found at ~/.claude.json.bak")
        return

    backup_content = backup.read_text()
    claude_json.write_text(backup_content)
    print("Restored ~/.claude.json from backup.")
    print("Restart Claude Code terminals to use stdio mode.")


def cmd_proxy(args):
    """Manage jit-proxy daemon."""
    subcmd = getattr(args, "proxy_command", None)

    if subcmd == "install":
        _jp_install(args)
    elif subcmd == "uninstall":
        _jp_uninstall(args)
    elif subcmd == "status":
        _jp_status(args)
    elif subcmd == "migrate-config":
        _jp_migrate_config(args)
    elif subcmd == "restore-config":
        _jp_restore_config(args)
    else:
        print("Usage: omega proxy {install|uninstall|status|migrate-config|restore-config}")


def _jp_install(args):
    """Install jit-proxy launchd daemon."""
    plist_template = (DATA_DIR / "com.omega.jit-proxy-daemon.plist").read_text()

    python_path = _resolve_python_path()
    omega_home = str(OMEGA_DIR)

    try:
        import omega
        pythonpath = str(Path(omega.__file__).parent.parent)
    except Exception:
        pythonpath = ""

    # Capture current PATH so backends (npx, uvx, x-twitter-mcp-server) are findable
    current_path = os.environ.get("PATH", "/usr/bin:/bin:/usr/sbin:/sbin")

    log_dir = OMEGA_DIR / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    plist_content = (
        plist_template
        .replace("__PYTHON_PATH__", python_path)
        .replace("__OMEGA_HOME__", omega_home)
        .replace("__PYTHONPATH__", pythonpath)
        .replace("__PATH__", current_path)
    )

    _JP_PLIST_DEST.parent.mkdir(parents=True, exist_ok=True)
    _JP_PLIST_DEST.write_text(plist_content)
    print(f"Plist written to {_JP_PLIST_DEST}")

    result = subprocess.run(
        ["launchctl", "load", str(_JP_PLIST_DEST)],
        capture_output=True, text=True,
    )
    if result.returncode == 0:
        print("jit-proxy daemon loaded. It will start automatically on login.")
        print(f"\nVerify: curl http://{_JP_HTTP_HOST}:{_JP_HTTP_PORT}/health")
        print("\nTo use with Claude Code, run: omega proxy migrate-config")
    else:
        print(f"launchctl load failed: {result.stderr.strip()}")
        sys.exit(1)


def _jp_uninstall(args):
    """Unload and remove jit-proxy daemon."""
    if _JP_PLIST_DEST.exists():
        subprocess.run(
            ["launchctl", "unload", str(_JP_PLIST_DEST)],
            capture_output=True, text=True,
        )
        _JP_PLIST_DEST.unlink()
        print("jit-proxy daemon unloaded and plist removed.")
        print("\nTo restore stdio config, run: omega proxy restore-config")
    else:
        print("No jit-proxy daemon plist found. Nothing to uninstall.")


def _jp_status(args):
    """Check jit-proxy daemon status."""
    import urllib.request
    import urllib.error

    result = subprocess.run(
        ["launchctl", "list", _JP_PLIST_LABEL],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        print("Daemon: not loaded")
    else:
        print("Daemon: loaded")
        for line in result.stdout.strip().split("\n"):
            if line.strip():
                print(f"  {line.strip()}")

    url = f"http://{_JP_HTTP_HOST}:{_JP_HTTP_PORT}/health"
    try:
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=3) as resp:
            data = json.loads(resp.read())
            print("\nHealth: OK")
            print(f"  PID: {data.get('pid')}")
            print(f"  RSS: {data.get('rss_mb')} MB")
            print(f"  Uptime: {data.get('uptime_s')}s")
            print(f"  Tools: {data.get('tool_count')}")
            backends = data.get("backends", {})
            for name, status in backends.items():
                connected = "connected" if status.get("connected") else "idle"
                print(f"  Backend {name}: {connected}")
    except Exception:
        print(f"\nHealth: unreachable ({url})")


def _jp_migrate_config(args):
    """Migrate ~/.claude.json jit-proxy entry from stdio to http."""
    claude_json = Path.home() / ".claude.json"
    if not claude_json.exists():
        print("No ~/.claude.json found.")
        return

    content = claude_json.read_text()
    config = json.loads(content)

    # Backup
    backup = claude_json.with_suffix(".json.bak")
    backup.write_text(content)
    print(f"Backup saved to {backup}")

    url = f"http://{_JP_HTTP_HOST}:{_JP_HTTP_PORT}/mcp"
    changed = 0

    # Global mcpServers (top-level)
    servers = config.get("mcpServers", {})
    if "jit-proxy" in servers:
        entry = servers["jit-proxy"]
        if entry.get("type") == "stdio":
            servers["jit-proxy"] = {
                "type": "http",
                "url": url,
            }
            changed += 1

    # Also check per-project entries (in case user moved it)
    projects = config.get("projects", {})
    for proj_path, proj_config in projects.items():
        proj_servers = proj_config.get("mcpServers", {})
        if "jit-proxy" in proj_servers:
            entry = proj_servers["jit-proxy"]
            if entry.get("type") == "stdio":
                proj_servers["jit-proxy"] = {
                    "type": "http",
                    "url": url,
                }
                changed += 1

    if changed > 0:
        claude_json.write_text(json.dumps(config, indent=2) + "\n")
        print(f"Migrated {changed} jit-proxy entry/entries from stdio to http.")
        print(f"MCP endpoint: {url}")
        print("\nRestart Claude Code terminals to use the daemon.")
    else:
        print("No stdio jit-proxy entries found to migrate.")


def _jp_restore_config(args):
    """Restore ~/.claude.json from backup."""
    claude_json = Path.home() / ".claude.json"
    backup = claude_json.with_suffix(".json.bak")

    if not backup.exists():
        print("No backup found at ~/.claude.json.bak")
        return

    backup_content = backup.read_text()
    claude_json.write_text(backup_content)
    print("Restored ~/.claude.json from backup.")
    print("Restart Claude Code terminals to apply.")


def cmd_embed_daemon(args):
    """Manage the shared embedding daemon."""
    try:
        from omega.embedding_daemon import is_daemon_running, get_daemon_pid, stop_daemon, main as daemon_main
    except ImportError:
        print("Embedding daemon requires omega-pro.")
        sys.exit(1)

    subcmd = args.embed_command
    if subcmd == "start":
        if is_daemon_running():
            pid = get_daemon_pid()
            print(f"Embedding daemon already running (PID {pid})")
        else:
            print("Starting embedding daemon...")
            daemon_main()
    elif subcmd == "stop":
        if stop_daemon():
            print("Embedding daemon stopped")
        else:
            print("No embedding daemon running")
    elif subcmd == "status":
        pid = get_daemon_pid()
        if pid:
            print(f"Embedding daemon running (PID {pid})")
            try:
                from omega.embedding_client import EmbeddingClient

                client = EmbeddingClient()
                if client._connect():
                    info = client.info()
                    if info:
                        print(f"  Model: {info.get('model', 'unknown')}")
                        print(f"  Backend: {info.get('backend', 'unknown')}")
                        print(f"  Cache: {info.get('cache_size', 0)}/{info.get('cache_max', 0)}")
                        print(f"  Requests: {info.get('request_count', 0)}")
                        print(f"  Uptime: {info.get('uptime_s', 0)}s")
                    client.close()
            except Exception as e:
                logger.debug("Embedding daemon info check failed: %s", e)
        else:
            print("Embedding daemon not running")
    else:
        print("Usage: omega embed-daemon {start|stop|status}")


def cmd_doctor(args):
    """Verify OMEGA installation: import, model, database, MCP, hooks."""
    from omega.cli_ui import print_header, print_section, print_status_line, print_summary

    use_json = _use_json(args)
    checks = []
    errors = 0
    warnings = 0

    def ok(msg):
        checks.append({"status": "ok", "message": msg})
        if not use_json:
            print_status_line("ok", msg)

    def fail(msg):
        nonlocal errors
        errors += 1
        checks.append({"status": "fail", "message": msg})
        if not use_json:
            print_status_line("fail", msg)

    def warn(msg):
        nonlocal warnings
        warnings += 1
        checks.append({"status": "warn", "message": msg})
        if not use_json:
            print_status_line("warn", msg)

    if not use_json:
        print_header("OMEGA Doctor")

    # 1. Package import
    if not use_json:
        print_section("Package Import")
    try:
        import omega

        ok(f"omega {omega.__version__} imported")
    except Exception as e:
        fail(f"Cannot import omega: {e}")
        if use_json:
            print(json.dumps({"checks": checks, "errors": errors, "warnings": warnings}, indent=2))
        else:
            print(f"\n{errors} error(s), {warnings} warning(s)")
        sys.exit(1)

    try:
        from omega.bridge import status as _s, auto_capture as _ac, query as _q  # noqa: F811,F401

        ok("omega.bridge imported (status, auto_capture, query)")
    except Exception as e:
        fail(f"Cannot import omega.bridge: {e}")

    try:
        from omega.server.handlers import HANDLERS

        ok(f"omega.server.handlers: {len(HANDLERS)} handlers registered")
    except Exception as e:
        fail(f"Cannot import handlers: {e}")

    try:
        from omega.server.tool_schemas import TOOL_SCHEMAS

        ok(f"omega.server.tool_schemas: {len(TOOL_SCHEMAS)} tools defined")
    except Exception as e:
        fail(f"Cannot import tool_schemas: {e}")

    # 2. ONNX model
    if not use_json:
        print_section("Embedding Model")
    bge_path = BGE_MODEL_DIR / "model.onnx"
    minilm_path = MINILM_MODEL_DIR / "model.onnx"
    if bge_path.exists():
        model_mb = bge_path.stat().st_size / (1024 * 1024)
        ok(f"bge-small-en-v1.5 model.onnx present ({model_mb:.0f} MB)")
        active_model_dir = BGE_MODEL_DIR
    elif minilm_path.exists():
        model_mb = minilm_path.stat().st_size / (1024 * 1024)
        ok(f"all-MiniLM-L6-v2 model.onnx present ({model_mb:.0f} MB)")
        warn("Using legacy model. Run 'omega setup --download-model' to upgrade to bge-small-en-v1.5")
        active_model_dir = MINILM_MODEL_DIR
    else:
        fail(f"model.onnx not found at {BGE_MODEL_DIR} or {MINILM_MODEL_DIR}")
        active_model_dir = BGE_MODEL_DIR

    tokenizer_path = active_model_dir / "tokenizer.json"
    if tokenizer_path.exists():
        ok("tokenizer.json present")
    else:
        fail(f"tokenizer.json not found at {active_model_dir}")

    try:
        from omega.embedding import generate_embedding, get_embedding_info

        info = get_embedding_info()
        if info.get("onnx_available"):
            ok("ONNX Runtime available")
        else:
            warn("ONNX Runtime not available, will use fallback")

        emb = generate_embedding("test embedding")
        if len(emb) == 384:
            ok(f"Embedding generation works (384-dim, backend={info.get('backend', 'unknown')})")
        else:
            fail(f"Embedding dimension wrong: {len(emb)} (expected 384)")
    except Exception as e:
        fail(f"Embedding generation failed: {e}")

    # 3. Database
    # Use a single lightweight read-only connection with short busy_timeout
    # to avoid blocking when the MCP server holds a WAL write lock.
    if not use_json:
        print_section("Database")
    db_path = OMEGA_DIR / "omega.db"
    _doctor_conn = None
    if db_path.exists():
        size_mb = db_path.stat().st_size / (1024 * 1024)
        ok(f"omega.db exists ({size_mb:.2f} MB)")
        try:
            import sqlite3 as _sqlite3
            _doctor_conn = _sqlite3.connect(str(db_path), timeout=5)
            _doctor_conn.execute("PRAGMA busy_timeout=5000")
            _doctor_conn.execute("PRAGMA query_only=ON")
            try:
                import sqlite_vec
                _doctor_conn.enable_load_extension(True)
                sqlite_vec.load(_doctor_conn)
                _doctor_conn.enable_load_extension(False)
                vec_enabled = True
            except Exception:
                vec_enabled = False
            mem_count = _doctor_conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
            ok(f"Database accessible: {mem_count} memories, {size_mb:.2f} MB")
            if vec_enabled:
                ok("sqlite-vec enabled (vector search)")
            else:
                warn("sqlite-vec not available (text-only search)")
        except Exception as e:
            fail(f"Database check failed: {e}")
    else:
        warn("omega.db not found (will be created on first use)")

    # 4. MCP registration (client-specific)
    client = getattr(args, "client", None)
    check_claude = client == "claude-code" or shutil.which("claude")
    if check_claude:
        if not use_json:
            print_section("MCP Server (Claude Code)")
        try:
            result = subprocess.run(["claude", "mcp", "list"], capture_output=True, text=True, timeout=5)
            if "omega-memory" in result.stdout:
                ok("omega-memory registered in Claude Code")
            else:
                fail("omega-memory NOT registered in Claude Code")
                if not use_json:
                    print("    Run: claude mcp add -s user omega-memory -- python3 -m omega.server.mcp_server")
        except FileNotFoundError:
            warn("Claude Code CLI not found (cannot verify MCP registration)")
        except Exception as e:
            warn(f"MCP check failed: {e}")
    else:
        if not use_json:
            print_section("MCP Server")
        python_path = _resolve_python_path()
        ok(f"MCP server available: {python_path} -m omega.server.mcp_server")

    # Claude Desktop config check
    if not use_json:
        print_section("Claude Desktop")
    if sys.platform == "darwin":
        desktop_config = Path.home() / "Library" / "Application Support" / "Claude" / "claude_desktop_config.json"
    else:
        appdata = os.environ.get("APPDATA", "")
        desktop_config = Path(appdata) / "Claude" / "claude_desktop_config.json" if appdata else None

    if desktop_config and desktop_config.exists():
        try:
            dc = json.loads(desktop_config.read_text(encoding="utf-8"))
            servers = dc.get("mcpServers", {})
            if "omega-memory" in servers:
                entry = servers["omega-memory"]
                cmd = entry.get("command", "")
                if cmd and Path(cmd).exists():
                    ok(f"Claude Desktop: omega-memory configured (python: {cmd})")
                elif cmd:
                    warn(f"Claude Desktop: omega-memory configured but python not found: {cmd}")
                else:
                    warn("Claude Desktop: omega-memory entry has no command")
            else:
                warn("Claude Desktop: omega-memory not registered")
                if not use_json:
                    print("    Run: omega setup --client claude-desktop")
        except (json.JSONDecodeError, OSError) as e:
            warn(f"Claude Desktop: cannot read config: {e}")
    elif desktop_config:
        ok("Claude Desktop: config not found (not installed or not configured)")
    else:
        ok("Claude Desktop: skipped (APPDATA not set)")

    # 5. FTS5 health
    if not use_json:
        print_section("FTS5 Index")
    if _doctor_conn:
        try:
            fts_count = _doctor_conn.execute("SELECT COUNT(*) FROM memories_fts").fetchone()[0]
            mem_count = _doctor_conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
            if fts_count > 0:
                ok(f"FTS5 index populated ({fts_count} entries, {mem_count} memories)")
                if abs(fts_count - mem_count) > mem_count * 0.1:
                    warn(f"FTS5 index drift: {fts_count} vs {mem_count} memories (>10% mismatch)")
            else:
                warn("FTS5 index empty (text search will use slower LIKE fallback)")
            # Integrity check (requires write access; use separate connection)
            try:
                import sqlite3 as _sqlite3
                _fts_conn = _sqlite3.connect(str(db_path), timeout=5)
                _fts_conn.execute("PRAGMA busy_timeout=5000")
                _fts_conn.execute("INSERT INTO memories_fts(memories_fts) VALUES('integrity-check')")
                ok("FTS5 integrity check passed")
                _fts_conn.close()
            except Exception as fts_err:
                if "readonly" in str(fts_err) or "locked" in str(fts_err):
                    ok("FTS5 index readable (integrity check skipped, DB busy)")
                else:
                    fail(f"FTS5 integrity check failed: {fts_err}")
                    if not use_json:
                        print("    Fix: INSERT INTO memories_fts(memories_fts) VALUES('rebuild')")
        except Exception as e:
            warn(f"FTS5 check skipped: {e}")

    # 5b. Vec index health
    if not use_json:
        print_section("Vector Index")
    if _doctor_conn:
        try:
            vec_count = _doctor_conn.execute("SELECT COUNT(*) FROM memories_vec").fetchone()[0]
            mem_count = _doctor_conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
            ok(f"Vec index: {vec_count} embeddings, {mem_count} memories")
            if vec_count > mem_count:
                orphans = vec_count - mem_count
                warn(f"Vec index has ~{orphans} potential orphaned embeddings (run 'omega consolidate' to clean)")
        except Exception as e:
            warn(f"Vec table not available: {e}")

    # 6. Coordination tables
    if not use_json:
        print_section("Coordination")
    if _doctor_conn:
        try:
            coord_tables = [
                "coord_sessions",
                "coord_file_claims",
                "coord_branch_claims",
                "coord_intents",
                "coord_snapshots",
                "coord_tasks",
                "coord_audit",
            ]
            found = 0
            for tbl in coord_tables:
                row = _doctor_conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (tbl,)).fetchone()
                if row:
                    found += 1
            if found == len(coord_tables):
                ok(f"All {found} coordination tables present")
            elif found > 0:
                warn(f"Only {found}/{len(coord_tables)} coordination tables found")
            else:
                warn("No coordination tables (run any coordination tool to create them)")

            # Check stale sessions
            try:
                cutoff = (datetime.now(timezone.utc) - timedelta(seconds=360)).isoformat()
                stale = _doctor_conn.execute(
                    "SELECT COUNT(*) FROM coord_sessions WHERE last_heartbeat < ?", (cutoff,)
                ).fetchone()[0]
                if stale > 0:
                    warn(f"{stale} stale session(s) (heartbeat >360s ago)")
                else:
                    ok("No stale sessions")
            except Exception as e:
                logger.debug("Stale session check failed: %s", e)
        except Exception as e:
            warn(f"Coordination check skipped: {e}")

    # 7. Memory quality
    if not use_json:
        print_section("Memory Quality")
    if _doctor_conn:
        try:
            rows = _doctor_conn.execute("SELECT metadata FROM memories WHERE metadata LIKE '%feedback_score%'").fetchall()
            if rows:
                scores = []
                flagged = 0
                for (meta_str,) in rows:
                    try:
                        meta = json.loads(meta_str)
                        scores.append(meta.get("feedback_score", 0))
                        if meta.get("flagged_for_review"):
                            flagged += 1
                    except Exception as e:
                        logger.debug("Feedback metadata parse failed: %s", e)
                if scores:
                    avg = sum(scores) / len(scores)
                    ok(f"{len(scores)} memories with feedback (avg score: {avg:.2f})")
                    if flagged > 0:
                        warn(f"{flagged} memory(ies) flagged for review (score <= -3)")
            else:
                ok("No feedback signals recorded yet")
        except Exception as e:
            warn(f"Quality check skipped: {e}")

    # 8. Recent hook errors
    if not use_json:
        print_section("Hook Health")
    hooks_log = OMEGA_DIR / "hooks.log"
    if hooks_log.exists():
        try:
            lines = hooks_log.read_text().strip().split("\n")
            error_lines = [line for line in lines if line.startswith("[") and ": OK " not in line]
            if error_lines:
                recent = error_lines[-5:]
                warn(f"{len(error_lines)} hook error(s) in log, last {len(recent)}:")
                if not use_json:
                    for line in recent:
                        print(f"    {line[:120]}")
            else:
                ok("No hook errors in log")
        except Exception as e:
            warn(f"Cannot read hooks.log: {e}")
    else:
        ok("No hooks.log (no errors recorded)")

    # 9. Hooks configuration (Claude Code-specific)
    check_hooks = client == "claude-code" or SETTINGS_JSON_PATH.exists()
    if check_hooks:
        if not use_json:
            print_section("Hooks (Claude Code)")
        if SETTINGS_JSON_PATH.exists():
            try:
                settings = json.loads(SETTINGS_JSON_PATH.read_text())
                hooks = settings.get("hooks", {})
                expected_events = ["SessionStart", "Stop", "PostToolUse"]
                for event in expected_events:
                    found = False
                    for entry in hooks.get(event, []):
                        for h in entry.get("hooks", []):
                            if "omega" in h.get("command", ""):
                                found = True
                                cmd_parts = h["command"].split()
                                if cmd_parts and not Path(cmd_parts[0]).exists():
                                    warn(f"{event} hook references {cmd_parts[0]} which doesn't exist")
                                break
                    if found:
                        ok(f"{event} hook configured")
                    else:
                        warn(f"{event} hook not configured")
            except Exception as e:
                warn(f"Cannot read settings.json: {e}")
        else:
            warn("settings.json not found (hooks not configured)")

    # 6. Python path
    if not use_json:
        print_section("Environment")
    python_path = _resolve_python_path()
    if Path(python_path).exists():
        ok(f"Python: {python_path}")
    else:
        fail(f"Python path does not exist: {python_path}")

    ok(f"OMEGA home: {OMEGA_DIR}")
    ok(f"Platform: {sys.platform}")

    # CLAUDE.md tier check
    if CLAUDE_MD_PATH.exists():
        claude_content = CLAUDE_MD_PATH.read_text()
        if OMEGA_BEGIN in claude_content:
            if "Multi-Agent Coordination" in claude_content:
                ok("CLAUDE.md: OMEGA Pro block installed")
            else:
                ok("CLAUDE.md: OMEGA Core block installed")
            backup = CLAUDE_MD_PATH.with_suffix(".md.pre-omega")
            if backup.exists():
                ok(f"CLAUDE.md: pre-OMEGA backup at {backup.name}")
        else:
            warn("CLAUDE.md exists but has no OMEGA block (run 'omega setup' to add)")
    else:
        warn("CLAUDE.md not found (run 'omega setup' to create)")

    # Cleanup
    if _doctor_conn:
        _doctor_conn.close()

    # Summary
    if use_json:
        print(json.dumps({"checks": checks, "errors": errors, "warnings": warnings}, indent=2))
    else:
        print()
        print_summary(errors, warnings)

    # Pro upgrade nudge for free users
    if not use_json:
        try:
            from omega_platform.license import is_pro
            if not is_pro():
                print("\n  Upgrade to Pro: 98 more tools. Run 'omega upgrade' or visit https://omegamax.co/pro?ref=cli-doctor")
        except Exception:
            print("\n  Upgrade to Pro: 98 more tools. Run 'omega upgrade' or visit https://omegamax.co/pro?ref=cli-doctor")

    if not use_json:
        _offer_email_capture()

    sys.exit(1 if errors > 0 else 0)


def cmd_knowledge(args):
    """Knowledge base management."""
    try:
        from omega.knowledge.engine import scan_directory, list_documents, search_documents  # noqa: F401
    except ImportError:
        print("Knowledge base requires omega-pro.")
        print("Install: pip install omega-pro")
        return

    subcmd = getattr(args, "kb_command", None)

    if subcmd == "scan":
        directory = args.dir
        result = scan_directory(directory)
        print(result)

    elif subcmd == "list":
        print(list_documents())

    elif subcmd == "search":
        query_text = " ".join(args.query)
        result = search_documents(query_text, limit=args.limit)
        print(result)

    elif subcmd == "sync-kb":
        try:
            from omega.knowledge.cloud_sync import sync_kb_queue
        except ImportError:
            print("Knowledge cloud sync requires omega-pro.")
            return
        result = sync_kb_queue(batch_size=args.batch_size)
        print(result)

    else:
        docs_dir = Path.home() / ".omega" / "documents"
        print("Usage: omega knowledge {scan|list|search}")
        print(f"\nDocuments folder: {docs_dir}")
        print("Drop PDF, markdown, or text files there for auto-ingestion.")
        print("Files are auto-scanned on each Claude Code session start.")


def cmd_cloud(args):
    """Cloud sync and Supabase management."""
    try:
        from omega.cloud.sync import get_sync  # noqa: F401
    except ImportError:
        print("Cloud sync requires omega-pro.")
        print("Install: pip install omega-pro")
        return

    from omega.cli_ui import print_header

    subcmd = getattr(args, "cloud_command", None)

    if subcmd == "setup":
        url = args.url
        key = args.key
        service_key = args.service_key or ""
        if not url or not key:
            print("Usage: omega cloud setup --url <SUPABASE_URL> --key <ANON_KEY>")
            print("\nGet these from: Supabase Dashboard → Settings → API")
            return
        try:
            from omega.cloud.setup import setup_supabase
        except ImportError:
            print("Cloud setup requires omega-pro.")
            return

        result = setup_supabase(url, key, service_key)
        print(result)

    elif subcmd == "sync":
        print_header("Cloud Sync")
        try:
            sync = get_sync()
            results = sync.sync_all()
            for table, info in results.items():
                status = info.get("status", "unknown")
                synced = info.get("synced", 0)
                print(f"  {table}: {synced} synced ({status})")
        except Exception as e:
            print(f"  Sync failed: {e}")

    elif subcmd == "status":
        try:
            print(get_sync().status())
        except Exception as e:
            print(f"Cloud not configured: {e}")

    elif subcmd == "schema":
        try:
            from omega.cloud.setup import get_schema_sql
        except ImportError:
            print("Cloud setup requires omega-pro.")
            return

        print(get_schema_sql())

    elif subcmd == "verify":
        try:
            from omega.cloud.setup import verify_connection
        except ImportError:
            print("Cloud verify requires omega-pro.")
            return

        print(verify_connection())

    elif subcmd == "pull":
        print_header("Cloud Pull")
        try:
            sync = get_sync()
            results = sync.pull_all()
            for table, info in results.items():
                status = info.get("status", "unknown")
                pulled = info.get("pulled", 0)
                skipped = info.get("skipped", 0)
                print(f"  {table}: {pulled} pulled, {skipped} skipped ({status})")
        except Exception as e:
            print(f"  Pull failed: {e}")

    else:
        print("Usage: omega cloud {setup|sync|pull|status|schema|verify}")
        print("\nCloud sync enables mobile access to OMEGA memories via Supabase.")


def cmd_mobile(args):
    """Mobile access setup and mcp-proxy management."""
    try:
        from omega.cloud.sync import get_sync  # noqa: F401
    except ImportError:
        print("Mobile access requires omega-pro (cloud sync).")
        print("Install: pip install omega-pro")
        return

    subcmd = getattr(args, "mobile_command", None)

    if subcmd == "setup":
        print("""
## OMEGA Mobile Access Setup

### Prerequisites
1. Install mcp-proxy: `pipx install mcp-proxy`
2. Install Tailscale: `brew install tailscale && tailscale up`

### Quick Start (4 steps)

1. Start OMEGA HTTP proxy:
   ```
   omega mobile serve
   ```

2. Expose via Tailscale:
   ```
   tailscale serve https / http://127.0.0.1:8089
   ```

3. Get your Tailscale hostname:
   ```
   tailscale status | head -1
   ```

4. Add to Claude mobile app:
   - Settings → MCP Servers → Add
   - URL: https://<your-tailscale-hostname>/mcp
   - All 70 OMEGA tools available from your phone!

### Security
- Tailscale uses WireGuard encryption (zero-trust mesh)
- Only your enrolled devices can connect
- No ports exposed to the public internet
- Encryption key stays on your Mac (profile decryption is local)

### Troubleshooting
- Verify: `curl http://127.0.0.1:8089/health`
- Tailscale: `tailscale status` (should show 'active')
- Logs: `omega logs -n 20`
""")

    elif subcmd == "serve":
        import subprocess
        import sys

        port = args.port
        host = args.host
        print(f"Starting OMEGA MCP proxy on {host}:{port}...")
        print(f"Connect via: http://{host}:{port}/mcp")
        print("Press Ctrl+C to stop.\n")

        try:
            subprocess.run(
                [
                    sys.executable, "-m", "mcp_proxy",
                    "--transport", "streamablehttp",
                    "--host", host,
                    "--port", str(port),
                    "--",
                    sys.executable, "-m", "omega.server.mcp_server",
                ],
                check=True,
            )
        except FileNotFoundError:
            print("Error: mcp-proxy not found. Install with: pipx install mcp-proxy")
        except KeyboardInterrupt:
            print("\nProxy stopped.")

    else:
        print("Usage: omega mobile {setup|serve}")
        print("\nMobile access via mcp-proxy + Tailscale.")


def _offer_email_capture():
    """Offer email capture for users approaching or at the memory cap."""
    try:
        import sqlite3
        from pathlib import Path
        db_path = Path.home() / ".omega" / "omega.db"
        if not db_path.exists():
            return
        conn = sqlite3.connect(str(db_path), timeout=5)
        count = conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
        conn.close()
        if count < 1800:
            return

        # Check if already captured
        telemetry_path = Path.home() / ".omega" / "telemetry.json"
        if telemetry_path.exists():
            import json as _json
            data = _json.loads(telemetry_path.read_text())
            if data.get("email_captured"):
                return

        print()
        print(f"  You have {count} memories" + (" (limit: 2,000)" if count < 2000 else " (at free tier limit)") + ".")
        print("  Get product updates and early access to new features.")
        email = input("  Email (or press Enter to skip): ").strip()

        if email and "@" in email:
            import urllib.request
            import json as _json
            data = _json.dumps({"email": email, "source": "cli_cap", "metadata": {"memory_count": count}}).encode()
            req = urllib.request.Request(
                "https://admin.omegamax.co/api/subscribe",
                data=data,
                headers={"Content-Type": "application/json"},
                method="POST"
            )
            try:
                urllib.request.urlopen(req, timeout=5)
                print("  Thanks! We'll keep you posted.")
                # Mark as captured in telemetry
                if telemetry_path.exists():
                    tdata = _json.loads(telemetry_path.read_text())
                    tdata["email_captured"] = True
                    telemetry_path.write_text(_json.dumps(tdata, indent=2))
            except Exception:
                print("  Saved. Thanks!")
    except Exception:
        pass


def cmd_upgrade(args):
    """Open the Pro purchase page in the browser."""
    import webbrowser

    url = "https://omegamax.co/pro?ref=cli-upgrade"
    print("Opening OMEGA Pro purchase page...")
    print(f"  {url}")
    print()
    print("After purchase, activate with:")
    print("  omega activate <your-license-key>")
    try:
        from omega.telemetry import track_event
        track_event("upgrade_opened")
    except Exception:
        pass
    webbrowser.open(url)


def _activate_license(key: str) -> bool:
    """Activate a Pro license key against the server API.

    Self-contained: uses only stdlib so activation works from the public
    omega-memory package without requiring the Pro wheel.
    """
    import json as _json
    import socket
    import urllib.error
    import urllib.request

    activate_url = "https://admin.omegamax.co/api/activate"
    cache_days = 7
    timeout = min(int(os.environ.get("OMEGA_LICENSE_TIMEOUT", "30")), 120)

    data = _json.dumps({"key": key}).encode()
    req = urllib.request.Request(
        activate_url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            result = _json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        if e.code == 404:
            print("License key not found. Double-check you copied the full key.")
        elif e.code == 403:
            print("This license is not active. It may have been cancelled.")
        elif e.code in (500, 502, 503, 504):
            print(f"Activation server error (HTTP {e.code}). Please try again shortly.")
        else:
            print(f"Unexpected HTTP error (HTTP {e.code}).")
        print("Contact hello@omegamax.co if this persists.")
        return False
    except urllib.error.URLError as e:
        print("Could not reach the activation server. Check your internet connection.")
        print(f"Detail: {e.reason}")
        return False
    except socket.timeout:
        print(f"Activation request timed out after {timeout}s. Try again.")
        return False
    except _json.JSONDecodeError:
        print("Activation server returned a malformed response. Contact hello@omegamax.co.")
        return False
    except Exception as e:
        print(f"Unexpected error: {type(e).__name__}: {e}")
        print("Contact hello@omegamax.co with this message.")
        return False

    if not result.get("valid"):
        reason = result.get("reason", "No reason given")
        print(f"Activation rejected: {reason}")
        return False

    # Parse expiry
    expires_at_str = result.get("expires_at", "")
    try:
        dt = datetime.fromisoformat(expires_at_str.replace("Z", "+00:00"))
        valid_until = dt.timestamp()
    except (ValueError, AttributeError):
        valid_until = time.time() + (cache_days * 86400)

    # Write license file
    omega_dir = Path.home() / ".omega"
    license_file = omega_dir / "license.json"
    try:
        omega_dir.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        print(f"Could not create {omega_dir}: {e}")
        return False

    license_data: dict = {
        "key": key,
        "valid_until": valid_until,
        "activated_at": time.time(),
    }
    signature = result.get("signature")
    if signature:
        license_data["signature"] = signature

    try:
        license_file.write_text(_json.dumps(license_data))
        license_file.chmod(0o600)
    except OSError as e:
        print(f"Could not write license file at {license_file}: {e}")
        return False

    return True


def _download_and_install_pro_wheel(key: str) -> bool:
    """Download the Pro wheel from the server and pip-install it.

    Uses a POST request with the license key in the body (not URL) to avoid
    leaking credentials in server logs. Verifies download integrity via
    SHA256 hash returned by the server.
    Returns True if the wheel was installed successfully.
    """
    import hashlib
    import json as _json
    import tempfile
    import urllib.error
    import urllib.request

    wheel_url = "https://admin.omegamax.co/api/pro/download-wheel"
    timeout = min(int(os.environ.get("OMEGA_LICENSE_TIMEOUT", "60")), 120)

    # Download the wheel (key in POST body, not URL query string)
    try:
        data = _json.dumps({"key": key}).encode()
        req = urllib.request.Request(
            wheel_url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            # Extract filename from Content-Disposition header
            cd = resp.headers.get("Content-Disposition", "")
            filename = "omega_memory_pro.whl"
            if "filename=" in cd:
                # Parse filename="omega_memory_pro-1.4.0-py3-none-any.whl"
                parts = cd.split("filename=")[-1].strip().strip('"').strip("'")
                if parts:
                    filename = parts

            # Server provides SHA256 hash for integrity verification
            expected_hash = resp.headers.get("X-Content-SHA256", "")

            wheel_bytes = resp.read()
    except urllib.error.HTTPError as e:
        if e.code == 503:
            print("  Pro wheel is not yet published. Contact hello@omegamax.co.")
        elif e.code in (403, 404):
            print("  Could not download Pro wheel. Contact hello@omegamax.co.")
        else:
            print(f"  Download failed (HTTP {e.code}). Contact hello@omegamax.co.")
        return False
    except Exception as e:
        print(f"  Download failed: {e}")
        return False

    # Verify integrity if server provided a hash
    if expected_hash:
        actual_hash = hashlib.sha256(wheel_bytes).hexdigest()
        if actual_hash != expected_hash:
            print("  ERROR: Wheel integrity check failed (SHA256 mismatch).")
            print("  The download may have been tampered with. Aborting.")
            return False

    # Save to temp file and pip install
    tmp_dir = tempfile.mkdtemp(prefix="omega_pro_")
    wheel_path = Path(tmp_dir) / filename
    installed = False
    try:
        wheel_path.write_bytes(wheel_bytes)

        # Use the same Python that's running this CLI (ensures same environment)
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", str(wheel_path), "--quiet"],
            capture_output=True,
            text=True,
            timeout=120,
        )
        if result.returncode != 0:
            stderr = result.stderr.strip()
            print(f"  pip install failed: {stderr}")
            print(f"  Wheel saved to: {wheel_path}")
            print(f"  Try manually: pip install {wheel_path}")
            return False

        installed = True
        return True
    except subprocess.TimeoutExpired:
        print(f"  pip install timed out. Try manually: pip install {wheel_path}")
        return False
    except Exception as e:
        print(f"  Install failed: {e}")
        print(f"  Wheel saved to: {wheel_path}")
        print(f"  Try manually: pip install {wheel_path}")
        return False
    finally:
        # Clean up temp file only on success; keep it on failure for manual install
        if installed:
            try:
                wheel_path.unlink(missing_ok=True)
                Path(tmp_dir).rmdir()
            except OSError:
                pass


def cmd_activate(args):
    """Activate a Pro license key."""
    key = args.key.strip()

    if not key.startswith("OMEGA-PRO-"):
        print("Invalid key format. Keys start with OMEGA-PRO-")
        sys.exit(1)

    # If omega_platform is already installed, use its richer activate() with
    # Ed25519 verification, clock-skew detection, and detailed diagnostics.
    try:
        from omega_platform.license import activate
        print("Activating license key...")
        if activate(key):
            print("License activated successfully! Pro modules will load on next MCP server start.")
            print("\nRestart Claude Code or your MCP client to load Pro tools.")
        else:
            print("Activation failed. Please check your key and try again.")
            print("If the problem persists, contact hello@omegamax.co")
            sys.exit(1)
        return
    except ImportError:
        pass

    # Standalone path: validate key, cache license, download + install Pro wheel.
    print("Activating license key...")
    if not _activate_license(key):
        sys.exit(1)

    print("  License validated.")
    print("  Downloading Pro package...")

    if _download_and_install_pro_wheel(key):
        print("  Pro package installed.")
        print()
        print("  Pro license activated! 69 Pro tools now available.")
        print()
        print("  Next steps:")
        print("    omega doctor                         # Verify installation")
        print("    omega status                         # Check Pro features")
        print()
        print("  Restart Claude Code or your MCP client to load Pro tools.")
    else:
        print()
        print("  License activated, but Pro package could not be installed automatically.")
        print("  Download it manually from your dashboard:")
        print()
        print("    https://admin.omegamax.co/pro/dashboard")
        print()
        print("  Then install:")
        print("    pip install ~/Downloads/omega_memory_pro-*.whl")
        print()
        print("  After installing, restart Claude Code or your MCP client.")


def cmd_license(args):
    """Show current license status."""
    # If omega_platform is available, use its richer implementation.
    try:
        from omega_platform.license import license_status, deactivate

        if getattr(args, "deactivate", False):
            deactivate()
            print("License removed.")
            return

        status = license_status()
        if status["active"]:
            print("Status:      Active")
            print(f"Key:         {status['key']}")
            print(f"Valid until:  {status['valid_until']}")
        else:
            if status["key"]:
                print("Status:      Expired")
                print(f"Key:         {status['key']}")
                print("\nRun 'omega activate <key>' to reactivate.")
            else:
                print("Status:      No license")
                print("\nUpgrade at https://omegamax.co/pro")
        return
    except ImportError:
        pass

    # Standalone: read the license file directly.
    license_file = Path.home() / ".omega" / "license.json"

    if getattr(args, "deactivate", False):
        try:
            license_file.unlink(missing_ok=True)
        except OSError:
            pass
        print("License removed.")
        return

    if not license_file.exists():
        print("Status:      No license")
        print("\nActivate with: omega activate <your-license-key>")
        print("Purchase at: https://omegamax.co/pro")
        return

    try:
        data = json.loads(license_file.read_text())
    except (json.JSONDecodeError, OSError):
        print("Status:      License file corrupt")
        print(f"\nRemove and re-activate:\n  rm {license_file}\n  omega activate <your-key>")
        return

    key = data.get("key", "unknown")
    valid_until = data.get("valid_until", 0)
    now = time.time()

    # Mask key for display
    if len(key) > 20:
        masked = key[:14] + "..." + key[-4:]
    else:
        masked = key

    if valid_until > now:
        valid_dt = datetime.fromtimestamp(valid_until, tz=timezone.utc).isoformat()
        print("Status:      Active")
        print(f"Key:         {masked}")
        print(f"Valid until:  {valid_dt}")
        pro_installed = False
        try:
            import omega_platform  # noqa: F401
            pro_installed = True
        except ImportError:
            pass
        if not pro_installed:
            print("\nPro wheel not installed. Download from https://admin.omegamax.co/pro/dashboard")
            print("Then: pip install ~/Downloads/omega_memory_pro-*.whl")
    else:
        print("Status:      Expired")
        print(f"Key:         {masked}")
        print("\nRun 'omega activate <key>' to reactivate.")


def cmd_export_obsidian(args):
    """Export memories as Obsidian-compatible markdown files."""
    from omega.obsidian_export import export_to_obsidian

    output_dir = getattr(args, "output_dir", "./omega-vault")
    project = getattr(args, "project", None)
    limit = getattr(args, "limit", 0)

    result = export_to_obsidian(
        output_dir=output_dir,
        project=project,
        limit=limit,
    )

    print(f"Exported {result['memories_exported']} memories "
          f"({result['edge_links_created']} edges) to {result['output_dir']}")
    print(f"Index file: {result['index_file']}")


def cmd_eval_retrieval(args):
    """Evaluate retrieval quality with probe queries."""
    from omega.evaluation.retrieval_eval import format_report, run_evaluation

    sample_size = getattr(args, "sample_size", 20)
    top_k = getattr(args, "top_k", 5)
    judge = getattr(args, "judge", False)
    model = getattr(args, "model", "claude-haiku-4-5-20251001")
    seed = getattr(args, "seed", 42)
    output_path = getattr(args, "output", None)
    use_json = _use_json(args)

    if judge:
        try:
            import anthropic  # noqa: F401
        except ImportError:
            print("Error: --judge requires the 'anthropic' package. Install with: pip install anthropic")
            sys.exit(1)
        if not os.environ.get("ANTHROPIC_API_KEY"):
            print("Error: --judge requires ANTHROPIC_API_KEY environment variable")
            sys.exit(1)

    print(f"Running retrieval evaluation ({sample_size} probes, top-{top_k}, mode={'judge' if judge else 'basic'})...")

    report = run_evaluation(
        sample_size=sample_size,
        top_k=top_k,
        judge=judge,
        model=model,
        seed=seed,
        output_path=output_path,
    )

    if use_json:
        from dataclasses import asdict

        print(json.dumps(asdict(report), indent=2, default=str))
    else:
        print(format_report(report))

    if output_path:
        print(f"\nReport saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        prog="omega",
        description="OMEGA — Persistent memory for AI coding agents",
        epilog="Pro: 98 more tools (coordination, routing, knowledge base). Run 'omega upgrade' for details.",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # --- Memory commands ---
    query_parser = subparsers.add_parser("query", help="Search memories by semantic similarity or exact phrase")
    query_parser.add_argument("query_text", nargs="+", help="Search text")
    query_parser.add_argument("--exact", action="store_true", help="Use FTS5 exact phrase search instead of semantic")
    query_parser.add_argument("--limit", type=int, default=10, help="Max results (default: 10)")
    query_parser.add_argument("--json", action="store_true", help="Output as JSON (also: OMEGA_JSON=1)")

    store_parser = subparsers.add_parser("store", help="Store a memory with a specified type")
    store_parser.add_argument("content", nargs="+", help="Memory content")
    store_parser.add_argument(
        "-t",
        "--type",
        default="memory",
        choices=["memory", "lesson", "decision", "error", "task", "preference"],
        help="Memory type (default: memory)",
    )
    store_parser.add_argument("--json", action="store_true", help="Output as JSON (also: OMEGA_JSON=1)")

    remember_parser = subparsers.add_parser("remember", help="Store a permanent user preference")
    remember_parser.add_argument("text", nargs="+", help="Preference text")
    remember_parser.add_argument("--json", action="store_true", help="Output as JSON (also: OMEGA_JSON=1)")

    timeline_parser = subparsers.add_parser("timeline", help="Show memory timeline grouped by day")
    timeline_parser.add_argument("--days", type=int, default=7, help="Number of days to show (default: 7)")
    timeline_parser.add_argument("--json", action="store_true", help="Output as JSON (also: OMEGA_JSON=1)")

    # --- Admin commands ---
    setup_parser = subparsers.add_parser("setup", help="Set up OMEGA: download model, initialize DB")
    setup_parser.add_argument(
        "--download-model",
        action="store_true",
        help="Download bge-small-en-v1.5 ONNX model (upgrade from all-MiniLM-L6-v2)",
    )
    setup_parser.add_argument(
        "--client", choices=["claude-code", "claude-desktop", "cursor", "windsurf", "cline", "codex", "antigravity", "venv"], help="Configure a specific client (MCP registration, hooks)"
    )
    setup_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would change without writing any files",
    )
    setup_parser.add_argument(
        "--hooks-only",
        action="store_true",
        help="Configure hooks and CLAUDE.md WITHOUT MCP server (saves ~600MB RAM per session)",
    )

    status_parser = subparsers.add_parser("status", help="Show memory count, store size, model status")
    status_parser.add_argument("--json", action="store_true", help="Output as JSON (also: OMEGA_JSON=1)")

    doctor_parser = subparsers.add_parser("doctor", help="Verify installation: import, model, database")
    doctor_parser.add_argument("--client", choices=["claude-code", "claude-desktop", "cursor", "windsurf", "cline", "codex", "antigravity", "venv"], help="Include client-specific checks (MCP, hooks)")
    doctor_parser.add_argument("--json", action="store_true", help="Output as JSON (also: OMEGA_JSON=1)")

    subparsers.add_parser("migrate", help="Copy MAGMA data to OMEGA (non-destructive)")
    migrate_db_parser = subparsers.add_parser("migrate-db", help="Migrate JSON graphs to SQLite backend")
    migrate_db_parser.add_argument("--force", action="store_true", help="Overwrite existing SQLite database")
    subparsers.add_parser("reingest", help="Load store.jsonl entries into graph system")
    consolidate_parser = subparsers.add_parser("consolidate", help="Deduplicate, prune, and optimize memory")
    consolidate_parser.add_argument(
        "--prune-days", type=int, default=30, help="Prune entries older than N days with 0 access (default: 30)"
    )
    subparsers.add_parser("backup", help="Back up omega.db to ~/.omega/backups/ (keeps last 5)")
    compact_parser = subparsers.add_parser("compact", help="Cluster and summarize related memories")
    compact_parser.add_argument(
        "-t",
        "--type",
        default="lesson_learned",
        choices=["lesson_learned", "decision", "error_pattern", "task_completion"],
        help="Event type to compact (default: lesson_learned)",
    )
    compact_parser.add_argument("--threshold", type=float, default=0.60, help="Similarity threshold (default: 0.60)")
    compact_parser.add_argument(
        "--dry-run", action="store_true", help="Show what would be compacted without changing data"
    )
    stats_parser = subparsers.add_parser("stats", help="Show memory type distribution and health summary")
    stats_parser.add_argument("--json", action="store_true", help="Output as JSON (also: OMEGA_JSON=1)")
    stats_parser.add_argument("--card", action="store_true", help="Show a shareable stats card")
    activity_parser = subparsers.add_parser("activity", help="Show recent session activity overview")
    activity_parser.add_argument("--days", type=int, default=7, help="Number of days to show (default: 7)")
    activity_parser.add_argument("--json", action="store_true", help="Output as JSON (also: OMEGA_JSON=1)")
    logs_parser = subparsers.add_parser("logs", help="Show recent hook errors from hooks.log")
    logs_parser.add_argument("-n", "--lines", type=int, default=50, help="Number of lines to show (default: 50)")
    validate_parser = subparsers.add_parser("validate", help="Validate omega.db integrity (SQLite + FTS5)")
    validate_parser.add_argument("--repair", action="store_true", help="Attempt to repair FTS5 index if corrupted")
    serve_parser = subparsers.add_parser("serve", help="Run MCP server (stdio or HTTP daemon)")
    serve_parser.add_argument("--daemon", action="store_true", help="Run as HTTP daemon (OMEGA_TRANSPORT=http)")
    serve_parser.add_argument("--no-condensed", action="store_true", help="Disable condensed mode (expose all tools individually instead of meta-tools)")
    serve_sub = serve_parser.add_subparsers(dest="serve_command", help="Daemon management")
    serve_sub.add_parser("install", help="Install launchd daemon and load it")
    serve_sub.add_parser("uninstall", help="Unload and remove launchd daemon")
    serve_sub.add_parser("status", help="Check daemon status and health")
    serve_sub.add_parser("migrate-config", help="Migrate ~/.claude.json from stdio to http")
    serve_sub.add_parser("restore-config", help="Restore ~/.claude.json from backup")

    # --- Proxy commands (jit-proxy daemon) ---
    proxy_parser = subparsers.add_parser("proxy", help="Manage jit-proxy daemon")
    proxy_sub = proxy_parser.add_subparsers(dest="proxy_command", help="Proxy daemon management")
    proxy_sub.add_parser("install", help="Install jit-proxy launchd daemon")
    proxy_sub.add_parser("uninstall", help="Unload and remove jit-proxy daemon")
    proxy_sub.add_parser("status", help="Check jit-proxy daemon status and health")
    proxy_sub.add_parser("migrate-config", help="Migrate ~/.claude.json jit-proxy from stdio to http")
    proxy_sub.add_parser("restore-config", help="Restore ~/.claude.json from backup")

    # --- Hooks commands ---
    hooks_parser = subparsers.add_parser("hooks", help="Manage Claude Code hooks")
    hooks_sub = hooks_parser.add_subparsers(dest="hooks_command", help="Hook subcommands")
    hooks_sub.add_parser("setup", help="Configure hooks in ~/.claude/settings.json")
    hooks_sub.add_parser("path", help="Print the hooks directory path")
    hooks_sub.add_parser("doctor", help="Check hook configuration health")

    # --- Embedding daemon commands ---
    embed_parser = subparsers.add_parser("embed-daemon", help="Manage shared embedding daemon")
    embed_sub = embed_parser.add_subparsers(dest="embed_command", help="Daemon subcommands")
    embed_sub.add_parser("start", help="Start the embedding daemon")
    embed_sub.add_parser("stop", help="Stop the embedding daemon")
    embed_sub.add_parser("status", help="Show daemon status")

    # --- License commands ---
    subparsers.add_parser("upgrade", help="Open Pro purchase page in browser")

    activate_parser = subparsers.add_parser("activate", help="Activate a Pro license key")
    activate_parser.add_argument("key", help="License key (OMEGA-PRO-...)")

    license_parser = subparsers.add_parser("license", help="Show Pro license status")
    license_parser.add_argument("--deactivate", action="store_true", help="Remove local license")

    # --- Reminder commands (experimental) ---
    remind_parser = subparsers.add_parser("remind", help="Manage time-based reminders (experimental)")
    remind_sub = remind_parser.add_subparsers(dest="remind_command", help="Reminder subcommands")

    remind_set_parser = remind_sub.add_parser("set", help="Set a new reminder")
    remind_set_parser.add_argument("text", nargs="+", help="Reminder text")
    remind_set_parser.add_argument("-d", "--duration", required=True, help="Duration: 1h, 30m, 2d, 1w, 1d12h")
    remind_set_parser.add_argument("--context", help="Optional context for the reminder")

    remind_list_parser = remind_sub.add_parser("list", help="List reminders")
    remind_list_parser.add_argument(
        "--status",
        choices=["pending", "fired", "dismissed", "all"],
        help="Filter by status (default: pending + fired)",
    )

    remind_check_parser = remind_sub.add_parser("check", help="Check for due reminders")
    remind_check_parser.add_argument("--notify", action="store_true", help="Send macOS notification for due reminders")

    remind_dismiss_parser = remind_sub.add_parser("dismiss", help="Dismiss a reminder")
    remind_dismiss_parser.add_argument("reminder_id", help="Reminder ID to dismiss")

    # --- Knowledge commands ---
    knowledge_parser = subparsers.add_parser("knowledge", aliases=["kb"], help="Knowledge base management")
    knowledge_sub = knowledge_parser.add_subparsers(dest="kb_command", help="Knowledge subcommands")
    scan_parser = knowledge_sub.add_parser("scan", help="Scan documents folder for new/changed files")
    scan_parser.add_argument("--dir", help="Custom directory to scan (default: ~/.omega/documents/)")
    knowledge_sub.add_parser("list", help="List all ingested documents")
    knowledge_search_parser = knowledge_sub.add_parser("search", help="Search ingested documents")
    knowledge_search_parser.add_argument("query", nargs="+", help="Search query")
    knowledge_search_parser.add_argument("--limit", type=int, default=5, help="Max results (default: 5)")
    sync_kb_parser = knowledge_sub.add_parser("sync-kb", help="Sync pending files from cloud KB queue")
    sync_kb_parser.add_argument("--batch-size", type=int, default=10, help="Max items to process (default: 10)")

    # --- Cloud commands ---
    cloud_parser = subparsers.add_parser("cloud", help="Cloud sync and mobile access")
    cloud_sub = cloud_parser.add_subparsers(dest="cloud_command", help="Cloud subcommands")

    cloud_setup_parser = cloud_sub.add_parser("setup", help="Configure Supabase connection")
    cloud_setup_parser.add_argument("--url", help="Supabase project URL")
    cloud_setup_parser.add_argument("--key", help="Supabase anon key")
    cloud_setup_parser.add_argument("--service-key", help="Supabase service role key (optional)")

    cloud_sub.add_parser("sync", help="Sync local data to Supabase cloud")
    cloud_sub.add_parser("status", help="Show cloud sync status")
    cloud_sub.add_parser("schema", help="Print Supabase SQL schema")
    cloud_sub.add_parser("verify", help="Verify Supabase connection")
    cloud_sub.add_parser("pull", help="Pull memories and documents from Supabase cloud")

    # --- Mobile commands ---
    mobile_parser = subparsers.add_parser("mobile", help="Mobile access via mcp-proxy + Tailscale")
    mobile_sub = mobile_parser.add_subparsers(dest="mobile_command", help="Mobile subcommands")
    mobile_sub.add_parser("setup", help="Print setup instructions for mobile access")
    mobile_serve_parser = mobile_sub.add_parser("serve", help="Start mcp-proxy HTTP server for mobile access")
    mobile_serve_parser.add_argument("--port", type=int, default=8089, help="HTTP port (default: 8089)")
    mobile_serve_parser.add_argument("--host", default="127.0.0.1", help="Bind address (default: 127.0.0.1)")

    # --- Obsidian export ---
    obsidian_parser = subparsers.add_parser("export-obsidian", help="Export memories as Obsidian-compatible markdown files")
    obsidian_parser.add_argument("--output-dir", default="./omega-vault", help="Root directory for exported vault (default: ./omega-vault)")
    obsidian_parser.add_argument("--project", help="Only export memories for this project")
    obsidian_parser.add_argument("--limit", type=int, default=0, help="Max memories to export (default: all)")

    # --- Evaluation commands ---
    eval_parser = subparsers.add_parser("eval-retrieval", help="Evaluate retrieval quality with probe queries")
    eval_parser.add_argument("--sample-size", type=int, default=20, help="Number of memories to probe (default: 20)")
    eval_parser.add_argument("--top-k", type=int, default=5, help="Results per probe (default: 5)")
    eval_parser.add_argument("--judge", action="store_true", help="Use LLM to generate queries and score relevance")
    eval_parser.add_argument("--model", default="claude-haiku-4-5-20251001", help="LLM model for judge mode")
    eval_parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible sampling (default: 42)")
    eval_parser.add_argument("--output", help="Save JSON report to this path")
    eval_parser.add_argument("--json", action="store_true", help="Output as JSON to stdout (also: OMEGA_JSON=1)")

    args = parser.parse_args()

    commands = {
        "query": cmd_query,
        "store": cmd_store,
        "remember": cmd_remember,
        "timeline": cmd_timeline,
        "setup": cmd_setup,
        "status": cmd_status,
        "doctor": cmd_doctor,
        "migrate": cmd_migrate,
        "migrate-db": cmd_migrate_db,
        "reingest": cmd_reingest,
        "consolidate": cmd_consolidate,
        "backup": cmd_backup,
        "compact": cmd_compact,
        "stats": cmd_stats,
        "activity": cmd_activity,
        "logs": cmd_logs,
        "validate": cmd_validate,
        "serve": cmd_serve,
        "proxy": cmd_proxy,
        "hooks": cmd_hooks,
        "embed-daemon": cmd_embed_daemon,
        "upgrade": cmd_upgrade,
        "activate": cmd_activate,
        "license": cmd_license,
        "remind": cmd_remind,
        "knowledge": cmd_knowledge,
        "kb": cmd_knowledge,
        "cloud": cmd_cloud,
        "mobile": cmd_mobile,
        "eval-retrieval": cmd_eval_retrieval,
        "export-obsidian": cmd_export_obsidian,
    }

    # Wire plugin CLI commands (omega-pro, etc.)
    try:
        from omega.plugins import discover_plugins
        for plugin in discover_plugins():
            for cmd_name, setup_func in getattr(plugin, "CLI_COMMANDS", []):
                if cmd_name not in commands:
                    try:
                        setup_func(subparsers)
                        commands[cmd_name] = getattr(plugin, f"cmd_{cmd_name}", None)
                    except Exception as e:
                        print(f"Warning: plugin CLI command '{cmd_name}' failed: {e}", file=sys.stderr)
    except Exception as e:
        logger.debug("Plugin CLI registration failed: %s", e)

    if args.command in commands:
        commands[args.command](args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
