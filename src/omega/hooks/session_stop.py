#!/usr/bin/env python3
"""OMEGA SessionStop hook — Generate and store session summary on exit."""
import json
import os
import sys
import time
import traceback
from datetime import datetime, timedelta, timezone
from pathlib import Path


# Critical tools that agents SHOULD call at least once per session.
# Scored: each hit = 1 point, total / len = percentage.
CRITICAL_TOOLS = [
    "omega_reflect",          # Contradiction/stale detection — 0 calls ever
    "omega_decision_query",   # Check active decisions before domain work — 0 calls
    "omega_file_check",       # Conflict check before edits — 5 calls / 931 edits
    "omega_checkpoint",       # Save state at 70% context — 4 calls ever
    "omega_coord_status",     # Check peers before taking work — 10 calls
]


def _build_utilization_report(tool_calls: list) -> dict:
    """Score which critical OMEGA tools the agent used this session."""
    called = set(tool_calls)
    # Normalize: strip mcp__omega-memory__ prefix if present
    normalized = set()
    for t in called:
        if t.startswith("mcp__omega-memory__"):
            normalized.add(t.replace("mcp__omega-memory__", ""))
        else:
            normalized.add(t)

    missed = [t for t in CRITICAL_TOOLS if t not in normalized]
    hit_count = len(CRITICAL_TOOLS) - len(missed)
    score = round(hit_count / len(CRITICAL_TOOLS) * 100) if CRITICAL_TOOLS else 100

    return {"score": score, "missed": missed, "hit": hit_count, "total": len(CRITICAL_TOOLS)}


def _get_session_tool_names(session_id: str) -> list:
    """Get list of tool names called in this session from coord_audit."""
    try:
        import sqlite3
        db_path = os.path.expanduser("~/.omega/omega.db")
        conn = sqlite3.connect(db_path, timeout=2)
        rows = conn.execute(
            "SELECT DISTINCT tool_name FROM coord_audit WHERE session_id = ?",
            (session_id,),
        ).fetchall()
        conn.close()
        return [r[0] for r in rows]
    except Exception:
        return []


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


def _get_activity_counts(session_id: str) -> dict:
    """Count memories by event_type for this session."""
    try:
        from omega.bridge import _get_store
        store = _get_store()
        return store.get_session_event_counts(session_id)
    except Exception:
        return {}


def _get_surfaced_count(session_id: str) -> int:
    """Read and clean up the surfacing counter file."""
    try:
        marker = Path.home() / ".omega" / f"session-{session_id}.surfaced"
        if marker.exists():
            count = marker.stat().st_size
            marker.unlink()
            return count
    except Exception:
        pass
    return 0


def _get_surfaced_details(session_id: str) -> tuple:
    """Read unique memory IDs and file count from surfaced.json."""
    unique_ids = 0
    unique_files = 0
    try:
        json_path = Path.home() / ".omega" / f"session-{session_id}.surfaced.json"
        if json_path.exists():
            data = json.loads(json_path.read_text())
            all_ids = set()
            for ids in data.values():
                all_ids.update(ids)
            unique_ids = len(all_ids)
            unique_files = len(data)
    except Exception:
        pass
    return unique_ids, unique_files


def _print_activity_report(session_id: str):
    """Print session memory activity summary with productivity recap."""
    if not session_id:
        return
    counts = _get_activity_counts(session_id)
    surfaced = _get_surfaced_count(session_id)
    surfaced_unique_ids, surfaced_unique_files = _get_surfaced_details(session_id)
    if not counts and surfaced == 0:
        return

    captured = sum(counts.values())
    parts = [f"{captured} captured"]
    _LABELS = {
        "error_pattern": ("error", "errors"),
        "decision": ("decision", "decisions"),
        "lesson_learned": ("lesson learned", "lessons learned"),
    }
    for key, (singular, plural) in _LABELS.items():
        n = counts.get(key, 0)
        if n:
            parts.append(f"{n} {plural if n > 1 else singular}")
    if surfaced:
        parts.append(f"{surfaced} surfaced")
    print(f"\n## Session complete — {' | '.join(parts)}")

    # Unique recall stats
    if surfaced_unique_ids > 0:
        print(f"  Recalled: {surfaced_unique_ids} unique memories across {surfaced_unique_files} file{'s' if surfaced_unique_files != 1 else ''}")

    # Weekly recap
    try:
        from omega.bridge import _get_store
        store = _get_store()
        total = store.node_count()

        from datetime import timedelta, timezone
        week_cutoff = (datetime.now(timezone.utc) - timedelta(days=7)).isoformat()
        row = store._conn.execute(
            "SELECT COUNT(DISTINCT session_id) FROM memories "
            "WHERE created_at >= ? AND session_id IS NOT NULL",
            (week_cutoff,),
        ).fetchone()
        weekly_sessions = row[0] if row else 0

        row2 = store._conn.execute(
            "SELECT COUNT(*) FROM memories WHERE created_at >= ?",
            (week_cutoff,),
        ).fetchone()
        weekly_memories = row2[0] if row2 else 0

        # Prior week count for growth
        prev_cutoff = (datetime.now(timezone.utc) - timedelta(days=14)).isoformat()
        row3 = store._conn.execute(
            "SELECT COUNT(*) FROM memories WHERE created_at >= ? AND created_at < ?",
            (prev_cutoff, week_cutoff),
        ).fetchone()
        prev_week_memories = row3[0] if row3 else 0

        recap_parts = []
        if weekly_sessions > 1:
            recap_parts.append(f"{weekly_sessions} sessions this week")
        if weekly_memories > 0:
            recap_parts.append(f"{weekly_memories} memories this week")
        recap_parts.append(f"{total} total")
        print(f"  Recap: {', '.join(recap_parts)}")

        # Week-over-week growth
        if prev_week_memories > 0 and weekly_memories > 0:
            growth_pct = ((weekly_memories - prev_week_memories) / prev_week_memories) * 100
            sign = "+" if growth_pct >= 0 else ""
            print(f"  Growth: {sign}{growth_pct:.0f}% vs last week")
    except Exception:
        pass

    # Utilization scorecard
    try:
        tool_names = _get_session_tool_names(session_id)
        report = _build_utilization_report(tool_names)
        if report["missed"]:
            print(f"  Utilization: {report['score']}% ({report['hit']}/{report['total']} critical tools used)")
            print(f"  Unused: {', '.join(report['missed'])}")
    except Exception:
        pass

    # Pro upgrade nudge -- frequency scales with memory count (closer to limit = more frequent)
    try:
        pro_available = False
        try:
            from omega.plugins import has_capability
            pro_available = has_capability("pro_tools")
        except Exception:
            pass
        if not pro_available:
            mem_count = 0
            try:
                from omega.bridge import _get_store
                _s = _get_store()
                mem_count = _s.node_count() if hasattr(_s, 'node_count') else 0
            except Exception:
                pass

            from omega.telemetry import _load as _telem_load
            tdata = _telem_load()
            session_total = tdata.get("sessions", {}).get("total", 0)

            # Graduated urgency: more memories = more frequent nudge
            show = False
            if mem_count >= 1800:
                show = True  # every session
                print(f"  {mem_count:,}/2,000 memories -- approaching free tier limit")
                print("  Search quality degrades at 2,000. Run 'omega upgrade' for unlimited.")
            elif mem_count >= 1500:
                show = session_total % 3 == 0  # every 3rd session
                if show:
                    print(f"  {mem_count:,}/2,000 memories (75%). Pro removes limits. Run 'omega upgrade'")
            elif mem_count >= 1000:
                show = session_total % 5 == 0  # every 5th session
                if show:
                    print(f"  {mem_count:,}/2,000 memories. Pro: unlimited + coordination + routing. Run 'omega upgrade'")
            elif mem_count >= 500:
                show = session_total % 10 == 0  # every 10th session
                if show:
                    print("  Pro: coordination, routing, and 96 more tools. Run 'omega upgrade'")
    except Exception:
        pass


def _build_summary(session_id: str, project: str) -> str:
    """Build a session summary from per-type targeted queries.

    Each category is queried independently with event_type filter.
    session_summary type is excluded entirely to prevent circular refs.
    """
    try:
        from omega.bridge import query_structured
    except ImportError:
        return "Session ended"

    decisions = query_structured(
        query_text="decisions made",
        limit=5,
        session_id=session_id,
        project=project,
        event_type="decision",
    )
    errors = query_structured(
        query_text="errors encountered",
        limit=3,
        session_id=session_id,
        project=project,
        event_type="error_pattern",
    )
    tasks = query_structured(
        query_text="completed tasks",
        limit=3,
        session_id=session_id,
        project=project,
        event_type="task_completion",
    )

    if not decisions and not errors and not tasks:
        return "Session ended (no captured activity)"

    parts = []
    if decisions:
        items = [m.get("content", "")[:120] for m in decisions[:3]]
        parts.append(f"Decisions ({len(decisions)}): " + "; ".join(items))
    if errors:
        items = [m.get("content", "")[:120] for m in errors[:3]]
        parts.append(f"Errors ({len(errors)}): " + "; ".join(items))
    if tasks:
        items = [m.get("content", "")[:120] for m in tasks[:3]]
        parts.append(f"Tasks ({len(tasks)}): " + "; ".join(items))

    if not parts:
        return "Session ended"

    return " | ".join(parts)[:600]


def _get_reflect_store():
    """Lazy import store for reflection. Separated for testability."""
    from omega.bridge import _get_store
    return _get_store()


# Lazy import: may be None if omega.reflect is not installed
try:
    from omega.reflect import find_contradictions
except ImportError:
    find_contradictions = None


def _auto_reflect(session_id: str, project: str) -> dict:
    """Run contradiction detection automatically at session end.
    Returns summary dict with contradictions_found count."""
    try:
        store = _get_reflect_store()
        result = find_contradictions(store, topic="recent decisions", limit=10)

        contradictions = result.get("contradictions", [])
        if contradictions:
            # Store a summary for the next session to see
            summary = f"Auto-reflect found {len(contradictions)} potential contradiction(s):\n"
            for c in contradictions[:3]:
                summary += f"- '{c.get('memory_a_content', '')[:80]}' vs '{c.get('memory_b_content', '')[:80]}'\n"

            try:
                from omega.bridge import auto_capture
                auto_capture(
                    content=summary,
                    event_type="lesson_learned",
                    session_id=session_id,
                    project=project,
                    metadata={"source": "auto_reflect", "contradiction_count": len(contradictions)},
                )
            except Exception:
                pass

        return {"contradictions_found": len(contradictions)}
    except Exception:
        return {"contradictions_found": 0}


def _auto_feedback_on_surfaced(session_id: str):
    """Auto-record 'helpful' feedback for memories surfaced during active work."""
    if not session_id:
        return
    json_path = Path.home() / ".omega" / f"session-{session_id}.surfaced.json"
    if not json_path.exists():
        return
    try:
        data = json.loads(json_path.read_text())
        # Collect all unique memory IDs across all files
        all_ids = set()
        for ids in data.values():
            all_ids.update(ids)

        if not all_ids:
            return

        from omega.bridge import record_feedback
        count = 0
        for mid in list(all_ids)[:10]:  # Cap at 10 feedback calls
            try:
                record_feedback(mid, "helpful", "Auto: surfaced during active work")
                count += 1
            except Exception:
                pass

        # Clean up the JSON file
        json_path.unlink(missing_ok=True)
    except ImportError:
        pass
    except Exception as e:
        _log_hook_error("auto_feedback_surfaced", e)
    finally:
        # Always try to clean up
        try:
            if json_path.exists():
                json_path.unlink()
        except Exception:
            pass


def _capture_usage_to_supabase(session_id: str, project_dir: str):
    """Push session usage data from ~/.claude.json to Supabase. Fire-and-forget."""
    if not session_id:
        return
    try:
        import urllib.request
        import urllib.error

        # Read ~/.claude.json for last* session metrics
        claude_json_path = Path.home() / ".claude.json"
        if not claude_json_path.exists():
            return

        claude_data = json.loads(claude_json_path.read_text())
        projects = claude_data.get("projects", {})

        # Find the project entry; keys are absolute project paths.
        project_entry = None
        for path_key, entry in projects.items():
            # Exact match or normalized match (avoid substring false positives)
            if project_dir and (path_key == project_dir or path_key.rstrip("/") == project_dir.rstrip("/")):
                project_entry = entry
                break

        if not project_entry:
            return

        last_session_id = project_entry.get("lastSessionId", "")
        # Only capture if this matches our session (avoid stale data)
        if last_session_id and last_session_id != session_id:
            # Fall back: use the data anyway if session_id looks like a subagent
            if not session_id.startswith("agent-"):
                return

        # Load Supabase credentials from env vars or ~/.omega/secrets.json
        sb_url = os.environ.get("SUPABASE_URL", "")
        sb_key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "")
        if not sb_url or not sb_key:
            secrets_file = Path.home() / ".omega" / "secrets.json"
            if secrets_file.exists():
                try:
                    secrets_data = json.loads(secrets_file.read_text())
                    if not sb_url:
                        sb_url = secrets_data.get("SUPABASE_URL", secrets_data.get("supabase_url", ""))
                    if not sb_key:
                        sb_key = secrets_data.get("SUPABASE_SERVICE_ROLE_KEY", secrets_data.get("supabase_key", ""))
                except (json.JSONDecodeError, OSError):
                    pass
        if not sb_url or not sb_key:
            return

        # Extract project name from path
        project_name = Path(project_dir).name if project_dir else None

        # Build model cost breakdown from lastModelUsage
        model_usage = project_entry.get("lastModelUsage", {})
        cost_by_model = {}
        for model_id, stats in model_usage.items():
            cost = stats.get("costUSD", 0)
            if cost:
                # Clean up model ID for display
                short_name = model_id
                if "opus" in model_id.lower():
                    short_name = "Claude Opus"
                elif "sonnet" in model_id.lower():
                    short_name = "Claude Sonnet"
                elif "haiku" in model_id.lower():
                    short_name = "Claude Haiku"
                cost_by_model[short_name] = round(cost, 6)

        # Compute session timestamps from duration
        duration_ms = project_entry.get("lastDuration", 0)
        duration_s = round(duration_ms / 1000, 2) if duration_ms else None
        now = datetime.now(timezone.utc)
        session_end = now.isoformat()
        session_start = None
        if duration_ms:
            session_start = (now - timedelta(milliseconds=duration_ms)).isoformat()

        # Try to read rich metadata from session-summaries.jsonl
        files_modified = []
        tasks_completed = []
        git_commits = []
        try:
            summaries_path = Path.home() / ".claude" / "session-summaries.jsonl"
            if summaries_path.exists():
                # Read last line matching this session_id
                for line in reversed(summaries_path.read_text().splitlines()):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        s = json.loads(line)
                        sid = s.get("sessionId") or s.get("id", "")
                        if sid == session_id:
                            files_modified = s.get("filesModified", [])
                            tasks_completed = s.get("tasksCompleted", [])
                            git_commits = s.get("gitCommits", [])
                            # Use more precise timestamps if available
                            if s.get("startTime"):
                                session_start = s["startTime"]
                            if s.get("endTime"):
                                session_end = s["endTime"]
                            break
                    except json.JSONDecodeError:
                        continue
        except Exception:
            pass

        row = {
            "session_id": session_id,
            "project_name": project_name,
            "project_path": project_dir,
            "total_cost_usd": project_entry.get("lastCost", 0),
            "input_tokens": project_entry.get("lastTotalInputTokens", 0),
            "output_tokens": project_entry.get("lastTotalOutputTokens", 0),
            "cache_read_tokens": project_entry.get("lastTotalCacheReadInputTokens", 0),
            "cache_miss_tokens": project_entry.get("lastTotalCacheCreationInputTokens", 0),
            "duration_seconds": duration_s,
            "cost_by_model": json.dumps(cost_by_model),
            "files_modified": json.dumps(files_modified[:50] if isinstance(files_modified, list) else []),
            "tasks_completed": json.dumps(tasks_completed[:20] if isinstance(tasks_completed, list) else []),
            "git_commits": json.dumps(git_commits[:20] if isinstance(git_commits, list) else []),
            "session_start": session_start,
            "session_end": session_end,
        }

        url = f"{sb_url}/rest/v1/session_usage?on_conflict=session_id"
        headers = {
            "apikey": sb_key,
            "Authorization": f"Bearer {sb_key}",
            "Content-Type": "application/json",
            "Prefer": "resolution=merge-duplicates",
        }
        body = json.dumps([row]).encode("utf-8")
        req = urllib.request.Request(url, data=body, headers=headers, method="POST")
        urllib.request.urlopen(req, timeout=3)

    except Exception as e:
        _log_hook_error("capture_usage_supabase", e)


def _build_project_status(session_id: str, project: str):
    """Build a project status snapshot from session activity.

    Returns structured text or None if insufficient data.
    """
    if not project:
        return None
    try:
        from omega.bridge import query_structured
    except ImportError:
        return None

    decisions = query_structured(
        query_text="decisions made",
        limit=5,
        session_id=session_id,
        project=project,
        event_type="decision",
    )
    tasks = query_structured(
        query_text="completed tasks",
        limit=5,
        session_id=session_id,
        project=project,
        event_type="task_completion",
    )

    if not decisions and not tasks:
        return None  # Not enough activity for a status snapshot

    parts = [f"Project: {Path(project).name}"]
    if decisions:
        items = [m.get("content", "")[:150] for m in decisions[:3]]
        parts.append("Key decisions: " + "; ".join(items))
    if tasks:
        items = [m.get("content", "")[:150] for m in tasks[:3]]
        parts.append("Completed: " + "; ".join(items))

    return " | ".join(parts)[:600]


def main():
    session_id = os.environ.get("SESSION_ID", "")
    project = os.environ.get("PROJECT_DIR", os.getcwd())

    _capture_usage_to_supabase(session_id, project)
    _auto_feedback_on_surfaced(session_id)
    _print_activity_report(session_id)

    # Auto-reflect: detect contradictions (Part C — omega_reflect has 0 agent calls)
    try:
        reflect_result = _auto_reflect(session_id, project)
        if reflect_result["contradictions_found"] > 0:
            print(f"  Auto-reflect: {reflect_result['contradictions_found']} contradiction(s) detected. Check next session start.")
    except Exception:
        pass

    # Cloud push fallback — when the hook_server is down (OOM), this fast_hook
    # path is the only session_stop that fires. Push to cloud here too.
    try:
        from omega_platform.cloud.sync import get_sync
        get_sync().sync_all()
        push_marker = Path.home() / ".omega" / "last-cloud-push"
        push_marker.write_text(datetime.now(timezone.utc).isoformat())
    except Exception:
        pass  # Fail-open: cloud push is best-effort

    if os.environ.get("OMEGA_NO_SESSION_SUMMARY", "").strip() == "1":
        return

    summary = _build_summary(session_id, project)

    try:
        from omega.bridge import auto_capture
        auto_capture(
            content=f"Session summary: {summary}",
            event_type="session_summary",
            metadata={"source": "session_stop_hook", "project": project},
            session_id=session_id,
            project=project,
            ttl_override=3600,  # Match hook server TTL — don't accumulate
        )
    except ImportError:
        pass
    except Exception as e:
        _log_hook_error("session_stop", e)
        print(f"OMEGA session_stop failed: {e}", file=sys.stderr)

    # Auto-generate project_status (will evolve existing if present)
    project_status_text = _build_project_status(session_id, project)
    if project_status_text:
        try:
            from omega.bridge import auto_capture as _ac
            _ac(
                content=project_status_text,
                event_type="project_status",
                session_id=session_id,
                project=project,
                metadata={"source": "session_stop_auto", "project": project},
            )
        except ImportError:
            pass
        except Exception as e:
            _log_hook_error("session_stop_project_status", e)


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


if __name__ == "__main__":
    _t0 = time.monotonic()
    main()
    _log_timing("session_stop", (time.monotonic() - _t0) * 1000)
