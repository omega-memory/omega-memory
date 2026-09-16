#!/usr/bin/env python3
"""OMEGA SessionStart hook — Welcome briefing with recent context."""
import os
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

from omega.hooks._output import emit

try:
    import fcntl
except ImportError:  # Windows: no fcntl, fall back to best-effort no-op locking.
    fcntl = None  # type: ignore[assignment]


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


def _try_acquire_periodic(marker_name: str, min_age_days: int):
    """Atomically check and claim a periodic task via file lock.

    Returns the old marker content (for rollback) if claimed, None if skipped.
    Uses fcntl.flock to prevent concurrent processes from racing.
    """
    omega_dir = Path.home() / ".omega"
    omega_dir.mkdir(parents=True, exist_ok=True)
    marker = omega_dir / marker_name
    lock_path = omega_dir / f"{marker_name}.lock"
    lock_fd = os.open(str(lock_path), os.O_WRONLY | os.O_CREAT, 0o600)
    if fcntl is not None:
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except (OSError, BlockingIOError):
            os.close(lock_fd)
            return None  # Another process holds the lock

    try:
        old_content = None
        if marker.exists():
            old_content = marker.read_text().strip()
            last = datetime.fromisoformat(old_content)
            if last.tzinfo is None:
                last = last.replace(tzinfo=timezone.utc)
            age_days = (datetime.now(timezone.utc) - last).days
            if age_days < min_age_days:
                return None

        # Write marker BEFORE the slow operation to block other processes
        marker.write_text(datetime.now(timezone.utc).isoformat())
        return old_content if old_content else ""
    except Exception:
        return None
    finally:
        if fcntl is not None:
            try:
                fcntl.flock(lock_fd, fcntl.LOCK_UN)
            except OSError:
                pass
        os.close(lock_fd)


def _rollback_marker(marker_name: str, old_content: str):
    """Restore a marker to its previous value on failure."""
    marker = Path.home() / ".omega" / marker_name
    if old_content == "":
        marker.unlink(missing_ok=True)
    else:
        marker.write_text(old_content)


def _maybe_auto_consolidate():
    """Run lightweight consolidation if >3 days since last run."""
    try:
        old = _try_acquire_periodic("last-consolidate", 3)
        if old is None:
            return
        try:
            from omega.bridge import consolidate
            consolidate(prune_days=7, max_summaries=30)
        except ImportError:
            _rollback_marker("last-consolidate", old)
        except Exception:
            _rollback_marker("last-consolidate", old)
            raise
    except ImportError:
        pass
    except Exception as e:
        _log_hook_error("auto_consolidate", e)


def _maybe_auto_backup():
    """Export a backup if >7 days since last backup. Keep last 4."""
    try:
        old = _try_acquire_periodic("last-backup", 7)
        if old is None:
            return
        try:
            backup_dir = Path.home() / ".omega" / "backups"
            backup_dir.mkdir(parents=True, exist_ok=True)
            dest = backup_dir / f"omega-{datetime.now(timezone.utc).strftime('%Y-%m-%d')}.json"
            from omega.bridge import export_memories
            export_memories(filepath=str(dest))
            backups = sorted(backup_dir.glob("omega-*.json"), key=lambda p: p.name, reverse=True)
            for b in backups[4:]:
                b.unlink()
        except ImportError:
            _rollback_marker("last-backup", old)
        except Exception:
            _rollback_marker("last-backup", old)
            raise
    except ImportError:
        pass
    except Exception as e:
        _log_hook_error("auto_backup", e)


def _maybe_auto_compact():
    """Run compaction of high-volume memory types if >3 days since last run."""
    try:
        old = _try_acquire_periodic("last-compact", 3)
        if old is None:
            return
        try:
            from omega.bridge import compact
            for etype in ("advisor_insight", "lesson_learned", "decision",
                          "observation", "session_summary", "handoff", "task_completion"):
                compact(event_type=etype, similarity_threshold=0.50, min_cluster_size=2)
        except ImportError:
            _rollback_marker("last-compact", old)
        except Exception:
            _rollback_marker("last-compact", old)
            raise
    except ImportError:
        pass
    except Exception as e:
        _log_hook_error("auto_compact", e)


def _maybe_analyze_behavior():
    """Run behavioral pattern extraction if >3 days since last run."""
    try:
        old = _try_acquire_periodic("last-behavioral-analysis", 3)
        if old is None:
            return
        try:
            from omega.behavioral import analyze_and_store
            analyze_and_store()
        except ImportError:
            _rollback_marker("last-behavioral-analysis", old)
        except Exception:
            _rollback_marker("last-behavioral-analysis", old)
            raise
    except ImportError:
        pass
    except Exception as e:
        _log_hook_error("behavioral_analysis", e)


def run(payload: dict) -> None:
    """Run periodic maintenance and emit the welcome briefing for one session."""
    project = payload.get("project") or payload.get("cwd") or os.getcwd()
    session_id = payload.get("session_id", "")

    # Auto-consolidation check (lightweight, max once per 3 days)
    _maybe_auto_consolidate()

    # Auto-compaction check (max once per 3 days)
    _maybe_auto_compact()

    # Auto-backup check (max once per 7 days)
    _maybe_auto_backup()

    # Behavioral pattern extraction (max once per 3 days)
    _maybe_analyze_behavior()

    try:
        from omega.bridge import welcome
        result = welcome(session_id=session_id, project=project)
    except ImportError:
        emit("OMEGA not installed. Run: pip install omega-memory && omega setup")
        return
    except Exception as e:
        _log_hook_error("session_start", e)
        emit(f"OMEGA welcome failed: {e}")
        return

    memory_count = result.get("memory_count", 0)
    recent = result.get("recent_memories", [])

    emit(f"## Welcome back! OMEGA ready — {memory_count} memories")

    # First-time user "Aha" moment
    if memory_count == 0:
        emit("")
        emit("OMEGA captures decisions, lessons, and errors automatically as you work.")
        emit("Next session, it surfaces relevant context when you edit the same files.")
        emit("")
        emit("**Quick start:**")
        emit('- Say "remember that we always use TypeScript strict mode" to store a preference')
        emit("- Make a decision and OMEGA captures it automatically")
        emit("- Encounter an error, and OMEGA stores the pattern for future recall")
        emit("")
        emit("After this session ends, you'll see exactly what was captured.")
    elif memory_count <= 10:
        emit(f"  OMEGA has {memory_count} memories from your first sessions. These will surface when you edit related files.")
        try:
            from omega.bridge import type_stats as _ts_first
            first_stats = _ts_first()
            stat_parts = []
            for k, v in sorted(first_stats.items(), key=lambda x: x[1], reverse=True):
                if v > 0 and k != "session_summary":
                    stat_parts.append(f"{v} {k.replace('_', ' ')}")
            if stat_parts:
                emit(f"  Captured so far: {', '.join(stat_parts[:4])}")
        except Exception:
            pass

    # Health pulse
    try:
        from datetime import timezone
        from omega.bridge import _get_store, status as omega_status
        health = omega_status()
        health_label = "ok" if health.get("ok") else health.get("status", "unknown")

        store = _get_store()
        edge_count = store.edge_count()
        last_ts = store.get_last_capture_time()
        if last_ts:
            last_dt = datetime.fromisoformat(last_ts.replace("Z", "+00:00"))
            if last_dt.tzinfo is None:
                last_dt = last_dt.replace(tzinfo=timezone.utc)
            delta = datetime.now(timezone.utc) - last_dt
            secs = delta.total_seconds()
            ago = (f"{int(secs)}s ago" if secs < 60
                   else f"{int(secs/60)}m ago" if secs < 3600
                   else f"{int(secs/3600)}h ago" if secs < 86400
                   else f"{int(secs/86400)}d ago")
        else:
            ago = "never"
        node_count = store.count()
        if node_count > 0:
            ratio = edge_count / node_count
            graph_label = "rich" if ratio >= 1.5 else ("good" if ratio >= 0.5 else "sparse")
            graph_info = f" | graph: {graph_label} ({edge_count:,} edges)"
        else:
            graph_info = ""
        emit(f"Health: {health_label} | Last capture: {ago}{graph_info}")
    except Exception:
        pass

    # Behavioral patterns (habits) — with confidence decay and status
    try:
        from omega.behavioral import effective_confidence
        from omega.bridge import _get_store as _get_store_habits
        habit_store = _get_store_habits()
        habits = habit_store.get_by_type("behavioral_pattern", limit=10)
        surfaced = []
        for h in habits:
            meta = h.metadata or {}
            if meta.get("suppressed"):
                continue
            raw_conf = meta.get("confidence", 0)
            last_ev = meta.get("last_evidence_at") or meta.get("captured_at", "")
            eff_conf = effective_confidence(raw_conf, last_ev)
            if eff_conf >= 0.7:
                status = "confirmed" if meta.get("user_confirmed") else "inferred"
                surfaced.append((h, eff_conf, status))
            if len(surfaced) >= 3:
                break
        if surfaced:
            emit("\n[HABITS] Inferred from your behavior:")
            for h, conf, status in surfaced:
                emit(f"  - {h.content} ({status}, {conf:.0%})")
    except ImportError:
        pass
    except Exception as e:
        _log_hook_error("behavioral_habits", e)

    # Clean up stale surfacing counter files (both .surfaced and .surfaced.json)
    try:
        omega_dir = Path.home() / ".omega"
        cutoff = time.time() - 86400
        for pattern in ("session-*.surfaced", "session-*.surfaced.json"):
            for f in omega_dir.glob(pattern):
                if f.stat().st_mtime < cutoff:
                    f.unlink()
    except Exception:
        pass

    # Cross-project lesson surfacing
    try:
        from omega.bridge import get_cross_project_lessons
        cross_lessons = get_cross_project_lessons(
            task=None,
            exclude_project=project,
            limit=3,
        )
        cross_only = [l for l in cross_lessons if l.get("cross_project")]
        if cross_only:
            emit("\n[CROSS-PROJECT] Lessons from other codebases:")
            for l in cross_only[:3]:
                content = l.get("content", "")[:120]
                source_proj = l.get("project", "unknown")
                emit(f"  - [{source_proj}] {content}")
    except ImportError:
        pass
    except Exception as e:
        _log_hook_error("cross_project_lessons", e)

    # Surface top project lessons
    try:
        from omega.bridge import get_cross_session_lessons
        project_lessons = get_cross_session_lessons(
            task=None,
            project_path=project,
            exclude_session=session_id,
            limit=3,
        )
        top_lessons = [l for l in project_lessons if (l.get("access_count", 0) or 0) > 0]
        if top_lessons:
            emit("\n[LESSONS] Top lessons for this project:")
            for l in top_lessons[:3]:
                content = l.get("content", "")[:120]
                emit(f"  - {content}")
    except ImportError:
        pass
    except Exception as e:
        _log_hook_error("project_lessons", e)

    # Weekly digest, type stats, preferences, recent memories available on-demand
    # via omega_weekly_digest, omega_type_stats, omega_list_preferences.


def main(payload: dict | None = None) -> None:
    """Standalone entry point: build the payload from the hook env vars when not given."""
    if payload is None:
        payload = {
            "session_id": os.environ.get("SESSION_ID", ""),
            "project": os.environ.get("PROJECT_DIR", os.getcwd()),
        }
    run(payload)


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
    _log_timing("session_start", (time.monotonic() - _t0) * 1000)
