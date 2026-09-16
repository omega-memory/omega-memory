"""
OMEGA Behavioral Pattern Extraction

Infers user preferences from observed behavior in coordination tables.
SQL-only extractors (no LLM calls). Stores patterns as behavioral_pattern
memories through the standard auto_capture pipeline.

Data sources: coord_audit, coord_git_events, coord_file_claims, coord_sessions,
              coord_handoffs, coord_tasks.
"""

import json as _json
import logging
import math
import sqlite3
from collections import Counter
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger("omega.behavioral")

# Minimum confidence to store a pattern
MIN_STORE_CONFIDENCE = 0.6
# Minimum confidence to surface in welcome flow
# Lowered from 0.7 to 0.5 so patterns remain visible after temporal decay
MIN_SURFACE_CONFIDENCE = 0.5

# Reinforcement delta when a pattern is re-discovered
REINFORCE_DELTA = 0.03
# Max confidence for unconfirmed patterns
MAX_UNCONFIRMED_CONFIDENCE = 0.95


def effective_confidence(raw_confidence: float, last_evidence_iso: str) -> float:
    """Apply temporal decay to raw confidence. Half-life: 30 days."""
    if not last_evidence_iso:
        return raw_confidence
    try:
        last = datetime.fromisoformat(last_evidence_iso.replace("Z", "+00:00"))
        if last.tzinfo is None:
            last = last.replace(tzinfo=timezone.utc)
        days = (datetime.now(timezone.utc) - last).total_seconds() / 86400
        decayed = raw_confidence * math.exp(-days / 43.3)  # 30-day half-life
        return max(round(decayed, 3), 0.0)
    except (ValueError, TypeError):
        return raw_confidence


def _compute_confidence(
    session_count: int,
    datapoint_count: int,
    consistency_ratio: float,
    min_sessions: int = 5,
    min_datapoints: int = 10,
) -> float:
    """Compute confidence score for a behavioral pattern.

    Formula: 0.3 * breadth(sessions) + 0.3 * volume(datapoints) + 0.4 * consistency(ratio)

    Each component is normalized to [0, 1]:
    - breadth: min(session_count / (min_sessions * 3), 1.0)
    - volume: min(datapoint_count / (min_datapoints * 5), 1.0)
    - consistency: clamp(consistency_ratio, 0, 1)
    """
    breadth = min(session_count / (min_sessions * 3), 1.0)
    volume = min(datapoint_count / (min_datapoints * 5), 1.0)
    consistency = max(0.0, min(consistency_ratio, 1.0))
    return round(0.3 * breadth + 0.3 * volume + 0.4 * consistency, 3)


def _is_subsequence(pattern: List[str], sequence: List[str]) -> bool:
    """Check if pattern is a subsequence of sequence."""
    it = iter(sequence)
    return all(item in it for item in pattern)


class BehavioralAnalyzer:
    """Infer user preferences from observed behavior in coordination tables."""

    def __init__(self, conn: Optional[sqlite3.Connection] = None):
        """Initialize with a coordination DB connection.

        If conn is None, gets the singleton CoordinationManager's connection.
        """
        self._conn = conn

    def _get_conn(self) -> sqlite3.Connection:
        if self._conn is not None:
            return self._conn
        try:
            from omega_platform.orchestrator.coordination import get_manager
            mgr = get_manager()
            return mgr.get_read_connection()
        except ImportError:
            return None

    def _table_exists(self, table_name: str) -> bool:
        row = self._get_conn().execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
            (table_name,),
        ).fetchone()
        return row is not None

    def analyze_tool_preferences(self, min_sessions: int = 3) -> List[Dict[str, Any]]:
        """Analyze tool usage ratios from coord_audit.

        Detects strong preferences like "Uses Grep 4x more than Bash for search".
        Requires data from at least min_sessions distinct sessions.
        """
        conn = self._get_conn()
        if not self._table_exists("coord_audit"):
            return []

        # Count distinct sessions
        row = conn.execute(
            "SELECT COUNT(DISTINCT session_id) FROM coord_audit WHERE session_id IS NOT NULL"
        ).fetchone()
        total_sessions = row[0] if row else 0
        if total_sessions < min_sessions:
            return []

        # Get tool usage counts per session
        rows = conn.execute("""
            SELECT tool_name, COUNT(*) as cnt, COUNT(DISTINCT session_id) as sess_cnt
            FROM coord_audit
            WHERE tool_name IS NOT NULL AND session_id IS NOT NULL
            GROUP BY tool_name
            ORDER BY cnt DESC
        """).fetchall()

        if not rows:
            return []

        tool_counts: Dict[str, int] = {}
        tool_sessions: Dict[str, int] = {}
        for tool_name, cnt, sess_cnt in rows:
            tool_counts[tool_name] = cnt
            tool_sessions[tool_name] = sess_cnt

        patterns = []

        # Detect dominant tool preferences via ratio analysis
        # Group by functional category
        search_tools = {t: c for t, c in tool_counts.items()
                        if any(k in t.lower() for k in ("grep", "search", "glob", "find"))}
        if len(search_tools) >= 2:
            sorted_search = sorted(search_tools.items(), key=lambda x: x[1], reverse=True)
            top, second = sorted_search[0], sorted_search[1]
            if second[1] > 0:
                ratio = top[1] / second[1]
                if ratio >= 3.0:
                    confidence = _compute_confidence(
                        session_count=tool_sessions.get(top[0], 0),
                        datapoint_count=top[1],
                        consistency_ratio=min(ratio / 5.0, 1.0),
                        min_sessions=min_sessions,
                    )
                    if confidence >= MIN_STORE_CONFIDENCE:
                        patterns.append({
                            "content": f"Uses {top[0]} {ratio:.0f}x more than {second[0]} for search",
                            "pattern_type": "tool_preference",
                            "pattern_key": f"tool_ratio:{top[0]}:{second[0]}",
                            "confidence": confidence,
                            "evidence_count": top[1] + second[1],
                            "evidence_sessions": total_sessions,
                        })

        # Detect most-used tools overall (top 3 by usage)
        top_tools = sorted(tool_counts.items(), key=lambda x: x[1], reverse=True)[:3]
        total_calls = sum(tool_counts.values())
        for tool_name, count in top_tools:
            share = count / total_calls if total_calls > 0 else 0
            if share >= 0.2:  # Tool accounts for 20%+ of all usage
                confidence = _compute_confidence(
                    session_count=tool_sessions.get(tool_name, 0),
                    datapoint_count=count,
                    consistency_ratio=share * 2,  # Scale 0.5 share to 1.0
                    min_sessions=min_sessions,
                )
                if confidence >= MIN_STORE_CONFIDENCE:
                    patterns.append({
                        "content": f"Heavy {tool_name} user ({share:.0%} of tool calls, {count} total across {tool_sessions.get(tool_name, 0)} sessions)",
                        "pattern_type": "tool_preference",
                        "pattern_key": f"tool_dominant:{tool_name}",
                        "confidence": confidence,
                        "evidence_count": count,
                        "evidence_sessions": tool_sessions.get(tool_name, 0),
                    })

        return patterns

    def analyze_git_style(self, min_commits: int = 5) -> List[Dict[str, Any]]:
        """Analyze git workflow patterns from coord_git_events.

        Detects commit frequency, conventional commit usage, branch patterns.
        """
        conn = self._get_conn()
        if not self._table_exists("coord_git_events"):
            return []

        # Count commits
        row = conn.execute(
            "SELECT COUNT(*) FROM coord_git_events WHERE event_type = 'commit'"
        ).fetchone()
        total_commits = row[0] if row else 0
        if total_commits < min_commits:
            return []

        # Commits per session
        rows = conn.execute("""
            SELECT session_id, COUNT(*) as cnt
            FROM coord_git_events
            WHERE event_type = 'commit' AND session_id IS NOT NULL
            GROUP BY session_id
        """).fetchall()
        session_count = len(rows)
        if session_count == 0:
            return []
        avg_commits = total_commits / session_count

        patterns = []

        # Commit frequency pattern
        freq_label = "frequently" if avg_commits >= 5 else ("moderately" if avg_commits >= 2 else "sparingly")
        confidence = _compute_confidence(
            session_count=session_count,
            datapoint_count=total_commits,
            consistency_ratio=min(avg_commits / 10.0, 1.0),
            min_sessions=5,
            min_datapoints=min_commits,
        )
        if confidence >= MIN_STORE_CONFIDENCE:
            patterns.append({
                "content": f"Commits {freq_label} (avg {avg_commits:.1f}/session across {session_count} sessions)",
                "pattern_type": "git_workflow",
                "pattern_key": "git_commit_frequency",
                "confidence": confidence,
                "evidence_count": total_commits,
                "evidence_sessions": session_count,
            })

        # Conventional commits detection
        messages = conn.execute("""
            SELECT message FROM coord_git_events
            WHERE event_type = 'commit' AND message IS NOT NULL
        """).fetchall()
        if messages:
            conventional_prefixes = ("feat:", "fix:", "chore:", "docs:", "style:",
                                     "refactor:", "test:", "ci:", "perf:", "build:")
            conventional_count = sum(
                1 for (msg,) in messages
                if any(msg.strip().lower().startswith(p) for p in conventional_prefixes)
            )
            conv_ratio = conventional_count / len(messages)
            if conv_ratio >= 0.5:
                confidence = _compute_confidence(
                    session_count=session_count,
                    datapoint_count=len(messages),
                    consistency_ratio=conv_ratio,
                    min_sessions=5,
                    min_datapoints=min_commits,
                )
                if confidence >= MIN_STORE_CONFIDENCE:
                    patterns.append({
                        "content": f"Uses conventional commit format ({conv_ratio:.0%} of {len(messages)} commits)",
                        "pattern_type": "git_workflow",
                        "pattern_key": "git_conventional_commits",
                        "confidence": confidence,
                        "evidence_count": len(messages),
                        "evidence_sessions": session_count,
                    })

            # Commit message length
            lengths = [len(msg) for (msg,) in messages if msg]
            if lengths:
                avg_len = sum(lengths) / len(lengths)
                length_label = "concise" if avg_len < 40 else ("moderate" if avg_len < 70 else "detailed")
                confidence = _compute_confidence(
                    session_count=session_count,
                    datapoint_count=len(lengths),
                    consistency_ratio=0.7,  # Length is inherently consistent
                    min_sessions=5,
                    min_datapoints=min_commits,
                )
                if confidence >= MIN_STORE_CONFIDENCE:
                    patterns.append({
                        "content": f"Writes {length_label} commit messages (avg {avg_len:.0f} chars across {len(lengths)} commits)",
                        "pattern_type": "git_workflow",
                        "pattern_key": "git_message_length",
                        "confidence": confidence,
                        "evidence_count": len(lengths),
                        "evidence_sessions": session_count,
                    })

        # Branch discipline
        branch_rows = conn.execute("""
            SELECT branch, COUNT(*) as cnt FROM coord_git_events
            WHERE event_type = 'commit' AND branch IS NOT NULL
            GROUP BY branch
        """).fetchall()
        if branch_rows:
            branch_counts = {r[0]: r[1] for r in branch_rows}
            branch_total = sum(branch_counts.values())
            main_count = branch_counts.get("main", 0) + branch_counts.get("master", 0)
            if branch_total > 0:
                main_ratio = main_count / branch_total
                if main_ratio >= 0.7:
                    style = "Trunk-based development"
                elif main_ratio <= 0.3:
                    style = "Feature-branch workflow"
                else:
                    style = "Mixed branching"
                confidence = _compute_confidence(
                    session_count=session_count,
                    datapoint_count=branch_total,
                    consistency_ratio=max(main_ratio, 1.0 - main_ratio),
                    min_sessions=5,
                    min_datapoints=min_commits,
                )
                if confidence >= MIN_STORE_CONFIDENCE:
                    patterns.append({
                        "content": f"{style}: {main_ratio:.0%} of commits on main/master",
                        "pattern_type": "git_workflow",
                        "pattern_key": "git_branch_style",
                        "confidence": confidence,
                        "evidence_count": branch_total,
                        "evidence_sessions": session_count,
                    })

        return patterns

    def analyze_session_patterns(self, min_sessions: int = 5) -> List[Dict[str, Any]]:
        """Analyze session timing patterns from coord_sessions.

        Detects work hours, average session duration, time-of-day preferences.
        """
        conn = self._get_conn()
        if not self._table_exists("coord_sessions"):
            return []

        rows = conn.execute("""
            SELECT started_at, last_heartbeat
            FROM coord_sessions
            WHERE started_at IS NOT NULL
            ORDER BY started_at DESC
            LIMIT 200
        """).fetchall()

        if len(rows) < min_sessions:
            return []

        patterns = []
        hours: List[int] = []
        durations: List[float] = []

        for started_at, last_hb in rows:
            try:
                start = datetime.fromisoformat(started_at.replace("Z", "+00:00"))
                # Convert to ICT (UTC+7) for the user's timezone
                hour_ict = (start.hour + 7) % 24
                hours.append(hour_ict)

                if last_hb:
                    end = datetime.fromisoformat(last_hb.replace("Z", "+00:00"))
                    duration_min = (end - start).total_seconds() / 60
                    if 1 <= duration_min <= 480:  # 1 min to 8 hours
                        durations.append(duration_min)
            except (ValueError, TypeError):
                continue

        session_count = len(rows)

        # Time-of-day preference
        if hours:
            hour_counts = Counter(hours)
            # Find peak 4-hour window
            best_start = 0
            best_count = 0
            for h in range(24):
                window_count = sum(hour_counts.get((h + i) % 24, 0) for i in range(4))
                if window_count > best_count:
                    best_count = window_count
                    best_start = h
            peak_ratio = best_count / len(hours)
            if peak_ratio >= 0.4:  # 40%+ of sessions in a 4-hour window
                end_hour = (best_start + 4) % 24
                period = "morning" if 5 <= best_start < 12 else (
                    "afternoon" if 12 <= best_start < 17 else (
                        "evening" if 17 <= best_start < 21 else "night"
                    )
                )
                confidence = _compute_confidence(
                    session_count=session_count,
                    datapoint_count=len(hours),
                    consistency_ratio=peak_ratio,
                    min_sessions=min_sessions,
                )
                if confidence >= MIN_STORE_CONFIDENCE:
                    patterns.append({
                        "content": f"Works primarily in the {period} ({best_start}:00-{end_hour}:00 ICT, {peak_ratio:.0%} of sessions)",
                        "pattern_type": "session_timing",
                        "pattern_key": "session_peak_hours",
                        "confidence": confidence,
                        "evidence_count": len(hours),
                        "evidence_sessions": session_count,
                    })

        # Average session duration
        if durations:
            avg_min = sum(durations) / len(durations)
            duration_label = (
                f"{avg_min:.0f}min" if avg_min < 60
                else f"{avg_min/60:.1f}h"
            )
            # Consistency: std dev relative to mean (lower = more consistent)
            if len(durations) >= 3:
                mean = avg_min
                variance = sum((d - mean) ** 2 for d in durations) / len(durations)
                std_dev = variance ** 0.5
                cv = std_dev / mean if mean > 0 else 1.0
                consistency = max(0.0, 1.0 - cv)  # Low CV = high consistency
            else:
                consistency = 0.5

            confidence = _compute_confidence(
                session_count=session_count,
                datapoint_count=len(durations),
                consistency_ratio=consistency,
                min_sessions=min_sessions,
            )
            if confidence >= MIN_STORE_CONFIDENCE:
                patterns.append({
                    "content": f"Average session duration: {duration_label} (across {len(durations)} sessions)",
                    "pattern_type": "session_timing",
                    "pattern_key": "session_avg_duration",
                    "confidence": confidence,
                    "evidence_count": len(durations),
                    "evidence_sessions": session_count,
                })

        # Day-of-week preference
        weekdays: List[int] = []
        for started_at, _ in rows:
            try:
                start = datetime.fromisoformat(started_at.replace("Z", "+00:00"))
                weekdays.append(start.weekday())  # 0=Mon, 6=Sun
            except (ValueError, TypeError):
                continue

        if weekdays:
            weekday_count = sum(1 for d in weekdays if d < 5)
            weekday_ratio = weekday_count / len(weekdays)
            day_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
            day_counts = Counter(weekdays)
            heaviest_day = day_names[day_counts.most_common(1)[0][0]]

            if weekday_ratio >= 0.8:
                label = f"Works primarily on weekdays ({weekday_ratio:.0%} Mon-Fri, heaviest on {heaviest_day})"
            elif weekday_ratio <= 0.4:
                label = f"Frequently works on weekends ({1 - weekday_ratio:.0%} Sat-Sun, heaviest on {heaviest_day})"
            else:
                label = f"Works throughout the week (heaviest on {heaviest_day})"

            confidence = _compute_confidence(
                session_count=session_count,
                datapoint_count=len(weekdays),
                consistency_ratio=max(weekday_ratio, 1.0 - weekday_ratio),
                min_sessions=min_sessions,
            )
            if confidence >= MIN_STORE_CONFIDENCE:
                patterns.append({
                    "content": label,
                    "pattern_type": "session_timing",
                    "pattern_key": "session_weekday_pattern",
                    "confidence": confidence,
                    "evidence_count": len(weekdays),
                    "evidence_sessions": session_count,
                })

        return patterns

    def analyze_co_edits(self, min_cooccurrence: int = 2) -> List[Dict[str, Any]]:
        """Analyze file co-edit patterns from coord_file_claims.

        Detects files that are always edited together.
        """
        conn = self._get_conn()
        if not self._table_exists("coord_file_claims"):
            return []

        # Get files claimed per session
        rows = conn.execute("""
            SELECT session_id, file_path
            FROM coord_file_claims
            WHERE session_id IS NOT NULL
        """).fetchall()

        if not rows:
            return []

        # Build session -> files mapping
        session_files: Dict[str, List[str]] = {}
        for session_id, file_path in rows:
            session_files.setdefault(session_id, []).append(file_path)

        # Count co-occurrences (pairs of files in same session)
        pair_counts: Counter = Counter()
        for files in session_files.values():
            unique_files = sorted(set(files))
            for i, f1 in enumerate(unique_files):
                for f2 in unique_files[i + 1:]:
                    pair_counts[(f1, f2)] += 1

        patterns = []
        total_sessions = len(session_files)

        for (f1, f2), count in pair_counts.most_common(5):
            if count < min_cooccurrence:
                break
            # Shorten file paths for readability
            f1_short = f1.rsplit("/", 1)[-1] if "/" in f1 else f1
            f2_short = f2.rsplit("/", 1)[-1] if "/" in f2 else f2
            co_ratio = count / total_sessions if total_sessions > 0 else 0

            confidence = _compute_confidence(
                session_count=count,  # Sessions where both appeared
                datapoint_count=count * 2,  # Each co-occurrence = 2 file claims
                consistency_ratio=co_ratio * 2,  # Scale: if 50% of sessions, that's very consistent
                min_sessions=min_cooccurrence,
                min_datapoints=min_cooccurrence * 2,
            )
            if confidence >= MIN_STORE_CONFIDENCE:
                patterns.append({
                    "content": f"{f1_short} + {f2_short} always co-edited ({count} sessions)",
                    "pattern_type": "co_edit_cluster",
                    "pattern_key": f"co_edit:{f1_short}:{f2_short}",
                    "confidence": confidence,
                    "evidence_count": count * 2,
                    "evidence_sessions": count,
                })

        return patterns

    def analyze_project_focus(self, min_sessions: int = 5) -> List[Dict[str, Any]]:
        """Analyze which projects the user spends the most time on.

        Detects dominant project focus and multi-project breadth.
        """
        conn = self._get_conn()
        if not self._table_exists("coord_sessions"):
            return []

        rows = conn.execute("""
            SELECT project,
                   COUNT(*) as session_count,
                   SUM(CAST((julianday(last_heartbeat) - julianday(started_at)) * 24 * 60 AS INTEGER)) as total_minutes
            FROM coord_sessions
            WHERE project IS NOT NULL AND started_at IS NOT NULL AND last_heartbeat IS NOT NULL
            GROUP BY project
            ORDER BY total_minutes DESC
        """).fetchall()

        if not rows:
            return []

        total_sessions = sum(r[1] for r in rows)
        if total_sessions < min_sessions:
            return []

        total_minutes = sum((r[2] or 0) for r in rows)
        patterns = []

        # Dominant project: one project has 50%+ of session time
        if rows[0][2] and total_minutes > 0:
            top_project = rows[0][0]
            top_sessions = rows[0][1]
            top_minutes = rows[0][2]
            share = top_minutes / total_minutes

            if share >= 0.5:
                basename = top_project.rsplit("/", 1)[-1] if "/" in top_project else top_project
                hours = top_minutes / 60
                confidence = _compute_confidence(
                    session_count=total_sessions,
                    datapoint_count=top_sessions,
                    consistency_ratio=share * 2,
                    min_sessions=min_sessions,
                )
                if confidence >= MIN_STORE_CONFIDENCE:
                    patterns.append({
                        "content": f"Primary project: {basename} ({share:.0%} of sessions, {hours:.0f}h total across {top_sessions} sessions)",
                        "pattern_type": "project_focus",
                        "pattern_key": f"project_focus:{basename}",
                        "confidence": confidence,
                        "evidence_count": top_sessions,
                        "evidence_sessions": total_sessions,
                    })

        # Multi-project breadth: 3+ projects each with 10%+ of sessions
        if total_sessions > 0:
            active_projects = [r for r in rows if r[1] / total_sessions >= 0.1]
            if len(active_projects) >= 3:
                confidence = _compute_confidence(
                    session_count=total_sessions,
                    datapoint_count=sum(r[1] for r in active_projects),
                    consistency_ratio=len(active_projects) / len(rows) if rows else 0,
                    min_sessions=min_sessions,
                )
                if confidence >= MIN_STORE_CONFIDENCE:
                    patterns.append({
                        "content": f"Active multi-project developer: works across {len(active_projects)} projects regularly",
                        "pattern_type": "project_focus",
                        "pattern_key": "project_breadth:multi",
                        "confidence": confidence,
                        "evidence_count": sum(r[1] for r in active_projects),
                        "evidence_sessions": total_sessions,
                    })

        return patterns

    def analyze_workflow_sequences(self, min_sessions: int = 3) -> List[Dict[str, Any]]:
        """Analyze common consecutive tool-call pairs from coord_audit.

        Detects strong tool sequences and handoff discipline.
        """
        conn = self._get_conn()
        if not self._table_exists("coord_audit"):
            return []

        rows = conn.execute("""
            SELECT session_id, tool_name
            FROM coord_audit
            WHERE session_id IS NOT NULL
            ORDER BY session_id, created_at, id
        """).fetchall()

        if not rows:
            return []

        # Group by session
        session_tools: Dict[str, List[str]] = {}
        for session_id, tool_name in rows:
            session_tools.setdefault(session_id, []).append(tool_name)

        if len(session_tools) < min_sessions:
            return []

        # Count consecutive pairs across sessions
        pair_counts: Counter = Counter()
        pair_sessions: Dict[tuple, set] = {}
        tool_sessions: Dict[str, set] = {}

        for sess_id, tools in session_tools.items():
            seen_tools = set()
            for i in range(len(tools) - 1):
                pair = (tools[i], tools[i + 1])
                pair_counts[pair] += 1
                pair_sessions.setdefault(pair, set()).add(sess_id)
                seen_tools.add(tools[i])
            if tools:
                seen_tools.add(tools[-1])
            for t in seen_tools:
                tool_sessions.setdefault(t, set()).add(sess_id)

        total_sessions = len(session_tools)
        patterns = []

        # Strong sequences: tool_a -> tool_b in 70%+ of sessions containing tool_a
        for pair, count in pair_counts.most_common(10):
            sessions_with_pair = len(pair_sessions.get(pair, set()))
            if sessions_with_pair < 3:
                continue
            sessions_with_tool_a = len(tool_sessions.get(pair[0], set()))
            if sessions_with_tool_a == 0:
                continue
            ratio = sessions_with_pair / sessions_with_tool_a
            if ratio >= 0.7:
                confidence = _compute_confidence(
                    session_count=sessions_with_tool_a,
                    datapoint_count=count,
                    consistency_ratio=ratio,
                    min_sessions=min_sessions,
                )
                if confidence >= MIN_STORE_CONFIDENCE:
                    patterns.append({
                        "content": f"Almost always follows {pair[0]} with {pair[1]} ({ratio:.0%} of {sessions_with_tool_a} sessions)",
                        "pattern_type": "workflow_sequence",
                        "pattern_key": f"workflow_sequence:{pair[0]}:{pair[1]}",
                        "confidence": confidence,
                        "evidence_count": count,
                        "evidence_sessions": sessions_with_tool_a,
                    })

        # Handoff discipline: omega_handoff before session_deregister
        handoff_pair = ("omega_handoff", "session_deregister")
        if handoff_pair in pair_sessions:
            sessions_with_deregister = len(tool_sessions.get("session_deregister", set()))
            sessions_with_handoff_before = len(pair_sessions[handoff_pair])
            if sessions_with_deregister >= 3:
                ratio = sessions_with_handoff_before / sessions_with_deregister
                if ratio >= 0.5:
                    confidence = _compute_confidence(
                        session_count=sessions_with_deregister,
                        datapoint_count=pair_counts[handoff_pair],
                        consistency_ratio=ratio,
                        min_sessions=min_sessions,
                    )
                    if confidence >= MIN_STORE_CONFIDENCE:
                        patterns.append({
                            "content": f"Disciplined handoff writer (creates handoff before ending in {ratio:.0%} of sessions)",
                            "pattern_type": "workflow_sequence",
                            "pattern_key": "workflow_handoff_discipline",
                            "confidence": confidence,
                            "evidence_count": pair_counts[handoff_pair],
                            "evidence_sessions": sessions_with_deregister,
                        })

        return patterns

    def analyze_workflow_sequences_deep(self, min_sessions: int = 3) -> List[Dict[str, Any]]:
        """Analyze length-3+ tool sequences using PrefixSpan.

        Extends analyze_workflow_sequences (pairs only) to find longer patterns.
        Requires prefixspan>=0.5.0 (optional dep, graceful fallback).
        """
        try:
            from prefixspan import PrefixSpan
        except ImportError:
            return []

        conn = self._get_conn()
        if not self._table_exists("coord_audit"):
            return []

        rows = conn.execute("""
            SELECT session_id, tool_name
            FROM coord_audit
            WHERE session_id IS NOT NULL AND tool_name IS NOT NULL
            ORDER BY session_id, created_at, id
        """).fetchall()

        if not rows:
            return []

        # Group by session
        session_tools: Dict[str, List[str]] = {}
        for session_id, tool_name in rows:
            session_tools.setdefault(session_id, []).append(tool_name)

        if len(session_tools) < min_sessions:
            return []

        # Build sequences (deduplicate consecutive repeats within each session)
        sequences = []
        for tools in session_tools.values():
            deduped = [tools[0]]
            for t in tools[1:]:
                if t != deduped[-1]:
                    deduped.append(t)
            if len(deduped) >= 3:
                sequences.append(deduped)

        if len(sequences) < min_sessions:
            return []

        # Run PrefixSpan for patterns of length 3+
        ps = PrefixSpan(sequences)
        # Find patterns appearing in at least min_sessions sequences
        ps.minlen = 3
        ps.maxlen = 5
        frequent = ps.frequent(min_sessions)

        patterns = []
        for support, pattern in frequent:
            if len(pattern) < 3:
                continue

            # Count sessions containing this pattern
            sessions_with_pattern = 0
            for tools in session_tools.values():
                if _is_subsequence(pattern, tools):
                    sessions_with_pattern += 1

            ratio = sessions_with_pattern / len(session_tools)
            if ratio < 0.3:
                continue

            confidence = _compute_confidence(
                session_count=sessions_with_pattern,
                datapoint_count=support,
                consistency_ratio=ratio,
                min_sessions=min_sessions,
            )

            if confidence < MIN_STORE_CONFIDENCE:
                continue

            arrow = " -> ".join(pattern)
            patterns.append({
                "content": (
                    f"Common workflow: {arrow} "
                    f"(in {sessions_with_pattern}/{len(session_tools)} sessions, "
                    f"{ratio:.0%} consistency)"
                ),
                "pattern_type": "workflow_sequence_deep",
                "pattern_key": f"workflow_deep:{':'.join(pattern)}",
                "confidence": confidence,
                "evidence_count": support,
                "evidence_sessions": sessions_with_pattern,
            })

        return patterns[:10]  # Cap at top 10 patterns

    def analyze_handoff_patterns(self, min_handoffs: int = 3) -> List[Dict[str, Any]]:
        """Analyze structured handoff data for thoroughness, blocker frequency, decision density."""
        conn = self._get_conn()
        if not self._table_exists("coord_handoffs"):
            return []

        row = conn.execute("""
            SELECT
              COUNT(*) as total,
              SUM(CASE WHEN completed_tasks IS NOT NULL AND completed_tasks != '[]' THEN 1 ELSE 0 END) as has_completed,
              SUM(CASE WHEN blocked_items IS NOT NULL AND blocked_items != '[]' THEN 1 ELSE 0 END) as has_blocked,
              SUM(CASE WHEN next_steps IS NOT NULL AND next_steps != '' THEN 1 ELSE 0 END) as has_next,
              SUM(CASE WHEN decisions_made IS NOT NULL AND decisions_made != '[]' THEN 1 ELSE 0 END) as has_decisions,
              COUNT(DISTINCT session_id) as session_count
            FROM coord_handoffs
        """).fetchone()

        if not row or row[0] < min_handoffs:
            return []

        total, has_completed, has_blocked, has_next, has_decisions, session_count = row
        patterns = []

        # Handoff thoroughness: % with 3+ fields populated
        thorough_count = 0
        all_handoffs = conn.execute("""
            SELECT completed_tasks, blocked_items, next_steps, decisions_made
            FROM coord_handoffs
        """).fetchall()
        total_decisions = 0
        for ct, bi, ns, dm in all_handoffs:
            fields_populated = sum([
                ct is not None and ct != "[]",
                bi is not None and bi != "[]",
                ns is not None and ns != "",
                dm is not None and dm != "[]",
            ])
            if fields_populated >= 3:
                thorough_count += 1
            # Count decisions
            if dm and dm != "[]":
                try:
                    decisions = _json.loads(dm)
                    if isinstance(decisions, list):
                        total_decisions += len(decisions)
                except (_json.JSONDecodeError, TypeError):
                    total_decisions += 1

        thoroughness_ratio = thorough_count / total if total > 0 else 0
        confidence = _compute_confidence(
            session_count=session_count,
            datapoint_count=total,
            consistency_ratio=thoroughness_ratio,
            min_sessions=3,
            min_datapoints=min_handoffs,
        )
        if confidence >= MIN_STORE_CONFIDENCE:
            patterns.append({
                "content": f"Thorough handoff writer: includes next_steps ({has_next}/{total}) and decisions ({has_decisions}/{total}) in {total} handoffs",
                "pattern_type": "handoff_quality",
                "pattern_key": "handoff_thoroughness",
                "confidence": confidence,
                "evidence_count": total,
                "evidence_sessions": session_count,
            })

        # Blocker frequency
        blocker_ratio = has_blocked / total if total > 0 else 0
        if blocker_ratio >= 0.3:
            confidence = _compute_confidence(
                session_count=session_count,
                datapoint_count=total,
                consistency_ratio=blocker_ratio,
                min_sessions=3,
                min_datapoints=min_handoffs,
            )
            if confidence >= MIN_STORE_CONFIDENCE:
                patterns.append({
                    "content": f"Frequently encounters blockers: {blocker_ratio:.0%} of handoffs report blocked items ({has_blocked}/{total})",
                    "pattern_type": "handoff_quality",
                    "pattern_key": "handoff_blocker_rate",
                    "confidence": confidence,
                    "evidence_count": total,
                    "evidence_sessions": session_count,
                })

        # Decision density
        if total > 0:
            avg_decisions = total_decisions / total
            if avg_decisions >= 1.0:
                confidence = _compute_confidence(
                    session_count=session_count,
                    datapoint_count=total_decisions,
                    consistency_ratio=min(avg_decisions / 5.0, 1.0),
                    min_sessions=3,
                    min_datapoints=min_handoffs,
                )
                if confidence >= MIN_STORE_CONFIDENCE:
                    patterns.append({
                        "content": f"Active decision-maker: averages {avg_decisions:.1f} decisions per session handoff",
                        "pattern_type": "handoff_quality",
                        "pattern_key": "handoff_decision_density",
                        "confidence": confidence,
                        "evidence_count": total_decisions,
                        "evidence_sessions": session_count,
                    })

        return patterns

    def analyze_task_completion_style(self, min_tasks: int = 5) -> List[Dict[str, Any]]:
        """Analyze task management patterns from coord_tasks."""
        conn = self._get_conn()
        if not self._table_exists("coord_tasks"):
            return []

        # Task outcome distribution
        outcome_rows = conn.execute("""
            SELECT status, COUNT(*) as cnt FROM coord_tasks
            WHERE status IN ('completed', 'failed', 'canceled')
            GROUP BY status
        """).fetchall()

        outcomes = {r[0]: r[1] for r in outcome_rows}
        total_resolved = sum(outcomes.values())
        if total_resolved < min_tasks:
            return []

        patterns = []
        completed = outcomes.get("completed", 0)
        completion_rate = completed / total_resolved if total_resolved > 0 else 0

        # Completion rate
        session_row = conn.execute(
            "SELECT COUNT(DISTINCT session_id) FROM coord_tasks WHERE session_id IS NOT NULL"
        ).fetchone()
        session_count = session_row[0] if session_row else 0

        rate_label = "High" if completion_rate >= 0.8 else ("Moderate" if completion_rate >= 0.5 else "Low")
        confidence = _compute_confidence(
            session_count=session_count,
            datapoint_count=total_resolved,
            consistency_ratio=completion_rate,
            min_sessions=3,
            min_datapoints=min_tasks,
        )
        if confidence >= MIN_STORE_CONFIDENCE:
            patterns.append({
                "content": f"{rate_label} task completion rate: {completion_rate:.0%} completed ({completed}/{total_resolved} tasks)",
                "pattern_type": "task_management",
                "pattern_key": "task_completion_rate",
                "confidence": confidence,
                "evidence_count": total_resolved,
                "evidence_sessions": session_count,
            })

        # Average duration
        duration_row = conn.execute("""
            SELECT AVG(CAST((julianday(completed_at) - julianday(claimed_at)) * 24 * 60 AS REAL)) as avg_min,
                   COUNT(*) as cnt
            FROM coord_tasks
            WHERE status = 'completed' AND claimed_at IS NOT NULL AND completed_at IS NOT NULL
        """).fetchone()

        if duration_row and duration_row[0] is not None and duration_row[1] >= 3:
            avg_min = duration_row[0]
            duration_label = f"{avg_min:.0f}min" if avg_min < 60 else f"{avg_min / 60:.1f}h"
            confidence = _compute_confidence(
                session_count=session_count,
                datapoint_count=duration_row[1],
                consistency_ratio=min(duration_row[1] / (min_tasks * 2), 1.0),
                min_sessions=3,
                min_datapoints=min_tasks,
            )
            if confidence >= MIN_STORE_CONFIDENCE:
                patterns.append({
                    "content": f"Average task duration: {duration_label} from claim to completion",
                    "pattern_type": "task_management",
                    "pattern_key": "task_avg_duration",
                    "confidence": confidence,
                    "evidence_count": duration_row[1],
                    "evidence_sessions": session_count,
                })

        return patterns

    def compare_agents(self, session_a: str, session_b: str) -> Dict[str, Any]:
        """Compare tool usage distributions between two agents on same project.

        Returns alignment score (0-1) and divergent tools. Uses coord_audit
        grouped by session_id to build tool frequency vectors, then computes
        cosine similarity.
        """
        conn = self._get_conn()
        if not self._table_exists("coord_audit"):
            return {"alignment": 0.0, "error": "No audit data"}

        rows_a = conn.execute(
            "SELECT tool_name, COUNT(*) FROM coord_audit WHERE session_id = ? GROUP BY tool_name",
            (session_a,),
        ).fetchall()
        rows_b = conn.execute(
            "SELECT tool_name, COUNT(*) FROM coord_audit WHERE session_id = ? GROUP BY tool_name",
            (session_b,),
        ).fetchall()

        # Compute tool alignment if audit data exists
        alignment = 0.0
        divergent: List[str] = []
        num_a_tools = 0
        num_b_tools = 0
        has_audit_data = bool(rows_a and rows_b)

        if has_audit_data:
            vec_a: Dict[str, int] = {r[0]: r[1] for r in rows_a}
            vec_b: Dict[str, int] = {r[0]: r[1] for r in rows_b}
            num_a_tools = len(vec_a)
            num_b_tools = len(vec_b)
            all_tools = set(vec_a) | set(vec_b)

            # Cosine similarity
            dot = sum(vec_a.get(t, 0) * vec_b.get(t, 0) for t in all_tools)
            mag_a = math.sqrt(sum(v * v for v in vec_a.values()))
            mag_b = math.sqrt(sum(v * v for v in vec_b.values()))
            if mag_a > 0 and mag_b > 0:
                alignment = round(dot / (mag_a * mag_b), 3)

            # Find divergent tools: present in one but not the other, or ratio > 5x
            for t in all_tools:
                a_count = vec_a.get(t, 0)
                b_count = vec_b.get(t, 0)
                if a_count == 0 or b_count == 0:
                    divergent.append(t)
                elif max(a_count, b_count) / max(min(a_count, b_count), 1) > 5:
                    divergent.append(t)

        # File overlap: Jaccard similarity of file claims
        file_overlap = 0.0
        if self._table_exists("coord_file_claims"):
            files_a_rows = conn.execute(
                "SELECT file_path FROM coord_file_claims WHERE session_id = ?",
                (session_a,),
            ).fetchall()
            files_b_rows = conn.execute(
                "SELECT file_path FROM coord_file_claims WHERE session_id = ?",
                (session_b,),
            ).fetchall()
            files_a_set = {r[0] for r in files_a_rows}
            files_b_set = {r[0] for r in files_b_rows}
            union = files_a_set | files_b_set
            if union:
                file_overlap = round(len(files_a_set & files_b_set) / len(union), 3)

        # Shared goals: goals both agents have tasks on
        shared_goals: List[int] = []
        if self._table_exists("coord_tasks"):
            goal_rows = conn.execute(
                """SELECT DISTINCT a.goal_id FROM coord_tasks a
                   JOIN coord_tasks b ON a.goal_id = b.goal_id
                   WHERE a.session_id = ? AND b.session_id = ?
                     AND a.goal_id IS NOT NULL""",
                (session_a, session_b),
            ).fetchall()
            shared_goals = [r[0] for r in goal_rows]

        result: Dict[str, Any] = {
            "alignment": alignment,
            "divergent_tools": sorted(divergent),
            "session_a_tools": num_a_tools,
            "session_b_tools": num_b_tools,
            "file_overlap": file_overlap,
            "shared_goals": shared_goals,
        }
        if not has_audit_data:
            result["error"] = "Insufficient data"
        return result

    def analyze_and_store(self) -> Dict[str, Any]:
        """Run all extractors and store discovered patterns.

        Supports pattern evolution: existing patterns get updated instead of
        duplicated, denied patterns are not re-created, and re-discovered
        patterns get reinforced.

        Returns summary dict with counts per extractor.
        """
        empty = {"total_extracted": 0, "stored": 0, "updated": 0, "skipped_denied": 0, "skipped_confidence": 0}
        if self._get_conn() is None:
            # Every extractor reads the coordination tables, which only the Pro
            # package creates. On a Core install there is nothing to analyze.
            logger.debug("Behavioral analysis skipped: no coordination database (Pro feature)")
            return empty

        all_patterns: List[Dict[str, Any]] = []

        for extractor_name, extractor_fn in [
            ("tool_preferences", self.analyze_tool_preferences),
            ("git_style", self.analyze_git_style),
            ("session_patterns", self.analyze_session_patterns),
            ("co_edits", self.analyze_co_edits),
            ("project_focus", self.analyze_project_focus),
            ("workflow_sequences", self.analyze_workflow_sequences),
            ("workflow_sequences_deep", self.analyze_workflow_sequences_deep),
            ("handoff_patterns", self.analyze_handoff_patterns),
            ("task_completion", self.analyze_task_completion_style),
        ]:
            try:
                patterns = extractor_fn()
                all_patterns.extend(patterns)
            except Exception as e:
                logger.warning("Behavioral extractor %s failed: %s", extractor_name, e)

        stored = 0
        updated = 0
        skipped_denied = 0
        skipped_confidence = 0

        for pattern in all_patterns:
            if pattern["confidence"] < MIN_STORE_CONFIDENCE:
                skipped_confidence += 1
                continue

            existing = self._find_existing_pattern(pattern["pattern_key"])
            if existing is not None:
                meta = existing.metadata or {}
                # Denied patterns: respect user denial, don't re-create
                if meta.get("user_confirmed") is False:
                    skipped_denied += 1
                    continue
                # Pattern evolution: update existing with new evidence
                try:
                    self._reinforce_pattern(existing, pattern)
                    updated += 1
                except Exception as e:
                    logger.warning("Failed to reinforce pattern: %s", e)
                continue

            try:
                self._store_pattern(pattern)
                stored += 1
            except Exception as e:
                logger.warning("Failed to store behavioral pattern: %s", e)

        result = {
            "total_extracted": len(all_patterns),
            "stored": stored,
            "updated": updated,
            "skipped_denied": skipped_denied,
            "skipped_confidence": skipped_confidence,
        }
        logger.info("Behavioral analysis complete: %s", result)
        return result

    def _find_existing_pattern(self, pattern_key: str) -> Optional[Any]:
        """Find an existing pattern with this key. Returns the memory node or None."""
        try:
            from omega.bridge import _get_store
            store = _get_store()
            existing = store.get_by_type("behavioral_pattern", limit=50)
            for mem in existing:
                meta = mem.metadata or {}
                if meta.get("pattern_key") == pattern_key:
                    return mem
        except Exception as e:
            logger.debug("Pattern detection failed: %s", e)
        return None

    def _reinforce_pattern(self, existing: Any, new_pattern: Dict[str, Any]) -> None:
        """Reinforce an existing pattern with new evidence.

        Increments evidence counts, bumps confidence by REINFORCE_DELTA
        (capped at MAX_UNCONFIRMED_CONFIDENCE for unconfirmed patterns),
        and updates last_evidence_at timestamp.
        """
        from omega.bridge import _get_store
        store = _get_store()
        meta = dict(existing.metadata or {})

        old_evidence = meta.get("evidence_count", 0)
        meta["evidence_count"] = max(old_evidence, new_pattern["evidence_count"])
        meta["evidence_sessions"] = max(
            meta.get("evidence_sessions", 0), new_pattern["evidence_sessions"]
        )
        meta["last_evidence_at"] = datetime.now(timezone.utc).isoformat()

        # Bump confidence
        old_conf = meta.get("confidence", 0)
        new_conf = old_conf + REINFORCE_DELTA
        if not meta.get("user_confirmed"):
            new_conf = min(new_conf, MAX_UNCONFIRMED_CONFIDENCE)
        meta["confidence"] = round(new_conf, 3)

        store.update_node(existing.id, metadata=meta)

    def _store_pattern(self, pattern: Dict[str, Any]) -> str:
        """Store a behavioral pattern through the standard pipeline."""
        from omega.bridge import auto_capture

        metadata = {
            "source": "behavioral_inference",
            "pattern_type": pattern["pattern_type"],
            "confidence": pattern["confidence"],
            "evidence_count": pattern["evidence_count"],
            "evidence_sessions": pattern["evidence_sessions"],
            "pattern_key": pattern["pattern_key"],
            "user_confirmed": None,
        }

        return auto_capture(
            content=pattern["content"],
            event_type="behavioral_pattern",
            metadata=metadata,
        )


# -----------------------------------------------------------------------
# Phase 4: Cross-pattern correlation rules
# -----------------------------------------------------------------------

CORRELATION_RULES = [
    {
        "type_a": "session_timing",
        "type_b": "handoff_quality",
        "key_a_contains": None,
        "key_b_contains": None,
        "template": "Your {a_label} sessions correlate with {b_label} handoffs (confidence {a_conf:.0%} / {b_conf:.0%})",
    },
    {
        "type_a": "session_timing",
        "type_b": "git_workflow",
        "key_a_contains": "weekday",
        "key_b_contains": "frequency",
        "template": "Your {a_label} work pattern aligns with {b_label} commit style",
    },
    {
        "type_a": "project_focus",
        "type_b": "task_management",
        "key_a_contains": "project_focus:",
        "key_b_contains": "completion_rate",
        "template": "Task completion is highest on your primary project ({a_label})",
    },
    {
        "type_a": "workflow_sequence",
        "type_b": "handoff_quality",
        "key_a_contains": "handoff_discipline",
        "key_b_contains": "thoroughness",
        "template": "Your handoff discipline habit correlates with thorough handoffs",
    },
]

# Phase 4: Recommendation rules
RECOMMENDATION_RULES = [
    {
        "id": "long_sessions",
        "condition": lambda patterns: any(
            p.get("pattern_key") == "session_avg_duration"
            and "h" in p.get("content", "")
            for p in patterns
        ),
        "recommendation": "Your sessions average over an hour. Research shows productivity peaks in 60-90min blocks. Consider using omega_checkpoint to save state and take breaks.",
        "category": "productivity",
        "based_on": ["session_avg_duration"],
    },
    {
        "id": "no_handoff_discipline",
        "condition": lambda patterns: (
            not any(p.get("pattern_key") == "workflow_handoff_discipline" for p in patterns)
            and any(p.get("pattern_type") == "session_timing" for p in patterns)
        ),
        "recommendation": "No handoff discipline detected. Creating handoffs before ending sessions helps your next session start faster.",
        "category": "workflow",
        "based_on": ["workflow_handoff_discipline"],
    },
    {
        "id": "high_blocker_rate",
        "condition": lambda patterns: any(
            p.get("pattern_key") == "handoff_blocker_rate"
            for p in patterns
        ),
        "recommendation": "You frequently encounter blockers. Consider breaking tasks into smaller units or flagging dependencies earlier.",
        "category": "planning",
        "based_on": ["handoff_blocker_rate"],
    },
    {
        "id": "low_completion_rate",
        "condition": lambda patterns: any(
            p.get("pattern_key") == "task_completion_rate"
            and "Low" in p.get("content", "")
            for p in patterns
        ),
        "recommendation": "Task completion rate is low. Consider smaller, more specific task definitions.",
        "category": "task_management",
        "based_on": ["task_completion_rate"],
    },
    {
        "id": "single_project_tunnel",
        "condition": lambda patterns: (
            any(p.get("pattern_key", "").startswith("project_focus:") for p in patterns)
            and not any(p.get("pattern_key") == "project_breadth:multi" for p in patterns)
        ),
        "recommendation": "You're deeply focused on one project. If other projects need attention, consider scheduling dedicated sessions for them.",
        "category": "focus",
        "based_on": ["project_focus"],
    },
    {
        "id": "large_commits",
        "condition": lambda patterns: any(
            p.get("pattern_key") == "git_message_length"
            and "detailed" in p.get("content", "")
            for p in patterns
        ),
        "recommendation": "Your commit messages are detailed (avg 70+ chars). This suggests large commits. Smaller, atomic commits reduce churn and simplify review.",
        "category": "git",
        "based_on": ["git_message_length"],
    },
    {
        "id": "weekend_work",
        "condition": lambda patterns: any(
            p.get("pattern_key") == "session_weekday_pattern"
            and "weekend" in p.get("content", "").lower()
            for p in patterns
        ),
        "recommendation": "You frequently work on weekends. If unintentional, consider setting session time boundaries.",
        "category": "wellbeing",
        "based_on": ["session_weekday_pattern"],
    },
    {
        "id": "strong_co_edit_clusters",
        "condition": lambda patterns: len([
            p for p in patterns if p.get("pattern_type") == "co_edit_cluster"
        ]) >= 3,
        "recommendation": "You have 3+ file co-edit clusters. These files change together so often that a single change script or test suite covering all of them would save time.",
        "category": "efficiency",
        "based_on": ["co_edit_cluster"],
    },
]

# Phase 4: Dimension phrase builders for profile summary
DIMENSION_PHRASES = {
    "session_timing": lambda c: (
        "morning worker" if "morning" in c else
        "afternoon worker" if "afternoon" in c else
        "evening worker" if "evening" in c else
        "night owl"
    ),
    "git_workflow": lambda c: (
        "atomic committer" if "frequently" in c else
        "batch committer" if "sparingly" in c else
        "steady committer"
    ),
    "project_focus": lambda c: f"focused on {c.split('Primary project: ')[-1].split(' (')[0]}" if "Primary" in c else "multi-project developer",
    "handoff_quality": lambda c: "thorough handoff writer" if "Thorough" in c else "concise handoff writer",
    "task_management": lambda c: "high task completer" if "High" in c else ("steady task manager" if "Moderate" in c else "exploratory worker"),
    "workflow_sequence": lambda c: "disciplined workflow" if "handoff" in c.lower() else "consistent workflow",
    "tool_preference": lambda c: c.split("(")[0].strip() if "(" in c else c[:60],
    "co_edit_cluster": lambda c: "pattern-aware file editor",
}


class _ProfileMixin:
    """Phase 4 methods for BehavioralAnalyzer: correlation, profile, recommendations."""

    def _get_active_patterns(self) -> List[Dict[str, Any]]:
        """Fetch all behavioral_pattern memories with effective confidence >= threshold."""
        try:
            from omega.bridge import _get_store
            store = _get_store()
            habits = store.get_by_type("behavioral_pattern", limit=50)
        except Exception as e:
            logger.debug("Behavioral profile failed: %s", e)
            return []

        results = []
        for h in habits:
            meta = h.metadata or {}
            if meta.get("suppressed"):
                continue
            raw_conf = meta.get("confidence", 0)
            last_ev = meta.get("last_evidence_at") or meta.get("captured_at", "")
            eff_conf = effective_confidence(raw_conf, last_ev)
            if eff_conf < MIN_SURFACE_CONFIDENCE:
                continue
            results.append({
                "id": h.id,
                "content": h.content,
                "pattern_type": meta.get("pattern_type", ""),
                "pattern_key": meta.get("pattern_key", ""),
                "confidence": eff_conf,
                "raw_confidence": raw_conf,
                "evidence_count": meta.get("evidence_count", 0),
                "evidence_sessions": meta.get("evidence_sessions", 0),
                "user_confirmed": meta.get("user_confirmed"),
            })
        return results

    def detect_correlations(self) -> List[Dict[str, Any]]:
        """Detect cross-pattern correlations from stored behavioral patterns."""
        patterns = self._get_active_patterns()
        if not patterns:
            return []

        # Group by pattern_type
        by_type: Dict[str, List[Dict[str, Any]]] = {}
        for p in patterns:
            by_type.setdefault(p["pattern_type"], []).append(p)

        if len(by_type) < 2:
            return []

        insights = []
        for rule in CORRELATION_RULES:
            type_a = rule["type_a"]
            type_b = rule["type_b"]
            if type_a not in by_type or type_b not in by_type:
                continue

            # Filter by key_contains if specified
            candidates_a = by_type[type_a]
            if rule.get("key_a_contains"):
                candidates_a = [p for p in candidates_a if rule["key_a_contains"] in p["pattern_key"]]
            candidates_b = by_type[type_b]
            if rule.get("key_b_contains"):
                candidates_b = [p for p in candidates_b if rule["key_b_contains"] in p["pattern_key"]]

            if not candidates_a or not candidates_b:
                continue

            a = max(candidates_a, key=lambda p: p["confidence"])
            b = max(candidates_b, key=lambda p: p["confidence"])

            # Build labels from content
            a_label = a["content"][:60].split("(")[0].strip()
            b_label = b["content"][:60].split("(")[0].strip()

            try:
                message = rule["template"].format(
                    a_label=a_label, b_label=b_label,
                    a_conf=a["confidence"], b_conf=b["confidence"],
                )
            except (KeyError, IndexError):
                message = f"{a_label} correlates with {b_label}"

            insights.append({
                "message": message,
                "pattern_types": [type_a, type_b],
                "confidence": round(min(a["confidence"], b["confidence"]) * 0.9, 3),
                "source_ids": [a["id"], b["id"]],
            })

        return insights

    def synthesize_profile(self) -> Dict[str, Any]:
        """Generate composite behavioral profile from all stored patterns."""
        patterns = self._get_active_patterns()
        if not patterns:
            return {
                "summary": "No behavioral patterns detected yet.",
                "dimensions": {},
                "insights": [],
                "recommendations": [],
                "pattern_count": 0,
                "avg_confidence": 0.0,
                "last_analysis": datetime.now(timezone.utc).isoformat(),
            }

        # Pick highest-confidence pattern per type
        best_by_type: Dict[str, Dict[str, Any]] = {}
        for p in patterns:
            ptype = p["pattern_type"]
            if ptype not in best_by_type or p["confidence"] > best_by_type[ptype]["confidence"]:
                best_by_type[ptype] = p

        # Build dimensions
        dimensions: Dict[str, Dict[str, Any]] = {}
        for ptype, p in sorted(best_by_type.items(), key=lambda x: -x[1]["confidence"]):
            dimensions[ptype] = {
                "pattern": p["content"][:200],
                "confidence": p["confidence"],
            }

        # Build summary from top dimensions
        phrases = []
        for ptype, p in sorted(best_by_type.items(), key=lambda x: -x[1]["confidence"]):
            builder = DIMENSION_PHRASES.get(ptype)
            if builder:
                try:
                    phrase = builder(p["content"])
                    if phrase:
                        phrases.append(phrase)
                except Exception as e:
                    logger.debug("Profile phrase generation failed for %s: %s", ptype, e)
            if len(phrases) >= 4:
                break

        summary = ", ".join(phrases) if phrases else "Behavioral profile building"
        # Capitalize first letter
        if summary:
            summary = summary[0].upper() + summary[1:]

        # Cross-pattern insights
        insights = self.detect_correlations()
        insight_messages = [i["message"] for i in insights]

        # Recommendations
        recs = self.generate_recommendations()

        avg_conf = sum(p["confidence"] for p in patterns) / len(patterns)

        return {
            "summary": summary,
            "dimensions": dimensions,
            "insights": insight_messages,
            "recommendations": recs,
            "pattern_count": len(patterns),
            "avg_confidence": round(avg_conf, 3),
            "last_analysis": datetime.now(timezone.utc).isoformat(),
        }

    def generate_recommendations(self) -> List[Dict[str, str]]:
        """Generate actionable recommendations from behavioral patterns."""
        patterns = self._get_active_patterns()
        if not patterns:
            return []

        results = []
        for rule in RECOMMENDATION_RULES:
            try:
                if rule["condition"](patterns):
                    results.append({
                        "id": rule["id"],
                        "recommendation": rule["recommendation"],
                        "category": rule["category"],
                        "based_on": rule["based_on"],
                    })
            except Exception:
                logger.debug("Recommendation rule %s failed", rule["id"], exc_info=True)
        return results


# Apply mixin to BehavioralAnalyzer
for _method_name in ("_get_active_patterns", "detect_correlations", "synthesize_profile", "generate_recommendations"):
    setattr(BehavioralAnalyzer, _method_name, getattr(_ProfileMixin, _method_name))


# Module-level convenience function
def analyze_and_store() -> Dict[str, Any]:
    """Run behavioral analysis and store patterns. Module-level entry point."""
    analyzer = BehavioralAnalyzer()
    return analyzer.analyze_and_store()
