#!/usr/bin/env python3.11
"""One-shot script to bulk-store system insights from memory/insights.md into OMEGA.

Each insight is stored as event_type="advisor_insight" with metadata.category="system_insight"
and subsystem-specific tags. Run once, then delete or ignore.
"""

import sys
import time

# Ensure omega is importable
sys.path.insert(0, "src")

from omega.bridge import store  # noqa: E402

INSIGHTS = [
    {
        "content": (
            "Hook ordering matters with shared timeouts: Claude Code runs stop hooks "
            "sequentially within a single timeout window. Put fast, critical operations first "
            "(coord cleanup ~50ms) to guarantee completion. Heavy operations (session summary "
            "~5-6s) go last -- if timeout kills them, nothing critical is lost. "
            "Config: the coordination stop hook+session_stop+assistant_capture."
        ),
        "tags": ["hooks", "coordination", "session_stop", "timeouts"],
    },
    {
        "content": (
            "Three-layer defense for stale sessions: (1) _BEST_EFFORT_HOOKS ensures "
            "the coordination stop hook runs even when daemon is down -- prevents most leaks. "
            "(2) Heartbeat-based cleanup covers both active and stopped statuses -- catches "
            "partial stop hook execution. (3) PID liveness check on already-stale sessions -- "
            "catches crash/kill -9. Each layer handles a different failure mode."
        ),
        "tags": ["coordination", "sessions", "stale_sessions", "heartbeat", "hooks"],
    },
    {
        "content": (
            "Three failure modes compound for session leaks: (a) Stop hook has 8s timeout but "
            "session_stop alone can take 5-6s, leaving the coordination stop hook to get killed. "
            "(b) the coordination stop hook wasn't in _BEST_EFFORT_HOOKS, so daemon-down = silently "
            "skipped. (c) _clean_stale_sessions checked heartbeat age but never PID liveness, "
            "and only checked status='active' missing stopped sessions."
        ),
        "tags": ["coordination", "sessions", "stale_sessions", "failure_modes"],
    },
    {
        "content": (
            "Fact splitting amplifies noise via broad regex: _split_atomic_facts in bridge.py "
            "used is/are/was/were regex matching ~90% of English sentences. Combined with running "
            "on agent-authored types (advisor_insight, session_summary, lesson_learned), it spawned "
            "user_fact children from content the user never wrote. Fix: restrict to user-authored "
            "types (decision, user_fact) + require first-person signal (we/our/my/i or "
            "infrastructure nouns)."
        ),
        "tags": ["memory_engine", "bridge", "fact_splitting", "noise"],
    },
    {
        "content": (
            "user_fact had no guardrails: Unlike every other memory type, user_fact had no dedup "
            "threshold, no evolution, permanent TTL, and no quality gate -- so noise accumulated "
            "without any self-correcting mechanism. Fix: added 0.80 Jaccard dedup threshold."
        ),
        "tags": ["memory_engine", "bridge", "user_fact", "dedup"],
    },
    {
        "content": (
            "Overdue detection existed but wasn't wired to alerts: The insights API computed "
            "overdue = (nowMs - lastRun) > interval * 1.5 for every schedule, and the admin "
            "dashboard displayed it visually. But /api/notify (the email alert scanner) only "
            "checked last_status = 'error' -- it never asked 'has this job simply stopped running?' "
            "One 30-line block in notify/route.ts closed the gap."
        ),
        "tags": ["alerting", "monitoring", "cron", "overdue_detection"],
    },
    {
        "content": (
            "Silent heartbeat auth failures caused 4-day outage: _send_heartbeat() in "
            "maintenance.py caught all exceptions with logger.debug() -- including 401/403 from "
            "Supabase key rotation. The schedules.last_run_at stopped updating, but nobody checked "
            "because the last_status was still 'ok' from before the failure. Fix: log WARNING on "
            "401/403 specifically."
        ),
        "tags": ["alerting", "monitoring", "heartbeat", "supabase", "auth_failure"],
    },
    {
        "content": (
            "vec0 appears broken but isn't: Running SELECT vec_version() on omega.db without "
            "loading the extension first gives 'no such function' -- looks like semantic search is "
            "down. In reality, OMEGA's _base.py properly calls sqlite_vec.load(conn) on every "
            "connection. Always load the extension before testing. 87% of memories have vectors; "
            "the 13% missing are low-priority gardener observations."
        ),
        "tags": ["diagnostics", "sqlite_vec", "vectors", "false_alarm"],
    },
    {
        "content": (
            "False alarms waste investigation time: Before reporting an issue as critical, verify "
            "it in the system's actual runtime path, not just via ad-hoc queries that skip "
            "initialization steps (extension loading, config, etc.)."
        ),
        "tags": ["diagnostics", "debugging", "false_alarm"],
    },
]


def main():
    print(f"Storing {len(INSIGHTS)} system insights into OMEGA...")
    for i, insight in enumerate(INSIGHTS):
        result = store(
            content=insight["content"],
            event_type="advisor_insight",
            metadata={
                "category": "system_insight",
                "tags": insight["tags"],
            },
            project="omega",
            entity_id="omega",
        )
        print(f"  [{i+1}/{len(INSIGHTS)}] {result.strip()[:100]}")
        # Brief pause to avoid hammering the store pipeline
        time.sleep(0.5)
    print("Done.")


if __name__ == "__main__":
    main()
