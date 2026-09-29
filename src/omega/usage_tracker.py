"""LLM usage and cost tracking.

Logs every LLM call with token counts, estimates costs per model,
and provides aggregated usage queries for the admin dashboard.
"""
import os
import sqlite3
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path


MODEL_PRICING = {  # per 1M tokens (USD)
    "claude-opus-4-6": {"input": 15.0, "output": 75.0, "cache_read": 1.5, "cache_write": 18.75},
    "claude-sonnet-4-6": {"input": 3.0, "output": 15.0, "cache_read": 0.3, "cache_write": 3.75},
    "claude-haiku-4-5": {"input": 0.8, "output": 4.0, "cache_read": 0.08, "cache_write": 1.0},
    "nomic-embed-text": {"input": 0.0, "output": 0.0},
}

SCHEMA = """
CREATE TABLE IF NOT EXISTS llm_usage (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT,
    tool_name TEXT NOT NULL,
    model TEXT NOT NULL,
    provider TEXT NOT NULL DEFAULT 'anthropic',
    input_tokens INTEGER DEFAULT 0,
    output_tokens INTEGER DEFAULT 0,
    cache_read_tokens INTEGER DEFAULT 0,
    cache_write_tokens INTEGER DEFAULT 0,
    estimated_cost_usd REAL DEFAULT 0.0,
    duration_ms INTEGER,
    project TEXT,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_llm_usage_session ON llm_usage(session_id);
CREATE INDEX IF NOT EXISTS idx_llm_usage_tool ON llm_usage(tool_name, created_at);
CREATE INDEX IF NOT EXISTS idx_llm_usage_created ON llm_usage(created_at);
"""


def _estimate_cost(model: str, input_tokens: int, output_tokens: int,
                   cache_read: int = 0, cache_write: int = 0) -> float:
    pricing = MODEL_PRICING.get(model, MODEL_PRICING.get("claude-sonnet-4-6", {}))
    cost = (
        input_tokens * pricing.get("input", 3.0) / 1_000_000
        + output_tokens * pricing.get("output", 15.0) / 1_000_000
        + cache_read * pricing.get("cache_read", 0.3) / 1_000_000
        + cache_write * pricing.get("cache_write", 3.75) / 1_000_000
    )
    return round(cost, 6)


def _window_start(days: int) -> str:
    """Start of a look-back window, in the isoformat() text created_at is stored as.

    Comparing that column with SQLite datetime() text is wrong: ' ' sorts before
    'T', so every call from the cutoff's calendar day counted as inside the window.
    """
    return (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()


class UsageTracker:
    def __init__(self, db_path: str | None = None):
        if db_path is None:
            omega_home = Path(os.environ.get("OMEGA_HOME", str(Path.home() / ".omega")))
            db_path = str(omega_home / "llm_usage.db")
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript(SCHEMA)
        self._conn.commit()
        self._lock = threading.Lock()
        # Ensure DB file is owner-only (not world-readable)
        try:
            os.chmod(db_path, 0o600)
        except OSError:
            pass

    def close(self) -> None:
        self._conn.close()

    def log_call(
        self,
        session_id: str | None,
        tool_name: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
        cache_read: int = 0,
        cache_write: int = 0,
        duration_ms: int | None = None,
        project: str | None = None,
    ) -> None:
        provider = "local" if model.startswith("nomic") else "anthropic"
        cost = _estimate_cost(model, input_tokens, output_tokens, cache_read, cache_write)
        now = datetime.now(timezone.utc).isoformat()
        with self._lock:
            self._conn.execute(
                """INSERT INTO llm_usage
                   (session_id, tool_name, model, provider, input_tokens, output_tokens,
                    cache_read_tokens, cache_write_tokens, estimated_cost_usd,
                    duration_ms, project, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (session_id, tool_name, model, provider, input_tokens, output_tokens,
                 cache_read, cache_write, cost, duration_ms, project, now),
            )
            self._conn.commit()

    def get_usage(self, days: int = 7, group_by: str = "model") -> list[dict]:
        valid_groups = {"model", "tool_name", "session_id", "project"}
        if group_by not in valid_groups:
            group_by = "model"
        with self._lock:
            rows = self._conn.execute(
                f"""SELECT {group_by},
                           SUM(input_tokens) as total_input_tokens,
                           SUM(output_tokens) as total_output_tokens,
                           SUM(estimated_cost_usd) as total_cost,
                           COUNT(*) as call_count
                    FROM llm_usage
                    WHERE created_at > ?
                    GROUP BY {group_by}
                    ORDER BY total_cost DESC""",
                (_window_start(days),),
            ).fetchall()
        return [
            {group_by: r[0], "total_input_tokens": r[1], "total_output_tokens": r[2],
             "total_cost_usd": r[3], "call_count": r[4]}
            for r in rows
        ]

    def get_cost_estimate(self, days: int = 30) -> dict:
        with self._lock:
            row = self._conn.execute(
                """SELECT SUM(estimated_cost_usd), SUM(input_tokens), SUM(output_tokens),
                          COUNT(*)
                   FROM llm_usage
                   WHERE created_at > ?""",
                (_window_start(days),),
            ).fetchone()
        return {
            "total_usd": round(row[0] or 0, 4),
            "total_input_tokens": row[1] or 0,
            "total_output_tokens": row[2] or 0,
            "total_calls": row[3] or 0,
            "period_days": days,
        }

    def get_top_tools(self, days: int = 7, limit: int = 10) -> list[dict]:
        with self._lock:
            rows = self._conn.execute(
                """SELECT tool_name,
                          SUM(input_tokens + output_tokens) as total_tokens,
                          SUM(estimated_cost_usd) as total_cost,
                          COUNT(*) as call_count
                   FROM llm_usage
                   WHERE created_at > ?
                   GROUP BY tool_name
                   ORDER BY call_count DESC
                   LIMIT ?""",
                (_window_start(days), limit),
            ).fetchall()
        return [
            {"tool_name": r[0], "total_tokens": r[1], "total_cost_usd": r[2], "call_count": r[3]}
            for r in rows
        ]
