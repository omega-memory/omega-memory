"""Read-only reliability diagnostics for an explicitly selected Core database.

This module deliberately uses a SQLite ``mode=ro`` connection.  It never
discovers a default OMEGA home, so a user must opt in by supplying a database
path to the CLI command.
"""

from __future__ import annotations

import json
import sqlite3
import statistics
from collections import Counter
from pathlib import Path
from typing import Any


_REQUIRED_TABLES = frozenset({"memories", "edges"})
_DOMINANT_PRIORITY_SHARE = 0.50


def _metadata(value: str | None) -> dict[str, Any]:
    """Return object metadata without making malformed rows fatal to an audit."""
    try:
        parsed = json.loads(value or "{}")
    except (TypeError, json.JSONDecodeError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _database_uri(path: Path) -> str:
    """Build a read-only URI for an existing local SQLite database."""
    if not path.exists() or not path.is_file():
        raise ValueError("audit requires an explicit existing SQLite database path")
    return f"{path.resolve().as_uri()}?mode=ro"


def _require_schema(connection: sqlite3.Connection) -> None:
    tables = {
        row[0]
        for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        )
    }
    missing = sorted(_REQUIRED_TABLES - tables)
    if missing:
        raise ValueError(f"database is not an OMEGA Core database; missing table(s): {', '.join(missing)}")


def audit_database(database: str | Path) -> dict[str, Any]:
    """Return a read-only reliability report for *database*.

    The report deliberately identifies records by node ID only: diagnostics
    must be actionable without copying memory contents into a release report.
    """
    path = Path(database)
    with sqlite3.connect(_database_uri(path), uri=True) as connection:
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA query_only = ON")
        _require_schema(connection)
        rows = list(
            connection.execute(
                """
                SELECT node_id, content, metadata, access_count, priority, status
                FROM memories
                ORDER BY node_id
                """
            )
        )
        edges = {
            (row["source_id"], row["target_id"])
            for row in connection.execute(
                "SELECT source_id, target_id FROM edges WHERE edge_type = 'supersedes'"
            )
        }

    access_counts = [max(0, int(row["access_count"] or 0)) for row in rows]
    mean_access = statistics.fmean(access_counts) if access_counts else 0.0
    median_access = statistics.median(access_counts) if access_counts else 0.0
    outlier_cutoff = max(3.0, mean_access * 3.0)
    access_outliers = [
        {"node_id": row["node_id"], "access_count": int(row["access_count"] or 0)}
        for row in rows
        if max(0, int(row["access_count"] or 0)) > outlier_cutoff
    ]

    priority_counts = Counter(int(row["priority"] or 0) for row in rows)
    valid_priority_counts = {priority: count for priority, count in priority_counts.items() if 1 <= priority <= 5}
    dominant_priority = None
    if rows and valid_priority_counts:
        priority, count = max(valid_priority_counts.items(), key=lambda item: (item[1], item[0]))
        share = count / len(rows)
        if share >= _DOMINANT_PRIORITY_SHARE:
            dominant_priority = {
                "priority": priority,
                "count": count,
                "share": round(share, 6),
                "records": [
                    {
                        "node_id": row["node_id"],
                        "access_count": int(row["access_count"] or 0),
                    }
                    for row in rows
                    if int(row["priority"] or 0) == priority
                ],
            }

    invalid_active_records: list[dict[str, str]] = []
    supersede_inconsistencies: list[dict[str, str]] = []
    node_ids = {str(row["node_id"]) for row in rows}
    for row in rows:
        node_id = str(row["node_id"])
        metadata = _metadata(row["metadata"])
        status = str(row["status"] or "active")
        priority = int(row["priority"] or 0)
        reasons: list[str] = []
        if status == "active" and not str(row["content"] or "").strip():
            reasons.append("empty_content")
        if status == "active" and not 1 <= priority <= 5:
            reasons.append("invalid_priority")
        metadata_priority = metadata.get("priority")
        if status == "active" and metadata_priority is not None and metadata_priority != priority:
            reasons.append("priority_metadata_mismatch")
        if status == "active" and metadata.get("superseded"):
            reasons.append("active_with_superseded_metadata")
        if reasons:
            invalid_active_records.append({"node_id": node_id, "reasons": reasons})

        metadata_superseded = bool(metadata.get("superseded"))
        if (status == "superseded") != metadata_superseded:
            supersede_inconsistencies.append(
                {"node_id": node_id, "reason": "status_metadata_mismatch"}
            )
        replacement_id = metadata.get("superseded_by")
        if replacement_id:
            replacement_id = str(replacement_id)
            if replacement_id not in node_ids:
                supersede_inconsistencies.append(
                    {"node_id": node_id, "reason": "replacement_missing"}
                )
            elif (replacement_id, node_id) not in edges:
                supersede_inconsistencies.append(
                    {"node_id": node_id, "reason": "replacement_edge_missing"}
                )
        elif status == "superseded" and metadata.get("superseded_reason") is None:
            supersede_inconsistencies.append(
                {"node_id": node_id, "reason": "replacement_or_reason_missing"}
            )

    for source_id, target_id in sorted(edges):
        target = next((row for row in rows if row["node_id"] == target_id), None)
        if target is None:
            supersede_inconsistencies.append(
                {"node_id": target_id, "reason": f"supersedes_edge_targets_missing_record:{source_id}"}
            )
        elif str(target["status"] or "active") != "superseded":
            supersede_inconsistencies.append(
                {"node_id": target_id, "reason": f"supersedes_edge_targets_active_record:{source_id}"}
            )

    return {
        "database": str(path),
        "read_only": True,
        "summary": {
            "memory_count": len(rows),
            "supersedes_edge_count": len(edges),
        },
        "access": {
            "distribution": {
                "min": min(access_counts, default=0),
                "max": max(access_counts, default=0),
                "mean": round(mean_access, 6),
                "median": median_access,
            },
            "outlier_cutoff": round(outlier_cutoff, 6),
            "outliers": access_outliers,
        },
        "priority": {
            "distribution": {str(priority): count for priority, count in sorted(priority_counts.items())},
            "dominance": dominant_priority,
        },
        "invalid_active_records": invalid_active_records,
        "supersede_inconsistencies": supersede_inconsistencies,
    }
