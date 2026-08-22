"""Handlers for the harness-agnostic structured-context wire contract.

context_assemble returns OMEGA's active context as a list of structured
sections plus a pre-rendered markdown blob. The six section ids map
honestly onto OMEGA's event_type model -- no invented categories.

Token counts are a cheap heuristic (len/4); honest about being an estimate
so callers know not to treat them as authoritative.
"""
from __future__ import annotations

import logging
import os
import inspect
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from omega.server.responses import mcp_error, mcp_response

logger = logging.getLogger("omega.server.context_handlers")


# ---------------------------------------------------------------------------
# Section definitions
# ---------------------------------------------------------------------------
_DEFAULT_SINCE_DAYS = 7

# Ordered: lower priority number = higher precedence in composed_markdown.
_SECTION_SPECS: List[Dict[str, Any]] = [
    {"id": "active_reminders",   "priority": 1, "event_type": None,               "limit": 10, "use_since": False, "order_by": "recency"},
    {"id": "active_constraints", "priority": 2, "event_type": "constraint",       "limit": 10, "use_since": False, "order_by": "recency"},
    {"id": "recent_decisions",   "priority": 3, "event_type": "decision",         "limit": 10, "use_since": True,  "order_by": "recency"},
    {"id": "top_lessons",        "priority": 4, "event_type": "lesson_learned",   "limit": 10, "use_since": False, "order_by": "recency"},
    {"id": "user_preferences",   "priority": 5, "event_type": "user_preference",  "limit": 8,  "use_since": False, "order_by": "recency"},
    {"id": "user_facts",         "priority": 6, "event_type": "user_fact",        "limit": 8,  "use_since": False, "order_by": "recency"},
]

_SECTION_TITLES = {
    "active_reminders":   "Active reminders",
    "active_constraints": "Active constraints",
    "recent_decisions":   "Recent decisions",
    "top_lessons":        "Top lessons",
    "user_preferences":   "User preferences",
    "user_facts":         "User facts",
}


def _estimate_tokens(text: str) -> int:
    """Cheap heuristic -- len/4 approximates a tiktoken count well enough for budgeting."""
    return max(0, len(text) // 4)


def _flatten_scope(scope: Optional[Dict[str, Any]]) -> Dict[str, Optional[str]]:
    if not isinstance(scope, dict):
        return {"entity_id": None, "project": None, "session_id": None}
    return {
        "entity_id": scope.get("entity_id"),
        "project": scope.get("project"),
        "session_id": scope.get("session_id"),
    }


def _parse_since(since: Optional[str]) -> str:
    """Return an ISO-8601 string for the lower bound. Falls back to 7 days ago."""
    if since:
        try:
            datetime.fromisoformat(since.replace("Z", "+00:00"))
            return since
        except (ValueError, AttributeError):
            logger.debug("context_assemble: invalid since=%r; using default", since)
    return (datetime.now(timezone.utc) - timedelta(days=_DEFAULT_SINCE_DAYS)).isoformat()


# ---------------------------------------------------------------------------
# Section builders
# ---------------------------------------------------------------------------
def _build_section(db, spec: Dict[str, Any], scope: Dict[str, Optional[str]], since_iso: str) -> Dict[str, Any]:
    """Build one section. Returns a dict with id/priority/content/tokens/count/items."""
    section_id = spec["id"]
    items: List[Dict[str, Any]] = []

    if section_id == "active_reminders":
        items = _query_active_reminders(scope["entity_id"])
    else:
        rows = _query_by_event_type(
            db,
            event_type=spec["event_type"],
            entity_id=scope["entity_id"],
            limit=spec["limit"],
            since_iso=since_iso if spec["use_since"] else None,
            order_by=spec["order_by"],
        )
        items = [{"id": r["id"], "content": r["content"], "created_at": r["created_at"]} for r in rows]

    content = _render_section_markdown(section_id, items)
    return {
        "id": section_id,
        "priority": spec["priority"],
        "content": content,
        "tokens": _estimate_tokens(content),
        "count": len(items),
        "items": items,
        **({"since": since_iso} if spec["use_since"] else {}),
    }


def _query_by_event_type(
    db,
    event_type: str,
    entity_id: Optional[str],
    limit: int,
    since_iso: Optional[str],
    order_by: str,
) -> List[Dict[str, Any]]:
    """Raw SQL filter against memories table. Skips superseded rows."""
    # Access history is audit data, never a context-selection signal.
    order_clause = "created_at DESC"
    sql = (
        "SELECT node_id, content, metadata, created_at "
        "FROM memories WHERE event_type = ? AND COALESCE(status, 'active') != 'superseded'"
    )
    params: List[Any] = [event_type]
    if entity_id:
        sql += " AND COALESCE(entity_id, '') = ?"
        params.append(entity_id)
    if since_iso:
        sql += " AND created_at >= ?"
        params.append(since_iso)
    sql += f" ORDER BY {order_clause} LIMIT ?"
    params.append(limit)

    with db._lock:
        rows = db._conn.execute(sql, params).fetchall()

    return [
        {"id": r[0], "content": r[1], "created_at": r[3]}
        for r in rows
    ]


def _query_active_reminders(entity_id: Optional[str]) -> List[Dict[str, Any]]:
    """Delegate to bridge.list_reminders for the active set."""
    try:
        from omega.bridge import list_reminders
        rows = list_reminders(status="pending", include_dismissed=False, entity_id=entity_id)
    except Exception as exc:
        logger.debug("context_assemble: list_reminders failed: %s", exc)
        return []
    # bridge.list_reminders returns dicts with key 'text' (not 'content');
    # accept either for forward-compat in case the bridge shape changes.
    return [
        {
            "id": r.get("id") or r.get("node_id") or "",
            "content": r.get("text") or r.get("content") or "",
            "remind_at": r.get("remind_at"),
        }
        for r in rows
    ]


# ---------------------------------------------------------------------------
# Markdown rendering
# ---------------------------------------------------------------------------
def _render_section_markdown(section_id: str, items: List[Dict[str, Any]]) -> str:
    if not items:
        return ""
    lines = [f"## {_SECTION_TITLES.get(section_id, section_id)}"]
    for item in items:
        snippet = (item.get("content") or "").strip().splitlines()[0] if item.get("content") else ""
        if len(snippet) > 200:
            snippet = snippet[:197] + "..."
        when = item.get("created_at") or item.get("remind_at") or ""
        prefix = f"- `{when[:10]}` " if when else "- "
        lines.append(f"{prefix}{snippet}")
    return "\n".join(lines)


def _compose_markdown(sections: List[Dict[str, Any]]) -> str:
    """Concatenate non-empty section bodies in priority order."""
    blocks = [s["content"] for s in sections if s["content"]]
    return "\n\n".join(blocks)


# ---------------------------------------------------------------------------
# Task-aware context packet
# ---------------------------------------------------------------------------
_PACKET_MODE_TO_SURFACING = {
    "before_edit": "FILE_EDIT",
    "planning": "PLANNING",
    "debug": "ERROR_DEBUG",
    "review": "REVIEW",
    "command": "GENERAL",
}

_PACKET_TYPE_WEIGHT = {
    "constraint": 3.0,
    "decision": 2.4,
    "error_pattern": 2.1,
    "lesson_learned": 1.8,
    "user_preference": 1.6,
    "user_fact": 1.2,
    "memory": 1.0,
}

_PACKET_SECTION_BY_TYPE = {
    "constraint": "Critical",
    "decision": "Prior Decisions",
    "error_pattern": "Known Failure Modes",
    "lesson_learned": "Lessons",
    "user_preference": "Preferences",
    "user_fact": "Facts",
}

_PACKET_EDGE_TYPE_WEIGHT = {
    "supersedes": 1.2,
    "same_entity": 1.12,
    "evolution": 1.12,
    "derived_from": 1.12,
    "evolves": 1.1,
    "contradicts": 1.05,
    "temporal_cluster": 1.03,
    "related": 1.0,
}

_PACKET_EDGE_SOURCE_WEIGHT = {
    "context_packet_miss_backfill": 1.1,
    "context_packet_eval_backfill": 1.06,
    "discover_connections": 1.03,
}

_PACKET_SOURCE_WEIGHT = {
    "seed": 1.2,
    "chain": 1.0,
}

_SENSITIVITY_ORDER = {"public": 0, "internal": 1, "confidential": 2, "restricted": 3}
_DEFAULT_PACKET_BUDGET = 800
_MIN_PACKET_BUDGET = 120
_MAX_PACKET_BUDGET = 4000
_PACKET_MAX_RENDERED_ITEMS = 8
_PACKET_MAX_RENDERED_WARNINGS = 1

_PACKET_WARNING_REASON_WEIGHT = {
    "contradicts": 3,
    "flagged_for_review": 2,
    "stale": 1,
}


def _coerce_packet_files(raw: Any) -> List[str]:
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise ValueError("files must be an array of strings")
    files = []
    for item in raw:
        if isinstance(item, str) and item.strip():
            files.append(item.strip())
    return files[:8]


def _coerce_packet_budget(raw: Any) -> int:
    try:
        value = int(raw) if raw is not None else _DEFAULT_PACKET_BUDGET
    except (TypeError, ValueError):
        value = _DEFAULT_PACKET_BUDGET
    return max(_MIN_PACKET_BUDGET, min(_MAX_PACKET_BUDGET, value))


def _packet_query_text(task: str, files: List[str]) -> str:
    parts: List[str] = []
    if task:
        parts.append(task)
    for file_path in files:
        parts.append(file_path)
        parts.append(os.path.basename(file_path))
        parent = os.path.basename(os.path.dirname(file_path))
        if parent:
            parts.append(parent)
    return " ".join(p for p in parts if p).strip()


def _memory_scope_row(db, node_id: str) -> Dict[str, Any]:
    try:
        with db._lock:
            columns = {
                row[1]
                for row in db._conn.execute("PRAGMA table_info(memories)").fetchall()
            }
            select_cols = [
                "status" if "status" in columns else "'active'",
                "sensitivity" if "sensitivity" in columns else "NULL",
                "entity_id" if "entity_id" in columns else "NULL",
                "project" if "project" in columns else "NULL",
                "project_id" if "project_id" in columns else "NULL",
            ]
            row = db._conn.execute(
                f"SELECT {', '.join(select_cols)} FROM memories WHERE node_id = ?",
                (node_id,),
            ).fetchone()
    except Exception:
        row = None
    if not row:
        return {}
    return {
        "status": row[0] or "active",
        "sensitivity": row[1],
        "entity_id": row[2],
        "project": row[3],
        "project_id": row[4],
    }


def _query_packet_scope_fallback(
    db,
    *,
    scope: Dict[str, Optional[str]],
    max_sensitivity: str,
    limit: int = 5,
) -> List[Dict[str, Any]]:
    """Return a small scoped fallback when primary retrieval abstains."""
    allowed_levels = [
        name for name, level in _SENSITIVITY_ORDER.items()
        if level <= _SENSITIVITY_ORDER.get(max_sensitivity, 3)
    ]
    if not allowed_levels:
        return []

    try:
        with db._lock:
            columns = {
                row[1]
                for row in db._conn.execute("PRAGMA table_info(memories)").fetchall()
            }
    except Exception:
        columns = set()

    sql = (
        "SELECT node_id, content, metadata, created_at, event_type "
        "FROM memories WHERE COALESCE(status, 'active') = 'active' "
        "AND event_type IN ('constraint', 'decision', 'error_pattern', 'lesson_learned', "
        "'user_preference', 'user_fact', 'memory')"
    )
    params: List[Any] = []
    if "sensitivity" in columns:
        placeholders = ",".join("?" for _ in allowed_levels)
        sql += " AND COALESCE(sensitivity, 'internal') IN (" + placeholders + ")"
        params.extend(allowed_levels)
    if scope.get("entity_id"):
        sql += " AND COALESCE(entity_id, '') = ?"
        params.append(scope["entity_id"])
    if scope.get("project"):
        sql += " AND COALESCE(project, '') = ?"
        params.append(scope["project"])
    sql += " ORDER BY created_at DESC LIMIT ?"
    params.append(limit)

    try:
        with db._lock:
            rows = db._conn.execute(sql, params).fetchall()
    except Exception:
        return []

    fallback = []
    for row in rows:
        metadata = row[2] or {}
        if isinstance(metadata, str):
            try:
                import json
                metadata = json.loads(metadata)
            except Exception:
                metadata = {}
        sensitivity = metadata.get("sensitivity") or "internal"
        if sensitivity not in allowed_levels:
            continue
        metadata.setdefault("event_type", row[4] or metadata.get("event_type") or "memory")
        fallback.append({
            "id": row[0],
            "content": row[1],
            "metadata": metadata,
            "created_at": row[3],
            "event_type": row[4],
            "relevance": 0.35,
        })
    return fallback


def _apply_contradiction_warnings(db, candidates: Dict[str, Dict[str, Any]]) -> None:
    """Mark the lower-ranked side of in-packet contradiction edges as a warning."""
    if len(candidates) < 2:
        return
    ids = list(candidates.keys())
    placeholders = ",".join("?" for _ in ids)
    try:
        with db._lock:
            rows = db._conn.execute(
                f"""SELECT source_id, target_id FROM edges
                    WHERE edge_type = 'contradicts'
                    AND source_id IN ({placeholders})
                    AND target_id IN ({placeholders})""",
                ids + ids,
            ).fetchall()
    except Exception:
        return

    for source_id, target_id in rows:
        source = candidates.get(source_id)
        target = candidates.get(target_id)
        if not source or not target:
            continue
        source_rank = (
            _PACKET_TYPE_WEIGHT.get(source.get("event_type"), 1.0),
            source.get("score", 0),
        )
        target_rank = (
            _PACKET_TYPE_WEIGHT.get(target.get("event_type"), 1.0),
            target.get("score", 0),
        )
        if source_rank < target_rank:
            weaker = source
            stronger = target
        else:
            weaker = target
            stronger = source
        weaker["_packet_warning_reason"] = "contradicts"
        stronger["_packet_contradiction_winner"] = True


def _edge_metadata_source(edge_metadata: Optional[Dict[str, Any]]) -> str:
    if not isinstance(edge_metadata, dict):
        return ""
    source = edge_metadata.get("source") or ""
    return str(source)


def _packet_edge_multiplier(edge_type: Optional[str], edge_metadata: Optional[Dict[str, Any]]) -> float:
    type_factor = _PACKET_EDGE_TYPE_WEIGHT.get(edge_type or "related", 1.0)
    source_factor = _PACKET_EDGE_SOURCE_WEIGHT.get(_edge_metadata_source(edge_metadata), 1.0)
    typed_bonus = 1.02 if isinstance(edge_metadata, dict) and edge_metadata.get("typed") else 1.0
    return type_factor * source_factor * typed_bonus


def _packet_chain_entry_rank(entry: Dict[str, Any]) -> tuple[float, float, int]:
    edge_weight = max(0.0, min(float(entry.get("weight") or 0.0), 1.0))
    edge_factor = _packet_edge_multiplier(entry.get("edge_type"), entry.get("edge_metadata"))
    hop = int(entry.get("hop") or 1)
    return (edge_weight * edge_factor, -edge_weight, -hop)


def _memory_to_packet_item(
    db,
    memory: Any,
    *,
    source: str,
    seed_id: Optional[str] = None,
    edge_type: Optional[str] = None,
    edge_metadata: Optional[Dict[str, Any]] = None,
    edge_weight: float = 1.0,
    hop: int = 0,
    base_score: float = 1.0,
) -> Optional[Dict[str, Any]]:
    node_id = getattr(memory, "id", None)
    if node_id is None and isinstance(memory, dict):
        node_id = memory.get("node_id") or memory.get("id")
    if not node_id:
        return None

    content = getattr(memory, "content", None)
    metadata = getattr(memory, "metadata", None)
    created_at = getattr(memory, "created_at", None)
    relevance = getattr(memory, "relevance", None)
    status = getattr(memory, "status", None)

    if isinstance(memory, dict):
        content = memory.get("content", content)
        metadata = memory.get("metadata", metadata)
        created_at = memory.get("created_at", created_at)
        relevance = memory.get("relevance", relevance)
        status = memory.get("status", status)

    metadata = metadata or {}
    scope_row = _memory_scope_row(db, node_id)
    event_type = metadata.get("event_type") or (memory.get("event_type") if isinstance(memory, dict) else None) or "memory"
    status = scope_row.get("status") or metadata.get("status") or status or "active"
    sensitivity = scope_row.get("sensitivity") or metadata.get("sensitivity") or "internal"

    type_weight = _PACKET_TYPE_WEIGHT.get(event_type, 1.0)
    hop_factor = 1.0 if hop <= 0 else 0.55 ** hop
    edge_strength = max(0.2, min(float(edge_weight or 0.0), 1.0))
    edge_factor = edge_strength * _packet_edge_multiplier(edge_type, edge_metadata)
    source_factor = _PACKET_SOURCE_WEIGHT.get(source, 1.0)
    rel = float(relevance if relevance is not None else base_score)
    score = rel * type_weight * source_factor * hop_factor * edge_factor

    if created_at and hasattr(created_at, "isoformat"):
        created_str = created_at.isoformat()
    else:
        created_str = str(created_at or "")

    return {
        "id": node_id,
        "content": str(content or ""),
        "event_type": event_type,
        "metadata": metadata,
        "created_at": created_str,
        "relevance": rel,
        "score": round(score, 4),
        "status": status,
        "sensitivity": sensitivity,
        "entity_id": scope_row.get("entity_id") or metadata.get("entity_id"),
        "project": scope_row.get("project") or metadata.get("project"),
        "project_id": scope_row.get("project_id") or metadata.get("project_id"),
        "source": source,
        "seed_id": seed_id,
        "edge_type": edge_type,
        "edge_metadata": edge_metadata or {},
        "edge_source": _edge_metadata_source(edge_metadata),
        "edge_weight": edge_weight,
        "hop": hop,
    }


def _admit_packet_item(
    item: Dict[str, Any],
    *,
    scope: Dict[str, Optional[str]],
    max_sensitivity: str,
) -> tuple[bool, Optional[str]]:
    if not item.get("content"):
        return False, "empty"

    status = item.get("status") or "active"
    if status == "superseded" or item.get("metadata", {}).get("superseded"):
        return False, "superseded"
    if item.get("metadata", {}).get("flagged_for_review"):
        return False, "flagged_for_review"
    if item.get("_packet_warning_reason") == "contradicts":
        return False, "contradicts"
    if item.get("edge_type") == "contradicts" and not item.get("_packet_contradiction_winner"):
        return False, "contradicts"

    item_level = _SENSITIVITY_ORDER.get(item.get("sensitivity", "internal"), 1)
    max_level = _SENSITIVITY_ORDER.get(max_sensitivity, 3)
    if item_level > max_level:
        return False, "sensitivity"

    scope_entity = scope.get("entity_id")
    item_entity = item.get("entity_id")
    if scope_entity and item_entity and item_entity != scope_entity and item.get("sensitivity") != "public":
        return False, "entity_scope"

    scope_project = scope.get("project")
    item_project = item.get("project")
    if scope_project and item_project and item_project != scope_project and item.get("sensitivity") != "public":
        return False, "project_scope"

    if status == "stale":
        return False, "stale"

    return True, None


def _packet_snippet(text: str, max_chars: int = 210) -> str:
    snippet = " ".join((text or "").strip().split())
    if len(snippet) > max_chars:
        snippet = snippet[: max_chars - 3].rstrip() + "..."
    return snippet


def _rank_packet_warnings(warnings: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Rank warnings by severity, then item score."""
    return sorted(
        warnings,
        key=lambda warning: (
            _PACKET_WARNING_REASON_WEIGHT.get(warning.get("reason", ""), 0),
            (warning.get("item") or {}).get("score", 0),
        ),
        reverse=True,
    )


def _packet_warning_counts(warnings: List[Dict[str, Any]]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for warning in warnings:
        reason = str(warning.get("reason") or "warning")
        counts[reason] = counts.get(reason, 0) + 1
    return counts


def _render_packet_warning_summary(warnings: List[Dict[str, Any]]) -> List[str]:
    if not warnings:
        return []
    ranked = _rank_packet_warnings(warnings)
    counts = _packet_warning_counts(warnings)
    count_bits = ", ".join(f"{count} {reason}" for reason, count in sorted(counts.items()))
    lines = [f"- {len(warnings)} warning(s) in structured receipt: {count_bits}."]
    for warning in ranked[:_PACKET_MAX_RENDERED_WARNINGS]:
        reason = warning.get("reason", "warning")
        item = warning.get("item", {})
        lines.append(f"- top {reason}: `{item.get('id', '')[:12]}` {_packet_snippet(item.get('content', ''), 140)}")
    hidden = max(len(ranked) - _PACKET_MAX_RENDERED_WARNINGS, 0)
    if hidden:
        lines.append(f"- {hidden} additional warning(s) omitted from prompt; available in `warnings` payload.")
    return lines


def _rendered_packet_warning_count(packet_markdown: str) -> int:
    """Count visible warning item lines, not summary/receipt lines."""
    count = 0
    in_warnings = False
    for line in packet_markdown.splitlines():
        if line.strip() == "Warnings:":
            in_warnings = True
            continue
        if in_warnings and (not line.strip() or line.startswith("Receipt:")):
            in_warnings = False
            continue
        if in_warnings and line.startswith("- top "):
            count += 1
    return count


def _rendered_packet_memory_ids(packet_markdown: str) -> List[str]:
    """Return memory IDs rendered as packet items, excluding warning receipts."""
    ids: List[str] = []
    in_warnings = False
    for line in packet_markdown.splitlines():
        if line.strip() == "Warnings:":
            in_warnings = True
            continue
        if in_warnings and (not line.strip() or line.startswith("Receipt:")):
            in_warnings = False
            continue
        if in_warnings:
            continue
        if line.startswith("- `"):
            end = line.find("`", 3)
            if end > 3:
                ids.append(line[3:end])
    return ids


def _render_context_packet(
    *,
    title: str,
    admitted: List[Dict[str, Any]],
    warnings: List[Dict[str, Any]],
    budget_tokens: int,
    include_receipt: bool,
) -> str:
    lines = [f"[MEMORY_CONTEXT] Context packet for {title}"]
    used_ids: set[str] = set()

    by_section: Dict[str, List[Dict[str, Any]]] = {}
    for item in admitted:
        section = _PACKET_SECTION_BY_TYPE.get(item["event_type"], "Relevant Context")
        by_section.setdefault(section, []).append(item)

    section_order = [
        "Critical",
        "Prior Decisions",
        "Known Failure Modes",
        "Lessons",
        "Preferences",
        "Facts",
        "Relevant Context",
    ]

    def can_add(candidate: str) -> bool:
        projected = "\n".join(lines + [candidate])
        return _estimate_tokens(projected) <= budget_tokens

    rendered_sections: set[str] = set()
    rendered_count = 0
    top_item = admitted[0] if admitted else None
    if top_item:
        top_section = _PACKET_SECTION_BY_TYPE.get(top_item["event_type"], "Relevant Context")
        heading = f"\n{top_section}:"
        if can_add(heading):
            lines.append(heading)
            rendered_sections.add(top_section)
        edge = ""
        if top_item.get("edge_type"):
            edge = f" ({top_item['edge_type']}, hop {top_item.get('hop', 0)})"
        candidate = f"- `{top_item['id'][:12]}`{edge}: {_packet_snippet(top_item['content'])}"
        if can_add(candidate):
            lines.append(candidate)
            used_ids.add(top_item["id"])
            rendered_count += 1

    for section in section_order:
        if rendered_count >= _PACKET_MAX_RENDERED_ITEMS:
            break
        items = by_section.get(section, [])
        if not items:
            continue
        heading = f"\n{section}:"
        if section not in rendered_sections and can_add(heading):
            lines.append(heading)
            rendered_sections.add(section)
        for item in items[:4]:
            if top_item and item["id"] == top_item["id"]:
                continue
            if rendered_count >= _PACKET_MAX_RENDERED_ITEMS:
                break
            edge = ""
            if item.get("edge_type"):
                edge = f" ({item['edge_type']}, hop {item.get('hop', 0)})"
            candidate = f"- `{item['id'][:12]}`{edge}: {_packet_snippet(item['content'])}"
            if not can_add(candidate):
                break
            lines.append(candidate)
            used_ids.add(item["id"])
            rendered_count += 1

    warning_lines = _render_packet_warning_summary(warnings)
    if warning_lines and can_add("\nWarnings:"):
        lines.append("\nWarnings:")
        for candidate in warning_lines:
            if can_add(candidate):
                lines.append(candidate)

    if include_receipt:
        packet_without_receipt = "\n".join(lines)
        estimated = _estimate_tokens(packet_without_receipt)
        raw_tokens = sum(_estimate_tokens(i.get("content", "")) for i in admitted if i["id"] in used_ids)
        saved = max(raw_tokens - estimated, 0)
        warning_note = f" · {len(warnings)} warning(s)" if warnings else ""
        receipt = (
            f"\nReceipt: Used {len(used_ids)} memories · ~{estimated} tokens"
            f"{warning_note} · saved ~{saved} tokens vs raw recall."
        )
        if can_add(receipt):
            lines.append(receipt)

    return "\n".join(lines)


def _packet_candidate_receipt(
    item: Dict[str, Any],
    *,
    rank: int,
    admitted: bool,
    reason: Optional[str],
) -> Dict[str, Any]:
    return {
        "id": item["id"],
        "rank": rank,
        "event_type": item.get("event_type") or "memory",
        "section": _PACKET_SECTION_BY_TYPE.get(item.get("event_type"), "Relevant Context"),
        "score": item.get("score", 0),
        "source": item.get("source") or "",
        "seed_id": item.get("seed_id"),
        "edge_type": item.get("edge_type"),
        "edge_source": item.get("edge_source") or "",
        "edge_weight": item.get("edge_weight"),
        "hop": item.get("hop", 0),
        "admitted": admitted,
        "admission_reason": "admitted" if admitted else (reason or "filtered"),
        "rendered": False,
        "render_rank": None,
    }


def build_context_packet(
    db,
    *,
    task: str = "",
    files: Optional[List[str]] = None,
    scope: Optional[Dict[str, Any]] = None,
    mode: str = "before_edit",
    budget_tokens: int = _DEFAULT_PACKET_BUDGET,
    max_sensitivity: str = "restricted",
    include_receipt: bool = True,
) -> Dict[str, Any]:
    """Build a task-aware context packet from retrieval seeds plus graph chains."""
    files = files or []
    scope_flat = _flatten_scope(scope)
    mode = mode if mode in _PACKET_MODE_TO_SURFACING else "before_edit"
    budget_tokens = _coerce_packet_budget(budget_tokens)
    max_sensitivity = max_sensitivity if max_sensitivity in _SENSITIVITY_ORDER else "restricted"

    query_text = _packet_query_text(task, files)
    if not query_text:
        query_text = "recent decisions lessons constraints"

    try:
        from omega.sqlite_store import SurfacingContext
        surfacing_context = getattr(SurfacingContext, _PACKET_MODE_TO_SURFACING[mode])
    except Exception:
        surfacing_context = None

    query_kwargs: Dict[str, Any] = {
        "limit": 8,
        "session_id": None,
        "project_path": scope_flat.get("project") or "",
        "scope": "project",
        "context_file": files[0] if files else "",
        "context_tags": None,
        "entity_id": scope_flat.get("entity_id"),
        "surfacing_context": surfacing_context,
        "max_sensitivity": max_sensitivity,
    }
    try:
        query_params = inspect.signature(db.query).parameters
    except Exception:
        query_params = {}
    if query_params and "max_sensitivity" not in query_params:
        query_kwargs.pop("max_sensitivity", None)
    seeds = db.query(query_text, **query_kwargs)
    if not seeds:
        seeds = _query_packet_scope_fallback(
            db,
            scope=scope_flat,
            max_sensitivity=max_sensitivity,
        )

    candidates: Dict[str, Dict[str, Any]] = {}
    chains: List[Dict[str, Any]] = []
    warnings: List[Dict[str, Any]] = []

    for seed in seeds[:5]:
        seed_id = getattr(seed, "id", None)
        if seed_id is None and isinstance(seed, dict):
            seed_id = seed.get("id") or seed.get("node_id")
        seed_item = _memory_to_packet_item(db, seed, source="seed", base_score=getattr(seed, "relevance", 1.0))
        if seed_item:
            candidates[seed_item["id"]] = seed_item
        if not seed_id:
            continue
        chain_nodes: List[Dict[str, Any]] = []
        try:
            related = db.get_related_chain(
                seed_id,
                max_hops=2,
                min_weight=0.25,
                exclude_ids=set(candidates.keys()),
                _include_results=True,
            )
        except Exception as exc:
            logger.debug("context_packet: graph expansion failed for %s: %s", getattr(seed, "id", ""), exc)
            related = []
        related = sorted(related, key=_packet_chain_entry_rank, reverse=True)
        for entry in related[:8]:
            result = entry.get("_result") or entry
            item = _memory_to_packet_item(
                db,
                result,
                source="chain",
                seed_id=seed_id,
                edge_type=entry.get("edge_type"),
                edge_metadata=entry.get("edge_metadata"),
                edge_weight=float(entry.get("weight") or 0.0),
                hop=int(entry.get("hop") or 1),
                base_score=getattr(seed, "relevance", 1.0),
            )
            if not item:
                continue
            if item["id"] not in candidates or item["score"] > candidates[item["id"]]["score"]:
                candidates[item["id"]] = item
            chain_nodes.append({
                "id": item["id"],
                "edge_type": item.get("edge_type"),
                "edge_source": item.get("edge_source"),
                "edge_weight": item.get("edge_weight"),
                "hop": item.get("hop"),
            })
        if chain_nodes:
            chains.append({"seed_id": seed_id, "nodes": chain_nodes[:5]})

    for seed_id in list(candidates.keys())[:5]:
        try:
            related = db.get_related_chain(
                seed_id,
                max_hops=1,
                min_weight=0.25,
                exclude_ids=set(candidates.keys()),
                _include_results=True,
            )
        except Exception:
            related = []
        chain_nodes = []
        related = sorted(related, key=_packet_chain_entry_rank, reverse=True)
        for entry in related[:4]:
            item = _memory_to_packet_item(
                db,
                entry.get("_result") or entry,
                source="chain",
                seed_id=seed_id,
                edge_type=entry.get("edge_type"),
                edge_metadata=entry.get("edge_metadata"),
                edge_weight=float(entry.get("weight") or 0.0),
                hop=int(entry.get("hop") or 1),
                base_score=candidates[seed_id].get("relevance", 1.0),
            )
            if not item:
                continue
            if item["id"] not in candidates or item["score"] > candidates[item["id"]]["score"]:
                candidates[item["id"]] = item
            chain_nodes.append({
                "id": item["id"],
                "edge_type": item.get("edge_type"),
                "edge_source": item.get("edge_source"),
                "edge_weight": item.get("edge_weight"),
                "hop": item.get("hop"),
            })
        if chain_nodes:
            chains.append({"seed_id": seed_id, "nodes": chain_nodes[:4]})

    if not any(
        _admit_packet_item(item, scope=scope_flat, max_sensitivity=max_sensitivity)[0]
        for item in candidates.values()
    ):
        for fallback in _query_packet_scope_fallback(
            db,
            scope=scope_flat,
            max_sensitivity=max_sensitivity,
        ):
            item = _memory_to_packet_item(
                db,
                fallback,
                source="scope_fallback",
                base_score=float(fallback.get("relevance") or 0.35),
            )
            if item and item["id"] not in candidates:
                candidates[item["id"]] = item

    _apply_contradiction_warnings(db, candidates)

    admitted: List[Dict[str, Any]] = []
    candidate_receipts: List[Dict[str, Any]] = []
    ranked_candidates = sorted(candidates.values(), key=lambda x: x["score"], reverse=True)
    for rank, item in enumerate(ranked_candidates, start=1):
        ok, reason = _admit_packet_item(item, scope=scope_flat, max_sensitivity=max_sensitivity)
        candidate_receipts.append(_packet_candidate_receipt(
            item,
            rank=rank,
            admitted=ok,
            reason=reason,
        ))
        if ok:
            admitted.append(item)
        elif reason in {"stale", "contradicts", "flagged_for_review"}:
            warnings.append({"reason": reason, "item": item})

    title = os.path.basename(files[0]) if files else (task[:48] or "current task")
    packet_markdown = _render_context_packet(
        title=title,
        admitted=admitted,
        warnings=warnings,
        budget_tokens=budget_tokens,
        include_receipt=include_receipt,
    )

    rendered_id_prefixes = _rendered_packet_memory_ids(packet_markdown)
    used_ids: List[str] = []
    for prefix in rendered_id_prefixes:
        matched = next(
            (
                item["id"]
                for item in admitted
                if item["id"].startswith(prefix) and item["id"] not in used_ids
            ),
            None,
        )
        if matched:
            used_ids.append(matched)
    render_rank = {node_id: idx for idx, node_id in enumerate(used_ids, start=1)}
    for receipt in candidate_receipts:
        if receipt["id"] in render_rank:
            receipt["rendered"] = True
            receipt["render_rank"] = render_rank[receipt["id"]]
    metrics = {
        "mode": mode,
        "seed_count": len(seeds),
        "candidate_count": len(candidates),
        "chain_count": len(chains),
        "memories_used": len(used_ids),
        "warnings_count": len(warnings),
        "rendered_warnings_count": _rendered_packet_warning_count(packet_markdown),
        "estimated_tokens": _estimate_tokens(packet_markdown),
        "estimated_tokens_saved": max(
            sum(_estimate_tokens(item["content"]) for item in admitted if item["id"] in used_ids)
            - _estimate_tokens(packet_markdown),
            0,
        ),
        "budget_tokens": budget_tokens,
    }
    try:
        recorder = getattr(db, "record_memory_accesses", None)
        if recorder is not None:
            recorder(used_ids)
    except Exception as exc:
        # Access accounting is best effort. A valid context packet must never
        # become an error because its audit write lost a lock race.
        logger.warning("context_packet: access accounting failed: %s", exc)
    return {
        "packet_markdown": packet_markdown,
        "chains": chains,
        "candidate_ids": list(candidates.keys()),
        "candidate_receipts": candidate_receipts,
        "memories_used": used_ids,
        "warnings": warnings,
        "estimated_tokens": metrics["estimated_tokens"],
        "estimated_tokens_saved": metrics["estimated_tokens_saved"],
        "metrics": metrics,
        "scope": {k: v for k, v in scope_flat.items() if v is not None},
        "mode": mode,
        "budget_tokens": budget_tokens,
    }


# ---------------------------------------------------------------------------
# Handler
# ---------------------------------------------------------------------------
async def handle_context_assemble(arguments: dict) -> dict:
    scope = _flatten_scope(arguments.get("scope"))
    since_iso = _parse_since(arguments.get("since"))

    section_filter = arguments.get("sections")
    if section_filter is not None and not isinstance(section_filter, list):
        return mcp_error("sections must be an array of section ids")

    requested = set(section_filter) if section_filter else {spec["id"] for spec in _SECTION_SPECS}
    specs = [spec for spec in _SECTION_SPECS if spec["id"] in requested]

    try:
        from omega.bridge._core import _get_store
        db = _get_store()
    except Exception as exc:
        logger.error("context_assemble: store unavailable: %s", exc, exc_info=True)
        return mcp_error(f"context_assemble failed: {exc}")

    sections: List[Dict[str, Any]] = []
    for spec in specs:
        try:
            sections.append(_build_section(db, spec, scope, since_iso))
        except Exception as exc:
            logger.debug("context_assemble: section %s failed: %s", spec["id"], exc)
            sections.append({
                "id": spec["id"],
                "priority": spec["priority"],
                "content": "",
                "tokens": 0,
                "count": 0,
                "items": [],
                "error": str(exc)[:200],
            })

    sections.sort(key=lambda s: s["priority"])
    composed = _compose_markdown(sections)

    return mcp_response({
        "sections": sections,
        "composed_markdown": composed,
        "scope": {k: v for k, v in scope.items() if v is not None},
        "since": since_iso,
    })


async def handle_context_packet(arguments: dict) -> dict:
    try:
        files = _coerce_packet_files(arguments.get("files"))
    except ValueError as exc:
        return mcp_error(str(exc))

    scope = arguments.get("scope")
    if scope is not None and not isinstance(scope, dict):
        return mcp_error("scope must be an object")

    try:
        from omega.bridge._core import _get_store
        db = _get_store()
    except Exception as exc:
        logger.error("context_packet: store unavailable: %s", exc, exc_info=True)
        return mcp_error(f"context_packet failed: {exc}")

    try:
        payload = build_context_packet(
            db,
            task=(arguments.get("task") or "").strip(),
            files=files,
            scope=scope,
            mode=arguments.get("mode") or "before_edit",
            budget_tokens=_coerce_packet_budget(arguments.get("budget_tokens")),
            max_sensitivity=arguments.get("max_sensitivity") or "restricted",
            include_receipt=arguments.get("include_receipt", True) is not False,
        )
    except Exception as exc:
        logger.error("context_packet failed: %s", exc, exc_info=True)
        return mcp_error(f"context_packet failed: {exc}")

    try:
        from omega.telemetry import track_context_packet
        track_context_packet(payload.get("metrics"), surface="mcp")
    except Exception:
        pass

    return mcp_response(payload)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
CONTEXT_HANDLERS = {
    "context_assemble": handle_context_assemble,
    "context_packet": handle_context_packet,
}
