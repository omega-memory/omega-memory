# Python API

`omega.bridge` public API -- 36 functions.

This is the main programmatic interface used internally by the MCP handlers. All functions are importable from `omega.bridge`.

```python
from omega.bridge import store, query, remember, auto_capture
```

---

## Core

| Function | Signature | Description |
|----------|-----------|-------------|
| `store` | `store(content, event_type="memory", metadata=None, session_id=None, entity_id=None) -> str` | Store a memory with metadata. Wraps `auto_capture` with a default event type. |
| `remember` | `remember(text, session_id=None) -> str` | Store a permanent memory as `user_preference` type. |
| `auto_capture` | `auto_capture(content, event_type, metadata=None, session_id=None, project=None, ttl_override=None, entity_id=None) -> str` | Primary ingestion function. Handles dedup (SHA256 + Jaccard), evolution (appends to similar memories), auto-tagging, auto-relating, and blocklist filtering. |
| `delete_memory` | `delete_memory(memory_id) -> dict` | Delete a memory by its node ID. Returns `{"success": bool}`. |
| `edit_memory` | `edit_memory(memory_id, new_content) -> dict` | Update a memory's content and regenerate its embedding. |

---

## Query

| Function | Signature | Description |
|----------|-----------|-------------|
| `query` | `query(text, limit=10, event_type=None, filter_tags=None, temporal_range=None, context_file=None, context_tags=None, entity_id=None, project=None, session_id=None, max_chars=200) -> str` | Semantic search with blended ranking (70% vector, 30% FTS5) and contextual re-ranking. Returns markdown-formatted results. |
| `query_structured` | `query_structured(text, limit=10, event_type=None, filter_tags=None, ...) -> list[dict]` | Same search pipeline as `query` but returns structured dicts instead of markdown. |
| `phrase_search` | `phrase_search(phrase, limit=10, event_type=None, project=None, case_sensitive=False, max_chars=200) -> str` | Exact substring match via FTS5 full-text search. `max_chars=0` renders each memory in full. |
| `find_similar_memories` | `find_similar_memories(memory_id, limit=5) -> str` | Find memories similar to a given memory by embedding distance. |

---

## Session

| Function | Signature | Description |
|----------|-----------|-------------|
| `welcome` | `welcome(session_id=None, project=None) -> dict` | Session briefing: recent memories, user profile, project context. |
| `get_session_context` | `get_session_context(session_id, project=None) -> dict` | Retrieve session context for handoff or resume. |
| `clear_session` | `clear_session(session_id) -> dict` | Delete all memories associated with a session. |
| `batch_store` | `batch_store(items) -> dict` | Store multiple memories in a single call. Each item is a dict with `content` and optional `event_type`, `metadata`, etc. |

---

## Health and Stats

| Function | Signature | Description |
|----------|-----------|-------------|
| `check_health` | `check_health(warn_mb=350, critical_mb=500, max_nodes=10000) -> dict` | System health check: memory usage, node counts, cache stats, warnings. |
| `status` | `status() -> dict` | Memory count, database size, model status, edge count. |
| `get_dedup_stats` | `get_dedup_stats() -> dict` | Deduplication statistics (duplicates found, evolved, blocked). |
| `type_stats` | `type_stats() -> dict` | Memory counts grouped by event type. |
| `session_stats` | `session_stats() -> dict` | Memory counts grouped by session (top 20). |
| `get_activity_summary` | `get_activity_summary(days=7) -> dict` | Recent session activity overview. |

---

## Profile

| Function | Signature | Description |
|----------|-----------|-------------|
| `get_profile` | `get_profile() -> dict` | User profile built from memory patterns. |
| `save_profile` | `save_profile(profile) -> bool` | Save or update user profile fields. |
| `extract_preferences` | `extract_preferences(text) -> dict` | Extract preference signals from text content. |
| `list_preferences` | `list_preferences() -> list[dict]` | List all stored user preferences. |

---

## Lessons

| Function | Signature | Description |
|----------|-----------|-------------|
| `get_cross_session_lessons` | `get_cross_session_lessons(task=None, limit=5, project_path=None, exclude_project=None, exclude_session=None) -> list` | Lessons across sessions, ranked by verification count and access frequency. |
| `get_cross_project_lessons` | `get_cross_project_lessons(task=None, limit=5, exclude_project=None, exclude_session=None) -> list` | Lessons across all projects. |

---

## Maintenance

| Function | Signature | Description |
|----------|-----------|-------------|
| `consolidate` | `consolidate(prune_days=30, max_summaries=50) -> str` | Prune stale low-value memories, cap session summaries, clean orphaned edges. Auto-backs up before running. |
| `compact` | `compact(event_type="lesson_learned", threshold=0.6, min_cluster=3, dry_run=False) -> str` | Cluster similar memories (Jaccard similarity) and create summary nodes. Originals marked as superseded. |
| `deduplicate` | `deduplicate() -> dict` | Remove exact content duplicates (SHA256 hash match). |
| `timeline` | `timeline(days=7, limit_per_day=10) -> str` | Memory timeline grouped by day. |
| `traverse` | `traverse(memory_id, max_hops=2, min_weight=0.0) -> str` | BFS traversal of the memory relationship graph (max 5 hops). |

---

## Export and Import

| Function | Signature | Description |
|----------|-----------|-------------|
| `export_memories` | `export_memories(filepath) -> str` | Export all memories to a JSON file. |
| `import_memories` | `import_memories(filepath, clear_existing=True) -> str` | Import memories from a JSON file. Optionally clears existing data first. |
| `reingest` | `reingest() -> dict` | Reload entries from legacy store.jsonl into the graph system. |

---

## Constraints

| Function | Signature | Description |
|----------|-----------|-------------|
| `check_constraints` | `check_constraints(file_path, project=None) -> list[dict]` | Check a file against stored project constraints. |
| `list_constraints` | `list_constraints(project=None) -> dict` | List all constraints for a project. |
| `save_constraints` | `save_constraints(constraints, project=None) -> dict` | Save or update project constraints. |

---

## Feedback

| Function | Signature | Description |
|----------|-----------|-------------|
| `record_feedback` | `record_feedback(memory_id, rating, reason=None) -> dict` | Rate a memory as helpful, unhelpful, or outdated. Affects future search ranking via feedback dampening. |

---

## Testing

| Function | Signature | Description |
|----------|-----------|-------------|
| `reset_memory` | `reset_memory() -> None` | Reset the memory store. For testing only -- destroys all data. |

---

## Notes

- All functions are importable from `omega.bridge`.
- The MCP handlers in `server/handlers.py` call these functions directly.
- `auto_capture` is the primary ingestion path -- `store` and `remember` are convenience wrappers around it.
- Functions returning `str` typically return markdown-formatted output suitable for display in agent conversations.
- Functions returning `dict` or `list` return structured data for programmatic use.
