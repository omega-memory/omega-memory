# MCP Tools Reference

All tools available through the OMEGA MCP server.

---

## Memory (24 tools)

| Tool | Description | Key Parameters |
|------|-------------|----------------|
| `omega_remember` | Store a permanent memory from user instruction | `text` |
| `omega_store` | Store typed memory with metadata | `content`, `event_type` (decision / lesson_learned / error_pattern / task_completion / session_summary / user_preference / checkpoint), `priority` (1-5), `session_id`, `entity_id` |
| `omega_query` | Semantic search with filters and re-ranking | `query`, `limit`, `max_chars` (content per result, default 200; 0 = full), `event_type`, `filter_tags`, `temporal_range`, `context_file`, `context_tags`, `entity_id`, `project`, `session_id` (who is asking), `scope` (`session` = only memories stored by `session_id`; default every session) |
| `omega_phrase_search` | Exact substring match via FTS5 | `phrase`, `limit`, `event_type`, `project`, `case_sensitive` |
| `omega_welcome` | Session briefing with recent memories and profile | `session_id`, `project` |
| `omega_profile` | Show user profile built from memory patterns | (none) |
| `omega_save_profile` | Save or update user profile fields | `profile` (object) |
| `omega_list_preferences` | List all stored user preferences | (none) |
| `omega_delete_memory` | Delete a specific memory by ID | `memory_id` |
| `omega_edit_memory` | Edit memory content | `memory_id`, `new_content` |
| `omega_lessons` | Cross-session lessons ranked by access count | `task`, `project_path`, `cross_project`, `exclude_project`, `exclude_session`, `limit` |
| `omega_feedback` | Rate a memory (helpful / unhelpful / outdated) | `memory_id`, `rating`, `reason` |
| `omega_clear_session` | Clear all memories for a session | `session_id` |
| `omega_similar` | Find memories similar to a given memory | `memory_id`, `limit` |
| `omega_timeline` | Memory timeline grouped by day | `days`, `limit_per_day` |
| `omega_traverse` | Walk the memory relationship graph | `memory_id`, `max_hops` (1-5), `min_weight` |
| `omega_consolidate` | Prune stale memories, cap session summaries | `prune_days`, `max_summaries` |
| `omega_compact` | Cluster and summarize related memories | `event_type`, `similarity_threshold`, `min_cluster_size`, `dry_run` |
| `omega_health` | Detailed system health check | `warn_mb`, `critical_mb`, `max_nodes` |
| `omega_backup` | Export or import memories for backup/restore | `filepath`, `mode` (export / import), `clear_existing` |
| `omega_type_stats` | Memory counts grouped by event type | (none) |
| `omega_session_stats` | Memory counts grouped by session (top 20) | (none) |
| `omega_checkpoint` | Save task state for cross-session continuity | `task_title` (required), `progress` (required), `plan`, `files_touched`, `decisions`, `key_context`, `next_steps`, `project`, `session_id` |
| `omega_resume_task` | Resume a previously checkpointed task | `task_title`, `project`, `limit`, `verbosity` (full / summary / minimal) |

---

Coordination, router, entity, knowledge, profile and oracle tools are available in [OMEGA Pro](https://omegamax.co/pro).

## Cross-Model Consultation (2 tools)

Consult a different LLM for a second opinion. Provider-aware: Claude agents get `omega_consult_gpt`, non-Anthropic agents get `omega_consult_claude`.

| Tool | Description | Key Parameters |
|------|-------------|----------------|
| `omega_consult_gpt` | Consult GPT for a second opinion (for Claude-based agents) | `prompt`, `context`, `system`, `temperature` (0.0-2.0), `max_tokens` (max: 16384) |
| `omega_consult_claude` | Consult Claude for a second opinion (for non-Anthropic agents) | `prompt`, `context`, `system`, `temperature` (0.0-2.0), `max_tokens` (max: 16384) |
