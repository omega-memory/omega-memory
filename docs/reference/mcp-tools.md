# MCP Tools Reference

All tools available through the OMEGA MCP server.

---

## Memory (24 tools)

| Tool | Description | Key Parameters |
|------|-------------|----------------|
| `omega_remember` | Store a permanent memory from user instruction | `text` |
| `omega_store` | Store typed memory with metadata | `content`, `event_type` (decision / lesson_learned / error_pattern / task_completion / session_summary / user_preference / checkpoint), `priority` (1-5), `session_id`, `entity_id` |
| `omega_query` | Semantic search with filters and re-ranking | `query`, `limit`, `max_chars` (content per result, default 200; 0 = full), `event_type`, `filter_tags`, `temporal_range`, `context_file`, `context_tags`, `entity_id`, `project`, `session_id` |
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

## Coordination (28 tools)

### Sessions

| Tool | Description | Key Parameters |
|------|-------------|----------------|
| `omega_session_register` | Register an agent session for multi-agent coordination | `session_id` (required), `project`, `task`, `capabilities` |
| `omega_session_heartbeat` | Update heartbeat to signal the agent is active | `session_id` |
| `omega_session_deregister` | End session, release all file and branch claims | `session_id` |
| `omega_sessions_list` | List all active agent sessions (auto-cleans stale) | (none) |
| `omega_session_snapshot` | Snapshot session state before risky operations | `session_id`, `reason` |
| `omega_session_recover` | Recover context from a crashed predecessor session | `project` |

### Files and Branches

| Tool | Description | Key Parameters |
|------|-------------|----------------|
| `omega_file_claim` | Claim exclusive file access | `session_id`, `file_path`, `task`, `force` |
| `omega_file_release` | Release a file claim | `session_id`, `file_path` |
| `omega_file_check` | Check who owns a file | `file_path` |
| `omega_branch_claim` | Claim exclusive branch access (protected branches blocked) | `session_id`, `project`, `branch`, `task` |
| `omega_branch_release` | Release a branch claim | `session_id`, `project`, `branch` |
| `omega_branch_check` | Check who owns a branch | `project`, `branch` |

### Intents

| Tool | Description | Key Parameters |
|------|-------------|----------------|
| `omega_intent_announce` | Broadcast planned work so peers can check for overlaps | `session_id`, `description`, `target_files`, `target_branch`, `intent_type`, `ttl_minutes` |
| `omega_intent_check` | Check if planned files or branch overlap with peer intents | `session_id` |

### Tasks

| Tool | Description | Key Parameters |
|------|-------------|----------------|
| `omega_task_create` | Create a coordination task | `session_id`, `title` (required), `description`, `priority`, `project`, `depends_on` |
| `omega_task_claim` | Claim a pending task to work on | `task_id`, `session_id` |
| `omega_task_complete` | Mark a task as completed | `task_id`, `session_id`, `result` |
| `omega_task_fail` | Mark a task as failed | `task_id`, `session_id`, `reason` |
| `omega_task_cancel` | Cancel a task | `task_id`, `session_id` |
| `omega_task_progress` | Update progress percentage (0-100) | `task_id`, `session_id`, `progress`, `status_note` |
| `omega_tasks_list` | List tasks with optional filters | `project`, `status` (pending / in_progress / completed / failed / canceled) |
| `omega_task_deps` | View dependency graph for a task | `task_id` |

### Messaging

| Tool | Description | Key Parameters |
|------|-------------|----------------|
| `omega_send_message` | Send a message to a specific agent or broadcast to project | `session_id`, `subject`, `body`, `to_session`, `msg_type` (request / inform / acknowledge / reject / complete), `ref_task_id`, `ttl_minutes` |
| `omega_inbox` | Check inbox for messages from other agents | `session_id`, `unread_only`, `msg_type`, `limit` |
| `omega_find_agents` | Find active sessions with a matching capability | `capability`, `project` |

### Dashboard

| Tool | Description | Key Parameters |
|------|-------------|----------------|
| `omega_coord_status` | Full coordination dashboard: sessions, claims, intents, conflicts | (none) |
| `omega_audit` | Query the coordination audit log | `session_id`, `tool_name`, `limit` |
| `omega_git_events` | Recent git events tracked by coordination | `project`, `event_type`, `limit` |

---

## Router (10 tools)

Optional module. Install with `pip install omega-memory[router]`.

| Tool | Description | Key Parameters |
|------|-------------|----------------|
| `omega_route_prompt` | Route a prompt to the optimal LLM based on intent and priority | `prompt`, `priority` (cost / speed / quality / balanced), `force_intent`, `estimated_tokens`, `session_id` |
| `omega_classify_intent` | Classify a prompt's intent without routing | `prompt`, `detailed` |
| `omega_router_status` | Show provider availability, routing stats, priority mode | (none) |
| `omega_set_priority_mode` | Set the routing priority mode | `mode` (cost / speed / quality / balanced) |
| `omega_get_model_config` | View routing configuration for an intent or all intents | `intent` (coding / creative / logic / exploration / simple_edit) |
| `omega_switch_model` | Switch to a different LLM with OMEGA memory preservation | `session_id`, `target_provider` (anthropic / openai / google / groq / xai), `target_model`, `retrieve_context` |
| `omega_get_current_model` | Get the current model for a session | `session_id` |
| `omega_router_context` | Get session routing context (model, provider, tokens, depth) | `session_id` |
| `omega_warm_router` | Pre-load the intent classifier to reduce first-route latency | (none) |
| `omega_router_benchmark` | Run a quick accuracy test with 6 sample prompts | (none) |

---

## Entity (8 tools)

Optional module. Install with `pip install omega-memory[entity]`.

| Tool | Description | Key Parameters |
|------|-------------|----------------|
| `omega_entity_create` | Create a corporate entity in the registry | `entity_id` (slug), `name`, `entity_type` (company / llc / s_corp / c_corp / foundation / startup / trust / partnership / sole_proprietorship / nonprofit / other), `jurisdiction`, `metadata` |
| `omega_entity_get` | Get detailed information about an entity | `entity_id` |
| `omega_entity_list` | List all registered entities | `entity_type`, `status` (active / acquired / dissolved / dormant / pending) |
| `omega_entity_update` | Update an entity's fields | `entity_id`, `name`, `status`, `jurisdiction`, `metadata` |
| `omega_entity_delete` | Soft-delete an entity (sets status to dissolved) | `entity_id` |
| `omega_entity_add_relationship` | Add a directed relationship between entities | `source_entity_id`, `target_entity_id`, `relationship_type` (parent_of / subsidiary_of / owned_by / acquired_by / partner_of / investor_in / operated_by), `metadata` |
| `omega_entity_relationships` | Query all relationships for an entity | `entity_id`, `direction` (outgoing / incoming), `relationship_type` |
| `omega_entity_tree` | Recursive hierarchy view of an entity and its children | `entity_id` |

Entity-scoped data: memories, profiles, and documents all accept an `entity_id` parameter to scope data to a specific entity.

---

## Knowledge (6 tools)

Optional module. Install with `pip install omega-memory[knowledge-pdf]` (Docling) or `omega-memory[knowledge-pdf-lite]` (pdfplumber only).

| Tool | Description | Key Parameters |
|------|-------------|----------------|
| `omega_ingest_document` | Ingest a document (PDF, webpage, markdown, text) into the knowledge base | `path_or_url`, `title`, `source_type` (pdf / webpage / markdown / text), `entity_id` |
| `omega_search_documents` | Search across ingested documents via vector similarity | `query`, `limit`, `source_type`, `entity_id` |
| `omega_list_documents` | List all documents in the knowledge base with chunk counts | (none) |
| `omega_remove_document` | Remove a document and all its chunks from the knowledge base | `source_path` |
| `omega_scan_documents` | Scan a directory for new or changed files and auto-ingest | `directory` (default: ~/.omega/documents/) |
| `omega_sync_kb` | Sync pending files from cloud queue (Supabase) into local knowledge base | `batch_size` (default: 10, max: 50) |

---

## Secure Profile (4 tools)

Optional module. Install with `pip install omega-memory[encrypt]`.

| Tool | Description | Key Parameters |
|------|-------------|----------------|
| `omega_profile_set` | Store an AES-256 encrypted personal profile field | `category` (identity / medical / financial / personal / professional / contacts / legal), `field_name`, `value`, `entity_id`, `metadata` |
| `omega_profile_get` | Decrypt and retrieve profile fields | `category`, `field_name`, `entity_id` |
| `omega_profile_search` | Search profile metadata and field names (not encrypted values) | `query`, `entity_id` |
| `omega_profile_list` | List all stored categories with field counts (no decryption) | `entity_id` |

---

## Oracle (4 tools)

Prediction intelligence with calibration tracking.

| Tool | Description | Key Parameters |
|------|-------------|----------------|
| `omega_oracle_record` | Record a prediction, wallet score, regime change, or signal snapshot | `record_type` (prediction / wallet_score / regime_change / signal_snapshot), `content`, `data` (structured metadata), `market_type` |
| `omega_oracle_resolve` | Mark a prediction as resolved with outcome | `market_id`, `outcome` (yes / no), `resolution_price` |
| `omega_oracle_analyze` | Compute analytical views over prediction history | `view` (calibration / signals / wallets / bias / playbook / briefing), `market_type`, `regime`, `days`, `limit` |
| `omega_oracle_status` | Dashboard: prediction count, Brier score, active regime, coverage | (none) |

---

## Cross-Model Consultation (2 tools)

Consult a different LLM for a second opinion. Provider-aware: Claude agents get `omega_consult_gpt`, non-Anthropic agents get `omega_consult_claude`.

| Tool | Description | Key Parameters |
|------|-------------|----------------|
| `omega_consult_gpt` | Consult GPT for a second opinion (for Claude-based agents) | `prompt`, `context`, `system`, `temperature` (0.0-2.0), `max_tokens` (max: 16384) |
| `omega_consult_claude` | Consult Claude for a second opinion (for non-Anthropic agents) | `prompt`, `context`, `system`, `temperature` (0.0-2.0), `max_tokens` (max: 16384) |
