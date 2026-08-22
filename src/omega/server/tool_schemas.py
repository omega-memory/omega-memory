"""OMEGA MCP Tool Schemas -- 16 tools for memory management.

Consolidated into 15 action-discriminated composites plus context_packet.
All original capabilities preserved; low-frequency operations grouped by intent.
omega_briefing and omega_habits remain as backward-compat aliases in handlers.
omega_lessons removed — cross-session lessons auto-surface via hooks on file edits.

Condensed Mode (CodeMode-inspired):
  When OMEGA_CONDENSED=1, only 5 tools are exposed: 3 standalone essentials
  (omega_welcome, omega_protocol, omega_store) + 2 meta-tools (omega_tools,
  omega_call). All other tools are accessible via omega_call(tool=..., args=...).
  This reduces schema token overhead by ~80%.
"""

TOOL_SCHEMAS = [
    {
        "name": "omega_store",
        "description": "Store a memory with optional type and metadata. Use when the user says 'remember this' or for programmatic capture (decisions, lessons, errors). Defaults to type 'memory' if event_type is omitted.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "content": {"type": "string", "description": "Memory content (also accepts 'text' as alias)"},
                "text": {"type": "string", "description": "Alias for content"},
                "event_type": {
                    "type": "string",
                    "description": "Type: memory (default), session_summary, task_completion, error_pattern, lesson_learned, decision, user_preference, constraint, advisor_insight",
                },
                "metadata": {"type": "object", "description": "Additional metadata"},
                "session_id": {"type": "string"},
                "project": {"type": "string"},
                "priority": {
                    "type": "integer",
                    "description": "Memory priority 1-5 (5=highest). Auto-set from event type if omitted.",
                    "minimum": 1,
                    "maximum": 5,
                },
                "entity_id": {
                    "type": "string",
                    "description": "Scope this memory to an entity (e.g., 'acme'). Omit for unscoped.",
                },
                "agent_type": {
                    "type": "string",
                    "description": "Agent type for sub-agent memory scoping (e.g., 'code-reviewer', 'test-runner').",
                },
                "derived_from": {
                    "type": "string",
                    "description": "Node ID of the parent memory this was derived from. Creates a 'derived_from' edge for lineage tracking.",
                },
                "source_uri": {
                    "type": "string",
                    "description": "External source reference (e.g., Slack URL, Google Doc ID, git commit SHA, X post URL). Enables provenance tracking.",
                },
                "status": {
                    "type": "string",
                    "enum": ["active", "superseded", "speculative", "archived"],
                    "description": "Memory lifecycle status. Default 'active'. Use 'speculative' for unverified claims, 'archived' for intentionally preserved but inactive.",
                },
                "sensitivity": {
                    "type": "string",
                    "enum": ["public", "internal", "confidential", "restricted"],
                },
                "project_id": {
                    "type": "string",
                    "description": "Scope this memory to an entity project.",
                },
                "items": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "content": {"type": "string", "description": "Memory content"},
                            "event_type": {"type": "string", "description": "Per-item memory type"},
                            "metadata": {"type": "object", "description": "Per-item additional metadata"},
                            "session_id": {"type": "string"},
                            "project": {"type": "string"},
                            "project_id": {"type": "string"},
                            "priority": {"type": "integer", "minimum": 1, "maximum": 5},
                            "entity_id": {"type": "string"},
                            "agent_type": {"type": "string"},
                            "derived_from": {"type": "string"},
                            "source_uri": {"type": "string"},
                            "status": {
                                "type": "string",
                                "enum": ["active", "superseded", "speculative", "archived"],
                            },
                            "sensitivity": {
                                "type": "string",
                                "enum": ["public", "internal", "confidential", "restricted"],
                            },
                        },
                        "required": ["content"],
                    },
                    "description": "Batch mode. Request-level scope and provenance fields are defaults for items that omit them; explicit per-item values win.",
                },
            },
            "anyOf": [
                {"required": ["content"]},
                {"required": ["text"]},
                {"required": ["items"]},
            ],
        },
    },
    {
        "name": "omega_query",
        "description": "Search memories. Modes: 'semantic' (default) for meaning-based search, 'phrase' for exact substring match, 'timeline' for recent memories grouped by day, 'browse' for listing by type/session/recent.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query (or exact phrase when mode='phrase'). Not required for mode='timeline' or mode='browse'."},
                "mode": {
                    "type": "string",
                    "enum": ["semantic", "phrase", "timeline", "browse", "trace", "unified"],
                    "description": "Search mode: 'semantic' (default), 'phrase' for exact match, 'timeline' for recent memories by day, 'browse' for listing, 'trace' for session tool call timeline, 'unified' for cross-searching memories + knowledge documents",
                },
                "limit": {"type": "integer", "default": 10},
                "event_type": {"type": "string", "description": "Filter by event type (also used as type filter in semantic mode for scoped search)"},
                "project": {"type": "string"},
                "session_id": {"type": "string"},
                "context_file": {"type": "string", "description": "Current file being edited (boosts results)"},
                "context_tags": {"type": "array", "items": {"type": "string"}, "description": "Context tags for boosting"},
                "filter_tags": {"type": "array", "items": {"type": "string"}, "description": "Hard filter: ALL tags must match (AND logic)"},
                "temporal_range": {"type": "array", "items": {"type": "string"}, "minItems": 2, "maxItems": 2, "description": "[start_iso, end_iso] date range filter"},
                "entity_id": {"type": "string", "description": "Filter to entity. Omit for all."},
                "agent_type": {"type": "string", "description": "Filter to agent type. Omit for all."},
                "case_sensitive": {"type": "boolean", "description": "Case-sensitive (only for mode='phrase', default false)", "default": False},
                "days": {"type": "integer", "description": "Days to look back (only for mode='timeline', default 7)", "default": 7},
                "limit_per_day": {"type": "integer", "description": "Max per day (only for mode='timeline', default 10)", "default": 10},
                "browse_by": {
                    "type": "string",
                    "enum": ["type", "session", "recent"],
                    "description": "Browse dimension (only for mode='browse'): 'type' lists by event_type, 'session' lists by session_id, 'recent' lists most recent memories",
                },
                "context": {
                    "type": "string",
                    "enum": ["general", "error_debug", "file_edit", "planning", "review"],
                    "description": "Retrieval context for tuned scoring. 'error_debug' boosts error patterns, 'planning' boosts decisions, 'review' boosts lessons, 'file_edit' boosts file-related memories.",
                },
                "perspective": {
                    "type": "string",
                    "enum": ["implementation", "critique", "verification"],
                    "description": "Behavioral diversity lens. Biases retrieval toward different memory types: 'implementation' boosts errors/lessons/code, 'critique' boosts constraints/preferences/contradictions, 'verification' boosts decisions/benchmarks/evaluations. Auto-set from session role in multi-agent mode.",
                },
                "strength_min": {
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 1.0,
                    "description": "Minimum strength score (0.0-1.0). Filters out weak/decayed memories.",
                },
                "memory_type": {
                    "type": "string",
                    "enum": ["episodic", "semantic", "procedural"],
                    "description": "Filter by memory type: 'episodic' (session events), 'semantic' (facts/decisions), 'procedural' (lessons/rules).",
                },
                "include_contradicted": {
                    "type": "boolean",
                    "description": "If true, return only memories that have been contradicted by newer memories. Useful for data quality auditing.",
                },
                "valid_at": {
                    "type": "string",
                    "description": "ISO datetime. Return only memories that were valid at this point in time. Enables temporal queries like 'what did we know before session X?'",
                },
                "status": {
                    "type": "string",
                    "enum": ["active", "superseded", "speculative", "archived"],
                    "description": "Filter by memory lifecycle status. Default: returns all statuses. Use 'active' to exclude superseded/archived.",
                },
            },
        },
    },
    {
        "name": "omega_welcome",
        "description": "Session startup briefing. Call at the beginning of every session to load recent context, active reminders, and user profile. Returns what the agent needs to continue where the last session left off.",
        "inputSchema": {"type": "object", "properties": {"session_id": {"type": "string"}, "project": {"type": "string"}}},
    },
    {
        "name": "omega_protocol",
        "description": "Retrieve your operating rules and behavioral guidelines for this session. Returns context-sensitive instructions covering memory usage, coordination, reminders, and workflow. In multi-agent mode, includes a session role (primary/challenger/verifier) for behavioral diversity. Call after omega_welcome at session start, or on-demand for a specific section.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "section": {"type": "string", "description": "Section: 'memory', 'coordination', 'coordination_gate', 'teamwork', 'context', 'reminders', 'diagnostics', 'entity', 'heuristics', 'git', 'what_next'. Groups: 'solo', 'multi_agent', 'full', 'minimal'."},
                "project": {"type": "string", "description": "Project path for context-sensitive rules."},
                "session_id": {"type": "string", "description": "Session ID for role assignment in multi-agent mode."},
            },
        },
    },
    {
        "name": "omega_checkpoint",
        "description": "Save a task checkpoint: captures current plan, progress, files touched, decisions, and key context. Enables seamless session continuity.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "task_title": {"type": "string", "description": "Brief title of the current task"},
                "plan": {"type": "string", "description": "Current plan or goals"},
                "progress": {"type": "string", "description": "What's been completed, in progress, remaining"},
                "files_touched": {"type": "object", "description": "Map of file paths to change summaries", "additionalProperties": {"type": "string"}},
                "decisions": {"type": "array", "items": {"type": "string"}, "description": "Key technical decisions"},
                "key_context": {"type": "string", "description": "Critical context for continuation"},
                "next_steps": {"type": "string", "description": "What to do next"},
                "session_id": {"type": "string"},
                "project": {"type": "string"},
            },
            "required": ["task_title", "progress"],
        },
    },
    {
        "name": "omega_resume_task",
        "description": "Resume a previously checkpointed task. Retrieves the latest checkpoint with full plan, progress, files, decisions, and next steps.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "task_title": {"type": "string", "description": "Title of the task to resume (semantic search)"},
                "project": {"type": "string", "description": "Project path to filter checkpoints"},
                "verbosity": {"type": "string", "enum": ["full", "summary", "minimal"], "description": "full=everything, summary=plan+progress+next, minimal=next steps only"},
                "limit": {"type": "integer", "description": "Number of checkpoints to retrieve (default 1)"},
            },
        },
    },
    {
        "name": "omega_memory",
        "description": "Manage a specific memory by ID: edit its content, delete it, mark it as superseded, mark it as helpful/unhelpful/outdated, find similar memories, traverse relationship edges, link two memories, or list flagged memories for review. Use when acting on an individual memory rather than searching broadly.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "action": {"type": "string", "enum": ["edit", "delete", "feedback", "similar", "traverse", "link", "flagged", "check_contradictions", "supersede"], "description": "Operation to perform"},
                "memory_id": {"type": "string", "description": "Memory node ID. For action='supersede', this is the old/outdated memory to retire. Required for most actions; not required for 'flagged' or 'check_contradictions'."},
                "new_content": {"type": "string", "description": "New content (for action='edit') or content to check (for action='check_contradictions')"},
                "priority": {"type": "integer", "minimum": 1, "maximum": 5, "description": "New priority from 1 (lowest) to 5 (highest), only for action='edit'. Edit requires new_content, priority, or both."},
                "rating": {"type": "string", "description": "helpful, unhelpful, or outdated (only for action='feedback')"},
                "reason": {"type": "string", "description": "Optional explanation (for action='feedback' or action='supersede')"},
                "limit": {"type": "integer", "description": "Max results (default 5)", "default": 5},
                "max_hops": {"type": "integer", "description": "Traversal depth 1-5 (default 2, only for action='traverse')", "default": 2},
                "min_weight": {"type": "number", "description": "Min edge weight 0.0-1.0 (default 0.0, only for action='traverse')", "default": 0.0},
                "edge_types": {"type": "array", "items": {"type": "string"}, "description": "Filter by edge type: related, contradicts, supersedes, evolves (only for action='traverse')"},
                "target_id": {"type": "string", "description": "For action='link', the destination memory. For action='supersede', the optional new replacement memory linked as replacement -> supersedes -> outdated memory_id."},
                "edge_type": {"type": "string", "enum": ["related", "contradicts", "supersedes", "evolves"], "description": "Edge type (only for action='link', default 'related')"},
                "weight": {"type": "number", "description": "Edge weight 0.0-1.0 (only for action='link', default 1.0)"},
            },
            "required": ["action"],
        },
    },
    {
        "name": "omega_profile",
        "description": "Read or update the user's persistent profile (name, preferences, working style) or list all stored preferences. The profile persists across sessions and informs agent behavior. Default action is 'read'.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "action": {"type": "string", "enum": ["read", "update", "list_preferences"], "description": "read (default), update, or list_preferences", "default": "read"},
                "update": {"type": "object", "description": "Profile fields to merge (only for action='update')"},
            },
        },
    },
    {
        "name": "omega_remind",
        "description": "Manage time-based reminders: set new reminders, list active ones, or dismiss by ID.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "action": {"type": "string", "enum": ["set", "list", "dismiss"], "description": "set (default), list, or dismiss", "default": "set"},
                "text": {"type": "string", "description": "What to be reminded about (for action='set')"},
                "duration": {"type": "string", "description": "When to remind, e.g. '1h', '30m', '2d' (for action='set')"},
                "context": {"type": "string", "description": "Optional context (for action='set')"},
                "session_id": {"type": "string"},
                "project": {"type": "string"},
                "status": {"type": "string", "enum": ["pending", "fired", "dismissed", "all"], "description": "Filter (for action='list')"},
                "reminder_id": {"type": "string", "description": "Reminder ID (for action='dismiss')"},
                "entity_id": {"type": "string", "description": "Scope reminders to entity (for action='list'). Omit for all."},
            },
        },
    },
    {
        "name": "omega_maintain",
        "description": "System housekeeping and constraint management. Use 'health' to check database size and integrity, 'consolidate' to prune stale memories, 'compact' to merge near-duplicates, 'discover_connections' to actively find and link related memories (generates cross-type insights), 'backup'/'restore' for data safety, 'clear_session' to purge a session's data, 'synthesize_insights' to generate system insights, 'backfill_embeddings' to fill missing vectors, 'job_status' to poll a previously submitted async job, 'list_constraints'/'check_constraint'/'save_constraints' to manage file constraint rules. Long-running actions (consolidate, compact, backup, restore, discover_connections, synthesize_insights, backfill_embeddings) return a job_id immediately and run in the background to avoid client RPC timeouts; poll with action='job_status'. Pass wait=true to block instead.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "action": {"type": "string", "enum": ["health", "consolidate", "compact", "discover_connections", "backup", "restore", "clear_session", "synthesize_insights", "backfill_embeddings", "job_status", "list_constraints", "check_constraint", "save_constraints"], "description": "Maintenance operation"},
                "warn_mb": {"type": "number", "description": "Warning threshold MB (health, default 350)", "default": 350},
                "critical_mb": {"type": "number", "description": "Critical threshold MB (health, default 800)", "default": 800},
                "max_nodes": {"type": "integer", "description": "Max expected nodes (health, default 10000)", "default": 10000},
                "prune_days": {"type": "integer", "description": "Prune zero-access older than N days (consolidate, default 14)", "default": 14},
                "max_summaries": {"type": "integer", "description": "Max session summaries (consolidate, default 50)", "default": 50},
                "event_type": {"type": "string", "description": "Type to compact (compact, default lesson_learned)", "default": "lesson_learned"},
                "similarity_threshold": {"type": "number", "description": "Jaccard similarity 0.0-1.0 (compact, default 0.6)", "default": 0.6},
                "min_cluster_size": {"type": "integer", "description": "Min cluster size (compact, default 3)", "default": 3},
                "dry_run": {"type": "boolean", "description": "Preview only (compact/discover_connections, default false)", "default": False},
                "lookback_hours": {"type": "integer", "description": "Hours to look back for discover_connections (default 24)", "default": 24},
                "filepath": {"type": "string", "description": "File path (backup/restore)"},
                "clear_existing": {"type": "boolean", "description": "Clear before restore (default true)", "default": True},
                "session_id": {"type": "string", "description": "Session to purge (clear_session)"},
                "file_path": {"type": "string", "description": "File path to check (only for action='check_constraint')"},
                "rules": {"type": "array", "items": {"type": "object"}, "description": "Constraint rules to save (only for action='save_constraints'). Each: {pattern, constraint, severity}"},
                "batch_size": {"type": "integer", "description": "Batch size (only for action='backfill_embeddings', default 50)", "default": 50},
                "wait": {"type": "boolean", "description": "If true, block until the action finishes and return the full result. If false (default for long-running actions), return a job_id immediately and run in the background.", "default": False},
                "job_id": {"type": "string", "description": "Job id returned by a prior async submission (only for action='job_status')"},
            },
            "required": ["action"],
        },
    },
    {
        "name": "omega_stats",
        "description": "View analytics and behavioral insights: memory breakdown by type, per-session statistics, weekly digest, forgetting audit log, deduplication stats, access rate trends, milestones, unified diagnostic report, tool utilization monitoring, and behavioral patterns (habits_list, habits_analyze, habits_profile, habits_confirm, habits_deny, habits_recommendations).",
        "inputSchema": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["types", "sessions", "digest", "forgetting_log", "dedup", "milestones", "access_rate", "diagnostic", "habits_list", "habits_confirm", "habits_deny", "habits_analyze", "habits_profile", "habits_recommendations", "graph_stats", "utilization"],
                    "description": "Which stats/insights to retrieve. 'diagnostic' returns a unified health/value report. 'utilization' shows tool usage vs defined tools. habits_* actions manage behavioral patterns.",
                },
                "days": {"type": "integer", "description": "Days for digest (default 7)", "default": 7},
                "limit": {"type": "integer", "description": "Max entries for forgetting_log (default 50)", "default": 50},
                "reason": {"type": "string", "description": "Filter forgetting_log by reason"},
                "pattern_id": {"type": "string", "description": "Memory ID of a behavioral pattern (for habits_confirm/habits_deny)"},
            },
            "required": ["action"],
        },
    },
    {
        "name": "omega_reflect",
        "description": "Analyze memory quality and knowledge evolution. 'contradictions' finds conflicting memories on a topic, 'evolution' traces how understanding changed over time, 'stale' surfaces old never-accessed memories for review.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["contradictions", "evolution", "stale"],
                    "description": "Analysis to perform",
                },
                "topic": {
                    "type": "string",
                    "description": "Topic to analyze (required for contradictions/evolution)",
                },
                "limit": {
                    "type": "integer",
                    "description": "Max memories to analyze (default 20 for contradictions/evolution, 30 for stale)",
                    "default": 20,
                },
                "days": {
                    "type": "integer",
                    "description": "Look-back window for stale action (default 30)",
                    "default": 30,
                },
                "min_age_days": {
                    "type": "integer",
                    "description": "Minimum age in days to be considered stale (default 14)",
                    "default": 14,
                },
                "entity_id": {
                    "type": "string",
                    "description": "Scope to entity. Omit for all.",
                },
            },
            "required": ["action"],
        },
    },
    {
        "name": "omega_consult_gpt",
        "description": "Consult GPT for a second opinion on hard problems. Use when stuck (10+ min or 3+ failed approaches), facing irreversible architecture decisions, debugging dead ends, cross-validating fragile solutions, or bridging domain expertise gaps. Do NOT use for simple tasks, speed-sensitive work, or when tests already pass.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": "The question or problem to consult GPT about.",
                },
                "context": {
                    "type": "string",
                    "description": "Supporting context: code snippets, error messages, constraints. Kept separate from prompt for clarity.",
                },
                "system": {
                    "type": "string",
                    "description": "Override the system prompt for domain-specific framing (default: generic second-opinion prompt).",
                },
                "temperature": {
                    "type": "number",
                    "description": "Sampling temperature: 0.0-0.3 for factual, 0.5-0.7 for design, 0.7-1.0 for brainstorming (default: 0.7).",
                    "minimum": 0.0,
                    "maximum": 2.0,
                },
                "max_tokens": {
                    "type": "integer",
                    "description": "Max response tokens (default: 4096, max: 16384).",
                    "minimum": 1,
                    "maximum": 16384,
                },
            },
            "required": ["prompt"],
        },
    },
    {
        "name": "omega_consult_claude",
        "description": "Consult Claude for a second opinion on hard problems (for non-Anthropic agents). Use when stuck (10+ min or 3+ failed approaches), facing irreversible architecture decisions, debugging dead ends, cross-validating fragile solutions, or bridging domain expertise gaps. Do NOT use for simple tasks, speed-sensitive work, or when tests already pass.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": "The question or problem to consult Claude about.",
                },
                "context": {
                    "type": "string",
                    "description": "Supporting context: code snippets, error messages, constraints. Kept separate from prompt for clarity.",
                },
                "system": {
                    "type": "string",
                    "description": "Override the system prompt for domain-specific framing (default: generic second-opinion prompt).",
                },
                "temperature": {
                    "type": "number",
                    "description": "Sampling temperature: 0.0-0.3 for factual, 0.5-0.7 for design, 0.7-1.0 for brainstorming (default: 0.7).",
                    "minimum": 0.0,
                    "maximum": 2.0,
                },
                "max_tokens": {
                    "type": "integer",
                    "description": "Max response tokens (default: 4096, max: 16384).",
                    "minimum": 1,
                    "maximum": 16384,
                },
            },
            "required": ["prompt"],
        },
    },
    {
        "name": "omega_review",
        "description": "Review a code diff with multi-agent specialist panel. Uses OMEGA memory for codebase context, team conventions, and past incident awareness. Returns findings sorted by severity with confidence scores.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "diff": {
                    "type": "string",
                    "description": "Unified diff text to review (from git diff, PR, or raw text)",
                },
                "repo": {
                    "type": "string",
                    "description": "Repository name for context lookup",
                },
                "mode": {
                    "type": "string",
                    "enum": ["strict", "normal", "verbose"],
                    "description": "Filtering mode: strict (critical+major only), normal (default, >=70% confidence), verbose (all findings)",
                    "default": "normal",
                },
                "agents": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Agent types to run: correctness, security, performance, consistency, blast_radius. Default: all.",
                },
                "summarize_only": {
                    "type": "boolean",
                    "description": "If true, return only a fast deterministic summary (no LLM review). Good for quick risk assessment.",
                    "default": False,
                },
                "session_id": {"type": "string"},
                "entity_id": {"type": "string"},
            },
            "required": ["diff"],
        },
    },
]


# ---------------------------------------------------------------------------
# Condensed Mode (CodeMode-inspired)
# ---------------------------------------------------------------------------

# Tools that remain as standalone even in condensed mode.
# These are called every session and benefit from zero-overhead direct invocation.
STANDALONE_TOOLS = ["omega_welcome", "omega_protocol", "omega_store"]

# Category mapping for tool discovery via omega_tools.
# Covers core, coordination, oracle, router, profile, knowledge, and entity tools.
TOOL_CATEGORIES = {
    # Core memory tools
    "omega_store": "memory",
    "omega_query": "query",
    "omega_welcome": "session",
    "omega_protocol": "session",
    "omega_checkpoint": "memory",
    "omega_resume_task": "memory",
    "omega_memory": "memory",
    "omega_profile": "session",
    "omega_remind": "operations",
    "omega_maintain": "maintenance",
    "omega_stats": "maintenance",
    "omega_reflect": "intelligence",
    "omega_consult_gpt": "intelligence",
    "omega_consult_claude": "intelligence",
    "omega_review": "intelligence",
    # Backward-compat aliases
    "omega_weekly_digest": "operations",
    "omega_remind_list": "operations",
    "omega_remind_dismiss": "operations",
    # Coordination tools
    "omega_session_register": "coordination",
    "omega_session_heartbeat": "coordination",
    "omega_session_deregister": "coordination",
    "omega_sessions_list": "coordination",
    "omega_session_snapshot": "coordination",
    "omega_session_recover": "coordination",
    "omega_file_claim": "coordination",
    "omega_file_release": "coordination",
    "omega_file_check": "coordination",
    "omega_branch_claim": "coordination",
    "omega_branch_release": "coordination",
    "omega_branch_check": "coordination",
    "omega_intent_announce": "coordination",
    "omega_intent_check": "coordination",
    "omega_coord_status": "coordination",
    "omega_coord_metrics": "coordination",
    "omega_task_create": "coordination",
    "omega_task_claim": "coordination",
    "omega_task_next": "coordination",
    "omega_task_complete": "coordination",
    "omega_task_cancel": "coordination",
    "omega_task_fail": "coordination",
    "omega_task_progress": "coordination",
    "omega_task_deps": "coordination",
    "omega_tasks_list": "coordination",
    "omega_update_task": "coordination",
    "omega_send_message": "coordination",
    "omega_inbox": "coordination",
    "omega_handoff": "coordination",
    "omega_find_agents": "coordination",
    "omega_audit": "coordination",
    "omega_git_events": "coordination",
    "omega_action_check": "coordination",
    "omega_action_claim": "coordination",
    "omega_action_complete": "coordination",
    "omega_goal": "coordination",
    "omega_goal_link": "coordination",
    "omega_drift_check": "coordination",
    "omega_smart_route": "coordination",
    "omega_decision_register": "coordination",
    "omega_decision_query": "coordination",
    "omega_decision_revoke": "coordination",
    "omega_council": "coordination",
    # Oracle tools
    "omega_oracle_record": "oracle",
    "omega_oracle_resolve": "oracle",
    "omega_oracle_analyze": "oracle",
    "omega_oracle_status": "oracle",
    # Router tools
    "omega_route_prompt": "router",
    "omega_classify_intent": "router",
    "omega_router_status": "router",
    "omega_set_priority_mode": "router",
    "omega_get_model_config": "router",
    "omega_switch_model": "router",
    "omega_get_current_model": "router",
    "omega_router_context": "router",
    "omega_warm_router": "router",
    "omega_router_benchmark": "router",
    # Profile tools
    "omega_profile_set": "profile",
    "omega_profile_get": "profile",
    "omega_profile_search": "profile",
    "omega_profile_list": "profile",
    # Knowledge tools
    "omega_ingest_document": "knowledge",
    "omega_search_documents": "knowledge",
    "omega_list_documents": "knowledge",
    "omega_remove_document": "knowledge",
    "omega_scan_documents": "knowledge",
    "omega_sync_kb": "knowledge",
    # Entity tools
    "omega_entity_create": "entity",
    "omega_entity_get": "entity",
    "omega_entity_list": "entity",
    "omega_entity_update": "entity",
    "omega_entity_delete": "entity",
    "omega_entity_add_relationship": "entity",
    "omega_entity_relationships": "entity",
    "omega_entity_tree": "entity",
}

CONDENSED_TOOL_SCHEMAS = [
    {
        "name": "omega_tools",
        "description": "List available OMEGA tools or get the full schema for a specific tool. Call with no args to see all tool names and descriptions. Call with tool='name' to get its full input schema so you know what arguments to pass to omega_call.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "tool": {
                    "type": "string",
                    "description": "Tool name to get full schema for. Omit to list all tools.",
                },
                "category": {
                    "type": "string",
                    "enum": ["memory", "query", "session", "maintenance", "intelligence",
                             "operations", "coordination", "oracle", "router", "profile",
                             "knowledge", "entity", "all"],
                    "description": "Filter by category. Default: all.",
                },
            },
        },
    },
    {
        "name": "omega_call",
        "description": "Execute any OMEGA tool by name. Use omega_tools() first to discover available tools and their parameters. Example: omega_call(tool='omega_query', args={'query': 'auth decisions', 'mode': 'semantic'})",
        "inputSchema": {
            "type": "object",
            "properties": {
                "tool": {
                    "type": "string",
                    "description": "Tool name to execute, e.g. 'omega_query', 'omega_checkpoint', 'omega_memory'",
                },
                "args": {
                    "type": "object",
                    "description": "Arguments to pass to the tool. Use omega_tools(tool='name') to see accepted parameters.",
                },
            },
            "required": ["tool"],
        },
    },
]

# Harness-agnostic context wire contract. Gate the lower-level generic
# context_assemble alias behind an env var, but expose context_packet by
# default because it is the task-aware memory packet used by current hook and
# daemon surfaces.
import os as _os
from omega.server.context_tool_schemas import CONTEXT_TOOL_SCHEMAS  # noqa: E402

TOOL_SCHEMAS.extend(
    schema for schema in CONTEXT_TOOL_SCHEMAS
    if schema.get("name") == "context_packet"
)
if _os.environ.get("OMEGA_CONTEXT_WIRE", "").lower() in ("1", "true", "yes"):
    TOOL_SCHEMAS.extend(
        schema for schema in CONTEXT_TOOL_SCHEMAS
        if schema.get("name") != "context_packet"
    )


def get_condensed_schemas(all_schemas: list[dict]) -> list[dict]:
    """Return condensed tool set: standalone tools + meta-tools.

    In condensed mode, only essential high-frequency tools are exposed directly.
    All other tools are accessible via omega_call/omega_tools meta-tools.
    """
    standalone = [s for s in all_schemas if s["name"] in STANDALONE_TOOLS]
    return standalone + CONDENSED_TOOL_SCHEMAS
