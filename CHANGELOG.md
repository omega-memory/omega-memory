# Changelog

All notable changes to OMEGA will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [1.5.10] - 2026-08-14

### Fixed
- Classify the Pro-provided `omega_task_release` operation in Core's write
  rate-limit tier so task ownership mutations receive the same protection as
  other coordination writes.

## [1.5.9] - 2026-08-14

### Added
- `SQLiteStore.get_thread_local_read_conn()` gives Pro analyzers a stable,
  per-thread read connection with WAL, timeout, foreign-key, and sqlite-vec
  parity.
- `OMEGA_DEDUP_THRESHOLDS` accepts a validated JSON object for overriding
  selected auto-capture dedup thresholds. Restart the MCP process after
  changing it.

### Fixed
- Batch `omega_store(items=[...], entity_id=...)` requests now inherit
  request-level scope and provenance fields for items that omit them, while
  explicit per-item values take precedence.
- Batch normalization preserves per-item fields and no longer mutates caller
  dictionaries while generating embeddings.

## [1.5.2] - 2026-06-11

### Fixed
- **Core/Pro boundary hardening**: Core no longer trusts local
  `omega_platform.license.is_pro()` checks to unlock memory caps, full
  retrieval, Pro tool loading, or upgrade-message suppression. Core now uses
  plugin capabilities (`pro_tools`, `full_retrieval`, `unlimited_memory`) for
  extension-provided behavior.

## [1.5.1] - 2026-06-11

### Fixed
- **Core/Pro compatibility for Pro 1.5.x**: added compatibility imports for
  `omega.bridge._core`, `omega.bridge._ingest`, and `omega.bridge._query` so
  Pro modules expecting the split bridge layout can run against public Core.
- **MCP server PID registry**: restored `omega.server.pid_registry` for local
  lock diagnostics, orphaned stdio server cleanup, and session registration
  startup paths.

## [1.4.14] - 2026-05-19

### Fixed
- **`omega_maintain` "Server disconnected" / RPC timeout in Claude Desktop**:
  long-running maintenance actions (`consolidate`, `compact`, `backup`,
  `restore`, `discover_connections`, `synthesize_insights`,
  `backfill_embeddings`) ran synchronous SQLite + ONNX work directly on the
  MCP server's asyncio event loop. On populated stores this blocked the stdio
  transport long enough for Claude Desktop to hit its ~4-minute RPC timeout
  and drop the connection. These actions now submit to the shared SQLite
  executor via an in-process JobRegistry and return a `job_id` in under
  500 ms; the event loop stays responsive. Poll with
  `omega_maintain action=job_status job_id=<id>` or pass `wait=true` to keep
  the legacy blocking behavior.

## [1.3.1] - 2026-03-17

### Added
- **Condensed mode**: ~80% context token savings, enabled by default (opt out with `--no-condensed`)
- **Behavioral learning**: Pattern analysis engine that learns tool preferences, git style, session patterns, co-edit graphs, and workflow sequences
- **Advisory engine**: Context-aware suggestions for file edits, errors, deployments, and session starts
- **CrewAI integration**: Use OMEGA as CrewAI's memory backend
- **Obsidian export**: `omega export-obsidian` command for Obsidian vault export
- **Stats card**: `omega stats --card` for shareable stats visualization
- **Framework support**: Added Codex CLI, Antigravity IDE, and venv Python resolution
- **`llms-install.md`**: Agent-autonomous installation guide
- **CLI reference**: Added `docs/cli-reference.md`
- **Code review (`omega_review`)**: Multi-agent specialist review panel with 5 agents
  (correctness, security, performance, consistency, blast radius). Hybrid static+LLM
  analysis: 12 deterministic pattern checks (zero false positives) plus LLM for novel
  issues. Memory-powered: uses OMEGA conventions, past incidents, and team preferences
  for context. Confidence gating with strict/normal/verbose modes. Fast `summarize_only`
  mode for risk assessment without LLM. Pre-commit hook at `hooks/pre_review.py`.
  Standalone engine available separately (see revue).

### Fixed
- **`_get_store()` crash** (#48): `handlers.py` imported non-existent `omega.store` module, breaking `omega_query(mode="browse")`, `omega_stats`, `omega_reflect`, and memory link/flag/supersede actions
- **Recursive `omega_call`**: Prevented recursive calls to meta-tools
- **5 test failures** (#44): Resolved failing tests
- **Windows install docs**: Collapsed into collapsible `<details>` block

### Changed
- `omega doctor --client` now supports `claude-desktop`, `cursor`, `windsurf`, `cline`, `codex`, `antigravity`, `venv` (was only `claude-code`)
- `export`/`import` CLI subcommands removed (use `omega export-obsidian` for exports)

## [1.2.0] - 2026-03-04

### Added
- **Hooks in wheel**: Claude Code hooks now ship inside the pip package; `omega hooks setup`
  auto-configures `~/.claude/settings.json` with correct paths
- **Multi-user auth**: Google Sign-In, role-based access control, self-service onboarding wizard,
  per-user data scoping across all admin API routes
- **Auth guards**: 46 API routes hardened with session verification
- **LLM provider abstraction**: Unified `llm.py` (Python) and `lib/llm.ts` (TypeScript) supporting
  Anthropic, OpenAI, and OpenAI-compatible providers via `OMEGA_LLM_PROVIDER`
- **LLM usage tracking**: Per-call cost tracking with session rollups and admin dashboard
- **Multi-model consultation**: `omega_consult_gpt` and `omega_consult_claude` tools for
  cross-model second opinions (replaces `omega_lessons`)
- **Growth engine**: Thompson Sampling bandit for content optimization, attribution engine
  correlating X metrics with GitHub/PyPI, content genome pipeline
- **Scheduled jobs infrastructure**: State machine with retry logic, approval gates,
  audit trail, SLA escalation, heartbeat monitoring, Vercel cron migration
- **Query expansion**: LLM-based retrieval augmentation with strong-signal short-circuit
  and position-aware reranking (QMD-inspired)
- **System insights**: Permanent-TTL insight memories with protocol surfacing, file-triggered
  hooks, and admin graph visualization
- **Trajectory distillation**: Auto-extracts session summaries at session stop with quality gate
- **Project registry**: Unified canonical projects table with status tracking and
  auto-generation at session stop
- **Schema v13**: Unique index on forgetting_log to prevent duplicate entries
- **Coordination upgrades**: File claim gap closed with read tracking, peer-claimed commit
  blocking, auto-handoff on session stop, message priority
- **Admin dashboard**: Projects tab, LLM Usage tab, Entities tab, coordination console,
  Growth tab, Settings expansion (Profile, Agent, Memory, Projects, Integrations)
- **Website**: Next.js 16.1.6, homepage restructure, /pro page redesign, 6 new blog posts,
  HeroGraph visualization
- **Automation**: Target scanner, scan-and-reply cron, daily summary email,
  @omega_memory multi-account infrastructure

### Changed
- sqlite_store.py split into mixin-based package (7 modules) for maintainability
- Oracle engine upgraded to thread-safe singleton
- Hook system abstracted for multi-client support (Claude Code, Cursor, Windsurf, Cline)
- Protocol adapted for multi-model providers with provider-specific notes
- Admin theme refined (violet accent, semantic signal colors)

### Fixed
- ~176 bug fixes across security, admin, knowledge, bridge, entities, growth, LLM routing
- Multi-user data isolation across 29 API routes
- Thread-safety issues in oracle and embedding engines
- Knowledge base: SSRF prevention, entity filtering, race conditions
- Bridge: full memory ID returned from dedup/evolve/reconfirm paths
- Cloud sync hardening and expanded test coverage

### Removed
- SayDo experimental module (archived)
- `omega_lessons` tool (replaced by `omega_consult_gpt`/`omega_consult_claude`)

## [1.1.0] - 2026-02-25

### Added
- Automatic entity extraction from conversations (Phase 3) with async processing
  and throttle controls
- Bi-temporal data model with `valid_from`/`valid_until` for point-in-time queries
- Memory strength scoring with decay, deduplication, and `strength_min` query filter
- Memory type classification — auto-classifies on store, filterable via `memory_type`
  parameter in omega_query
- Contradiction detection surfaced in store output
- Intelligence cards — compact [OMEGA] cards for memory, decision, and learning events
  at NORMAL+ transparency
- Entity graph relationships wired into retrieval scoring
- Campaign automation v3.0 (Layers 1-3) with automation modules and proposal feed
- MCP server instructions for automatic memory usage by connected agents
- Session awareness and agent discipline protocol sections (v1.3.0)
- Admin dashboard: 3D Memory Graph visualization with bloom and clustering
- Admin dashboard: Interactive entity knowledge graph
- Admin dashboard: Skills Graph 3D visualization with manifest and API
- Admin dashboard: KnowledgeBase rewrite with folder tree, markdown preview,
  and breadcrumb navigation
- Admin dashboard: proposals feed, ambient awareness layer,
  historical coordination view
- Admin dashboard: Recharts-based Insights tab with memory sparklines and
  project charts
- Security hardening — shared validation module, coordinator handler hardening,
  write-tool rate limiting, CI dependency auditing, Dependabot
- Windows installer improvements and repair utility
- Website: downloads page, competitive positioning, OpenAI comparison blog post,
  dual-account tweet generation

### Changed
- Extracted shared `mcp_response`/`mcp_error` helpers to reduce duplication
  across server handlers
- Generate button switched from SSE to job polling for reliability
- Tweet pipeline migrated to EST with slot optimization and reply queue
- Heavy cron jobs migrated from Vercel to GitHub Actions

### Fixed
- SQLite lock contention reduced with `BEGIN IMMEDIATE` transactions and
  WAL checkpoint logic
- Connection leak in `record_metric` eliminated with hardened close/reconnect
- Stale embedding backend state causing `tuple.encode()` crash resolved
- Hook resilience improved — retry on startup race, skip informational hooks
  in fallback mode
- JIT proxy: bypass MCP SDK 1.26 `outputSchema` validation errors
- Router intent classifier no longer unconditionally overwrites session task
- Cloud sync: per-document error isolation with timeout config
- FTS5 query sanitization hardened against malformed input

## [1.0.0] - 2026-02-13

### Added
- Open-core plugin architecture (OmegaPlugin base class, discover_plugins())
- Graceful degradation for all optional modules (coordination, router, entity, knowledge, profile, cloud)
- Apache-2.0 license
- GitHub Actions CI/CD (test matrix: Python 3.11, 3.12, 3.13)
- PyPI publish workflow with trusted publishers
- CONTRIBUTING.md, SECURITY.md, NOTICE
- Issue and PR templates

### Changed
- License: MIT → Apache-2.0
- Author: OMEGA Memory Maintainers
- Hook server: conditional coordination handler registration
- CLI: graceful "requires omega-pro" messages for commercial modules
- MCP server: commercial tool schemas loaded only when modules available
- Version bump: 0.6.1 → 1.0.0

## [0.6.1] - 2026-02-11

### Removed
- **Phoenix module deleted** — 1,531 lines source + 1,152 lines tests. Fully disconnected dead code: no hooks triggered it, no workflows called its 6 MCP tools, respawn requests wrote to JSON files nothing read. Session context handoff already handled by coordinator's snapshot/recover system.
- 6 MCP tools removed: `omega_phoenix_check`, `omega_phoenix_request`, `omega_phoenix_complete`, `omega_phoenix_requests`, `omega_phoenix_handoff`, `omega_phoenix_metrics`
- `[phoenix]` optional dependency group from pyproject.toml

### Changed
- MCP tool count: 60 (was 66)
- Test count: 1406 across 29 test files (was 1447 across 31)
- Renamed `_phoenix_recovery` → `_session_resume` in hook_server.py (pure coordinator logic, no Phoenix dependency)
- `[PHOENIX]` label → `[RESUME]` in coord handler output

## [0.5.0] - 2026-02-10

### Added
- Router auto-warmup on session start, auto-classify on every user prompt
- Router provider status surfaced in welcome briefing
- Groq re-added for simple_edit intent (speed mode)
- 1122+ tests across 28 test files, 0 lint errors

### Changed
- Router classifier status checked on start, hot-reload on config change

## [0.4.3] - 2026-02-10

### Changed
- **Router rewired**: xAI/Grok-4 as primary for exploration intent (research/AI trends), Google/Gemini as fallbacks, Groq re-added for simple_edit (speed mode)
- **Router secrets**: API keys loaded from `~/.omega/secrets.json` — 5/5 providers active (Anthropic, OpenAI, Google, xAI, Groq)
- CI matrix: dropped Python 3.10 (unsupported), added 3.13
- Removed phantom `litellm` dependency from router extras (was never imported)
- Fixed commitizen `changelog_start_rev` to existing tag `v0.3.0`

### Fixed
- 105 ruff lint errors across src/ and tests/ (unused imports, f-strings without placeholders)
- `pre_push_guard` test subprocess import path
- `auto_claim_branch` path validation
- Feedback score inflation: clamped to valid range
- Unbounded state growth in coordination audit log
- Stale Haiku model ID updated in router defaults
- Deprecated `datetime.utcnow()` replaced with timezone-aware alternative
- Missing thread lock in coordination cleanup path

### Added
- 31 new tests: migration, reingest, reembed, lessons, classifier, concurrency coverage gaps
- Updated SCORECARD.md to v0.4.3 with post-cleanup metrics (242 memories, 4/5 providers)
- **README rewrite for public audience**: problem statement, 60-second quickstart, comparison table (vs Mem0/Zep/Copilot Memory), collapsible advanced details, contributing section, PyPI badge
- README: corrected test count (1074→1102), tool counts (22 memory + 25 coord), hooks (11)
- CONTRIBUTING.md: updated test count (1074→1102)
- SECURITY.md: added 0.3.x and 0.4.x to supported versions

### Removed
- 162 stale Gnosis-era memory artifacts (506→242 memories)
- Dead code paths identified in diagnostic audit

## [0.4.2] - 2026-02-10

### Added
- **Unified hook system**: all 11 hooks via `fast_hook.py` → daemon UDS dispatch
- `pre_push_guard` migrated from standalone script to daemon handler (12 handlers total)
- `_SLOW_HOOKS` set with configurable timeout in `fast_hook.py`
- 37 new batch protocol tests (daemon batch, client batch, fallback short-circuit, log format)
- 4 UAT test suites: memory (10 scenarios), router (4), cross-module (4) — 2,153 lines
- 83 new tests in `test_cli.py` and `test_graphs_coverage.py`

### Fixed
- Thread lock added to `get_node()` in sqlite_store.py (race condition on access_count)
- Null-guard on access_count arithmetic in bridge.py dedup path
- 19 bare `except: pass` replaced with specific exceptions + debug logging
- `cleanup_old_requests()` now uses `max_age_days` parameter (was ignored)
- Merged 3 separate `store.query()` calls into 1 in `auto_capture()` (saves 2 embedding generations)

### Removed
- Dead `HAS_NUMPY` variable from graphs.py
- Dead `get_model_history()` stub from router/engine.py

## [0.4.1] - 2026-02-10

### Fixed
- `omega doctor` now loads sqlite-vec extension before checking vec index
- Circuit breaker cooldown recovery: embeddings resume after transient failures
- Orphaned vec index entries cleaned up on startup

## [0.4.0] - 2026-02-10

### Added
- **Router module** (10 MCP tools): multi-LLM intent classification and routing
  - ONNX prototype classifier (<2ms, no ML deps beyond existing bge-small)
  - 5 intents: coding, creative, logic, exploration, simple_edit
  - 5 providers: Anthropic, OpenAI, Google, Groq, xAI
  - 4 priority modes: cost, speed, quality, balanced
  - Context affinity tracking with switch penalties
  - Large context override (>100K tokens → Gemini)
  - `omega_route_prompt`, `omega_classify_intent`, `omega_router_status`, `omega_set_priority_mode`, `omega_switch_model`, `omega_get_model_config`, `omega_get_current_model`, `omega_router_context`, `omega_warm_router`, `omega_router_benchmark`
- `filter_tags` parameter for `omega_query` with AND-logic hard filtering
- CLI `compact` and `stats` subcommands
- `scripts/migrate_magma.py` for Gnosis MAGMA → OMEGA migration
- README rewrite covering all tools, architecture diagram, install flow

### Changed
- MCP tool count: 47 → 57 (+10)
- Optional module loading: `try/except ImportError` in mcp_server.py
- `pyproject.toml`: router/full optional dependency groups
- Total tests: 779

## [0.3.2] - 2026-02-10

### Added
- `filter_tags` parameter on `query_structured` (AND-logic, 3x over-fetch)
- 4 utilization gaps closed: auto-tags on store, confidence thresholds, plan capture triggers
- `pre_push_guard` enhancements: checkout target parsing, branch claim checks

### Changed
- Hooks manifest aligned with active coordination hooks
- Utilization scorecard: grade improved from C+ to A- (gap 0.3)

## [0.3.1] - 2026-02-10

### Added
- **24 coordination MCP tools re-enabled**: sessions, file/branch claims, intents, tasks, messaging, audit
- `pre_task_guard` hook: blocks edits when file's task is assigned to another agent
- Utilization scorecard tracking (SCORECARD.md)

### Changed
- **Tool optimization**: 44 → 22 MCP tools (23% → 91% utilization)
  - Phase 1: Disconnected redundant coordination schemas (44 → 25)
  - Phase 2: Merged `omega_status` into `omega_health`, `omega_export`/`omega_import` into `omega_backup`, `omega_cross_project_lessons` into `omega_lessons` (25 → 22)
  - Phase 3: Activated 8 dormant tools via hook wiring (session_start: type_stats, list_preferences, auto-backup; session_stop: session_stats, timeline; surface: traverse, phrase_search)
- MCP idle timeout: 600s → 3600s (1 hour)

## [0.3.0] - 2026-02-10

### Added
- **Real multi-agent file enforcement**: PreToolUse `pre_file_guard` hook blocks Edit/Write/NotebookEdit via `sys.exit(2)` when the target file is claimed by another agent session
- **Claim TTL auto-expiry**: `CLAIM_TTL_SECONDS = 600` — file claims expire after 10 minutes of inactivity, independently of the 30-minute stale session timeout
- **Force-claim override**: `claim_file(force=True)` lets agents explicitly steal claims when coordination breaks down, with full audit trail via `log_audit(tool_name="file_claim_force")`
- `force` boolean parameter added to `omega_file_claim` MCP tool schema
- `_clean_expired_claims()` method runs during periodic stale session cleanup
- TTL-aware `check_file()` auto-deletes expired claims on read
- Daemon parity: `handle_pre_file_guard` handler in `hook_server.py` dispatch table
- `pre_file_guard` added to `fast_hook.py` fallback scripts table
- 21 new tests in `test_pre_file_guard.py` covering blocking, self-claim, TTL expiry, force-claim, fail-open, notebook support
- Atomic write for `profile.json` via tempfile + `os.replace` to prevent corruption on crash

### Fixed
- **Deadlock in `claim_file`**: `log_audit()` was called inside `with self._lock:` — since `threading.Lock` is non-reentrant, this caused silent hangs. Audit call moved outside the lock block

### Design Decisions
- **Fail-open**: OMEGA unavailability never blocks edits — coordination is opt-in safety
- **Standalone hook, not daemon-routed**: PreToolUse is on the critical path; standalone is safer if daemon crashes
- **No enforcement in single-agent mode**: Empty `SESSION_ID` skips all file guard checks

## [0.2.8] - 2026-02-10

### Added
- SECURITY.md with vulnerability reporting policy
- CHANGELOG.md in Keep-A-Changelog format (backfilled v0.2.0–v0.2.7)
- CLI memory commands: `omega query`, `omega store`, `omega remember`, `omega timeline`
- LLM-agnostic setup: `omega setup --client claude-code` (decoupled from Claude Code)
- `omega_task_cancel` MCP tool with handler and schema (was dead code — coordination method existed but had no MCP path)

### Changed
- Removed 6 zero-utilization MCP tools (`deduplicate`, `extract_preferences`, `constraints`, `batch_store`, `reload`, `dedup_stats`)
- Removed 3 overhead hooks (`post_edit_test`, `pre_edit_surface`, `track_file_read`), reducing per-edit Python processes from 5 to 3
- Export/import paths restricted to `~/.omega/` (was entire home directory)
- Error messages in MCP handlers no longer leak internal details
- `~/.omega/` directory created with mode `0o700` (owner-only access)
- Encryption key file created atomically with `O_EXCL` to prevent TOCTOU race
- Thread-safe SQLiteStore singleton via double-check locking
- Hardcoded `/opt/homebrew/bin/python3` replaced with dynamic resolution in hooks
- Silent vec-index delete failures now logged at DEBUG level
- Dedup regex compiled once at module level instead of per-query
- Hook log rotation at 5 MB cap to prevent disk fill
- `pre_push_guard` now blocks pushes on divergence via `sys.exit(2)` (was advisory-only)
- `auto_claim_file` now surfaces `[CONFLICT]` warnings instead of silently swallowing claim conflicts
- Hook timeouts increased: `coord_session_start` 3s → 10s, `coord_session_stop` 3s → 8s
- Git fetch subprocess timeout reduced from 15s to 5s to fit within hook timeouts
- Replaced deprecated `datetime.utcnow()` with `datetime.now(timezone.utc)` in hooks and coordination
- Coordination tool count: 25 handlers (was 24)

### Removed
- Dead `save()`/`load()` compatibility stubs from SQLiteStore
- Undefined `_LOAD_ATTEMPT_COUNT` global from graphs.py

### Fixed
- Numeric MCP handler parameters now clamped to safe bounds
- `.gitignore` hardened with `.env`, `*.db`, `*.log`, `*.key`, `hook.sock`, `.omega/`
- `cancel_task` now enforces owner check — previously any session could cancel another's in-progress work
- `claim_file`/`claim_branch` catch `sqlite3.IntegrityError` for cross-process race safety
- `check_file` reads now guarded by `_lock` for consistency with write locking discipline
- `check_inbox` read + mark-as-read unified under single lock to prevent concurrent read races
- `_snapshot_session` preserves session capabilities in metadata for `_auto_reregister` recovery
- Push event logging no longer truncates commit hashes (was `[:12]`, causing hash comparison mismatches)

## [0.2.7] - 2026-02-10

### Added
- Memory visibility UX: capture confirmations, scored surfacing, health pulse, session activity summary
- Auto-feedback on surfaced memories at session stop
- Auto-compaction of lessons every 14 days at session start
- Cross-project lesson surfacing at session start
- File-extension-to-tag mapping for contextual re-ranking in edit surfacing
- `fast_hook.py` stdin bridging for fallback hook scripts
- Public SQLiteStore API: `edge_count()`, `get_last_capture_time()`, `get_session_event_counts()`
- 61 tests for hook UX output formatting

### Changed
- Disabled CoreML provider to prevent native memory leak (~700KB/op); CPU-only ONNX used instead
- Session summaries TTL reduced from LONG_TERM (2 weeks) to SHORT_TERM (1 day)
- Hooks no longer access `SQLiteStore._conn` directly (use public API)

### Fixed
- Daemon/standalone hook parity for scored surfacing, capture confirmations, and auto-feedback
- Stale surfacing file cleanup (both `.surfaced` and `.surfaced.json` files older than 24h)
- Idle watchdog task reference saved to prevent GC cancellation
- Resolved 23 test failures from broken hook imports
- Error dedup in hooks: cap at 5 errors/session, deduplicate by first-100-chars hash

## [0.2.4] - 2026-02-09

### Added
- UDS hook server for fast hook dispatch (~5ms vs ~750ms cold start)
- Graph traversal (`omega_traverse`) with BFS over edges table, max 5 hops
- Memory compaction (`omega_compact`) with Jaccard clustering and consolidated summaries
- Contextual re-ranking with `context_file` and `context_tags` boost
- Auto-claim file hook for implicit coordination
- Orphan process cleanup and proactive stale session GC
- 24 tests for UDS hook server, idle watchdog, and stale cleanup debounce
- UAT suite for registration, coordination, and conflict avoidance

### Fixed
- Coordination lifecycle, edge creation, and TTL gaps
- Batch embedding falls back to single-item ONNX before hash

## [0.2.3] - 2026-02-09

### Added
- bge-small-en-v1.5 as primary embedding model (384-dim, better quality than all-MiniLM-L6-v2)
- Periodic TTL garbage collection (at most once per hour via `time.monotonic()`)
- Git-aware coordination: detect uncoordinated agents via git state
- Observability: `omega doctor`, FTS5 repair, backup, timing, plan capture

### Changed
- Python minimum version raised to 3.11+ (3.10 EOL)
- Test isolation improvements for safety

### Fixed
- bge-small-en-v1.5 HuggingFace download URLs corrected

## [0.2.2] - 2026-02-09

### Added
- Intelligence layer: constraints, cross-project lessons, smart surfacing
- Task management with deadlock detection and audit log
- Defensive hooks: read tracking and read-before-write warning
- Session recovery: snapshot/recover crashed sessions

### Fixed
- Flaky tests caused by embedding circuit-breaker leak
- Naive/aware datetime comparison bug in `query()`
- 4 documented feature gaps closed

## [0.2.1] - 2026-02-09

### Added
- `omega_similar`: find memories similar to a given one
- `omega_timeline`: show memories grouped by day
- `omega_consolidate`: memory hygiene at scale (dedup, prune, optimize)
- Auto-tags: extract languages, tools, file paths, project names at store time
- Auto-relate: create `related` edges on store (similarity >= 0.45)

### Fixed
- 3 critical bugs found during UAT testing

## [0.2.0] - 2026-02-09

### Added
- SQLite + sqlite-vec backend replacing in-memory graphs + JSONL sidecar
- FTS5 full-text search for phrase queries
- Multi-agent coordination system (12 tools, 38 tests)
- Encryption at rest with `cryptography` library
- `omega_remember`, `omega_store`, `omega_query`, `omega_welcome`, `omega_profile`
- Export/import for backup and restore
- Batch store for multiple memories in one call
- 37 handler tests covering all 21 handlers

### Changed
- Complete storage rewrite from JSONL to SQLite
- Parameterized SQL throughout (no string interpolation)

[1.0.0]: https://github.com/omega-memory/omega/compare/v0.6.1...v1.0.0
[0.6.1]: https://github.com/omega-memory/omega/compare/v0.5.0...v0.6.1
[0.5.0]: https://github.com/omega-memory/omega/compare/v0.4.3...v0.5.0
[0.4.3]: https://github.com/omega-memory/omega/compare/v0.4.1...v0.4.3
[0.4.2]: https://github.com/omega-memory/omega/compare/v0.4.1...v0.4.3
[0.4.1]: https://github.com/omega-memory/omega/compare/v0.3.0...v0.4.1
[0.4.0]: https://github.com/omega-memory/omega/compare/v0.3.0...v0.4.1
[0.3.2]: https://github.com/omega-memory/omega/compare/v0.3.0...v0.4.1
[0.3.1]: https://github.com/omega-memory/omega/compare/v0.3.0...v0.4.1
[0.3.0]: https://github.com/omega-memory/omega/compare/v0.2.8...v0.3.0
[0.2.8]: https://github.com/omega-memory/omega/compare/v0.2.7...v0.2.8
[0.2.7]: https://github.com/omega-memory/omega/compare/v0.2.2...v0.2.7
[0.2.4]: https://github.com/omega-memory/omega/compare/v0.2.2...v0.2.7
[0.2.3]: https://github.com/omega-memory/omega/compare/v0.2.2...v0.2.7
[0.2.2]: https://github.com/omega-memory/omega/compare/v0.2.1...v0.2.2
[0.2.1]: https://github.com/omega-memory/omega/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/omega-memory/omega/releases/tag/v0.2.0
