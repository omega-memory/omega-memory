# OMEGA

**AI agents that remember, coordinate, and learn. All on your machine.** Your agent's brain shouldn't live on someone else's server.

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![PyPI](https://img.shields.io/pypi/v/omega-memory.svg)](https://pypi.org/project/omega-memory/)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-1123%20passing-brightgreen.svg)]()

---

## The Problem

AI coding agents are stateless. Every new session starts from zero. And the "solutions" want you to send your codebase context to their cloud.

- **Context loss.** Agents forget every decision, preference, and architectural choice between sessions. Developers spend 10-30 minutes per session re-explaining context that was already established.
- **Repeated mistakes.** Without learning from past sessions, agents make the same errors over and over. They don't remember what worked, what failed, or why a particular approach was chosen.
- **Cloud memory = someone else's database.** Services like Mem0 require API keys and send your data to their servers. When they change pricing, get acquired, or go down, your agent's accumulated intelligence disappears.

OMEGA solves this. Memory, coordination, and learning that runs entirely on your machine. No cloud. No API keys. No vendor lock-in.

<!-- TODO: terminal GIF showing memory recall across sessions -->

## Quick Install

```bash
pip install omega-memory[server]    # Full install (memory + MCP server)
omega setup                         # Downloads model, registers MCP, installs hooks
omega doctor                        # Verify everything works
```

<details>
<summary><strong>Library-only install (no MCP server)</strong></summary>

If you only need OMEGA as a Python library for scripts, CI/CD, or automation:

```bash
pip install omega-memory    # Core only, no MCP server process
```

```python
from omega import store, query, remember

store("Always use TypeScript strict mode", "user_preference")
results = query("TypeScript preferences")
```

This gives you the full storage and retrieval API without running an MCP server (~50 MB lighter, no background process). Hooks still work:

```bash
omega setup --hooks-only    # Auto-capture + memory surfacing, no MCP server (~600MB RAM saved)
```

</details>

### From Source

```bash
git clone https://github.com/omega-memory/omega.git
cd omega
pip install -e ".[server,dev]"
omega setup
```

`omega setup` will:
1. Create `~/.omega/` directory
2. Download the ONNX embedding model (~90 MB) to `~/.cache/omega/models/`
3. Register `omega-memory` as an MCP server with Claude Code
4. Install session hooks into `~/.claude/settings.json`
5. Add an OMEGA block to `~/.claude/CLAUDE.md`

## 60-Second Quickstart

OMEGA works through natural language — no API calls, no configuration. Just talk to Claude.

**1. Tell Claude to remember something:**
> "Remember that the auth system uses JWT tokens, not session cookies"

Claude stores this as a permanent memory with semantic embeddings.

**2. Close the session. Open a new one.**

**3. Ask about it:**
> "What did I decide about authentication?"

OMEGA surfaces the relevant memory automatically:
```
Found 1 relevant memory:
  [decision] "The auth system uses JWT tokens, not session cookies"
  Stored 2 days ago | accessed 3 times
```

That's it. Memories persist across sessions, accumulate over time, and are surfaced automatically when relevant — even if you don't explicitly ask.

## Key Features

- **Memory & Learning** — Stores decisions, lessons, error patterns, and preferences with semantic search. Claude recalls what matters without you re-explaining everything each session. 25 memory tools including compaction, consolidation, timeline, graph traversal, and context virtualization (checkpoint/resume).

- **Multi-Agent Coordination** *(omega-pro)* — File and branch locking, session management, task queues with dependencies, intent broadcasting, and agent-to-agent messaging. 29 coordination tools that prevent agents from overwriting each other's work.

- **Intelligent LLM Routing** *(omega-pro)* — Classifies tasks and routes to the optimal model. Coding → Claude Sonnet. Quick edit → Llama 8b at 1/60th the cost. 1M token context → Gemini Flash. 5 providers, 4 priority modes, sub-2ms intent classification.

- **Knowledge Base** *(omega-pro)* — Ingest PDFs, markdown, web pages, and text files into a searchable knowledge base with semantic chunking.

- **Entity Registry** *(omega-pro)* — Multi-entity corporate memory with relationships, hierarchies, and entity-scoped memories/profiles/documents.

- **Secure Profile** *(omega-pro)* — AES-256 encrypted personal data storage with macOS Keychain integration.

## How OMEGA Compares

| Feature | OMEGA | Mem0 | Zep | Copilot Memory |
|---------|:-----:|:----:|:---:|:--------------:|
| Your data stays on your machine | Yes | No | No | No |
| No API keys or cloud dependency | Yes | No | No | No |
| Multi-agent coordination | Yes *(pro)* | No | No | Partial |
| Graph memory included free | Yes | $249/mo | No | No |
| LLM routing | Yes *(pro)* | No | No | No |
| Document ingestion (RAG) | Yes *(pro)* | No | Yes | No |
| Free & open source | Yes (Apache 2.0) | Freemium | Freemium | Bundled |

## Architecture

```
               ┌─────────────────────┐
               │    Claude Code       │
               │  (or any MCP host)   │
               └──────────┬──────────┘
                          │ stdio/MCP
               ┌──────────▼──────────┐
               │   OMEGA MCP Server   │
               │   25 core tools      │
               └──┬──────────────────┘
                  │
         ┌────────▼──────────────┐
         │ Core Memory Engine    │
         │ (semantic search,     │
         │  embeddings, graphs)  │
         └─────┬─────────────────┘
               │
               ▼
         ┌──────────────────────────────────────┐
         │         omega.db (SQLite)             │
         │  memories | edges | embeddings        │
         └──────────────────────────────────────┘
```

Single database, modular handlers. Optional modules (coordination, router, entity, knowledge, profile) are available via [omega-pro](https://github.com/omega-memory) and register into the same server process. No separate daemons, no microservices.

## MCP Tools Reference

OMEGA runs as an MCP server inside Claude Code. The core package provides 25 memory tools. [omega-pro](https://github.com/omega-memory) adds coordination, routing, entity, knowledge, and profile tools.

### Memory (25 tools)

| Tool | What it does |
|------|-------------|
| `omega_store` | Store typed memory (decision, lesson, error, summary) |
| `omega_query` | Semantic search with tag filters and contextual re-ranking |
| `omega_welcome` | Session briefing with recent memories and profile |
| `omega_profile` | Read or update user profile |
| `omega_delete_memory` | Delete a specific memory by ID |
| `omega_edit_memory` | Edit the content of a memory |
| `omega_list_preferences` | List all stored user preferences |
| `omega_health` | Detailed health check with memory usage and recommendations |
| `omega_backup` | Export or import memories for backup/restore |
| `omega_lessons` | Cross-session lessons ranked by access count |
| `omega_feedback` | Record feedback on a surfaced memory |
| `omega_clear_session` | Clear all memories for a specific session |
| `omega_similar` | Find memories similar to a given one |
| `omega_timeline` | Memories grouped by day |
| `omega_consolidate` | Prune stale memories, cap summaries, clean edges |
| `omega_traverse` | Walk the relationship graph |
| `omega_compact` | Cluster and summarize related memories |
| `omega_checkpoint` | Save task state for cross-session continuity |
| `omega_resume_task` | Resume a previously checkpointed task |
| `omega_remind` | Set a time-based reminder |
| `omega_remind_list` | List active reminders |
| `omega_remind_dismiss` | Dismiss a reminder |
| `omega_type_stats` | Memory counts grouped by event type |
| `omega_session_stats` | Memory counts grouped by session |
| `omega_weekly_digest` | Weekly knowledge digest with stats and trends |

### Additional tools with omega-pro

| Module | Tools | Description |
|--------|------:|-------------|
| Coordination | 29 | File/branch locking, sessions, tasks, messaging, audit |
| Router | 10 | LLM routing, intent classification, model switching |
| Entity | 8 | Corporate entities, relationships, hierarchies |
| Knowledge | 5 | Document ingestion, semantic search, RAG |
| Profile | 3 | AES-256 encrypted personal data storage |

## CLI

| Command | Description |
|---------|-------------|
| `omega setup` | Create dirs, download model, register MCP, install hooks (`--hooks-only` to skip MCP) |
| `omega doctor` | Verify installation health |
| `omega status` | Memory count, store size, model status |
| `omega query <text>` | Search memories by semantic similarity |
| `omega store <text>` | Store a memory with a specified type |
| `omega timeline` | Show memory timeline grouped by day |
| `omega activity` | Show recent session activity overview |
| `omega stats` | Memory type distribution and health summary |
| `omega consolidate` | Deduplicate, prune, and optimize memory |
| `omega compact` | Cluster and summarize related memories |
| `omega backup` | Back up omega.db (keeps last 5) |
| `omega validate` | Validate database integrity |
| `omega logs` | Show recent hook errors |
| `omega migrate-db` | Migrate legacy JSON to SQLite |

<details>
<summary><strong>Advanced Details</strong></summary>

### Hooks (7 processes, 11 handlers)

All hooks dispatch via `fast_hook.py` → daemon UDS socket, with fail-open semantics.

| Hook | Matcher | Handlers | Purpose |
|------|---------|----------|---------|
| SessionStart | all | `session_start` | Welcome briefing, session resume |
| Stop | all | `session_stop` | Summary |
| UserPromptSubmit | all | `auto_capture` | Auto-capture lessons/decisions |
| PostToolUse | Edit/Write/NotebookEdit | `surface_memories` | Surface relevant memories |
| PostToolUse | Bash/Read | `surface_memories` | Surface relevant memories |

> With omega-pro, additional coordination handlers register automatically: session lifecycle, file/branch claim guards, heartbeat, and git push guards.

### Storage

| Path | Purpose |
|------|---------|
| `~/.omega/omega.db` | SQLite database (memories, embeddings, edges) |
| `~/.omega/profile.json` | User profile |
| `~/.omega/hooks.log` | Hook error log |
| `~/.cache/omega/models/bge-small-en-v1.5-onnx/` | ONNX embedding model |

### Search Pipeline

1. **Vector similarity** via sqlite-vec (cosine distance, 384-dim bge-small-en-v1.5)
2. **Full-text search** via FTS5 (fast keyword matching)
3. **Type-weighted scoring** (decisions/lessons weighted 2x)
4. **Contextual re-ranking** (boosts by tag, project, and content match)
5. **Deduplication** at query time

### Memory Lifecycle

- **Dedup**: SHA256 hash (exact) + embedding similarity 0.85+ (semantic) + Jaccard per-type
- **Evolution**: Similar content (55-95%) appends new insights to existing memories
- **TTL**: Session summaries expire after 1 day, lessons/preferences are permanent
- **Auto-relate**: Creates `related` edges (similarity >= 0.45) to top-3 similar memories
- **Compaction**: Clusters and summarizes related memories

### Memory Footprint

- Startup: ~31 MB RSS
- After first query (ONNX model loaded): ~337 MB RSS
- Database: ~10.5 MB for ~242 memories

### What Gets Modified

`omega setup` modifies these files outside `~/.omega/`:

- `~/.claude.json` — Adds `omega-memory` MCP server entry
- `~/.claude/settings.json` — Adds hook entries
- `~/.claude/CLAUDE.md` — Adds a managed `<!-- OMEGA:BEGIN -->` block

All changes are idempotent.

</details>

## Troubleshooting

**`omega doctor` shows FAIL on import:**
- Ensure `pip install -e ".[server]"` from the repo root
- Check `python3 -c "import omega"` works

**MCP server fails to start:**
- Run `pip install omega-memory[server]` (the `[server]` extra includes the MCP package)

**MCP server not registered:**
```bash
claude mcp add omega-memory -- python3 -m omega.server.mcp_server
```

**Hooks not firing:**
- Check `~/.claude/settings.json` has OMEGA hook entries
- Check `~/.omega/hooks.log` for errors

## Development

```bash
pip install -e ".[server,dev]"
pytest tests/                # 2198+ tests
ruff check src/              # Lint
```

## Uninstall

```bash
claude mcp remove omega-memory
rm -rf ~/.omega ~/.cache/omega
pip uninstall omega-memory
```

Manually remove OMEGA entries from `~/.claude/settings.json` and the `<!-- OMEGA:BEGIN -->` block from `~/.claude/CLAUDE.md`.

## Contributing

- [Contributing Guide](CONTRIBUTING.md)
- [Security Policy](SECURITY.md)
- [Changelog](CHANGELOG.md)
- [Report a Bug](https://github.com/omega-memory/omega/issues)

## License

Apache-2.0. See [LICENSE](LICENSE).
