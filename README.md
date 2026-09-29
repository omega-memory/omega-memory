# OMEGA

**Cross-model memory for AI agents. Local-first. Works with Claude, GPT, Gemini, Cursor, Claw Code, and any MCP client.** Your agent's brain shouldn't live on someone else's server, or be locked to one provider.

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![PyPI](https://img.shields.io/pypi/v/omega-memory.svg)](https://pypi.org/project/omega-memory/)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)

---

## The Problem

AI coding agents are stateless. Every new session starts from zero. The "solutions" either lock you into one model provider or send your codebase context to their cloud.

- **Context loss.** Agents forget every decision, preference, and architectural choice between sessions. Developers spend 10-30 minutes per session re-explaining context that was already established.
- **Repeated mistakes.** Without learning from past sessions, agents make the same errors over and over. They don't remember what worked, what failed, or why a particular approach was chosen.
- **Cloud memory = someone else's database.** Services like Mem0 require API keys and send your data to their servers. When they change pricing, get acquired, or go down, your agent's accumulated intelligence disappears.
- **Vendor lock-in.** Anthropic's Memory Tool only works with Claude. OpenAI's memory only works with GPT. Switch models, lose your memory.

OMEGA solves this. Memory, coordination, and learning that runs entirely on your machine. Works with every major LLM and coding agent. No cloud. No API keys. No vendor lock-in.

<!-- TODO: terminal GIF showing memory recall across sessions -->
<!-- mcp-name: io.github.omega-memory/omega-memory -->

## Quick Install

OMEGA needs **Python 3.11 or newer**. Check with `python3 --version`: macOS ships Python 3.9, which is too old. If pip answers `No matching distribution found for omega-memory`, that is the reason; install a newer Python first (for example `brew install python@3.12`, or from [python.org](https://www.python.org/downloads/)).

Install OMEGA into its own environment. Homebrew's Python refuses a plain `pip install` (PEP 668), and an isolated install keeps OMEGA's dependencies away from your projects. Pick one:

```bash
# pipx
pipx install "omega-memory[server]"

# or uv
uv tool install "omega-memory[server]"

# or a virtual environment
python3.12 -m venv ~/.venvs/omega     # any Python 3.11 or newer
~/.venvs/omega/bin/pip install "omega-memory[server]"
export PATH="$HOME/.venvs/omega/bin:$PATH"
```

Keep the quotes around `"omega-memory[server]"`: zsh, the default macOS shell, treats bare square brackets as a pattern and fails. The `[server]` extra installs the MCP server, which also runs the hook daemon; without it Claude Code gets no OMEGA tools and no hooks.

Then:

```bash
omega setup     # downloads the embedding model, registers the MCP server, installs hooks
omega doctor    # checks every step, and names the fix for anything that failed
```

`omega setup --dry-run` shows what setup would change without writing or downloading anything.

### Claude Desktop

```bash
omega setup --client claude-desktop
```

This registers OMEGA as an MCP server in Claude Desktop's config. Restart Claude Desktop to activate.

### Cursor, Windsurf, Cline, Codex, Antigravity

```bash
omega setup --client cursor      # or: windsurf, cline, codex, antigravity
```

Codex and Antigravity get their config file written; for the others setup prints the JSON block to paste. For any other MCP client, `omega setup --client venv` prints the command and arguments to use. Hooks (automatic capture and surfacing) are available with Claude Code only.

<details>
<summary><strong>Library-only install (no MCP server)</strong></summary>

If you only need OMEGA as a Python library for scripts, CI/CD, or automation, install it into your project's environment:

```bash
pip install omega-memory    # Core only, no MCP server
```

```python
from omega import store, query, remember

store("Always use TypeScript strict mode", "user_preference")
results = query("TypeScript preferences")
```

This gives you the full storage and retrieval API without running an MCP server. Hooks need the MCP server (the hook daemon runs inside it), so a library-only install has no automatic capture or surfacing.

</details>

### From Source

```bash
git clone https://github.com/omega-memory/omega-memory.git
cd omega-memory
python3.12 -m venv .venv && source .venv/bin/activate
pip install -e ".[server,dev]"
omega setup
```

`omega setup` will:
1. Create `~/.omega/` (or `$OMEGA_HOME` if you set it)
2. Download the ONNX embedding model, bge-small-en-v1.5 (~130 MB), and a reranker (~90 MB) to `~/.cache/omega/models/`
3. Register `omega-memory` as an MCP server (Claude Code auto-detected, or specify `--client`)
4. Install session hooks into `~/.claude/settings.json`
5. Add an OMEGA block to `~/.claude/CLAUDE.md`

If a step fails, the summary marks it `[FAIL]` with the reason and setup exits with an error.

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

- **Memory & Learning** — Stores decisions, lessons, error patterns, and preferences with semantic search. Claude recalls what matters without you re-explaining everything each session. Tools cover compaction, consolidation, timeline, graph traversal, and checkpoint/resume.

- **Multi-Agent Coordination** *(omega-pro)* — File and branch locking, session management, task queues with dependencies, intent broadcasting, and agent-to-agent messaging, so agents don't overwrite each other's work.

- **Intelligent LLM Routing** *(omega-pro)* — Classifies tasks and routes to the optimal model. Coding → Claude Sonnet. Quick edit → Llama 8b at 1/60th the cost. 1M token context → Gemini Flash. 5 providers, 4 priority modes, sub-2ms intent classification.

- **Knowledge Base** *(omega-pro)* — Ingest PDFs, markdown, web pages, and text files into a searchable knowledge base with semantic chunking.

- **Entity Registry** *(omega-pro)* — Multi-entity corporate memory with relationships, hierarchies, and entity-scoped memories/profiles/documents.

- **Secure Profile** *(omega-pro)* — AES-256 encrypted personal data storage with macOS Keychain integration.

## How OMEGA Compares

| Feature | OMEGA | Anthropic Memory | Mem0 | Zep |
|---------|:-----:|:----------------:|:----:|:---:|
| Works with any LLM/agent | **Yes** | Claude only | Yes | Yes |
| Your data stays on your machine | **Yes** | Partial* | No | No |
| No cloud dependency | **Yes** | No (needs API) | No | No |
| Semantic search + knowledge graph | **Yes** | No (file CRUD) | $249/mo | Yes |
| Multi-agent coordination | **Yes** *(pro)* | Research preview | No | No |
| Works with Claude Code, Cursor, Claw Code | **Yes** | Claude only | Partial | No |
| Free & open source | **Yes** (Apache 2.0) | No | Freemium | Freemium |

*Anthropic's Memory Tool stores data client-side but requires Claude API calls for all memory operations. OMEGA runs entirely on-device, including embeddings (ONNX).*

**Anthropic Memory is for Anthropic. OMEGA is for everyone.**

## Architecture

```
     Claude Code  ·  Cursor  ·  Claw Code  ·  Any MCP Client
               │         │         │              │
               └─────────┴─────┬───┴──────────────┘
                               │ stdio/MCP
               ┌───────────────▼─────────────┐
               │   OMEGA MCP Server   │
               │   core memory tools  │
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

Single database, modular handlers. Optional modules (coordination, router, entity, knowledge, profile) are available with [OMEGA Pro](https://omegamax.co/pro) and register into the same server process. No separate daemons, no microservices.

## MCP Tools Reference

OMEGA runs as an MCP server inside Claude Code. By default it shows the client five tools (`omega_store`, `omega_welcome`, `omega_protocol`, `omega_tools`, `omega_call`) to save context: `omega_tools` lists everything available and `omega_call` runs any of it. Set `OMEGA_CONDENSED=0` in the server's environment to expose every tool directly.

### Core tools

| Tool | What it does |
|------|-------------|
| `omega_store` | Store a memory (decision, lesson, error, preference, ...) |
| `omega_query` | Search memories: semantic, exact phrase, timeline, or browse |
| `omega_welcome` | Session briefing: recent context, reminders, profile |
| `omega_protocol` | Operating rules for the session |
| `omega_checkpoint` | Save task state for cross-session continuity |
| `omega_resume_task` | Resume a checkpointed task |
| `omega_memory` | Edit, delete, supersede, rate, or link one memory; find similar ones |
| `omega_profile` | Read or update the user profile and preferences |
| `omega_remind` | Set, list, or dismiss time-based reminders |
| `omega_maintain` | Health, consolidation, compaction, backup and restore |
| `omega_stats` | Type breakdown, session stats, weekly digest |
| `omega_reflect` | Find contradictions; trace how a topic's decisions evolved |
| `omega_review` | Review a diff with memory of your codebase's conventions |
| `context_packet` | Compact, task-aware memory packet for the current work |
| `omega_consult_gpt`, `omega_consult_claude` | Ask another model for a second opinion (needs your own API key for that provider) |

[OMEGA Pro](https://omegamax.co/pro) adds coordination, routing, entity, knowledge base, and profile tools.

## CLI

| Command | Description |
|---------|-------------|
| `omega setup` | Create dirs, download model, register MCP, install hooks (`--dry-run` to preview) |
| `omega doctor` | Verify installation health |
| `omega hooks setup` / `omega hooks doctor` | Install or check the Claude Code hooks only |
| `omega status` | Memory count, store size, model status |
| `omega query <text>` | Search memories by semantic similarity |
| `omega store <text>` | Store a memory with a specified type |
| `omega remember <text>` | Store a permanent preference |
| `omega timeline` | Show memory timeline grouped by day |
| `omega activity` | Show recent session activity overview |
| `omega stats` | Memory type distribution and health summary |
| `omega consolidate` | Deduplicate, prune, and optimize memory |
| `omega compact` | Cluster and summarize related memories |
| `omega backup` | Back up omega.db (keeps last 5) |
| `omega validate` | Validate database integrity |
| `omega logs` | Show recent hook errors |
| `omega migrate-db` | Migrate legacy JSON to SQLite |
| `omega serve` | Run the MCP server (`omega serve --help` for the HTTP daemon) |

### Free and Pro

Core is free and open source. On the free tier, once the store holds 2,000 memories search switches from semantic search to a plain text match, and new memories stop being stored at 5,000. `omega status` shows where you are. OMEGA Pro removes both limits.

<details>
<summary><strong>Advanced Details</strong></summary>

### Hooks

All hooks run `fast_hook.py`, which hands the work to the hook daemon inside the MCP server over `~/.omega/hook.sock`. If no server is running, the hooks return at once without doing anything, except reply capture, which runs on its own; none of them blocks Claude Code.

| Hook | Matcher | Handler | Purpose |
|------|---------|---------|---------|
| SessionStart | all | `session_start` | Welcome briefing, session resume |
| Stop | all | `assistant_capture` | Capture decisions and fixes from the reply |
| Stop | all | `session_stop` | Session summary |
| UserPromptSubmit | all | `auto_capture` | Auto-capture lessons/decisions |
| PostToolUse | Edit/Write/NotebookEdit/Bash/Read | `surface_memories` | Surface relevant memories |

Memory text the hooks inject is labelled as stored data, not instructions.

> With OMEGA Pro, additional coordination handlers register automatically: session lifecycle, file/branch claim guards, heartbeat, and git push guards.

### HTTP daemon (optional)

`omega serve install` runs one shared server under launchd instead of one per session (macOS). It listens on `127.0.0.1:8377`, refuses requests whose Host or Origin is not local, and requires a bearer key kept in `~/.omega/mcp_api_key`. `omega serve migrate-config` points Claude Code at the daemon and writes that key into its config; `omega serve restore-config` switches back.

### Storage

| Path | Purpose |
|------|---------|
| `~/.omega/omega.db` | SQLite database (memories, embeddings, edges); set `OMEGA_HOME` to move `~/.omega` |
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
- Install the server extra into the same environment as OMEGA: `pip install "omega-memory[server]"` (or `pipx install --force "omega-memory[server]"`)

**MCP server not registered:**
- Run `omega setup` again, or run the `claude mcp add -s user omega-memory -- ...` command that `omega doctor` prints: it names the Python from OMEGA's own environment

**Hooks not firing:**
- Run `omega hooks doctor`; `omega hooks setup` repairs missing or outdated entries
- Check `~/.omega/hooks.log` for errors

## Development

```bash
pip install -e ".[server,dev]"
pytest tests/                # Test suite
ruff check src/              # Lint
```

## Uninstall

```bash
claude mcp remove -s user omega-memory
omega serve uninstall        # only if you installed the HTTP daemon
rm -rf ~/.omega ~/.cache/omega
pipx uninstall omega-memory  # or: uv tool uninstall omega-memory, or delete the venv
```

Manually remove OMEGA entries from `~/.claude/settings.json` and the `<!-- OMEGA:BEGIN -->` block from `~/.claude/CLAUDE.md`.

## Contributing

- [Contributing Guide](CONTRIBUTING.md)
- [Security Policy](SECURITY.md)
- [Changelog](CHANGELOG.md)
- [Report a Bug](https://github.com/omega-memory/omega-memory/issues)

## License

Apache-2.0. See [LICENSE](LICENSE).
