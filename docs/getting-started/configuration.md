---
title: Configuration
description: Storage paths, environment variables, hooks, and MCP server settings
---

# Configuration

## Storage paths

OMEGA stores all data locally. Here's where everything lives:

| Path | Purpose |
|------|---------|
| `~/.omega/omega.db` | SQLite database (memories, coordination, entities) |
| `~/.omega/profile.json` | User profile (greeting name, timezone, role) |
| `~/.omega/secrets.json` | Router API keys (chmod 600, never committed) |
| `~/.omega/hooks.log` | Hook error log (for debugging hook failures) |
| `~/.omega/documents/` | Drop folder for auto-ingestion (knowledge base) |
| `~/.omega/backups/` | Automatic weekly backups |
| `~/.cache/omega/models/bge-small-en-v1.5-onnx/` | Primary ONNX embedding model |
| `~/.cache/omega/models/all-MiniLM-L6-v2-onnx/` | Fallback embedding model |

!!! tip "Portable storage"
    Set the `OMEGA_HOME` environment variable to move the storage directory anywhere. The cache directory for models stays at `~/.cache/omega/` regardless.

## Files modified outside `~/.omega`

`omega setup` modifies three files in your Claude Code configuration:

### `~/.claude.json` — MCP server registration

Adds an `omega-memory` entry to the `mcpServers` section:

```json
{
  "mcpServers": {
    "omega-memory": {
      "command": "python3",
      "args": ["-m", "omega.server.mcp_server"],
      "env": {},
      "timeout": 3600
    }
  }
}
```

This tells Claude Code how to spawn the OMEGA MCP server.

### `~/.claude/settings.json` — Hook entries

Adds 7 hook entries that power automatic memory capture, surfacing, and coordination. See the [Hooks](#hooks) section below for details.

### `~/.claude/CLAUDE.md` — Agent instructions

Adds a managed block between `<!-- OMEGA:BEGIN -->` and `<!-- OMEGA:END -->` markers with instructions for using memory and coordination tools. This block is updated on each `omega setup` run.

## Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `OMEGA_HOME` | `~/.omega` | Override the storage directory for database, profile, secrets, and logs |
| `OMEGA_IDLE_TIMEOUT` | `3600` | MCP server idle timeout in seconds. Server auto-shuts down after this period of inactivity |
| `NO_COLOR` | (unset) | When set, disables Rich formatting in CLI output |

=== "Set temporarily"

    ```bash
    OMEGA_HOME=/path/to/storage omega doctor
    ```

=== "Set permanently (zsh)"

    ```bash
    echo 'export OMEGA_HOME=/path/to/storage' >> ~/.zshrc
    source ~/.zshrc
    ```

## Embedding model

The factory model is `bge-small-en-v1.5` — English-only, 384 dimensions. These
variables are read **once at import**, so restart OMEGA after changing any of
them.

| Variable | Default | Description |
|----------|---------|-------------|
| `OMEGA_EMBEDDING_MODEL` | `bge-small-en-v1.5` | Model name. Also selects the default model directory |
| `OMEGA_EMBEDDING_DIM` | `384` | Vector dimension. Must match both the model and the existing vector tables |
| `OMEGA_EMBEDDING_POOLING` | `mean` | `mean` or `cls`. CLS-trained models such as `bge-m3` need `cls` |
| `OMEGA_EMBEDDING_MODEL_DIR` | `~/.cache/omega/models/<model>-onnx` | Explicit ONNX model directory |
| `OMEGA_EMBEDDING_PAD_TOKEN` | `[PAD]` | Tokenizer padding token |
| `OMEGA_EMBEDDING_PAD_ID` | `0` | Tokenizer padding id |

!!! warning "The dimension must match the existing store"

    `OMEGA_EMBEDDING_DIM` has to agree with the dimension the vector tables were
    created with. OMEGA verifies this when it opens a store and **refuses to
    start** on a mismatch, rather than accepting writes it cannot vectorize.

    Changing the dimension on a store that already holds data requires
    re-embedding it — the existing vectors are not convertible.

Running a multilingual model, for example:

```bash
export OMEGA_EMBEDDING_MODEL=bge-m3
export OMEGA_EMBEDDING_DIM=1024
export OMEGA_EMBEDDING_POOLING=cls
```

## Hooks

OMEGA uses 7 hook processes (batched from 11 handlers) that run automatically during Claude Code sessions. All hooks are **fail-open** — if a hook errors, it logs to `~/.omega/hooks.log` and lets the operation proceed.

| # | Hook Event | Trigger | What it does |
|---|-----------|---------|--------------|
| 1 | `SessionStart` | Session opens | Delivers welcome briefing, registers coordination session, syncs git state, resumes checkpointed tasks |
| 2 | `Stop` | Session closes | Captures session summary, deregisters session, releases all file and branch claims |
| 3 | `UserPromptSubmit` | Every user message | Auto-captures decisions and lessons from conversation patterns |
| 4 | `PostToolUse` (Edit/Write) | After file edits | Surfaces relevant memories, sends coordination heartbeat, auto-claims edited file |
| 5 | `PostToolUse` (Bash/Read) | After bash/read | Surfaces relevant memories, sends coordination heartbeat |
| 6 | `PreToolUse` (Bash) | Before bash commands | Guards against git push divergence and unclaimed branch pushes |
| 7 | `PreToolUse` (Edit/Write) | Before file edits | Guards against editing files claimed by other agents, checks task assignment |

### Disabling hooks

To disable a specific hook, remove its entry from `~/.claude/settings.json`. To disable all hooks:

```bash
omega setup --uninstall-hooks
```

To reinstall them later:

```bash
omega setup --install-hooks
```

!!! warning "Disabling hooks reduces functionality"
    Without hooks, OMEGA still works as an MCP tool server — you can manually query and store memories. But automatic capture, surfacing, coordination guards, and session management won't function.

## MCP server

OMEGA runs as a **stdio MCP server** spawned by Claude Code on demand. Key characteristics:

- **Transport**: stdio (stdin/stdout JSON-RPC). No network ports, no HTTP.
- **Lifecycle**: Claude Code spawns the server process when it first needs an OMEGA tool. The server stays alive for the duration of the session.
- **Idle timeout**: Auto-shuts down after 1 hour (3600s) of inactivity. Configurable via `OMEGA_IDLE_TIMEOUT`.
- **Memory**: ~31MB at startup, ~337MB after first query (loads ONNX embedding model into RAM).
- **Tool count**: Up to 70 tools depending on installed extras (24 memory + 28 coordination + 10 router + 8 entity).

### Checking server status

```bash
# Verify MCP registration
omega doctor

# Check if the server process is running
ps aux | grep omega.server.mcp_server
```

## Router configuration (optional)

If you installed `omega-memory[router]`, configure API keys in `~/.omega/secrets.json`:

```json
{
  "anthropic_api_key": "<anthropic-api-key>",
  "openai_api_key": "<openai-api-key>",
  "google_api_key": "<google-api-key>",
  "groq_api_key": "<groq-api-key>",
  "xai_api_key": "<xai-api-key>"
}
```

!!! warning "Protect your secrets"
    `omega setup` creates `secrets.json` with `chmod 600` (owner read/write only). Never commit this file to version control.

You can also set API keys as environment variables:

```bash
export ANTHROPIC_API_KEY="<anthropic-api-key>"
export OPENAI_API_KEY="<openai-api-key>"
export GOOGLE_API_KEY="<google-api-key>"
export GROQ_API_KEY="<groq-api-key>"
export XAI_API_KEY="<xai-api-key>"
```

## Cloud sync configuration (optional)

If you installed `omega-memory[cloud]`, configure Supabase credentials:

```bash
omega cloud setup
```

This prompts for your Supabase URL and anon key, stored in `~/.omega/secrets.json`. Sync runs automatically:

- **Pull**: Once per day at session start
- **Push**: At session end after `sync_all`

## Auto-maintenance

OMEGA runs background maintenance tasks on a schedule, tracked by marker files in `~/.omega/`:

| Task | Cadence | Marker file | What it does |
|------|---------|-------------|--------------|
| Consolidate | 7 days | `last-consolidate` | Prunes stale low-value memories, caps session summaries |
| Compact | 14 days | `last-compact` | Merges similar memories into consolidated nodes |
| Backup | 7 days | `last-backup` | Exports full database to `~/.omega/backups/` |
| Doctor | 7 days | `last-doctor` | Runs health checks, logs warnings |
| Cloud pull | 1 day | `last-cloud-pull` | Syncs from Supabase (if configured) |
| Cloud push | per session | `last-cloud-push` | Syncs to Supabase at session end |

All maintenance runs at session start and is non-blocking.

## Next steps

- **[Quickstart](quickstart.md)** — Try OMEGA hands-on with your first memories.
- **[MCP Tools Reference](../reference/mcp-tools.md)** — Full documentation for all 70 MCP tools.
