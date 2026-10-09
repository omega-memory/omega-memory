---
title: Installation
description: Install OMEGA and set up persistent memory for Claude Code
---

# Installation

## Install from PyPI

=== "Core"

    ```bash
    pip install "omega-memory[server]"
    ```

    Installs the memory engine and the MCP server. This is everything most users need. Keep the quotes: zsh treats bare square brackets as a pattern.

=== "With encryption"

    ```bash
    pip install "omega-memory[encrypt]"
    ```

    Adds encrypted local storage with macOS Keychain integration.

=== "Everything"

    ```bash
    pip install "omega-memory[full]"
    ```

    Installs every optional extra: `server` and `encrypt`.

Multi-agent coordination, LLM routing, entities, the knowledge base, encrypted profiles and cloud sync are available in [OMEGA Pro](https://omegamax.co/pro).

## Install from source

```bash
git clone https://github.com/omega-memory/omega-memory.git
cd omega-memory
pip install -e ".[dev]"
omega setup
```

The `[dev]` extra includes test dependencies (pytest, ruff, etc.) in addition to core functionality.

## Requirements

| Requirement | Details |
|-------------|---------|
| **Python** | 3.11 or higher |
| **Disk** | ~90MB for the BGE-Small ONNX embedding model |
| **RAM** | ~31MB at startup, ~337MB after first query (CPU-only ONNX inference) |
| **OS** | macOS, Linux (Windows untested) |
| **Claude Code** | Required for MCP integration and hooks |

## Run setup

After installing, run the setup wizard:

```bash
omega setup
```

This performs 5 steps:

1. **Creates `~/.omega/`** — The storage directory for your database, profile, secrets, and logs.
2. **Downloads the ONNX embedding model** — Fetches `bge-small-en-v1.5` (~90MB) to `~/.cache/omega/models/` for local semantic search. No API calls needed.
3. **Registers the MCP server** — Adds an `omega-memory` entry to `~/.claude.json` so Claude Code can spawn OMEGA on demand via stdio transport.
4. **Installs hooks** — Adds 7 hook entries to `~/.claude/settings.json` for automatic memory capture, surfacing, coordination, and guard rails.
5. **Updates CLAUDE.md** — Adds a managed `<!-- OMEGA:BEGIN -->` block to `~/.claude/CLAUDE.md` with instructions for using memory and coordination tools.

!!! tip "Setup is idempotent"
    You can run `omega setup` multiple times safely. It will update existing configuration without duplicating entries.

## Verify the installation

```bash
omega doctor
```

This checks:

- Python version and OMEGA package version
- SQLite database exists and is accessible
- Embedding model is downloaded and loadable
- MCP server entry is registered in `~/.claude.json`
- Hooks are installed in `~/.claude/settings.json`
- CLAUDE.md contains the OMEGA block

Example output:

```
OMEGA Doctor — v0.6.1
─────────────────────
[OK] Python 3.12.4
[OK] Database: ~/.omega/omega.db (254 memories)
[OK] Embedding model: bge-small-en-v1.5-onnx
[OK] MCP server registered in ~/.claude.json
[OK] 7 hooks installed in ~/.claude/settings.json
[OK] CLAUDE.md has OMEGA block

All checks passed.
```

!!! warning "If doctor reports issues"
    Run `omega setup` again to repair missing configuration. If the embedding model download fails, check your internet connection — the model is fetched once from Hugging Face and cached locally.

## Uninstalling

To remove OMEGA completely:

```bash
omega setup --uninstall   # Removes hooks, MCP entry, and CLAUDE.md block
pip uninstall omega-memory
rm -rf ~/.omega            # Delete all stored memories (irreversible)
rm -rf ~/.cache/omega      # Delete cached embedding models
```

## Next steps

- **[Quickstart](quickstart.md)** — Store your first memory and see it come back in a new session.
- **[Configuration](configuration.md)** — Customize storage paths, hooks, and environment variables.
