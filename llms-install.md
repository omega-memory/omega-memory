# omega-memory

> Persistent memory for AI coding agents. Local-first, 25 MCP tools, zero cloud dependencies.

## OBJECTIVE

Install OMEGA as an MCP server so the AI coding agent has persistent memory across sessions.

## TODO

- [ ] Install the omega-memory Python package
- [ ] Run omega setup to configure MCP server and hooks
- [ ] Verify installation with omega doctor

## Step 1: Install

Requires Python 3.11+.

```bash
pip install omega-memory[server]
```

If `pip install` fails with a Python version error, try `pip3.11 install omega-memory[server]` or `python3.11 -m pip install omega-memory[server]`.

## Step 2: Setup

```bash
omega setup
```

This will:
1. Create `~/.omega/` directory
2. Download the ONNX embedding model (~90 MB)
3. Register omega-memory as an MCP server with the detected client (Claude Code, Cursor, Windsurf, Cline)
4. Install session hooks for auto-capture

For a specific client, pass `--client`:

```bash
omega setup --client claude-code
omega setup --client cursor
omega setup --client windsurf
omega setup --client cline
omega setup --client codex
```

## Step 3: Verify

```bash
omega doctor
```

Expected output:
```
OMEGA Doctor
  Python: 3.11.x
  Package: omega-memory 1.3.0
  Database: ~/.omega/omega.db (OK)
  Embedding model: bge-small-en-v1.5-onnx (OK)
  MCP server: registered (OK)
  Hooks: installed (OK)
All checks passed.
```

## DONE WHEN

- `omega doctor` reports all checks passed
- The MCP server is registered and the agent can call omega_store and omega_query tools

## EXECUTE NOW
