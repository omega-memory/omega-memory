---
name: omega-memory
description: "Persistent cross-session memory for AI coding agents via OMEGA's MCP server. Store decisions and lessons, query past context with semantic search, checkpoint and resume incomplete tasks, and manage reminders and profiles. Use when the user mentions OMEGA, persistent memory, remembering context, recalling past sessions, picking up where they left off, or MCP memory tools."
compatibility: "Python 3.11+, Claude Code, Cursor, Windsurf, Zed, Cline, Codex"
metadata:
  version: "1.0.0"
  requires_binaries: "python3, pip3"
---

# OMEGA Memory

Persistent memory for AI coding agents. Your agent remembers decisions, learns from mistakes, and picks up where it left off.

## Installation

```bash
pip3 install omega-memory[server]
omega setup
```

The `omega setup` command auto-configures your MCP client (Claude Code, Cursor, Windsurf, or Zed). No API keys needed — runs fully local with CPU-only embeddings.

## Session Workflow

Follow this sequence every session:

1. **Start** — Call `omega_welcome()` to get a context briefing with recent memories, active reminders, and profile
2. **Protocol** — Call `omega_protocol()` to retrieve operating rules and behavioral guidelines
3. **Follow** — Apply the protocol instructions returned
4. **Query** — Before non-trivial tasks, call `omega_query("prior decisions about [area]")` to check for existing context
5. **Work** — Complete the task using surfaced context
6. **Store** — Save key outcomes: `omega_store("Chose PostgreSQL over SQLite for multi-user support", "decision")`
7. **Checkpoint** — If the task is incomplete, call `omega_checkpoint()` to save state for the next session

## MCP Tools

OMEGA exposes 12 tools via MCP. The most common are shown with example calls.

### Core Tools

**`omega_store(content, event_type)`** — Save a typed memory.

```
omega_store("Auth system uses JWT tokens, not session cookies", "decision")
omega_store("User prefers Tailwind over styled-components", "user_preference")
omega_store("ConnectionResetError on large uploads — fixed by chunking to 5MB", "lesson_learned")
```

**`omega_query(query, mode?)`** — Search memories by meaning.

```
omega_query("how did we handle authentication?")
omega_query("pytest", mode="phrase")
omega_query("what happened this week?", mode="timeline", days=7)
```

**`omega_welcome(project?)`** — Session start briefing with recent context and active reminders.

**`omega_checkpoint()`** — Save current task state. The next `omega_welcome()` restores it.

**`omega_resume_task(task_id)`** — Resume a previously checkpointed task with full context.

### Additional Tools

| Tool | Purpose |
|------|---------|
| `omega_protocol` | Retrieve operating rules and behavioral guidelines |
| `omega_lessons` | Cross-session lessons ranked by access count |
| `omega_profile` | Read or update the user profile |
| `omega_memory` | Manage a specific memory (edit, delete, feedback, similar, traverse) |
| `omega_remind` | Set, list, or dismiss time-based reminders |
| `omega_maintain` | System housekeeping (health, consolidate, compact, backup, restore) |
| `omega_stats` | Analytics: type breakdown, session stats, weekly digest, access rates |

## Best Practices

- **Always start with `omega_welcome()`** — it loads critical context and prevents re-explaining
- **Query before acting** — check if a decision was already made before proposing a new one
- **Store decisions with reasoning** — "chose X because Y" is far more useful than just "chose X"
- **Use specific event types** — they control TTL and deduplication (decisions last 90 days, checkpoints 7 days)
- **Don't over-store** — skip raw tool output, build logs, and anything shorter than a sentence

## Links

- **PyPI:** [omega-memory](https://pypi.org/project/omega-memory/)
- **GitHub:** [omega-memory/omega-memory](https://github.com/omega-memory/omega-memory)
- **Website:** [omegamax.co](https://omegamax.co)
