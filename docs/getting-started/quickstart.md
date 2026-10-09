---
title: Quickstart
description: Store your first memory and see OMEGA in action
---

# Quickstart

## 60-second demo

OMEGA works through natural conversation with Claude Code. No special commands needed.

**Session 1** — Tell Claude something worth remembering:

> "Remember that our API rate limit is 100 requests per minute per user."

Claude stores this as a memory via OMEGA. Close the session.

**Session 2** — Ask about it later:

> "What's our API rate limit?"

Claude surfaces the memory automatically:

> "Based on a previous decision, your API rate limit is 100 requests per minute per user."

That's it. The memory persisted across sessions with zero manual work.

## Storing memories

OMEGA captures memories in two ways: **explicitly** (you tell it) and **automatically** (hooks detect patterns).

### Explicit storage

Use natural language with Claude Code:

=== "Decisions"

    > "Remember that authentication uses JWT tokens with RS256 signing."

    Stored as a `decision` — high priority, surfaces when auth topics come up.

=== "Lessons learned"

    > "Remember: never use `git add .` in this repo — it picks up generated files."

    Stored as a `lesson_learned` — surfaces when similar patterns are detected.

=== "Preferences"

    > "Remember I prefer tabs over spaces and 120-character line width."

    Stored as a `user_preference` — surfaces during code formatting discussions.

### Automatic capture

OMEGA's hooks watch your conversations and auto-capture:

- **Decisions** — When Claude detects language like "let's go with X" or "the approach is Y"
- **Lessons** — When debugging sessions resolve with an insight
- **Session summaries** — Created automatically when a session ends

!!! tip "You don't need to say 'remember'"
    The auto-capture hook (`UserPromptSubmit`) detects decision and lesson patterns in your conversation. Explicit "remember" commands are for things the hooks might miss.

## Querying memories

### In conversation

Just ask Claude naturally:

> "What did we decide about the database schema?"
> "Have I seen this error before?"
> "What's the architecture of the auth system?"

OMEGA uses semantic search — you don't need to remember exact wording. A query about "authentication approach" will find a memory stored as "auth uses JWT tokens."

### From the CLI

```bash
# Semantic search across all memories
omega query "authentication"

# Store a memory directly
omega store "API rate limit is 100 req/min" --type decision

# See what was captured recently
omega timeline --days 7

# Memory statistics
omega stats
```

## What happens automatically

OMEGA's 7 hook processes run in the background during every Claude Code session. Here's what they do without any action from you:

### Session start

When you open Claude Code, OMEGA delivers a welcome briefing:

```
## Welcome back! OMEGA ready — 254 memories | my-project | main
[CONTEXT] Recent: deployed v2.1, fixed auth bug, added rate limiting
[TODO] Next: implement webhook retry logic
```

### Memory surfacing

When you edit or read files, OMEGA surfaces relevant memories:

```
[MEMORY] (2 days ago) Decision: webhook payloads use HMAC-SHA256 signatures
[MEMORY] (5 days ago) Lesson: retry logic needs exponential backoff with jitter
```

### File claims (multi-agent)

When you edit a file, OMEGA automatically claims it so other agents know not to touch it. When another agent already owns a file, you'll see a warning before the edit proceeds.

### Session end

When you close the session, OMEGA captures a summary of what was accomplished and releases all file and branch claims.

Multi-agent coordination across sessions (file claims, branch ownership, task assignment) is available in [OMEGA Pro](https://omegamax.co/pro).

## Next steps

- **[Configuration](configuration.md)** — Customize storage paths, hooks, and environment variables.
- **[MCP Tools Reference](../reference/mcp-tools.md)** — Every MCP tool with its parameters.
