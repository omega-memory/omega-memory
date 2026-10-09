# Architecture

---

## System Overview

```
               +---------------------+
               |    Claude Code       |
               |  (or any MCP host)   |
               +----------+----------+
                          | stdio/MCP
               +----------v----------+
               |   OMEGA MCP Server   |
               |   70 tools total     |
               +--+---------------+---+
                  |               |
         +--------v--+      +----v---+
         | Core       |      | Router |
         | (memory +  |      | (opt)  |
         |  coord)    |      |        |
         +-----+------+      +---+----+
               |                 |
               v                 v
         +--------------------------------------+
         |         omega.db (SQLite)             |
         |  memories | edges | coord_* tables    |
         +--------------------------------------+
```

Single database, modular handlers. The router is an optional MCP tool set that registers into the same server process. No separate daemons, no microservices.

The MCP server communicates over stdio. Claude Code spawns the process on demand with a 3600-second idle timeout.

---

## Search Pipeline

Query processing follows five stages:

### Stage 1: Vector Similarity

Embed the query with bge-small-en-v1.5 (384-dim) and find nearest neighbors via `sqlite-vec` using cosine distance. Returns candidate set with raw vector scores.

### Stage 2: Full-Text Search

Run the query through FTS5 with BM25 scoring. This catches keyword matches that may be distant in embedding space.

### Stage 3: Blended Ranking

Combine the two result sets: **70% vector score, 30% text score**. Candidates appearing in both lists get the blended score; single-source candidates keep their weighted score.

### Stage 4: Type-Weighted Scoring

Apply event-type weights. Decisions and lessons are weighted 2x; session summaries are weighted lower. Priority field (1-5) further adjusts scores.

### Stage 5: Contextual Re-Ranking

Final adjustments based on context signals:

- **Tag and project overlap**: boost memories matching the query's tags or project
- **Word overlap**: Jaccard similarity on significant words (Phase 2.5 boost, capped at 50%)
- **Lightweight stemming**: `_word_overlap` normalizes word forms before comparison
- **Feedback dampening**: memories rated "unhelpful" or "outdated" are penalized
- **Temporal hard penalty**: very old, unaccessed memories get a 0.05x multiplier
- **Abstention floor**: results below 0.35 (vector) or 0.5 (text) are discarded

---

## Memory Lifecycle

### Ingestion

Content arrives via `auto_capture()` -- the primary ingestion function.

1. Unicode normalization (NFC)
2. Blocklist check (system noise patterns)
3. Auto-tag extraction (language, tools, project)
4. Embedding generation (bge-small-en-v1.5 ONNX)
5. Store with metadata, timestamps, TTL, priority

### Deduplication

Three layers prevent redundant storage:

- **Exact**: SHA256 content hash match
- **Semantic**: embedding cosine similarity >= 0.85
- **Per-type Jaccard**: word-level Jaccard similarity threshold per event type

### Evolution

When incoming content is similar (55-95%) to an existing memory, OMEGA appends new insights to the existing memory rather than creating a duplicate. This implements Zettelkasten-style knowledge growth.

### Auto-Relate

After storing, `_auto_relate` creates edges to the top 3 most similar existing memories (similarity >= 0.45). Edge types include `related`, `evolved_from`, `superseded_by`.

### TTLs

| Event Type | TTL |
|------------|-----|
| `session_summary` | 1 day (86400s) |
| `checkpoint` | 7 days (604800s) |
| `lesson_learned`, `user_preference` | Permanent |
| `decision` | Permanent |
| `error_pattern`, `task_completion` | Permanent |

### Compaction

`compact()` clusters related memories by Jaccard similarity, creates summary nodes, and marks originals as superseded. Reduces noise while preserving knowledge.

---

## Embedding System

| Property | Value |
|----------|-------|
| Primary model | bge-small-en-v1.5 (384 dimensions) |
| Fallback model | all-MiniLM-L6-v2 (384 dimensions) |
| Runtime | ONNX Runtime, CPU-only |
| CoreML | Disabled (memory leak in Apple's ANE runtime) |
| Model location | `~/.cache/omega/models/{model}-onnx/` |
| RAM after first query | ~337 MB |
| Circuit breaker | Fails open after transient ONNX errors |

Embeddings are generated in `embedding.py`. The ONNX session is initialized lazily on first use and cached for the process lifetime.

---

## Graph Structure

Memories form a directed graph:

- **Nodes**: each memory is a node with `node_id`, content, embedding, metadata
- **Edges**: directed relationships between nodes

Edge types:

| Type | Meaning |
|------|---------|
| `related` | Semantically similar content |
| `evolved_from` | New version of an older memory |
| `superseded_by` | Original replaced by compacted summary |

Edges carry a `weight` (0.0 to 1.0) reflecting relationship strength. Traversal uses BFS with a configurable maximum of 5 hops and optional minimum weight filter.

---

## Hook System

Hooks connect Claude Code lifecycle events to OMEGA processing.

### Architecture

```
Claude Code hook event
        |
        v
  fast_hook.py (thin dispatcher)
        |
        v (Unix Domain Socket)
  hook_server.py (daemon, 12 handlers)
        |
        v
  bridge.py / coordination.py
```

- **Dispatch latency**: ~5ms via UDS (vs ~750ms cold Python start)
- **Fail-open**: if the daemon is unreachable, hooks return success and agent work continues unblocked
- **Daemon timeout**: pre_push_guard has a 20-second timeout; all others are fast

### Hook Handlers

| # | Hook | Matcher | Batch | Purpose |
|---|------|---------|-------|---------|
| 1 | SessionStart | all | session_start + plugin hooks | Welcome + register + git sync + session resume |
| 2 | Stop | all | session_stop + plugin hooks | Summary + deregister + release claims |
| 3 | UserPromptSubmit | all | auto_capture | Lesson/decision auto-capture |
| 4 | PostToolUse | Edit, Write, NotebookEdit | surface_memories + coord_heartbeat + auto_claim_file | Surface context + heartbeat + file claim |
| 5 | PostToolUse | Bash, Read | surface_memories + coord_heartbeat | Surface context + heartbeat |
| 6 | PreToolUse | Bash | pre_push_guard | Git push divergence + branch claim guard |
| 7 | PreToolUse | Edit, Write, NotebookEdit | pre_file_guard + pre_task_guard | File claim guard + task guard |

---

## Database Schema

All data lives in a single SQLite database at `~/.omega/omega.db`.

### Core Tables

```sql
-- Memory storage
CREATE TABLE memories (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    node_id         TEXT UNIQUE NOT NULL,
    content         TEXT NOT NULL,
    metadata        TEXT,              -- JSON blob
    created_at      TEXT NOT NULL,
    last_accessed   TEXT,
    access_count    INTEGER DEFAULT 0,
    ttl_seconds     INTEGER,
    session_id      TEXT,
    event_type      TEXT,
    project         TEXT,
    content_hash    TEXT,              -- SHA256 for dedup
    priority        INTEGER DEFAULT 3, -- 1 (low) to 5 (high)
    referenced_date TEXT,              -- temporal anchoring
    entity_id       TEXT               -- entity scope
);

-- Relationship graph
CREATE TABLE edges (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    source_id   TEXT NOT NULL,
    target_id   TEXT NOT NULL,
    edge_type   TEXT NOT NULL,
    weight      REAL DEFAULT 1.0,
    metadata    TEXT,
    created_at  TEXT NOT NULL DEFAULT (datetime('now')),
    UNIQUE(source_id, target_id, edge_type)
);

-- Vector similarity (sqlite-vec extension)
CREATE VIRTUAL TABLE memories_vec
    USING vec0(embedding float[384] distance_metric=cosine);

-- Full-text search (FTS5)
CREATE VIRTUAL TABLE memories_fts
    USING fts5(content, content='memories', content_rowid='id');
```

### Coordination Tables

```sql
CREATE TABLE coord_sessions (...);       -- Active agent sessions
CREATE TABLE coord_file_claims (...);    -- Exclusive file locks
CREATE TABLE coord_branch_claims (...);  -- Exclusive branch locks
CREATE TABLE coord_intents (...);        -- Announced work plans
CREATE TABLE coord_snapshots (...);      -- Session state snapshots
CREATE TABLE coord_tasks (...);          -- Coordination tasks
CREATE TABLE coord_task_deps (...);      -- Task dependency graph
CREATE TABLE coord_audit (...);          -- Audit log of all coord operations
CREATE TABLE coord_git_events (...);     -- Tracked git push/merge events
CREATE TABLE coord_messages (...);       -- Inter-agent messages
```

### Schema Versioning

The `schema_version` table tracks migrations:

- **v1**: Initial schema (memories, edges, FTS5, vec)
- **v2**: Added `priority` and `referenced_date` columns
- **v3**: Added `entity_id` column for multi-entity support
