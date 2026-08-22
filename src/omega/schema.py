"""OMEGA schema initialization and migrations.

Extracted from sqlite_store.py to isolate the ~400-line schema setup
from the storage engine. All migrations and table definitions live here.
"""
import logging
import sqlite3
from typing import Tuple

logger = logging.getLogger("omega.schema")

SCHEMA_VERSION = 15


def init_schema(
    conn: sqlite3.Connection,
    vec_available: bool,
    embedding_dim: int,
) -> Tuple[bool, bool]:
    """Initialize or migrate the OMEGA schema.

    Creates all tables, indexes, triggers, and runs migrations.

    Args:
        conn: SQLite connection (must have sqlite-vec loaded if vec_available).
        vec_available: Whether sqlite-vec extension is loaded.
        embedding_dim: Embedding vector dimension (typically 384).

    Returns:
        (vec_available, fts_available) after schema init.
    """
    c = conn
    fts_available = False

    # ----------------------------------------------------------------
    # Schema version table
    # ----------------------------------------------------------------
    c.execute("""
        CREATE TABLE IF NOT EXISTS schema_version (
            version INTEGER NOT NULL
        )
    """)

    row = c.execute("SELECT version FROM schema_version LIMIT 1").fetchone()
    if row is None:
        c.execute("INSERT INTO schema_version (version) VALUES (?)", (SCHEMA_VERSION,))

    # ----------------------------------------------------------------
    # Migrations
    # ----------------------------------------------------------------

    # v1 -> v2: add priority and referenced_date columns
    if row and row[0] < 2:
        try:
            c.execute("ALTER TABLE memories ADD COLUMN priority INTEGER DEFAULT 3")
        except sqlite3.OperationalError:
            pass
        try:
            c.execute("ALTER TABLE memories ADD COLUMN referenced_date TEXT")
        except sqlite3.OperationalError:
            pass
        c.execute("CREATE INDEX IF NOT EXISTS idx_memories_priority ON memories(priority)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_memories_referenced_date ON memories(referenced_date)")
        c.execute("UPDATE schema_version SET version = 2")
        c.commit()
        logger.info("Schema migrated v1 -> v2: added priority, referenced_date columns")

    # v2 -> v3: add entity_id column
    current_version = c.execute("SELECT version FROM schema_version LIMIT 1").fetchone()
    if current_version and current_version[0] < 3:
        try:
            c.execute("ALTER TABLE memories ADD COLUMN entity_id TEXT")
        except sqlite3.OperationalError:
            pass
        c.execute("CREATE INDEX IF NOT EXISTS idx_memories_entity_id ON memories(entity_id)")
        c.execute("UPDATE schema_version SET version = 3")
        c.commit()
        logger.info("Schema migrated v2 -> v3: added entity_id column")

    # v3 -> v4: add agent_type column
    current_version = c.execute("SELECT version FROM schema_version LIMIT 1").fetchone()
    if current_version and current_version[0] < 4:
        try:
            c.execute("ALTER TABLE memories ADD COLUMN agent_type TEXT")
        except sqlite3.OperationalError:
            pass
        c.execute("CREATE INDEX IF NOT EXISTS idx_memories_agent_type ON memories(agent_type)")
        c.execute("UPDATE schema_version SET version = 4")
        c.commit()
        logger.info("Schema migrated v3 -> v4: added agent_type column")

    # v4 -> v5: add canonical_hash column (#6 Engram)
    current_version = c.execute("SELECT version FROM schema_version LIMIT 1").fetchone()
    if current_version and current_version[0] < 5:
        try:
            c.execute("ALTER TABLE memories ADD COLUMN canonical_hash TEXT")
        except sqlite3.OperationalError:
            pass
        c.execute("CREATE INDEX IF NOT EXISTS idx_memories_canonical_hash ON memories(canonical_hash)")
        c.execute("UPDATE schema_version SET version = 5")
        c.commit()
        logger.info("Schema migrated v4 -> v5: added canonical_hash column")

    # v5 -> v6: add forgetting_log table
    current_version = c.execute("SELECT version FROM schema_version LIMIT 1").fetchone()
    if current_version and current_version[0] < 6:
        c.execute("""
            CREATE TABLE IF NOT EXISTS forgetting_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                node_id TEXT NOT NULL,
                content_preview TEXT,
                event_type TEXT,
                reason TEXT NOT NULL,
                deleted_at TEXT NOT NULL,
                metadata TEXT
            )
        """)
        c.execute("CREATE INDEX IF NOT EXISTS idx_forgetting_log_deleted_at ON forgetting_log(deleted_at)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_forgetting_log_reason ON forgetting_log(reason)")
        c.execute("UPDATE schema_version SET version = 6")
        c.commit()
        logger.info("Schema migrated v5 -> v6: added forgetting_log table")

    # v6 -> v7: add entity_index table for Say/Do tracking
    current_version = c.execute("SELECT version FROM schema_version LIMIT 1").fetchone()
    if current_version and current_version[0] < 7:
        c.execute("""
            CREATE TABLE IF NOT EXISTS entity_index (
                entity_name TEXT NOT NULL PRIMARY KEY,
                entity_type TEXT DEFAULT 'person',
                statement_count INTEGER DEFAULT 0,
                outcome_count INTEGER DEFAULT 0,
                contradiction_score REAL DEFAULT 0.0,
                follow_through_rate REAL DEFAULT 0.0,
                first_seen TEXT NOT NULL,
                last_updated TEXT NOT NULL,
                metadata TEXT
            )
        """)
        c.execute("CREATE INDEX IF NOT EXISTS idx_entity_index_type ON entity_index(entity_type)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_entity_index_score ON entity_index(contradiction_score)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_entity_index_updated ON entity_index(last_updated)")
        c.execute("UPDATE schema_version SET version = 7")
        c.commit()
        logger.info("Schema migrated v6 -> v7: added entity_index table for Say/Do tracking")

    # v7 -> v8: add end_date, extracted_keywords; update FTS triggers
    current_version = c.execute("SELECT version FROM schema_version LIMIT 1").fetchone()
    if current_version and current_version[0] < 8:
        try:
            c.execute("ALTER TABLE memories ADD COLUMN end_date TEXT")
        except sqlite3.OperationalError:
            pass
        try:
            c.execute("ALTER TABLE memories ADD COLUMN extracted_keywords TEXT")
        except sqlite3.OperationalError:
            pass
        c.execute("CREATE INDEX IF NOT EXISTS idx_memories_end_date ON memories(end_date)")
        c.execute("DROP TRIGGER IF EXISTS memories_ai")
        c.execute("DROP TRIGGER IF EXISTS memories_ad")
        c.execute("DROP TRIGGER IF EXISTS memories_au")
        c.execute("UPDATE schema_version SET version = 8")
        c.commit()
        logger.info("Schema migrated v7 -> v8: added end_date, extracted_keywords columns")

    # v8 -> v9: add retrieval_count column
    current_version = c.execute("SELECT version FROM schema_version LIMIT 1").fetchone()
    if current_version and current_version[0] < 9:
        try:
            c.execute("ALTER TABLE memories ADD COLUMN retrieval_count INTEGER DEFAULT 0")
        except sqlite3.OperationalError:
            pass
        c.execute("UPDATE schema_version SET version = 9")
        c.commit()
        logger.info("Schema migrated v8 -> v9: added retrieval_count column")

    # v9 -> v10: add maintenance_dlq table
    current_version = c.execute("SELECT version FROM schema_version LIMIT 1").fetchone()
    if current_version and current_version[0] < 10:
        c.execute("""
            CREATE TABLE IF NOT EXISTS maintenance_dlq (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                stage_name TEXT NOT NULL,
                error_class TEXT NOT NULL DEFAULT 'transient',
                error_message TEXT,
                remediation_attempts INTEGER DEFAULT 0,
                max_remediation INTEGER DEFAULT 3,
                status TEXT NOT NULL DEFAULT 'pending',
                next_retry_at TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
        """)
        c.execute("CREATE INDEX IF NOT EXISTS idx_maintenance_dlq_status ON maintenance_dlq(status)")
        c.execute("UPDATE schema_version SET version = 10")
        c.commit()
        logger.info("Schema migrated v9 -> v10: added maintenance_dlq table")

    # v10 -> v11: add memory_type column for cognitive classification
    current_version = c.execute("SELECT version FROM schema_version LIMIT 1").fetchone()
    if current_version and current_version[0] < 11:
        try:
            c.execute("ALTER TABLE memories ADD COLUMN memory_type TEXT DEFAULT 'semantic'")
        except sqlite3.OperationalError:
            pass
        c.execute("CREATE INDEX IF NOT EXISTS idx_memories_memory_type ON memories(memory_type)")
        # Backfill existing memories
        c.execute("""
            UPDATE memories SET memory_type = 'episodic'
            WHERE event_type IN (
                'session_summary', 'task_completion', 'coordination_snapshot',
                'session_respawn', 'merge_claim', 'merge_release',
                'file_claimed', 'file_released', 'branch_claimed',
                'branch_released', 'code_chunk', 'file_summary', 'file_conflict'
            )
        """)
        c.execute("""
            UPDATE memories SET memory_type = 'procedural'
            WHERE event_type IN (
                'lesson_learned', 'reflexion', 'self_reflection',
                'outcome_evaluation', 'reminder'
            )
        """)
        c.execute("UPDATE schema_version SET version = 11")
        c.commit()
        logger.info("Schema migrated v10 -> v11: added memory_type column with backfill")

    # v11 -> v12: add bi-temporal columns (valid_from, valid_until)
    current_version = c.execute("SELECT version FROM schema_version LIMIT 1").fetchone()
    if current_version and current_version[0] < 12:
        try:
            c.execute("ALTER TABLE memories ADD COLUMN valid_from TEXT")
        except sqlite3.OperationalError:
            pass
        try:
            c.execute("ALTER TABLE memories ADD COLUMN valid_until TEXT")
        except sqlite3.OperationalError:
            pass
        c.execute("UPDATE memories SET valid_from = COALESCE(referenced_date, created_at)")
        c.execute("""
            UPDATE memories SET valid_until = json_extract(metadata, '$.superseded_at')
            WHERE json_extract(metadata, '$.superseded') = 1
            AND json_extract(metadata, '$.superseded_at') IS NOT NULL
        """)
        c.execute("CREATE INDEX IF NOT EXISTS idx_memories_valid_from ON memories(valid_from)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_memories_valid_until ON memories(valid_until)")
        c.execute("UPDATE schema_version SET version = 12")
        c.commit()
        logger.info("Schema migrated v11 -> v12: added bi-temporal columns with backfill")

    # v12 -> v13: unique index on forgetting_log to prevent multi-process duplicate entries
    current_version = c.execute("SELECT version FROM schema_version LIMIT 1").fetchone()
    if current_version and current_version[0] < 13:
        # Clean pre-existing duplicates before creating unique index
        c.execute("""DELETE FROM forgetting_log
                     WHERE rowid NOT IN (
                         SELECT MIN(rowid) FROM forgetting_log GROUP BY node_id, reason
                     )""")
        c.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_forgetting_log_node_reason ON forgetting_log(node_id, reason)")
        c.execute("UPDATE schema_version SET version = 13")
        c.commit()
        logger.info("Schema migrated v12 -> v13: added unique index on forgetting_log(node_id, reason)")

    # v13 -> v14: add derived_from, source_uri, status columns (context graph)
    current_version = c.execute("SELECT version FROM schema_version LIMIT 1").fetchone()
    if current_version and current_version[0] < 14:
        try:
            c.execute("ALTER TABLE memories ADD COLUMN derived_from TEXT")
        except sqlite3.OperationalError:
            pass
        try:
            c.execute("ALTER TABLE memories ADD COLUMN source_uri TEXT")
        except sqlite3.OperationalError:
            pass
        try:
            c.execute("ALTER TABLE memories ADD COLUMN status TEXT DEFAULT 'active'")
        except sqlite3.OperationalError:
            pass
        c.execute("CREATE INDEX IF NOT EXISTS idx_memories_derived_from ON memories(derived_from)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_memories_source_uri ON memories(source_uri)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_memories_status ON memories(status)")
        # Backfill status from metadata.superseded
        c.execute("""
            UPDATE memories SET status = 'superseded'
            WHERE json_extract(metadata, '$.superseded') = 1
            AND status IS NULL OR status = 'active'
        """)
        c.execute("UPDATE schema_version SET version = 14")
        c.commit()
        logger.info("Schema migrated v13 -> v14: added derived_from, source_uri, status columns")

    # v14 -> v15: add an edit timestamp independent from access tracking
    current_version = c.execute("SELECT version FROM schema_version LIMIT 1").fetchone()
    if current_version and current_version[0] < 15:
        try:
            c.execute("ALTER TABLE memories ADD COLUMN updated_at TEXT")
        except sqlite3.OperationalError:
            pass
        c.execute(
            "UPDATE memories SET updated_at = created_at WHERE updated_at IS NULL"
        )
        c.execute("UPDATE schema_version SET version = 15")
        c.commit()
        logger.info("Schema migrated v14 -> v15: added memory updated_at column")

    # ----------------------------------------------------------------
    # Table definitions (idempotent CREATE IF NOT EXISTS)
    # ----------------------------------------------------------------

    c.execute("""
        CREATE TABLE IF NOT EXISTS memories (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            node_id TEXT UNIQUE NOT NULL,
            content TEXT NOT NULL,
            metadata TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            last_accessed TEXT,
            access_count INTEGER DEFAULT 0,
            ttl_seconds INTEGER,
            session_id TEXT,
            event_type TEXT,
            project TEXT,
            content_hash TEXT,
            priority INTEGER DEFAULT 3,
            referenced_date TEXT,
            entity_id TEXT,
            agent_type TEXT,
            canonical_hash TEXT,
            end_date TEXT,
            extracted_keywords TEXT,
            retrieval_count INTEGER DEFAULT 0,
            memory_type TEXT DEFAULT 'semantic',
            valid_from TEXT,
            valid_until TEXT,
            derived_from TEXT,
            source_uri TEXT,
            status TEXT DEFAULT 'active'
        )
    """)

    # Indexes
    for col in (
        "node_id",
        "event_type",
        "session_id",
        "project",
        "created_at",
        "content_hash",
        "priority",
        "referenced_date",
        "entity_id",
        "agent_type",
        "canonical_hash",
        "end_date",
        "last_accessed",
        "ttl_seconds",
        "memory_type",
        "valid_from",
        "valid_until",
        "derived_from",
        "source_uri",
        "status",
    ):
        # SECURITY: col from hardcoded tuple above, not user input
        c.execute(f"""
            CREATE INDEX IF NOT EXISTS idx_memories_{col}
            ON memories({col})
        """)

    # Compound indexes for frequent query patterns
    c.execute("""
        CREATE INDEX IF NOT EXISTS idx_memories_event_access
        ON memories(event_type, access_count)
    """)

    # sqlite-vec virtual table
    if vec_available:
        try:
            c.execute(f"""
                CREATE VIRTUAL TABLE IF NOT EXISTS memories_vec
                USING vec0(embedding float[{embedding_dim}] distance_metric=cosine)
            """)
        except Exception as e:
            logger.warning(f"Failed to create vec table: {e}")
            vec_available = False

    # FTS5 full-text search index
    try:
        c.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts
            USING fts5(content, content='memories', content_rowid='id')
        """)
        fts_available = True
    except Exception as e:
        logger.debug(f"FTS5 not available: {e}")
        fts_available = False

    # Edges table (temporal, causal)
    c.execute("""
        CREATE TABLE IF NOT EXISTS edges (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source_id TEXT NOT NULL,
            target_id TEXT NOT NULL,
            edge_type TEXT NOT NULL,
            weight REAL DEFAULT 1.0,
            metadata TEXT,
            created_at TEXT NOT NULL DEFAULT (datetime('now')),
            UNIQUE(source_id, target_id, edge_type)
        )
    """)
    for col in ("source_id", "target_id", "edge_type"):
        c.execute(f"""
            CREATE INDEX IF NOT EXISTS idx_edges_{col}
            ON edges({col})
        """)

    # Forgetting audit log
    c.execute("""
        CREATE TABLE IF NOT EXISTS forgetting_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            node_id TEXT NOT NULL,
            content_preview TEXT,
            event_type TEXT,
            reason TEXT NOT NULL,
            deleted_at TEXT NOT NULL,
            metadata TEXT
        )
    """)
    c.execute("CREATE INDEX IF NOT EXISTS idx_forgetting_log_deleted_at ON forgetting_log(deleted_at)")
    c.execute("CREATE INDEX IF NOT EXISTS idx_forgetting_log_reason ON forgetting_log(reason)")

    # Cloud delete queue
    c.execute("""
        CREATE TABLE IF NOT EXISTS cloud_delete_queue (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            local_id INTEGER NOT NULL,
            deleted_at TEXT NOT NULL
        )
    """)

    # Entity index table for Say/Do contradiction tracking
    c.execute("""
        CREATE TABLE IF NOT EXISTS entity_index (
            entity_name TEXT NOT NULL PRIMARY KEY,
            entity_type TEXT DEFAULT 'person',
            statement_count INTEGER DEFAULT 0,
            outcome_count INTEGER DEFAULT 0,
            contradiction_score REAL DEFAULT 0.0,
            follow_through_rate REAL DEFAULT 0.0,
            first_seen TEXT NOT NULL DEFAULT (datetime('now')),
            last_updated TEXT NOT NULL DEFAULT (datetime('now')),
            metadata TEXT
        )
    """)
    c.execute("CREATE INDEX IF NOT EXISTS idx_entity_index_type ON entity_index(entity_type)")
    c.execute("CREATE INDEX IF NOT EXISTS idx_entity_index_score ON entity_index(contradiction_score)")
    c.execute("CREATE INDEX IF NOT EXISTS idx_entity_index_updated ON entity_index(last_updated)")

    # FTS5 sync triggers (include extracted_keywords for enriched BM25)
    if fts_available:
        try:
            c.execute("""
                CREATE TRIGGER IF NOT EXISTS memories_ai AFTER INSERT ON memories BEGIN
                    INSERT INTO memories_fts(rowid, content)
                    VALUES (new.id, new.content || ' ' || COALESCE(new.extracted_keywords, ''));
                END
            """)
            c.execute("""
                CREATE TRIGGER IF NOT EXISTS memories_ad AFTER DELETE ON memories BEGIN
                    INSERT INTO memories_fts(memories_fts, rowid, content)
                    VALUES ('delete', old.id, old.content || ' ' || COALESCE(old.extracted_keywords, ''));
                END
            """)
            c.execute("""
                CREATE TRIGGER IF NOT EXISTS memories_au AFTER UPDATE OF content ON memories BEGIN
                    INSERT INTO memories_fts(memories_fts, rowid, content)
                    VALUES ('delete', old.id, old.content || ' ' || COALESCE(old.extracted_keywords, ''));
                    INSERT INTO memories_fts(rowid, content)
                    VALUES (new.id, new.content || ' ' || COALESCE(new.extracted_keywords, ''));
                END
            """)
            # Populate FTS from existing data if empty
            fts_count = c.execute("SELECT COUNT(*) FROM memories_fts").fetchone()[0]
            mem_count = c.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
            if fts_count == 0 and mem_count > 0:
                c.execute(
                    "INSERT INTO memories_fts(rowid, content) "
                    "SELECT id, content || ' ' || COALESCE(extracted_keywords, '') FROM memories"
                )
                logger.info(f"Populated FTS5 index with {mem_count} existing memories")
        except Exception as e:
            logger.debug(f"FTS5 trigger setup failed: {e}")
            fts_available = False

    # Memory clusters table (pattern learner)
    c.execute("""
        CREATE TABLE IF NOT EXISTS memory_clusters (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            cluster_id INTEGER NOT NULL,
            label TEXT NOT NULL,
            member_count INTEGER NOT NULL,
            centroid BLOB,
            representative_keywords TEXT,
            representative_memory_ids TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            superseded INTEGER DEFAULT 0
        )
    """)
    c.execute("CREATE INDEX IF NOT EXISTS idx_memory_clusters_superseded ON memory_clusters(superseded)")
    c.execute("CREATE INDEX IF NOT EXISTS idx_memory_clusters_cluster_id ON memory_clusters(cluster_id)")

    # Thompson sampling arms table
    c.execute("""
        CREATE TABLE IF NOT EXISTS thompson_arms (
            arm_id TEXT PRIMARY KEY,
            arm_type TEXT NOT NULL,
            alpha REAL DEFAULT 1.0,
            beta REAL DEFAULT 1.0,
            total_trials INTEGER DEFAULT 0,
            total_successes INTEGER DEFAULT 0,
            last_updated TEXT NOT NULL,
            context TEXT
        )
    """)
    c.execute("CREATE INDEX IF NOT EXISTS idx_thompson_arms_type ON thompson_arms(arm_type)")

    c.commit()

    return vec_available, fts_available
