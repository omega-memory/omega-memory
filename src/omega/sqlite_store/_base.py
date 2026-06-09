"""SQLiteStore base class -- lifecycle, schema, and configuration."""

import logging
import os
import sqlite3
import threading
import time as _time
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from omega import json_compat as json
from omega.db_utils import retry_on_locked as _retry_on_locked
from omega.schema import SCHEMA_VERSION  # noqa: F401 -- re-exported
from omega.schema import init_schema as _init_schema_fn

from ._types import (
    EMBEDDING_DIM,
    MemoryResult,
    _HOT_CACHE_SIZE,
)

logger = logging.getLogger("omega.sqlite_store")


# Free-tier capacity gates. Module-level so a runtime monkey-patch of a
# SQLiteStore instance attribute cannot raise the cap; the property on
# SQLiteStoreBase always re-reads these unless an explicit override field
# is set (tests only).
#
# Pro path: when omega_platform.license.is_pro() returns True the
# OMEGA_MAX_NODES env var is honored (default 50000). Pro users keep the
# existing env-tunable behavior.
#
# Free path: the env var is IGNORED. The hard write cap is _CORE_HARD_LIMIT
# (5000). Per-instance grandfathering on the property lifts the ceiling to
# the current count for installs that already exceed 5K, so existing Free
# users do not suddenly hit a write wall after upgrading omega-memory.
_CORE_HARD_LIMIT = 5000


def _get_effective_max_nodes() -> int:
    """Hard write cap for this process.

    Pro: ``OMEGA_MAX_NODES`` env (default 50000). Free: ``_CORE_HARD_LIMIT``;
    the env var is ignored so a free user cannot set ``OMEGA_MAX_NODES=99999``
    to bypass the cap.
    """
    try:
        from omega_platform.license import is_pro

        if is_pro():
            return int(os.environ.get("OMEGA_MAX_NODES", "50000"))
    except ImportError:
        pass
    except Exception as e:
        logger.debug("is_pro() check failed in _get_effective_max_nodes: %s", e)
    return _CORE_HARD_LIMIT


class SQLiteStoreBase:
    """SQLite-backed memory store with sqlite-vec for vector search.

    Drop-in replacement for OmegaMemory in bridge.py. All data lives on disk
    in a single SQLite database file.
    """

    # Type weights for query scoring (same as OmegaMemory)
    _TYPE_WEIGHTS = {
        "constraint": 3.0,
        "checkpoint": 2.5,
        "reminder": 3.0,
        "decision": 2.0,
        "lesson_learned": 2.0,
        "error_pattern": 2.0,
        "user_preference": 2.0,
        "task_completion": 1.4,
        "reflexion": 1.3,
        "outcome_evaluation": 1.3,
        "self_reflection": 1.3,
        "session_summary": 1.2,
        "preference_generated": 1.1,
        "advisor_action_outcome": 1.1,
        "advisor_insight": 1.5,
        "sota_research": 1.4,
        "research_report": 1.3,
        "benchmark_update": 1.3,
        "sota_scan": 1.1,
        "file_conflict": 1.0,
        "merge_claim": 0.8,
        "merge_release": 0.8,
        "file_claimed": 0.7,
        "file_released": 0.7,
        "branch_claimed": 0.7,
        "branch_released": 0.7,
        "session_respawn": 0.5,
        "coordination_snapshot": 0.2,
        "test": 0.4,
        "code_chunk": 0.1,
        "file_summary": 0.05,
        # Say/Do contradiction tracking
        "public_statement": 1.5,
        "outcome_resolution": 1.5,
        "contradiction_detected": 2.0,
        "entity_profile_update": 1.0,
        # Experiential memory: distilled trajectories
        "skill_template": 2.0,
    }

    # Memory type classification: maps event_type -> cognitive category
    _MEMORY_TYPE_MAP = {
        # Episodic: raw experiences and session events
        "session_summary": "episodic",
        "task_completion": "episodic",
        "coordination_snapshot": "episodic",
        "session_respawn": "episodic",
        "merge_claim": "episodic",
        "merge_release": "episodic",
        "file_claimed": "episodic",
        "file_released": "episodic",
        "branch_claimed": "episodic",
        "branch_released": "episodic",
        "code_chunk": "episodic",
        "file_summary": "episodic",
        "file_conflict": "episodic",
        # Procedural: learned behavioral patterns and rules
        "lesson_learned": "procedural",
        "reflexion": "procedural",
        "self_reflection": "procedural",
        "outcome_evaluation": "procedural",
        "reminder": "procedural",
        "skill_template": "procedural",
        # Semantic: extracted facts and stable knowledge (everything else)
        "constraint": "semantic",
        "checkpoint": "semantic",
        "decision": "semantic",
        "user_preference": "semantic",
        "error_pattern": "semantic",
        "sota_research": "semantic",
        "research_report": "semantic",
        "benchmark_update": "semantic",
        "entity_profile_update": "semantic",
        "public_statement": "semantic",
        "outcome_resolution": "semantic",
        "contradiction_detected": "semantic",
        "sota_scan": "semantic",
        "preference_generated": "semantic",
        "advisor_action_outcome": "semantic",
        "advisor_insight": "semantic",
        "test": "semantic",
    }

    # Perspective-based type weight multipliers (behavioral diversity).
    # Each perspective boosts certain event types to bias retrieval toward
    # different knowledge categories, producing genuinely different results
    # for concurrent sessions querying the same topic.
    _PERSPECTIVE_BOOSTS: Dict[str, Dict[str, float]] = {
        "implementation": {
            "error_pattern": 1.8,
            "lesson_learned": 1.5,
            "decision": 1.3,
            "task_completion": 1.4,
            "code_chunk": 2.0,
            "file_summary": 2.0,
        },
        "critique": {
            "constraint": 1.8,
            "user_preference": 1.8,
            "contradiction_detected": 2.0,
            "lesson_learned": 1.3,
            "error_pattern": 1.5,
            "decision": 0.8,  # Slightly deprioritize existing decisions to question them
        },
        "verification": {
            "decision": 1.8,
            "lesson_learned": 1.5,
            "benchmark_update": 1.8,
            "outcome_evaluation": 1.8,
            "sota_research": 1.5,
            "error_pattern": 1.3,
        },
    }

    # Default priority per event type (1=lowest, 5=highest)
    _DEFAULT_PRIORITY = {
        "constraint": 5,
        "checkpoint": 5,
        "reminder": 5,
        "user_preference": 5,
        "error_pattern": 4,
        "lesson_learned": 4,
        "decision": 4,
        "task_completion": 3,
        "reflexion": 3,
        "outcome_evaluation": 3,
        "self_reflection": 3,
        "sota_research": 3,
        "research_report": 3,
        "advisor_insight": 3,
        "skill_template": 4,
        "session_summary": 2,
        "coordination_snapshot": 1,
        "session_respawn": 1,
        "file_summary": 1,
        "code_chunk": 1,
    }

    # ------------------------------------------------------------------
    # Decay curves — memories lose relevance over time unless accessed
    # factor = max(floor, exp(-lambda * days_since_last_access))
    # ------------------------------------------------------------------
    _DECAY_LAMBDAS = {
        "constraint": 0.0,              # No decay — permanent
        "user_preference": 0.0,         # No decay — permanent
        "error_pattern": 0.0,           # No decay — permanent
        "reminder": 0.0,               # No decay — permanent
        "lesson_learned": 0.005,       # 50% at ~139 days
        "skill_template": 0.01,         # 50% at ~69 days — slower decay than decisions
        "advisor_insight": 0.015,        # 50% at ~46 days — codebase knowledge ages with code
        "decision": 0.015,            # 50% at ~46 days (never-accessed); access reduces decay
        "task_completion": 0.015,      # 50% at ~46 days
        "checkpoint": 0.02,            # 50% at ~35 days
        "memory": 0.02,               # 50% at ~35 days
        "session_summary": 0.05,       # 50% at ~14 days
        "coordination_snapshot": 0.10,  # 50% at ~7 days
    }
    _DECAY_FLOOR = 0.35  # Floor for memories with access_count > 0
    _DECAY_FLOOR_NEVER_ACCESSED = 0.15  # Lower floor for never-accessed (reduces zombie noise)

    # Abstention thresholds — minimum quality for results to survive
    _MIN_VEC_SIMILARITY = 0.60  # Minimum cosine similarity for vec results (raised from 0.50)
    _MIN_TEXT_RELEVANCE = 0.35  # Minimum raw word overlap ratio for text-only results
    _MIN_COMPOSITE_SCORE = 0.10  # Absolute floor on composite score (catches temporal penalty)
    _MIN_VEC_CANDIDATES = 20  # Floor on vector candidate pool (prevents small limit from dropping good matches)

    # Per-event-type retrieval profiles (ALMA-inspired).
    # Reweight scoring phases based on what works best for each memory type.
    # Tuple order: (vec, text, word_overlap, context, graph)
    _RETRIEVAL_PROFILES = {
        # --- Event-type profiles (production MCP queries) ---
        "error_pattern":    (0.3, 1.5, 2.0, 0.5, 0.3),  # Stack traces need keyword match
        "decision":         (0.8, 0.6, 0.5, 1.0, 2.0),  # Decisions chain to prior decisions
        "lesson_learned":   (1.5, 0.8, 0.5, 0.8, 1.0),  # Abstract knowledge = semantic
        "user_preference":  (0.6, 1.0, 1.5, 0.3, 0.3),  # Keyword + preference boost
        # --- LongMemEval question-type profiles (benchmark) ---
        "single-session-assistant":  (1.0, 1.0, 1.0, 1.0, 1.0),  # 98.2% — at ceiling
        "single-session-user":       (1.0, 1.1, 1.2, 1.0, 1.0),  # 94.3% — slight word boost
        "knowledge-update":          (0.8, 1.3, 1.5, 1.0, 1.0),  # 85.9% — word match for exact fact
        "single-session-preference": (0.8, 1.0, 1.5, 1.0, 1.0),  # 83.3% — preference keywords
        "multi-session":             (1.3, 1.0, 1.3, 1.0, 1.0),  # 74.1% — broad vec + word recall
        "temporal-reasoning":        (1.0, 1.3, 1.3, 1.0, 1.0),  # ~70% — text/word for event IDs
        # --- Fallback ---
        "_default":         (0.7, 1.5, 1.3, 1.0, 1.0),  # Text-dominant: +9pp MRR over balanced baseline (rrf_weight_tuner)
    }

    _INFRASTRUCTURE_TYPES = frozenset(
        {
            "file_summary",
            "code_chunk",
            "session_respawn",
            "coordination_snapshot",
            "session_summary",  # exclude from user-facing queries
        }
    )

    DEFAULT_EMBEDDING_DEDUP_THRESHOLD = 0.88
    DEFAULT_JACCARD_DEDUP_THRESHOLD = 0.80

    # Input size limits (configurable via env vars)
    _MAX_CONTENT_SIZE = int(os.environ.get("OMEGA_MAX_CONTENT_SIZE", "1000000"))  # 1MB

    @property
    def _MAX_NODES(self) -> int:
        """Effective hard write cap. Re-reads license state per access.

        Pro: ``OMEGA_MAX_NODES`` env (default 50000).
        Free: ``_CORE_HARD_LIMIT`` (5000); the env var is ignored so a free
        user cannot set ``OMEGA_MAX_NODES=99999`` to bypass the cap.

        Grandfathering for Free installs that already exceed the Free cap
        is applied at the capacity-check site in ``_store.py`` (see
        ``_apply_grandfather`` and the store() write path). It lives there
        instead of inside this property so the existing capacity-check
        COUNT(*) query can be reused — running two sequential COUNT(*)
        queries inside a single store() call upset older SQLite libraries
        (Ubuntu CI 3.40-class) by leaving a prepared statement in
        progress between them.

        Test override: set ``self._max_nodes_override = <int>`` on the
        instance to force a specific cap value (bypasses the Pro check).
        Production callers should not set this attribute.
        """
        override = getattr(self, "_max_nodes_override", None)
        if override is not None:
            return override
        return _get_effective_max_nodes()

    def _apply_grandfather(self, base_max: int, count: int) -> int:
        """Return the effective per-instance cap, raising the Free ceiling
        for installs that already exceed _CORE_HARD_LIMIT. Caches the
        result on the instance so subsequent writes don't keep growing
        the ceiling — the intent is "your existing memories stay
        writable, but you do not get new growth headroom on Free."
        Pro callers (base_max != _CORE_HARD_LIMIT) get base_max
        unchanged. ``count`` is passed in by the caller's already-needed
        capacity-check query so this method does no I/O.
        """
        if base_max != _CORE_HARD_LIMIT:
            return base_max
        grand = getattr(self, "_grandfather_max", None)
        if grand is None:
            grand = max(base_max, count)
            self._grandfather_max = grand
        return grand

    def __init__(self, db_path=None, decompose_queries: bool = True):
        omega_home = Path(os.environ.get("OMEGA_HOME", str(Path.home() / ".omega")))
        self.db_path = Path(db_path) if db_path else (omega_home / "omega.db")
        self.db_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)

        self._lock = threading.RLock()
        self._cache_lock = threading.Lock()  # Protects _query_cache and _recent_query_context
        self._vec_available = False
        self._decompose_queries = decompose_queries
        self._cache_generation: int = 0
        self._query_cache: OrderedDict = OrderedDict()  # key -> (timestamp, results)
        self._conn = self._connect()
        self._init_schema()

        # Merged retrieval profiles: built-in + plugin overrides
        self._retrieval_profiles_merged: Dict[str, tuple] = dict(self._RETRIEVAL_PROFILES)
        # Plugin score modifiers: list of fn(node_id, score, metadata) -> score
        self._score_modifiers: list = []
        # A/B feedback tracking: LRU cache of recent query contexts per memory
        self._recent_query_context: OrderedDict = OrderedDict()  # node_id -> {query_text, query_hint, score, vec_sim, ts}
        self._QUERY_CONTEXT_MAX = 50

        # WAL checkpoint: PASSIVE every N writes, TRUNCATE every M writes.
        # PASSIVE is non-blocking but can't reclaim pages held by readers.
        # TRUNCATE is aggressive (resets WAL file) and prevents unbounded growth
        # when 4+ MCP server processes hold persistent connections.
        self._wal_write_count = 0
        self._wal_checkpoint_failures = 0
        _WAL_CHECKPOINT_INTERVAL = 10  # PASSIVE checkpoint every 10 writes
        raw_interval = int(os.environ.get("OMEGA_WAL_CHECKPOINT_INTERVAL", str(_WAL_CHECKPOINT_INTERVAL)))
        self._WAL_CHECKPOINT_INTERVAL = max(1, min(raw_interval, 1000))  # clamp to [1, 1000]
        self._WAL_TRUNCATE_INTERVAL = 100  # TRUNCATE checkpoint every 100 writes

        # Stats dict for bridge.py compatibility
        self.stats: Dict[str, Any] = {
            "stores": 0,
            "queries": 0,
            "hits": 0,
            "misses": 0,
            "auto_evictions": 0,
            "content_dedup_skips": 0,
            "memory_evolutions": 0,
            "embedding_dedup_skips": 0,
        }

        # Load persisted stats
        self._load_stats()

        # Last contradiction detection results (consume-once, set by store())
        self._last_contradiction_results: list = []

        # Agency Tax: operation latency tracking (arxiv 2602.19320 §4.4)
        self._op_timings: Dict[str, list] = {}
        self._op_counts: Dict[str, int] = {}

        # Format error tracking (backbone resilience, arxiv 2602.19320 §5.2)
        self._format_error_count: int = 0
        self._total_write_count: int = 0

        # Maintenance backlog tracking (throughput collapse detection)
        self._writes_since_consolidation: int = 0
        self._last_consolidation_ts: float = 0.0
        self._consolidation_history: list = []

        # Engram-inspired caches (#2)
        self._hot_memories: Dict[str, MemoryResult] = {}
        self._hot_cache_ts: float = 0.0
        self._session_cache: Dict[str, List[MemoryResult]] = {}
        self._prefetch_cache: Dict[str, List[MemoryResult]] = {}
        self._refresh_hot_cache()

        # Background threads spawned by bridge helpers (auto-relate,
        # entity-extraction). close() joins these with a timeout to prevent
        # use-after-close segfaults from in-flight sqlite-vec queries.
        self._background_threads: list[threading.Thread] = []
        self._bg_threads_lock = threading.Lock()

        # Deferred startup: integrity check, WAL checkpoint, and auto-backup
        # run in a background thread to avoid blocking MCP server init for 30+ seconds.
        self._deferred_startup_done = False
        self._deferred_thread = threading.Thread(
            target=self._deferred_startup, daemon=True, name="omega-deferred-init"
        )
        self._deferred_thread.start()

    def _deferred_startup(self) -> None:
        """Run expensive startup tasks in background thread.

        Includes integrity check, WAL checkpoint, and auto-backup.
        These previously blocked MCP server init for 30+ seconds on
        databases with 500+ memories.
        """
        try:
            # Integrity check: detect DB corruption
            with self._lock:
                try:
                    result = self._conn.execute("PRAGMA integrity_check").fetchone()
                    if result and result[0] != "ok":
                        logger.critical(
                            "DATABASE INTEGRITY CHECK FAILED: %s — creating backup before proceeding",
                            result[0][:200],
                        )
                        self._emergency_backup()
                except Exception as e:
                    logger.critical("Database integrity check error: %s", e)

                # WAL checkpoint: clear bloated WAL from multi-process contention
                try:
                    result = self._conn.execute("PRAGMA wal_checkpoint(PASSIVE)").fetchone()
                    if result and result[1] > 0:
                        logger.info("Startup WAL checkpoint: %d/%d pages checkpointed", result[1], result[2])
                except Exception as e:
                    logger.debug("Startup WAL checkpoint failed (non-fatal): %s", e)

            # Auto-backup (uses its own locking internally)
            self._auto_backup_if_stale()
        except Exception as e:
            logger.debug("Deferred startup failed (non-fatal): %s", e)
        finally:
            self._deferred_startup_done = True

    def _auto_backup_if_stale(self) -> None:
        """Create automatic backup if the most recent one is >24h old. Keeps max 5."""
        try:
            backup_dir = self.db_path.parent / "backups"
            if not backup_dir.exists():
                backup_dir.mkdir(parents=True, exist_ok=True, mode=0o700)

            # Check most recent backup age
            backups = sorted(backup_dir.glob("omega-auto-*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
            if backups:
                newest_mtime = backups[0].stat().st_mtime
                age_hours = (_time.time() - newest_mtime) / 3600
                if age_hours < 24:
                    return  # Recent backup exists

            # Check if store has any data worth backing up
            count = self._conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
            if count == 0:
                return  # Empty store, nothing to back up

            # Create backup
            ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
            backup_path = backup_dir / f"omega-auto-{ts}.json"
            result = self.export_to_file(backup_path)
            logger.info("Auto-backup created: %s (%d nodes)", backup_path.name, result.get("nodes", 0))

            # Rotate: keep max 5 auto-backups
            auto_backups = sorted(backup_dir.glob("omega-auto-*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
            for old_backup in auto_backups[5:]:
                old_backup.unlink()
                logger.debug("Rotated old backup: %s", old_backup.name)
        except Exception as e:
            logger.debug("Auto-backup skipped: %s", e)

    def register_plugin_profiles(self, profiles: Dict[str, tuple]) -> None:
        """Register retrieval profiles from a plugin. Plugin profiles override
        built-in defaults for the same event_type key."""
        for key, weights in profiles.items():
            if isinstance(weights, (tuple, list)) and len(weights) == 5:
                self._retrieval_profiles_merged[key] = tuple(weights)
            else:
                logger.warning("Plugin profile %s has invalid shape, skipping", key)

    def register_score_modifier(self, modifier) -> None:
        """Register a plugin score modifier: fn(node_id, score, metadata) -> score."""
        self._score_modifiers.append(modifier)

    def _connect(self) -> sqlite3.Connection:
        """Create a new SQLite connection with optimal settings."""
        from omega.crypto import secure_connect

        conn = secure_connect(
            self.db_path,
            timeout=30,
            check_same_thread=False,
            isolation_level="IMMEDIATE",
        )
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        # HTTP daemon mode: reduce memory footprint (mmap regions + page cache
        # were consuming ~200 MB across connections; see vmmap analysis).
        _is_http = os.environ.get("OMEGA_TRANSPORT", "").lower() == "http"
        conn.execute(f"PRAGMA cache_size={-4000 if _is_http else -16000}")  # 4MB / 16MB
        conn.execute(f"PRAGMA mmap_size={0 if _is_http else 33554432}")  # 0 / 32MB
        conn.execute("PRAGMA busy_timeout=30000")  # 30s — handles multi-process contention
        conn.execute("PRAGMA journal_size_limit=8388608")  # 8MB — cap WAL growth under multi-process contention
        conn.execute("PRAGMA foreign_keys=ON")

        # Try to load sqlite-vec extension
        try:
            import sqlite_vec

            conn.enable_load_extension(True)
            sqlite_vec.load(conn)
            conn.enable_load_extension(False)
            self._vec_available = True
        except (ImportError, Exception) as e:
            logger.warning(f"sqlite-vec not available, falling back to brute-force: {e}")
            self._vec_available = False

        return conn

    def _init_schema(self) -> None:
        """Create tables if they don't exist. Delegates to omega.schema."""
        self._vec_available, self._fts_available = _init_schema_fn(
            self._conn, self._vec_available, EMBEDDING_DIM
        )

    def _emergency_backup(self) -> None:
        """Create an emergency backup of the DB file when integrity check fails."""
        try:
            import shutil
            backup_dir = self.db_path.parent / "backups"
            backup_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
            ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
            backup_path = backup_dir / f"omega-emergency-{ts}.db"
            shutil.copy2(self.db_path, backup_path)
            logger.warning("Emergency backup saved to %s", backup_path)
        except Exception as e:
            logger.error("Emergency backup failed: %s", e, exc_info=True)

    # ------------------------------------------------------------------
    # Resilient commit — retries on multi-process lock contention
    # ------------------------------------------------------------------

    def _commit(self) -> None:
        """Commit with retry on 'database is locked'.

        WAL mode + busy_timeout=30s handles most contention, but under
        heavy multi-process load (3+ MCP servers) the timeout can still
        expire.  This retries with exponential backoff before giving up.
        """
        _retry_on_locked(self._conn.commit)
        self._maybe_wal_checkpoint()

    def _maybe_wal_checkpoint(self) -> None:
        """Run WAL checkpoints: PASSIVE every N writes, TRUNCATE every M writes.

        With multiple MCP server processes holding persistent connections,
        PASSIVE checkpoints get starved (can't reclaim pages held by readers).
        Periodic TRUNCATE resets the WAL file, preventing unbounded growth
        (observed: 43MB WAL vs 19MB DB with 6 concurrent processes).
        """
        self._wal_write_count += 1
        if self._wal_write_count >= self._WAL_TRUNCATE_INTERVAL:
            # TRUNCATE: aggressive, resets WAL file size
            self._wal_write_count = 0
            try:
                result = self._conn.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()
                if result:
                    busy, checkpointed, total = result
                    logger.debug("WAL TRUNCATE checkpoint: %d/%d pages checkpointed (%d busy)",
                                 checkpointed, total, busy)
                self._wal_checkpoint_failures = 0
            except Exception as e:
                self._wal_checkpoint_failures += 1
                if self._wal_checkpoint_failures >= 3:
                    logger.warning(
                        "WAL TRUNCATE checkpoint failed %d consecutive times: %s. "
                        "A stale OMEGA process may be preventing checkpoints. "
                        "Check: ps aux | grep omega",
                        self._wal_checkpoint_failures, e,
                    )
                else:
                    logger.debug("WAL TRUNCATE checkpoint failed (non-fatal): %s", e)
        elif self._wal_write_count % self._WAL_CHECKPOINT_INTERVAL == 0:
            # PASSIVE: gentle, non-blocking
            try:
                result = self._conn.execute("PRAGMA wal_checkpoint(PASSIVE)").fetchone()
                if result:
                    busy, checkpointed, total = result
                    if checkpointed > 0:
                        logger.debug("WAL PASSIVE checkpoint: %d/%d pages checkpointed (%d busy)",
                                     checkpointed, total, busy)
            except Exception as e:
                logger.debug("WAL PASSIVE checkpoint failed (non-fatal): %s", e)

    def _exec(self, sql, params=None):
        """Run SQL with retry on 'database is locked'.

        Supplements _commit() retry: individual SQL statements can also fail
        with 'database is locked' under heavy multi-process contention when
        busy_timeout expires.  Retries with exponential backoff.
        """
        if params is not None:
            return _retry_on_locked(self._conn.execute, sql, params)
        return _retry_on_locked(self._conn.execute, sql)

    def _invalidate_query_cache(self, new_content: Optional[str] = None) -> None:
        """Invalidate query cache after writes.

        Clears all cached query results. O(1) via generation counter bump
        and dict.clear() instead of O(n*m) trigram overlap computation.
        The new_content parameter is accepted for API compatibility but
        no longer used for selective invalidation (the trigram overhead
        exceeded the cache-hit benefit).
        """
        with self._cache_lock:
            self._cache_generation += 1
            self._query_cache.clear()
            self._session_cache.clear()
            self._prefetch_cache.clear()
        self._hot_cache_ts = 0.0  # Force hot cache refresh on next query (#2)

    def _record_timing(self, op_name: str, duration_ms: float) -> None:
        """Record operation latency for agency tax tracking."""
        if op_name not in self._op_timings:
            self._op_timings[op_name] = []
            self._op_counts[op_name] = 0
        self._op_timings[op_name].append(duration_ms)
        self._op_counts[op_name] += 1
        # Keep only last 100 timings per operation
        if len(self._op_timings[op_name]) > 100:
            self._op_timings[op_name] = self._op_timings[op_name][-100:]

    def _row_to_result(self, row: tuple) -> MemoryResult:
        """Convert a database row to a MemoryResult.

        Accepts 7-element rows (standard) or 9-element rows (with valid_from, valid_until).
        """
        if len(row) >= 9:
            node_id, content, metadata_json, created_at, access_count, last_accessed, ttl_seconds, vf, vu = row[:9]
        else:
            node_id, content, metadata_json, created_at, access_count, last_accessed, ttl_seconds = row[:7]
            vf = None
            vu = None

        meta = json.loads(metadata_json) if metadata_json else {}

        created = self._parse_dt(created_at) or datetime.now(timezone.utc)
        last_acc = self._parse_dt(last_accessed)
        valid_from_dt = self._parse_dt(vf) if vf else None
        valid_until_dt = self._parse_dt(vu) if vu else None

        return MemoryResult(
            id=node_id,
            content=content,
            metadata=meta,
            created_at=created,
            access_count=access_count or 0,
            last_accessed=last_acc,
            ttl_seconds=ttl_seconds,
            valid_from=valid_from_dt,
            valid_until=valid_until_dt,
            derived_from=meta.get("derived_from"),
            source_uri=meta.get("source_uri"),
            status=meta.get("status", "active"),
        )

    @staticmethod
    def _parse_dt(value: Optional[str]) -> Optional[datetime]:
        """Parse an ISO datetime string to an aware UTC datetime.

        Handles naive strings (no tz), Z-suffix, and +00:00 suffix.
        Returns None when *value* is falsy.
        """
        if not value:
            return None
        # Python 3.11+ fromisoformat supports 'Z' natively, but we keep this
        # workaround for existing DB records written with the Z-suffix format.
        if value.endswith("Z"):
            value = value[:-1] + "+00:00"
        dt = datetime.fromisoformat(value)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt

    # ------------------------------------------------------------------
    # Stats persistence
    # ------------------------------------------------------------------

    def _load_stats(self) -> None:
        """Load stats from a sidecar file if it exists."""
        stats_path = self.db_path.parent / "stats.json"
        if stats_path.exists():
            try:
                loaded = json.loads(stats_path.read_text())
                self.stats.update(loaded)
            except (json.JSONDecodeError, OSError) as e:
                logger.debug("Stats load failed: %s", e)

    def _save_stats(self) -> None:
        """Persist stats to a sidecar file."""
        stats_path = self.db_path.parent / "stats.json"
        try:
            self.stats["total_nodes"] = self.node_count()
            stats_path.write_text(json.dumps(self.stats))
        except (OSError, TypeError) as e:
            logger.debug("Stats save failed: %s", e)

    # ------------------------------------------------------------------
    # Engram-inspired hot cache
    # ------------------------------------------------------------------

    def _refresh_hot_cache(self) -> None:
        """Refresh the hot memory cache (#2) with top memories by access_count."""
        try:
            rows = self._conn.execute(
                """SELECT node_id, content, metadata, created_at,
                          access_count, last_accessed, ttl_seconds
                   FROM memories WHERE access_count > 0
                     AND json_extract(metadata, '$.superseded') IS NULL
                   ORDER BY access_count DESC LIMIT ?""",
                (_HOT_CACHE_SIZE,),
            ).fetchall()
            new_hot: Dict[str, MemoryResult] = {}
            for row in rows:
                result = self._row_to_result(row)
                if not result.is_expired():
                    new_hot[result.id] = result
            self._hot_memories = new_hot
            self._hot_cache_ts = _time.monotonic()
        except Exception as e:
            logger.debug("Hot cache refresh failed: %s", e)

    def register_background_thread(self, t: threading.Thread) -> None:
        """Track a background thread that holds a reference to this store.

        close() joins all registered threads with a timeout before closing
        the sqlite connection. Without this, an in-flight sqlite-vec query
        on a background thread can segfault after the connection is closed.
        """
        with self._bg_threads_lock:
            self._background_threads.append(t)

    def unregister_background_thread(self, t: threading.Thread) -> None:
        """Remove a finished background thread from the tracking list."""
        with self._bg_threads_lock:
            try:
                self._background_threads.remove(t)
            except ValueError:
                pass

    def close(self) -> None:
        """Close the database connection."""
        # Wait for deferred startup thread to finish before closing the connection
        if hasattr(self, "_deferred_thread") and self._deferred_thread.is_alive():
            self._deferred_thread.join(timeout=5.0)
        # Join bridge-spawned background threads (auto-relate, entity-extraction)
        # before closing the connection to prevent use-after-close segfaults.
        if hasattr(self, "_background_threads"):
            with self._bg_threads_lock:
                threads = list(self._background_threads)
            for t in threads:
                if t.is_alive():
                    t.join(timeout=2.0)
        self._save_stats()
        try:
            # Flush WAL before closing — helps other processes checkpoint
            self._conn.execute("PRAGMA wal_checkpoint(PASSIVE)")
        except Exception:
            pass
        try:
            self._conn.close()
        except Exception as e:
            logger.debug("Database close failed: %s", e)

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass  # Silence errors during GC — no logger guarantee
