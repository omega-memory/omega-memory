"""Maintenance, health, graph, entity, and I/O mixin for SQLiteStore."""

import gc
import logging
import os
import time as _time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from omega import json_compat as json
from omega.exceptions import EmbeddingError, StorageError, ValidationError

from ._types import (
    _serialize_f32,
)

logger = logging.getLogger("omega.sqlite_store")


class MaintenanceMixin:
    """Maintenance, health, graph, entity, and I/O methods for SQLiteStore."""

    # ------------------------------------------------------------------
    # Forgetting / cleanup
    # ------------------------------------------------------------------

    def _log_forgetting(self, node_id: str, content: str, event_type: str,
                        reason: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Log a deletion/forgetting event to the audit trail.

        MUST be called inside an already-acquired `with self._lock:` block.
        The lock is non-reentrant — never acquire it again here.
        """
        now = datetime.now(timezone.utc).isoformat()
        content_preview = content[:200] if content else None
        meta_json = json.dumps(metadata) if metadata else None
        try:
            self._exec(
                """INSERT OR IGNORE INTO forgetting_log
                   (node_id, content_preview, event_type, reason, deleted_at, metadata)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (node_id, content_preview, event_type, reason, now, meta_json),
            )
        except Exception as e:
            logger.debug("Failed to log forgetting event for %s: %s", node_id, e)

    def _queue_cloud_delete(self, local_id: int) -> None:
        """Queue a local rowid for deletion from Supabase on next sync.

        MUST be called inside an already-acquired `with self._lock:` block.
        """
        try:
            now = datetime.now(timezone.utc).isoformat()
            self._exec(
                "INSERT INTO cloud_delete_queue (local_id, deleted_at) VALUES (?, ?)",
                (local_id, now),
            )
        except Exception as e:
            logger.debug("Failed to queue cloud delete for local_id=%s: %s", local_id, e)

    def _log_forgetting_external(self, node_id: str, content: str, event_type: str,
                                   reason: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Log a forgetting event from outside a lock context (e.g., bridge.py).

        Acquires the lock internally. For use by external callers only —
        internal methods that already hold the lock must use _log_forgetting().
        """
        with self._lock:
            self._log_forgetting(node_id, content, event_type, reason, metadata)
            self._commit()

    def queue_cloud_delete_by_node_id(self, node_id: str) -> None:
        """Queue a cloud deletion by node_id (for external callers like bridge.py).

        Acquires the lock internally. Looks up the rowid from node_id.
        """
        with self._lock:
            row = self._conn.execute(
                "SELECT id FROM memories WHERE node_id = ?", (node_id,)
            ).fetchone()
            if row:
                self._queue_cloud_delete(row[0])
                self._commit()

    def cleanup_expired(self) -> int:
        """Remove expired memories. Returns count removed."""
        with self._lock:
            now = datetime.now(timezone.utc).isoformat()
            # Find expired: created_at + ttl_seconds < now
            rows = self._conn.execute(
                """SELECT id, node_id, content, event_type FROM memories
                   WHERE ttl_seconds IS NOT NULL
                   AND datetime(created_at, '+' || ttl_seconds || ' seconds') < ?""",
                (now,),
            ).fetchall()

            if not rows:
                return 0

            ids_to_delete = []
            node_ids_to_delete = []
            for rowid, node_id, content, et in rows:
                et = et or ""
                self._log_forgetting(node_id, content or "", et, "ttl_expired")
                self._queue_cloud_delete(rowid)
                if self._vec_available:
                    try:
                        self._conn.execute("DELETE FROM memories_vec WHERE rowid = ?", (rowid,))
                    except Exception as e:
                        logger.debug("Failed to delete vec embedding rowid=%s: %s", rowid, e)
                ids_to_delete.append(rowid)
                node_ids_to_delete.append(node_id)

            if ids_to_delete:
                ph = ",".join("?" * len(ids_to_delete))
                self._conn.execute(f"DELETE FROM memories WHERE id IN ({ph})", ids_to_delete)
                self._conn.execute(
                    f"DELETE FROM edges WHERE source_id IN ({ph}) OR target_id IN ({ph})",
                    node_ids_to_delete + node_ids_to_delete,
                )
            self._commit()
            return len(rows)

    def evict_lru(self, count: int = 1) -> int:
        """Evict least recently used memories."""
        with self._lock:
            rows = self._conn.execute(
                """SELECT id, node_id, content, event_type FROM memories
                   ORDER BY COALESCE(last_accessed, created_at) ASC
                   LIMIT ?""",
                (count,),
            ).fetchall()

            evicted = 0
            for rowid, node_id, content, et in rows:
                et = et or ""
                self._log_forgetting(node_id, content or "", et, "lru_evicted")
                self._queue_cloud_delete(rowid)
                self._conn.execute("DELETE FROM memories WHERE id = ?", (rowid,))
                if self._vec_available:
                    try:
                        self._conn.execute("DELETE FROM memories_vec WHERE rowid = ?", (rowid,))
                    except Exception as e:
                        logger.debug("Failed to delete vec embedding rowid=%s: %s", rowid, e)
                self._conn.execute("DELETE FROM edges WHERE source_id = ? OR target_id = ?", (node_id, node_id))
                evicted += 1

            if evicted:
                self._commit()
            return evicted

    def prune_forgetting_log(self, max_age_days: int = 90) -> int:
        """Remove forgetting log entries older than max_age_days. Returns count removed."""
        cutoff = (datetime.now(timezone.utc) - timedelta(days=max_age_days)).isoformat()
        with self._lock:
            cursor = self._conn.execute(
                "DELETE FROM forgetting_log WHERE deleted_at < ?", (cutoff,)
            )
            self._commit()
            return cursor.rowcount

    def get_forgetting_log(self, limit: int = 50, reason: Optional[str] = None) -> List[Dict[str, Any]]:
        """Retrieve recent forgetting log entries."""
        if reason:
            rows = self._conn.execute(
                """SELECT node_id, content_preview, event_type, reason, deleted_at, metadata
                   FROM forgetting_log WHERE reason = ?
                   ORDER BY deleted_at DESC LIMIT ?""",
                (reason, limit),
            ).fetchall()
        else:
            rows = self._conn.execute(
                """SELECT node_id, content_preview, event_type, reason, deleted_at, metadata
                   FROM forgetting_log ORDER BY deleted_at DESC LIMIT ?""",
                (limit,),
            ).fetchall()

        results = []
        for row in rows:
            entry = {
                "node_id": row[0],
                "content_preview": row[1],
                "event_type": row[2],
                "reason": row[3],
                "deleted_at": row[4],
            }
            if row[5]:
                try:
                    entry["metadata"] = json.loads(row[5])
                except Exception:
                    entry["metadata"] = row[5]
            results.append(entry)
        return results

    # ------------------------------------------------------------------
    # Consolidation / decay
    # ------------------------------------------------------------------

    def consolidate(
        self,
        prune_days: int = 14,
        max_summaries: int = 50,
    ) -> Dict[str, Any]:
        """Consolidate memories: prune stale low-value entries, cap session summaries.

        Prunes:
        1. Memories with 0 access older than prune_days (excluding protected types)
        2. Oldest session summaries beyond max_summaries cap
        3. Orphaned edges pointing to deleted nodes
        4. Orphaned vec embeddings without matching memory rows

        Returns dict with counts of what was removed.
        """
        protected_types = frozenset(
            {
                "user_preference",
                "error_pattern",
                "behavioral_pattern",
                "constraint",
                "reminder",
            }
        )
        stats = {"pruned_stale": 0, "pruned_summaries": 0, "pruned_edges": 0, "pruned_vec_orphans": 0}
        _t0_consolidate = _time.monotonic()

        cutoff = (datetime.now(timezone.utc) - timedelta(days=prune_days)).isoformat()

        with self._lock:
            # Phase 0: Fast decision pruning — zero-access, older than 14 days, priority < 5
            decision_cutoff = (datetime.now(timezone.utc) - timedelta(days=14)).isoformat()
            decision_rows = self._conn.execute(
                """SELECT id, node_id, content, metadata FROM memories
                    WHERE event_type = 'decision'
                    AND access_count = 0
                    AND created_at < ?
                    AND COALESCE(priority, 3) < 5""",
                (decision_cutoff,),
            ).fetchall()

            p0_ids = []
            p0_node_ids = []
            for rowid, node_id, content, meta_json in decision_rows:
                self._log_forgetting(node_id, content or "", "decision", "consolidation_phase0_pruned")
                self._queue_cloud_delete(rowid)
                if self._vec_available:
                    try:
                        self._conn.execute("DELETE FROM memories_vec WHERE rowid = ?", (rowid,))
                    except Exception as e:
                        logger.debug("Failed to delete vec embedding rowid=%s: %s", rowid, e)
                p0_ids.append(rowid)
                p0_node_ids.append(node_id)
                stats["pruned_stale"] += 1

            if p0_ids:
                ph = ",".join("?" * len(p0_ids))
                self._conn.execute(f"DELETE FROM memories WHERE id IN ({ph})", p0_ids)
                self._conn.execute(
                    f"DELETE FROM edges WHERE source_id IN ({ph}) OR target_id IN ({ph})",
                    p0_node_ids + p0_node_ids,
                )

            # Phase 1: Prune stale zero-access memories (not protected types)
            # Decisions with priority >= 4 are protected; lower-priority
            # zero-access decisions older than prune_days get pruned.
            placeholders = ",".join("?" * len(protected_types))
            rows = self._conn.execute(
                f"""SELECT id, node_id, content, event_type FROM memories
                    WHERE access_count = 0
                    AND created_at < ?
                    AND (event_type IS NULL OR event_type NOT IN ({placeholders}))
                    AND NOT (event_type = 'decision' AND COALESCE(priority, 3) >= 4)""",
                (cutoff, *protected_types),
            ).fetchall()

            p1_ids = []
            p1_node_ids = []
            for rowid, node_id, content, et in rows:
                et = et or ""
                self._log_forgetting(node_id, content or "", et, "consolidation_pruned")
                self._queue_cloud_delete(rowid)
                if self._vec_available:
                    try:
                        self._conn.execute("DELETE FROM memories_vec WHERE rowid = ?", (rowid,))
                    except Exception as e:
                        logger.debug("Failed to delete vec embedding rowid=%s: %s", rowid, e)
                p1_ids.append(rowid)
                p1_node_ids.append(node_id)
                stats["pruned_stale"] += 1

            if p1_ids:
                ph = ",".join("?" * len(p1_ids))
                self._conn.execute(f"DELETE FROM memories WHERE id IN ({ph})", p1_ids)
                self._conn.execute(
                    f"DELETE FROM edges WHERE source_id IN ({ph}) OR target_id IN ({ph})",
                    p1_node_ids + p1_node_ids,
                )

            # Phase 2: Cap session summaries — keep newest max_summaries, prune rest
            summary_rows = self._conn.execute(
                """SELECT id, node_id, content FROM memories
                   WHERE event_type = 'session_summary'
                   ORDER BY created_at DESC"""
            ).fetchall()

            if len(summary_rows) > max_summaries:
                to_prune = summary_rows[max_summaries:]
                p2_ids = []
                p2_node_ids = []
                for rowid, node_id, content in to_prune:
                    self._log_forgetting(
                        node_id, content or "", "session_summary",
                        "consolidation_pruned", {"reason_detail": "session_summary_cap"},
                    )
                    self._queue_cloud_delete(rowid)
                    if self._vec_available:
                        try:
                            self._conn.execute("DELETE FROM memories_vec WHERE rowid = ?", (rowid,))
                        except Exception as e:
                            logger.debug("Vec cleanup during summary prune failed for rowid %s: %s", rowid, e)
                    p2_ids.append(rowid)
                    p2_node_ids.append(node_id)
                    stats["pruned_summaries"] += 1

                if p2_ids:
                    ph = ",".join("?" * len(p2_ids))
                    self._conn.execute(f"DELETE FROM memories WHERE id IN ({ph})", p2_ids)
                    self._conn.execute(
                        f"DELETE FROM edges WHERE source_id IN ({ph}) OR target_id IN ({ph})",
                        p2_node_ids + p2_node_ids,
                    )

            # Phase 3: Prune orphaned edges
            orphaned = self._conn.execute(
                """SELECT e.id FROM edges e
                   LEFT JOIN memories m1 ON e.source_id = m1.node_id
                   LEFT JOIN memories m2 ON e.target_id = m2.node_id
                   WHERE m1.node_id IS NULL OR m2.node_id IS NULL"""
            ).fetchall()
            if orphaned:
                self._conn.execute(
                    f"DELETE FROM edges WHERE id IN ({','.join('?' * len(orphaned))})", [r[0] for r in orphaned]
                )
                stats["pruned_edges"] = len(orphaned)

            # Phase 4: Prune orphaned vec embeddings
            if self._vec_available:
                try:
                    orphaned_vec = self._conn.execute(
                        """SELECT vec.rowid FROM memories_vec vec
                           LEFT JOIN memories m ON vec.rowid = m.id
                           WHERE m.id IS NULL"""
                    ).fetchall()
                    if orphaned_vec:
                        orphan_ids = [row[0] for row in orphaned_vec]
                        ph = ",".join("?" * len(orphan_ids))
                        try:
                            self._conn.execute(f"DELETE FROM memories_vec WHERE rowid IN ({ph})", orphan_ids)
                        except Exception as e:
                            logger.debug("Vec orphan batch delete failed: %s — falling back to per-row", e)
                            for oid in orphan_ids:
                                try:
                                    self._conn.execute("DELETE FROM memories_vec WHERE rowid = ?", (oid,))
                                except Exception as e:
                                    logger.debug("Vec orphan per-row delete failed for rowid %s: %s", oid, e)
                        stats["pruned_vec_orphans"] = len(orphaned_vec)
                        logger.info("Pruned %d orphaned vec embeddings", len(orphaned_vec))
                except Exception as e:
                    logger.debug("Vec orphan check failed: %s", e)

            self._commit()

        # Phase 5: Strength decay -- mark weak old memories as superseded
        decay_stats = self.apply_strength_decay()
        stats["decayed_memories"] = decay_stats["decayed"]

        # Phase 6: Entity deduplication -- merge entities with matching names
        merge_stats = self.merge_duplicate_entities()
        stats["merged_entities"] = merge_stats["merged"]

        # Auto-prune old forgetting log entries
        self.prune_forgetting_log()

        stats["node_count_after"] = self.node_count()

        # Maintenance backlog tracking (arxiv 2602.19320 — throughput collapse)
        _consolidate_ms = (_time.monotonic() - _t0_consolidate) * 1000
        self._record_timing("consolidate", _consolidate_ms)
        self._consolidation_history.append({
            "duration_ms": round(_consolidate_ms, 1),
            "writes_processed": self._writes_since_consolidation,
            "timestamp": _time.time(),
        })
        if len(self._consolidation_history) > 10:
            self._consolidation_history = self._consolidation_history[-10:]
        self._writes_since_consolidation = 0
        self._last_consolidation_ts = _time.monotonic()

        return stats

    def apply_strength_decay(
        self,
        min_strength: float = 0.05,
        min_age_days: int = 30,
    ) -> dict:
        """Mark weak, old, unaccessed memories as superseded (ACT-R forgetting curve).

        Computes strength = type_weight * feedback_factor * decay_factor for each
        non-protected, non-superseded memory older than min_age_days with 0 access,
        plus "zombie" memories that are accessed but consistently rated down.
        Memories below min_strength get marked superseded with reason 'strength_decay'.

        Access history does not change how fast a record decays — that is
        ranking, and 1.5.13 deliberately removed access from it. But access does
        still decide whether a record is *eligible* for automatic forgetting at
        all. Those are different questions, and conflating them silently made
        previously-immune records disappear from search.
        """
        protected_types = frozenset({
            "user_preference", "error_pattern", "behavioral_pattern",
            "constraint", "reminder",
        })
        cutoff = (datetime.now(timezone.utc) - timedelta(days=min_age_days)).isoformat()
        stats: Dict[str, int] = {"decayed": 0, "scanned": 0}

        with self._lock:
            # Scan two populations:
            # 1. Never-accessed old memories (original behavior)
            # 2. Accessed but negatively-rated memories (feedback_score <= -3)
            #    These are "zombie" memories — surfaced but consistently ignored.
            #
            # Widening this to every record older than the cutoff is what let
            # auto-consolidation supersede records a user had actually used.
            # The approved design bounds how much access can influence ranking
            # and decay rate; it does not remove lifecycle protection, and
            # "bulk rewriting supersede records" is an explicit non-goal.
            rows = self._conn.execute(
                """SELECT node_id, content, metadata, created_at, access_count,
                          last_accessed, event_type
                   FROM memories
                   WHERE created_at < ?
                   AND (access_count = 0
                        OR json_extract(metadata, '$.feedback_score') <= -3)""",
                (cutoff,),
            ).fetchall()

        for row in rows:
            node_id, content, meta_json, created_at, access_count, last_accessed, event_type = row
            meta = json.loads(meta_json) if meta_json else {}
            stats["scanned"] += 1

            if meta.get("superseded"):
                continue
            if event_type in protected_types:
                continue
            # A recently used record is not a forgetting candidate, however
            # weak its computed strength.
            if last_accessed:
                la_dt = self._parse_dt(last_accessed)
                cutoff_dt = datetime.now(timezone.utc) - timedelta(days=min_age_days)
                if la_dt and la_dt > cutoff_dt:
                    continue

            type_weight = self._TYPE_WEIGHTS.get(event_type, 1.0)
            decay = self._compute_decay_factor(
                event_type or "", last_accessed, created_at, access_count,
            )
            fb_score = meta.get("feedback_score", 0)
            fb = self._compute_fb_factor(fb_score)
            strength = type_weight * fb * decay

            if strength < min_strength:
                with self._lock:
                    self._log_forgetting(
                        node_id, content or "", event_type or "", "strength_decay",
                    )
                    meta["superseded"] = True
                    meta["superseded_reason"] = "strength_decay"
                    meta["superseded_at"] = datetime.now(timezone.utc).isoformat()
                    self._conn.execute(
                        "UPDATE memories SET metadata = ? WHERE node_id = ?",
                        (json.dumps(meta), node_id),
                    )
                    self._commit()
                stats["decayed"] += 1

        return stats

    def merge_duplicate_entities(self) -> dict:
        """Merge entities with matching lowercased names.

        Transfers memories from duplicate entity_id to primary (first-seen).
        """
        stats: Dict[str, int] = {"merged": 0}
        try:
            from omega_platform.entity.engine import get_entity_manager
            em = get_entity_manager(Path(self.db_path) if hasattr(self, 'db_path') else None)
        except Exception as e:
            logger.debug("Entity engine unavailable for merge: %s", e)
            return stats

        try:
            entity_ids = em.list_entity_ids()
        except Exception as e:
            logger.debug("list_entity_ids failed: %s", e)
            return stats

        if len(entity_ids) < 2:
            return stats

        name_groups: Dict[str, list] = {}
        for eid, name in entity_ids:
            key = name.strip().lower()
            name_groups.setdefault(key, []).append(eid)

        for name_key, eids in name_groups.items():
            if len(eids) < 2:
                continue
            primary = eids[0]
            for duplicate in eids[1:]:
                with self._lock:
                    self._conn.execute(
                        "UPDATE memories SET entity_id = ? WHERE entity_id = ?",
                        (primary, duplicate),
                    )
                    self._commit()
                try:
                    em.delete_entity(duplicate)
                except Exception as e:
                    logger.debug("Failed to delete duplicate entity %s: %s", duplicate, e)
                stats["merged"] += 1
                logger.info(
                    "Merged entity %s into %s (name: %s)", duplicate, primary, name_key,
                )

        return stats

    # ------------------------------------------------------------------
    # Reembedding
    # ------------------------------------------------------------------

    def reembed_all(self, batch_size: int = 32) -> Dict[str, int]:
        """Regenerate all embeddings using the current ML model.

        Use this to fix corrupted (hash-fallback) embeddings or after
        switching embedding models. Only runs if an ML backend is available.

        Returns dict with counts of updated, skipped, and failed nodes.
        """
        from omega.embedding import generate_embeddings_batch, get_active_backend

        backend = get_active_backend()
        if backend is None:
            # Force a load attempt
            from omega.embedding import _get_embedding_model

            _get_embedding_model()
            backend = get_active_backend()
        if backend is None:
            raise EmbeddingError("Cannot reembed: no ML embedding backend available")

        rows = self._conn.execute("SELECT id, content FROM memories ORDER BY id").fetchall()

        updated = 0
        failed = 0
        for i in range(0, len(rows), batch_size):
            batch_rows = rows[i : i + batch_size]
            texts = [r[1] for r in batch_rows]
            ids = [r[0] for r in batch_rows]

            try:
                embeddings = generate_embeddings_batch(texts)
                with self._lock:
                    for mem_id, emb in zip(ids, embeddings):
                        try:
                            self._conn.execute("DELETE FROM memories_vec WHERE rowid = ?", (mem_id,))
                            self._conn.execute(
                                "INSERT INTO memories_vec (rowid, embedding) VALUES (?, ?)",
                                (mem_id, _serialize_f32(emb)),
                            )
                            updated += 1
                        except Exception as e:
                            logger.warning(f"reembed failed for id={mem_id}: {e}")
                            failed += 1
                    self._commit()
            except Exception as e:
                logger.warning(f"reembed batch failed: {e}")
                failed += len(batch_rows)

        logger.info(f"reembed_all: updated={updated}, failed={failed}")
        return {"updated": updated, "failed": failed, "total": len(rows)}

    def backfill_embeddings(self, batch_size: int = 50) -> dict:
        """Backfill missing embeddings for memories not in memories_vec.

        Generates embeddings for memories that were stored when the embedding
        model was unavailable (hash-fallback). Returns stats dict.
        """
        if not self._vec_available:
            return {"error": "vec not available", "backfilled": 0, "missing": 0}

        # Find memories missing from vec table
        missing = self._conn.execute(
            """SELECT m.id, m.node_id, m.content
               FROM memories m
               LEFT JOIN memories_vec v ON m.id = v.rowid
               WHERE v.rowid IS NULL
               LIMIT ?""",
            (batch_size,),
        ).fetchall()

        if not missing:
            return {"backfilled": 0, "remaining": 0, "failed": 0, "status": "all memories have embeddings"}

        from omega.embedding import generate_embedding, get_active_backend, is_embedding_degraded

        # Probe embedding system: abort only if a real backend exists but degraded
        # (i.e., real backend is broken). When no backend exists (test/bootstrap),
        # hash embeddings are acceptable for backfill.
        _probe = generate_embedding("probe")
        if is_embedding_degraded() and get_active_backend() is not None:
            return {"error": "embedding backend degraded", "backfilled": 0, "missing": len(missing)}

        # Generate embeddings outside lock (CPU-bound)
        to_insert = []
        failed = 0
        for rowid, node_id, content in missing:
            try:
                embedding = generate_embedding(content)
                if embedding:
                    to_insert.append((rowid, _serialize_f32(embedding)))
            except Exception as e:
                logger.debug("backfill embedding failed for %s: %s", node_id, e)
                failed += 1

        # Insert all in single transaction
        if to_insert:
            with self._lock:
                for rowid, emb_bytes in to_insert:
                    try:
                        self._conn.execute(
                            "INSERT OR IGNORE INTO memories_vec (rowid, embedding) VALUES (?, ?)",
                            (rowid, emb_bytes),
                        )
                    except Exception as e:
                        logger.debug("backfill vec insert failed for rowid %s: %s", rowid, e)
                        failed += 1
                self._commit()

        # Check if more remain
        remaining = self._conn.execute(
            """SELECT COUNT(*) FROM memories m
               LEFT JOIN memories_vec v ON m.id = v.rowid
               WHERE v.rowid IS NULL""",
        ).fetchone()[0]

        return {
            "backfilled": len(to_insert),
            "failed": failed,
            "remaining": remaining,
            "status": "complete" if remaining == 0 else f"{remaining} still missing",
        }

    # ------------------------------------------------------------------
    # Health / metrics
    # ------------------------------------------------------------------

    def get_agency_tax(self) -> Dict[str, Any]:
        """Return agency tax metrics: median/p95/p99 latency per operation.

        Agency tax (arxiv 2602.19320 §4.4) measures the latency overhead
        of memory operations.  MemoryOS = 32s (unusable), SimpleMem = 1.06s.
        OMEGA targets < 500ms for writes, < 200ms for reads.
        """
        result: Dict[str, Any] = {}
        for op, timings in self._op_timings.items():
            if not timings:
                continue
            s = sorted(timings)
            n = len(s)
            result[op] = {
                "count": self._op_counts.get(op, n),
                "median_ms": round(s[n // 2], 1),
                "p95_ms": round(s[int(n * 0.95)], 1) if n >= 20 else round(s[-1], 1),
                "p99_ms": round(s[int(n * 0.99)], 1) if n >= 100 else round(s[-1], 1),
                "mean_ms": round(sum(s) / n, 1),
            }
        return result

    def record_format_error(self, operation: str, error: str) -> None:
        """Record a format/JSON parsing error (backbone resilience tracking)."""
        self._format_error_count += 1
        logger.debug("Format error in %s: %s (total: %d)", operation, error, self._format_error_count)

    def get_format_error_rate(self) -> float:
        """Return format error rate: errors / total writes."""
        if self._total_write_count == 0:
            return 0.0
        return self._format_error_count / self._total_write_count

    def get_maintenance_backlog(self) -> Dict[str, Any]:
        """Return maintenance backlog metrics (throughput collapse detection).

        When writes_since_consolidation grows unbounded, the memory system
        accumulates stale entries, degrading retrieval quality (arxiv 2602.19320 §5.3).
        """
        return {
            "writes_since_consolidation": self._writes_since_consolidation,
            "last_consolidation_age_s": (
                round(_time.monotonic() - self._last_consolidation_ts, 1)
                if self._last_consolidation_ts > 0
                else None
            ),
            "recent_consolidations": self._consolidation_history[-5:],
            "backlog_critical": self._writes_since_consolidation > 500,
        }

    def check_memory_health(
        self,
        warn_mb: float = 1200,
        critical_mb: float = 2000,
        max_nodes: int = 10000,
    ) -> Dict[str, Any]:
        """Check memory health. Returns health dict."""
        count = self.node_count()
        db_size_mb = self.db_path.stat().st_size / (1024 * 1024) if self.db_path.exists() else 0

        # Measure current memory footprint.
        # On macOS, `ps -o rss=` includes MADV_FREE (reusable) pages, grossly
        # over-reporting (e.g. 6 GB reported vs 1.4 GB actual).  The `footprint`
        # command returns true dirty+compressed memory.  Fall back to ps/ru_maxrss.
        rss_mb = 0.0
        rss_is_peak = False
        try:
            import subprocess as _sp
            import sys as _sys

            if _sys.platform == "darwin":
                # footprint output: "... Footprint: 1382 MB ..."
                _fp_out = _sp.check_output(
                    ["footprint", str(os.getpid())],
                    text=True, timeout=5, stderr=_sp.DEVNULL,
                )
                import re as _re
                _m = _re.search(r"Footprint:\s+([\d.]+)\s+MB", _fp_out)
                if _m:
                    rss_mb = float(_m.group(1))
                else:
                    raise StorageError("footprint parse failed")
            else:
                _ps_out = _sp.check_output(
                    ["ps", "-o", "rss=", "-p", str(os.getpid())],
                    text=True, timeout=5,
                )
                rss_mb = int(_ps_out.strip()) / 1024  # KB → MB
        except Exception as e:
            logger.debug("RSS memory check via ps failed: %s", e)
            # Fall back to ru_maxrss (peak RSS, not current)
            try:
                import resource
                import sys as _sys

                rss_raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                if _sys.platform == "darwin":
                    rss_mb = rss_raw / (1024 * 1024)  # bytes → MB
                else:
                    rss_mb = rss_raw / 1024  # KB → MB
                rss_is_peak = True
            except (ImportError, OSError):
                rss_mb = 0

        status = "healthy"
        warnings = []
        recommendations = []

        if rss_mb > critical_mb:
            status = "critical"
            peak_note = " (peak, not current)" if rss_is_peak else ""
            warnings.append(
                f"RSS memory at {rss_mb:.0f} MB{peak_note} (critical threshold: {critical_mb} MB). "
                "Note: ONNX embedding model loads ~300 MB into memory on first query "
                "and auto-unloads after 10 min idle."
            )
            # Evict caches to reclaim memory
            try:
                self._invalidate_query_cache()
                from omega.embedding import _EMBEDDING_CACHE
                _EMBEDDING_CACHE.clear()
                recommendations.append("Caches evicted to reduce memory pressure")
            except Exception as e:
                logger.debug("Cache eviction during health check failed: %s", e)
        elif rss_mb > warn_mb:
            status = "warning"
            peak_note = " (peak, not current)" if rss_is_peak else ""
            warnings.append(
                f"RSS memory at {rss_mb:.0f} MB{peak_note} (warn threshold: {warn_mb} MB). "
                "Note: ONNX embedding model loads ~300 MB on first query; "
                "auto-unloads after 10 min idle."
            )

        if count > max_nodes:
            warnings.append(f"Node count {count} exceeds max {max_nodes}")
            recommendations.append("Run omega consolidate to deduplicate and prune")

        # Access rate tracking
        zero_access = self._conn.execute(
            "SELECT COUNT(*) FROM memories WHERE access_count = 0"
        ).fetchone()[0]
        never_accessed_pct = (zero_access / count * 100) if count > 0 else 0
        if never_accessed_pct > 80:
            warnings.append(f"{never_accessed_pct:.0f}% of memories never accessed")
            recommendations.append("Run omega_maintain(action='consolidate') to prune stale memories")

        # Embedding quality check — detect silent hash fallback
        embedding_degraded = False
        try:
            from omega.embedding import is_embedding_degraded
            embedding_degraded = is_embedding_degraded()
            if embedding_degraded:
                if status != "critical":
                    status = "warning"
                warnings.append(
                    "Embedding model degraded — using hash fallback. "
                    "Vector search returns meaningless results. "
                    "Restart the server or check ONNX model installation."
                )
                recommendations.append("Run: omega setup --download-model")
        except ImportError:
            pass

        # Maintenance backlog warning (arxiv 2602.19320 — throughput collapse)
        backlog = self.get_maintenance_backlog()
        if backlog["backlog_critical"]:
            if status != "critical":
                status = "warning"
            warnings.append(
                f"Maintenance backlog: {backlog['writes_since_consolidation']} writes "
                "since last consolidation (threshold: 500). Memory may be stale."
            )
            recommendations.append("Run omega_maintain(action='consolidate') to clear backlog")

        return {
            "status": status,
            "memory_mb": rss_mb,
            "db_size_mb": round(db_size_mb, 2),
            "node_count": count,
            "never_accessed_pct": round(never_accessed_pct, 1),
            "zero_access_count": zero_access,
            "embedding_degraded": embedding_degraded,
            "warnings": warnings,
            "recommendations": recommendations,
            "usage": {
                "stores": self.stats.get("stores", 0),
                "queries": self.stats.get("queries", 0),
                "vec_enabled": self._vec_available,
            },
            "format_error_rate": self.get_format_error_rate(),
            "agency_tax": self.get_agency_tax(),
            "maintenance_backlog": backlog,
        }

    # ------------------------------------------------------------------
    # Feedback
    # ------------------------------------------------------------------

    def record_feedback(self, node_id: str, rating: str, reason: Optional[str] = None) -> Dict[str, Any]:
        """Record feedback on a memory node."""
        with self._lock:
            row = self._conn.execute(
                "SELECT content, metadata FROM memories WHERE node_id = ?", (node_id,)
            ).fetchone()
            if not row:
                return {"error": f"Memory node {node_id} not found"}

            content = row[0] or ""
            meta = json.loads(row[1]) if row[1] else {}

            if "feedback_signals" not in meta:
                meta["feedback_signals"] = []
            if "feedback_score" not in meta:
                meta["feedback_score"] = 0

            was_flagged = meta.get("flagged_for_review", False)

            score_delta = {"helpful": 1, "unhelpful": -1, "outdated": -2}.get(rating, 0)
            meta["feedback_score"] += score_delta
            signal = {
                "rating": rating,
                "reason": reason,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            # Attach retrieval context if available (A/B tracking)
            with self._cache_lock:
                retrieval_ctx = self._recent_query_context.get(node_id)
            if retrieval_ctx:
                signal["retrieval_context"] = retrieval_ctx
            meta["feedback_signals"].append(signal)

            if meta["feedback_score"] <= -3:
                meta["flagged_for_review"] = True
                # Log to forgetting audit trail when crossing the threshold
                if not was_flagged:
                    et = meta.get("event_type", "")
                    self._log_forgetting(
                        node_id, content, et, "feedback_flagged",
                        {"feedback_score": meta["feedback_score"], "reason": reason},
                    )

            self._conn.execute("UPDATE memories SET metadata = ? WHERE node_id = ?", (json.dumps(meta), node_id))
            self._commit()

        # Update Thompson arms outside lock (non-critical)
        try:
            from omega.thompson import ThompsonBandit
            bandit = ThompsonBandit(store=self)
            et = meta.get("event_type", "") or ""
            arm_id = f"event_type:{et}" if et else "event_type:unknown"
            success = rating == "helpful"
            bandit.record_outcome(arm_id, "event_type", success)
        except Exception as e:
            logger.debug("Thompson sampling update skipped: %s", e)

        return {
            "node_id": node_id,
            "rating": rating,
            "new_score": meta["feedback_score"],
            "total_signals": len(meta["feedback_signals"]),
            "flagged": meta.get("flagged_for_review", False),
            "cache_invalidated": 0,
        }

    def batch_record_feedback(self, items: List[tuple]) -> int:
        """Record feedback for multiple memories in a single transaction.

        Each item is (node_id, rating, reason). Skips missing nodes.
        Returns count of successfully updated memories.
        """
        updated = 0
        with self._lock:
            for node_id, rating, reason in items:
                row = self._conn.execute(
                    "SELECT content, metadata FROM memories WHERE node_id = ?", (node_id,)
                ).fetchone()
                if not row:
                    continue

                content = row[0] or ""
                meta = json.loads(row[1]) if row[1] else {}

                if "feedback_signals" not in meta:
                    meta["feedback_signals"] = []
                if "feedback_score" not in meta:
                    meta["feedback_score"] = 0

                was_flagged = meta.get("flagged_for_review", False)

                score_delta = {"helpful": 1, "unhelpful": -1, "outdated": -2}.get(rating, 0)
                meta["feedback_score"] += score_delta
                signal = {
                    "rating": rating,
                    "reason": reason,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
                meta["feedback_signals"].append(signal)

                if meta["feedback_score"] <= -3 and not was_flagged:
                    meta["flagged_for_review"] = True
                    et = meta.get("event_type", "")
                    self._log_forgetting(
                        node_id, content, et, "feedback_flagged",
                        {"feedback_score": meta["feedback_score"], "reason": reason},
                    )

                self._conn.execute(
                    "UPDATE memories SET metadata = ? WHERE node_id = ?",
                    (json.dumps(meta), node_id),
                )
                updated += 1

            if updated:
                self._commit()
        return updated

    # ------------------------------------------------------------------
    # Graph / entity
    # ------------------------------------------------------------------

    def add_edge(
        self,
        source_id: str,
        target_id: str,
        edge_type: str = "related",
        weight: float = 1.0,
        metadata: Optional[Dict] = None,
    ) -> bool:
        """Insert an edge between two memories (thread-safe, idempotent)."""
        now = datetime.now(timezone.utc).isoformat()
        meta_str = json.dumps(metadata) if metadata else None
        with self._lock:
            try:
                self._exec(
                    """INSERT OR IGNORE INTO edges
                       (source_id, target_id, edge_type, weight, metadata, created_at)
                       VALUES (?, ?, ?, ?, ?, ?)""",
                    (source_id, target_id, edge_type, round(weight, 3), meta_str, now),
                )
                self._commit()
                return True
            except Exception as e:
                logger.debug(f"add_edge failed: {e}")
                return False

    def get_related_chain(
        self,
        start_id: str,
        max_hops: int = 2,
        min_weight: float = 0.0,
        edge_types: Optional[List[str]] = None,
        exclude_ids: Optional[Set[str]] = None,
        _include_results: bool = False,
    ) -> List[Dict[str, Any]]:
        """Traverse relationship edges from a starting memory up to max_hops.

        Returns a list of dicts with: node_id, content, hop, weight, edge_type.
        Nodes are deduplicated (each appears at its shortest hop distance).

        Args:
            exclude_ids: Node IDs to skip during traversal.
            _include_results: If True, include MemoryResult under "_result" key
                (internal use by find_relevant spreading activation).
        """
        if max_hops < 1 or max_hops > 5:
            max_hops = min(max(max_hops, 1), 5)

        visited: Dict[str, Dict[str, Any]] = {}
        skip = set(exclude_ids) if exclude_ids else set()
        frontier = {start_id}

        for hop in range(1, max_hops + 1):
            if not frontier:
                break
            next_frontier: Set[str] = set()
            for node_id in frontier:
                # Query edges in both directions (undirected graph)
                rows = self._conn.execute(
                    """SELECT source_id, target_id, edge_type, weight, metadata
                       FROM edges
                       WHERE (source_id = ? OR target_id = ?)
                       AND weight >= ?""",
                    (node_id, node_id, min_weight),
                ).fetchall()

                for source, target, etype, weight, edge_meta_raw in rows:
                    neighbor = target if source == node_id else source
                    if neighbor == start_id or neighbor in visited or neighbor in skip:
                        continue
                    if edge_types and etype not in edge_types:
                        continue
                    # Fetch the memory content
                    mem_row = self._conn.execute(
                        """SELECT node_id, content, metadata, created_at,
                                  access_count, last_accessed, ttl_seconds
                           FROM memories WHERE node_id = ?""",
                        (neighbor,),
                    ).fetchone()
                    if not mem_row:
                        continue

                    result = self._row_to_result(mem_row)
                    try:
                        edge_metadata = json.loads(edge_meta_raw) if edge_meta_raw else {}
                    except Exception:
                        edge_metadata = {}
                    entry = {
                        "node_id": neighbor,
                        "content": result.content,
                        "metadata": result.metadata,
                        "created_at": result.created_at.isoformat() if result.created_at else "",
                        "hop": hop,
                        "weight": weight,
                        "edge_type": etype,
                        "edge_metadata": edge_metadata,
                    }
                    if _include_results:
                        entry["_result"] = result
                    visited[neighbor] = entry
                    next_frontier.add(neighbor)

            frontier = next_frontier

        # Sort by hop (nearest first), then by weight (strongest first)
        results = sorted(visited.values(), key=lambda x: (x["hop"], -x["weight"]))
        return results

    # ------------------------------------------------------------------
    # Say/Do entity index helpers
    # ------------------------------------------------------------------

    def write_entity_index(
        self,
        entity_name: str,
        entity_type: str = "person",
        statement_count: int = 0,
        outcome_count: int = 0,
        contradiction_score: float = 0.0,
        follow_through_rate: float = 0.0,
        metadata: Optional[Dict] = None,
    ) -> bool:
        """Upsert an entity into the entity_index table."""
        now = datetime.now(timezone.utc).isoformat()
        meta_str = json.dumps(metadata) if metadata else None
        with self._lock:
            try:
                self._conn.execute(
                    """INSERT INTO entity_index
                       (entity_name, entity_type, statement_count, outcome_count,
                        contradiction_score, follow_through_rate, first_seen, last_updated, metadata)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                       ON CONFLICT(entity_name) DO UPDATE SET
                           entity_type = excluded.entity_type,
                           statement_count = excluded.statement_count,
                           outcome_count = excluded.outcome_count,
                           contradiction_score = excluded.contradiction_score,
                           follow_through_rate = excluded.follow_through_rate,
                           last_updated = excluded.last_updated,
                           metadata = excluded.metadata""",
                    (entity_name, entity_type, statement_count, outcome_count,
                     contradiction_score, follow_through_rate, now, now, meta_str),
                )
                self._commit()
                return True
            except Exception as e:
                logger.debug(f"write_entity_index failed: {e}")
                return False

    def get_entity_index(self, entity_name: str) -> Optional[Dict[str, Any]]:
        """Get an entity's profile from the entity_index table."""
        row = self._conn.execute(
            """SELECT entity_name, entity_type, statement_count, outcome_count,
                      contradiction_score, follow_through_rate, first_seen, last_updated, metadata
               FROM entity_index WHERE entity_name = ?""",
            (entity_name,),
        ).fetchone()
        if not row:
            return None
        return {
            "entity_name": row[0],
            "entity_type": row[1],
            "statement_count": row[2],
            "outcome_count": row[3],
            "contradiction_score": row[4],
            "follow_through_rate": row[5],
            "first_seen": row[6],
            "last_updated": row[7],
            "metadata": json.loads(row[8]) if row[8] else {},
        }

    def get_entity_list(
        self,
        entity_type: Optional[str] = None,
        min_statements: int = 0,
        order_by: str = "last_updated",
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        """List entities from the entity_index table."""
        sql = "SELECT entity_name, entity_type, statement_count, outcome_count, contradiction_score, follow_through_rate, first_seen, last_updated FROM entity_index WHERE 1=1"
        params: list = []
        if entity_type:
            sql += " AND entity_type = ?"
            params.append(entity_type)
        if min_statements > 0:
            sql += " AND statement_count >= ?"
            params.append(min_statements)
        valid_orders = {"last_updated", "contradiction_score", "statement_count", "entity_name"}
        col = order_by if order_by in valid_orders else "last_updated"
        desc = " DESC" if col != "entity_name" else ""
        sql += f" ORDER BY {col}{desc} LIMIT ?"
        params.append(limit)
        rows = self._conn.execute(sql, params).fetchall()
        return [
            {
                "entity_name": r[0],
                "entity_type": r[1],
                "statement_count": r[2],
                "outcome_count": r[3],
                "contradiction_score": r[4],
                "follow_through_rate": r[5],
                "first_seen": r[6],
                "last_updated": r[7],
            }
            for r in rows
        ]

    def get_entity_nodes(self, entity_name: str, event_type: Optional[str] = None, limit: int = 100) -> List[Dict[str, Any]]:
        """Get memory nodes associated with an entity via metadata."""
        sql = "SELECT node_id, content, metadata, created_at, event_type FROM memories WHERE metadata LIKE ?"
        params: list = [f'%"entity_name":"{entity_name}"%']
        if event_type:
            sql += " AND event_type = ?"
            params.append(event_type)
        sql += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)
        rows = self._conn.execute(sql, params).fetchall()
        return [
            {
                "node_id": r[0],
                "content": r[1],
                "metadata": json.loads(r[2]) if r[2] else {},
                "created_at": r[3],
                "event_type": r[4],
            }
            for r in rows
        ]

    def get_edges_by_type(self, edge_type: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Get edges filtered by type."""
        rows = self._conn.execute(
            """SELECT source_id, target_id, edge_type, weight, metadata, created_at
               FROM edges WHERE edge_type = ? ORDER BY created_at DESC LIMIT ?""",
            (edge_type, limit),
        ).fetchall()
        return [
            {
                "source_id": r[0],
                "target_id": r[1],
                "edge_type": r[2],
                "weight": r[3],
                "metadata": json.loads(r[4]) if r[4] else {},
                "created_at": r[5],
            }
            for r in rows
        ]

    # ------------------------------------------------------------------
    # Export / Import
    # ------------------------------------------------------------------

    def export_to_file(self, filepath: Path) -> Dict[str, Any]:
        """Export all memories to a JSON file."""
        rows = self._conn.execute(
            """SELECT node_id, content, metadata, created_at,
                      access_count, last_accessed, ttl_seconds
               FROM memories ORDER BY created_at"""
        ).fetchall()

        nodes = []
        sessions = set()
        for row in rows:
            result = self._row_to_result(row)
            nodes.append(
                {
                    "id": result.id,
                    "content": result.content,
                    "metadata": result.metadata,
                    "created_at": result.created_at.isoformat(),
                    "access_count": result.access_count,
                    "last_accessed": result.last_accessed.isoformat() if result.last_accessed else None,
                    "ttl_seconds": result.ttl_seconds,
                }
            )
            sid = result.metadata.get("session_id")
            if sid:
                sessions.add(sid)

        export_data = {
            "version": "omega-sqlite-v1",
            "exported_at": datetime.now(timezone.utc).isoformat(),
            "node_count": len(nodes),
            "session_count": len(sessions),
            "nodes": nodes,
        }

        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        export_json = json.dumps(export_data, indent=2)
        encrypted = False

        # Encrypt export if OMEGA_ENCRYPT is enabled
        from omega.crypto import is_enabled as crypto_enabled, encrypt as crypto_encrypt
        if crypto_enabled():
            encrypted_content = crypto_encrypt(export_json)
            if encrypted_content != export_json:  # Encryption actually happened
                export_bytes = encrypted_content.encode("utf-8")
                encrypted = True
            else:
                export_bytes = export_json.encode("utf-8")
        else:
            export_bytes = export_json.encode("utf-8")

        # Write with restricted permissions (0o600) — export contains memory content
        _flags = os.O_CREAT | os.O_WRONLY | os.O_TRUNC
        if hasattr(os, "O_NOFOLLOW"):
            _flags |= os.O_NOFOLLOW
        fd = os.open(str(filepath), _flags, 0o600)
        try:
            os.write(fd, export_bytes)
        finally:
            os.close(fd)

        result = {
            "filepath": str(filepath),
            "node_count": len(nodes),
            "session_count": len(sessions),
            "file_size_kb": filepath.stat().st_size / 1024,
            "exported_at": export_data["exported_at"],
        }
        if encrypted:
            result["encrypted"] = True
        return result

    def import_from_file(self, filepath: Path, clear_existing: bool = True) -> Dict[str, Any]:
        """Import memories from a JSON file. Auto-detects encrypted exports."""
        if Path(filepath).is_symlink():
            raise ValidationError("Import file must not be a symlink")
        raw_content = Path(filepath).read_text()

        # Auto-detect and decrypt encrypted exports (prefixed with "ENC:")
        if raw_content.startswith("ENC:"):
            from omega.crypto import decrypt as crypto_decrypt
            raw_content = crypto_decrypt(raw_content)

        data = json.loads(raw_content)
        nodes = data.get("nodes", [])

        # Serialize against the deferred-startup thread (PRAGMA integrity_check
        # + WAL checkpoint share this connection). Without the lock, that
        # thread's cursor can still be mid-statement at the SQLite C-API
        # level when executescript() below issues its implicit COMMIT,
        # producing "cannot commit transaction - SQL statements in progress"
        # on Ubuntu CI 3.40. _lock is an RLock, so self.store() below
        # re-enters fine.
        with self._lock:
            if clear_existing:
                # Older SQLite (Ubuntu CI 3.40) is strict about open prepared
                # statements at COMMIT time. Force-finalize any Cursor objects
                # that may still be unreaped from init (_refresh_hot_cache,
                # schema migrations) or deferred startup. CPython's refcount
                # GC usually handles this synchronously, but a brief delay
                # between cursor scope-out and reclamation is enough to lose
                # the race. gc.collect() makes it deterministic; on newer
                # SQLite it's a harmless ~ms no-op.
                gc.collect()
                # Atomic clear via executescript: BEGIN/DELETE/COMMIT batched
                # through sqlite3_exec, which finalizes each prepared statement
                # before the next runs.
                script = ["BEGIN;", "DELETE FROM memories;"]
                if self._vec_available:
                    script.append("DELETE FROM memories_vec;")
                script.extend(["DELETE FROM edges;", "COMMIT;"])
                try:
                    self._conn.executescript(" ".join(script))
                except Exception as e:
                    logger.debug("Import clear failed, rolling back: %s", e)
                    try:
                        self._conn.executescript("ROLLBACK;")
                    except Exception:
                        pass
                    raise

            imported = 0
            for node_data in nodes:
                try:
                    self.store(
                        content=node_data["content"],
                        session_id=node_data.get("metadata", {}).get("session_id"),
                        metadata=node_data.get("metadata"),
                        ttl_seconds=node_data.get("ttl_seconds"),
                        skip_inference=True,
                    )
                    imported += 1
                except Exception as e:
                    logger.debug(f"Import failed for node: {e}")

        return {
            "filepath": str(filepath),
            "node_count": imported,
            "session_count": data.get("session_count", 0),
        }
