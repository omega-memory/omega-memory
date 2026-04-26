"""CRUD operations mixin for SQLiteStore."""

import hashlib
import logging
import os
import time as _time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from omega import json_compat as json
from omega.exceptions import StorageError
from ._types import MemoryResult, _serialize_f32, _canonicalize

logger = logging.getLogger("omega.sqlite_store")


class StoreMixin:
    """CRUD operations for SQLiteStore — store, get, update, delete, batch."""

    def store(
        self,
        content: str,
        session_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        embedding: Optional[List[float]] = None,
        dependencies: Optional[List[str]] = None,
        ttl_seconds: Optional[int] = None,
        graphs: Optional[List[str]] = None,
        skip_inference: bool = False,
        entity_id: Optional[str] = None,
        agent_type: Optional[str] = None,
        derived_from: Optional[str] = None,
        source_uri: Optional[str] = None,
        status: Optional[str] = None,
    ) -> str:
        """Store a memory. Returns the node ID."""
        _t0_agency = _time.monotonic()
        self._total_write_count += 1
        if not content:
            raise StorageError("content must be a non-empty string")
        if len(content) > self._MAX_CONTENT_SIZE:
            raise StorageError(
                f"Content size ({len(content):,} bytes) exceeds limit ({self._MAX_CONTENT_SIZE:,} bytes). "
                "Override with OMEGA_MAX_CONTENT_SIZE env var."
            )
        meta = dict(metadata or {})
        if session_id:
            meta["session_id"] = session_id

        # Auto-generate embedding if not provided (outside lock — CPU-bound)
        if embedding is None:
            from omega.embedding import generate_embedding, get_embedding_model_info, is_embedding_degraded, get_active_backend

            embedding = generate_embedding(content)
            # Discard hash-fallback embeddings only when a real backend has been
            # established (degradation from real -> hash). When no backend exists
            # (test/bootstrap), hash embeddings are acceptable.
            if is_embedding_degraded() and get_active_backend() is not None:
                logger.warning("store: hash-fallback embedding discarded — text search only")
                embedding = None
            try:
                model_info = get_embedding_model_info()
                meta["_embedding_model"] = model_info["model_name"]
                meta["_embedding_model_version"] = model_info["model_version"]
            except Exception as e:
                logger.debug("Could not attach embedding model info: %s", e)

        content_hash = hashlib.sha256(content.encode()).hexdigest()
        canonical_hash = hashlib.sha256(_canonicalize(content).encode()).hexdigest()

        # Pre-compute embedding dedup outside lock to reduce contention
        # (vec_query is the expensive part; the UPDATE+commit stays inside the lock)
        _vec_dedup_candidate = None
        if embedding and not skip_inference and self._vec_available:
            try:
                similar = self._vec_query(embedding, limit=1)
                if similar:
                    top_rowid, distance = similar[0]
                    similarity = 1.0 - distance
                    if similarity >= self.DEFAULT_EMBEDDING_DEDUP_THRESHOLD:
                        _vec_dedup_candidate = (top_rowid, similarity)
            except Exception as e:
                logger.debug("Embedding dedup pre-check failed: %s", e)

        with self._lock:
            # Capacity check (inside lock — queries shared connection)
            self._capacity_warning = None
            if self._MAX_NODES > 0:
                count = self._exec("SELECT COUNT(*) FROM memories").fetchone()[0]
                if count >= self._MAX_NODES:
                    raise StorageError(
                        f"Node count ({count:,}) has reached the limit ({self._MAX_NODES:,}). "
                        "Run omega_consolidate to prune, or raise OMEGA_MAX_NODES env var."
                    )
                if count >= int(self._MAX_NODES * 0.9):
                    self._capacity_warning = (
                        f"Memory store is at {count:,}/{self._MAX_NODES:,} ({count*100//self._MAX_NODES}% capacity). "
                        "Consider running omega_consolidate or omega_compact to free space."
                    )
                    logger.warning(self._capacity_warning)
            self._invalidate_query_cache(new_content=content)
            # Canonical dedup (#6): catch reformatted duplicates
            canonical_existing = self._exec(
                """SELECT node_id, id FROM memories WHERE canonical_hash = ?
                   AND (ttl_seconds IS NULL
                        OR datetime(created_at, '+' || ttl_seconds || ' seconds') > datetime('now'))
                   LIMIT 1""",
                (canonical_hash,),
            ).fetchone()
            if canonical_existing:
                self._exec(
                    "UPDATE memories SET access_count = access_count + 1 WHERE node_id = ?",
                    (canonical_existing[0],),
                )
                self._commit()
                self.stats.setdefault("dedup_canonical", 0)
                self.stats["dedup_canonical"] += 1
                self._record_timing("write", (_time.monotonic() - _t0_agency) * 1000)
                return canonical_existing[0]

            # Exact-match dedup via content hash (skip expired memories)
            existing = self._exec(
                """SELECT node_id, id FROM memories WHERE content_hash = ?
                   AND (ttl_seconds IS NULL
                        OR datetime(created_at, '+' || ttl_seconds || ' seconds') > datetime('now'))
                   LIMIT 1""",
                (content_hash,),
            ).fetchone()
            if existing:
                self._exec(
                    "UPDATE memories SET access_count = access_count + 1 WHERE node_id = ?", (existing[0],)
                )
                self._commit()
                self.stats.setdefault("dedup_exact", 0)
                self.stats["dedup_exact"] += 1
                self._record_timing("write", (_time.monotonic() - _t0_agency) * 1000)
                return existing[0]

            # Embedding-based dedup (vec_query already done outside lock)
            if _vec_dedup_candidate:
                try:
                    top_rowid, _similarity = _vec_dedup_candidate
                    row = self._exec(
                        "SELECT node_id FROM memories WHERE id = ?", (top_rowid,)
                    ).fetchone()
                    if row:
                        self._exec(
                            "UPDATE memories SET access_count = access_count + 1 WHERE id = ?", (top_rowid,)
                        )
                        self._commit()
                        self.stats["embedding_dedup_skips"] += 1
                        self._record_timing("write", (_time.monotonic() - _t0_agency) * 1000)
                        return row[0]
                except Exception as e:
                    logger.debug("Embedding dedup check failed: %s", e)

            # Generate node ID
            node_id = f"mem-{uuid.uuid4().hex[:12]}"

            event_type = meta.get("event_type") or meta.get("type")
            project = meta.get("project") or os.getcwd()
            now = datetime.now(timezone.utc).isoformat()

            # Determine priority from metadata or event type default
            priority = meta.get("priority") or self._DEFAULT_PRIORITY.get(event_type, 3)
            referenced_date = meta.get("referenced_date")

            # Wire entity_id from metadata if not passed directly
            effective_entity_id = entity_id or meta.get("entity_id")

            # Wire agent_type from metadata if not passed directly
            effective_agent_type = agent_type or meta.get("agent_type")

            # P5: Extract keywords for enhanced BM25 retrieval
            extracted_keywords = self._extract_keywords(content)

            # Classify memory type from event_type
            memory_type = self._MEMORY_TYPE_MAP.get(event_type, "semantic")
            meta["memory_type"] = memory_type

            # Bi-temporal: valid_from defaults to referenced_date or created_at
            valid_from = referenced_date or now

            # Context graph: wire derived_from, source_uri, status from params or metadata
            effective_derived_from = derived_from or meta.get("derived_from")
            effective_source_uri = source_uri or meta.get("source_uri")
            effective_status = status or meta.get("status") or "active"

            _insert_cur = self._exec(
                """INSERT INTO memories
                   (node_id, content, metadata, created_at, access_count,
                    ttl_seconds, session_id, event_type, project, content_hash,
                    priority, referenced_date, entity_id, agent_type, canonical_hash,
                    extracted_keywords, memory_type, valid_from,
                    derived_from, source_uri, status)
                   VALUES (?, ?, ?, ?, 0, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    node_id,
                    content,
                    json.dumps(meta),
                    now,
                    ttl_seconds,
                    session_id,
                    event_type,
                    project,
                    content_hash,
                    priority,
                    referenced_date,
                    effective_entity_id,
                    effective_agent_type,
                    canonical_hash,
                    extracted_keywords,
                    memory_type,
                    valid_from,
                    effective_derived_from,
                    effective_source_uri,
                    effective_status,
                ),
            )

            # Get the rowid for the vec table — use cursor.lastrowid to avoid a
            # SELECT race condition under concurrent writes (WAL mode + Waitress
            # multi-thread): the follow-up SELECT could return None if another
            # thread's transaction hasn't been seen yet by this connection.
            rowid = _insert_cur.lastrowid

            # Insert embedding into vec table
            if embedding and self._vec_available:
                try:
                    self._exec(
                        "INSERT INTO memories_vec (rowid, embedding) VALUES (?, ?)", (rowid, _serialize_f32(embedding))
                    )
                except Exception as e:
                    logger.debug(f"Vec insert failed: {e}")

            # Add causal edges if dependencies provided
            if dependencies:
                for dep_id in dependencies:
                    self._exec(
                        """INSERT INTO edges (source_id, target_id, edge_type, created_at)
                           VALUES (?, ?, 'causal', ?)""",
                        (node_id, dep_id, now),
                    )

            # Add derived_from edge if lineage is specified
            if effective_derived_from:
                self._exec(
                    """INSERT OR IGNORE INTO edges (source_id, target_id, edge_type, created_at)
                       VALUES (?, ?, 'derived_from', ?)""",
                    (node_id, effective_derived_from, now),
                )

            self._commit()
            self.stats["stores"] += 1
            self._writes_since_consolidation += 1

        # Post-store: contradiction detection (outside lock — CPU-bound)
        # Finds existing memories that contradict the new one and annotates both.
        if not skip_inference and embedding and self._vec_available:
            try:
                self._last_contradiction_results = self._check_contradictions(
                    node_id, content, embedding
                )
            except Exception as e:
                logger.debug("Contradiction check failed (non-blocking): %s", e)

        self._record_timing("write", (_time.monotonic() - _t0_agency) * 1000)
        return node_id

    def get_last_contradiction_results(self) -> list:
        """Return contradiction results from the most recent store() call. Consume-once."""
        results = self._last_contradiction_results
        self._last_contradiction_results = []
        return results

    def get_node(self, node_id: str, track_access: bool = True) -> Optional[MemoryResult]:
        """Get a node by ID.

        Args:
            node_id: The memory node ID to retrieve.
            track_access: If True (default), increment access_count and
                update last_accessed. Pass False for internal lookups
                (e.g. contradiction checks, validation) to avoid
                inflating access counts.
        """
        with self._lock:
            row = self._exec(
                """SELECT node_id, content, metadata, created_at, access_count,
                          last_accessed, ttl_seconds
                   FROM memories WHERE node_id = ?""",
                (node_id,),
            ).fetchone()
            if not row:
                return None

            if track_access:
                # Update access tracking
                now = datetime.now(timezone.utc).isoformat()
                self._exec(
                    "UPDATE memories SET access_count = access_count + 1, last_accessed = ? WHERE node_id = ?",
                    (now, node_id),
                )
                self._commit()

            return self._row_to_result(row)

    def delete_node(self, node_id: str) -> bool:
        """Delete a node and its edges."""
        self._invalidate_query_cache()
        with self._lock:
            # Get rowid + content + event_type for audit log before deletion
            row = self._exec(
                "SELECT id, content, metadata FROM memories WHERE node_id = ?", (node_id,)
            ).fetchone()
            if not row:
                return False

            rowid = row[0]
            content = row[1] or ""
            meta = json.loads(row[2]) if row[2] else {}
            event_type = meta.get("event_type", "")

            # Log to forgetting audit trail before deleting
            self._log_forgetting(node_id, content, event_type, "user_deleted")
            self._queue_cloud_delete(rowid)

            self._exec("DELETE FROM memories WHERE node_id = ?", (node_id,))
            self._exec("DELETE FROM edges WHERE source_id = ? OR target_id = ?", (node_id, node_id))

            if self._vec_available:
                try:
                    self._exec("DELETE FROM memories_vec WHERE rowid = ?", (rowid,))
                except Exception as e:
                    logger.debug("Failed to delete vec embedding rowid=%s: %s", rowid, e)

            self._commit()
        return True

    def node_count(self) -> int:
        """Return total number of memories."""
        row = self._conn.execute("SELECT COUNT(*) FROM memories").fetchone()
        return row[0] if row else 0

    def edge_count(self) -> int:
        """Return total number of edges."""
        row = self._conn.execute("SELECT COUNT(*) FROM edges").fetchone()
        return row[0] if row else 0

    def get_last_capture_time(self) -> Optional[str]:
        """Return ISO timestamp of the most recent memory, or None."""
        row = self._conn.execute("SELECT created_at FROM memories ORDER BY created_at DESC LIMIT 1").fetchone()
        return row[0] if row else None

    def get_session_event_counts(self, session_id: str) -> Dict[str, int]:
        """Count memories by event_type for a given session."""
        rows = self._conn.execute(
            "SELECT event_type, COUNT(*) "
            "FROM memories WHERE session_id = ? AND event_type IS NOT NULL "
            "GROUP BY event_type",
            (session_id,),
        ).fetchall()
        return {r[0]: r[1] for r in rows if r[0]}

    def update_node(
        self,
        node_id: str,
        content: Optional[str] = None,
        metadata: Optional[Dict] = None,
        access_count: Optional[int] = None,
    ) -> bool:
        """Update fields on an existing node."""
        self._invalidate_query_cache()
        sets = []
        params = []
        new_embedding = None
        if content is not None:
            sets.append("content = ?")
            params.append(content)
            sets.append("content_hash = ?")
            params.append(hashlib.sha256(content.encode()).hexdigest())
            sets.append("canonical_hash = ?")
            params.append(hashlib.sha256(_canonicalize(content).encode()).hexdigest())
            # Re-embed to keep vec table in sync (CPU-bound, done outside lock)
            if self._vec_available:
                try:
                    from omega.embedding import generate_embedding, get_active_backend, is_embedding_degraded

                    new_embedding = generate_embedding(content)
                    if is_embedding_degraded() and get_active_backend() is not None:
                        new_embedding = None  # Hash fallback from real backend — don't store
                except Exception as e:
                    logger.debug("update_node: re-embed failed: %s", e)
        if metadata is not None:
            sets.append("metadata = ?")
            params.append(json.dumps(metadata))
            # Update denormalized columns
            sets.append("event_type = ?")
            params.append(metadata.get("event_type") or metadata.get("type"))
            sets.append("session_id = ?")
            params.append(metadata.get("session_id"))
            sets.append("project = ?")
            params.append(metadata.get("project"))
        if access_count is not None:
            sets.append("access_count = ?")
            params.append(access_count)

        if not sets:
            return False

        with self._lock:
            params.append(node_id)
            self._exec(f"UPDATE memories SET {', '.join(sets)} WHERE node_id = ?", params)
            # Update vec embedding if content changed
            if new_embedding is not None:
                row = self._exec("SELECT id FROM memories WHERE node_id = ?", (node_id,)).fetchone()
                if row:
                    try:
                        self._exec("DELETE FROM memories_vec WHERE rowid = ?", (row[0],))
                        self._exec(
                            "INSERT INTO memories_vec (rowid, embedding) VALUES (?, ?)",
                            (row[0], _serialize_f32(new_embedding)),
                        )
                    except Exception as e:
                        logger.debug("update_node: vec update failed: %s", e)
            self._commit()
        return True

    # ------------------------------------------------------------------
    # Batch operations
    # ------------------------------------------------------------------

    def batch_store(self, items: List[Dict[str, Any]]) -> List[str]:
        """Store multiple memories efficiently."""
        if not items:
            return []

        # Batch-generate embeddings for items without them
        items_needing = [(i, item) for i, item in enumerate(items) if item.get("embedding") is None]
        if items_needing:
            try:
                from omega.embedding import generate_embeddings_batch, get_active_backend

                texts = [item["content"] for _, item in items_needing]
                embeddings = generate_embeddings_batch(texts)
                backend = get_active_backend()
                if backend is not None:
                    # Real ML embeddings — store in vec table
                    for (idx, item), emb in zip(items_needing, embeddings):
                        item["embedding"] = emb
                else:
                    # Hash fallback — do NOT store in vec table (incompatible with ML embeddings)
                    logger.warning(
                        f"batch_store: skipping {len(texts)} embeddings (hash fallback — "
                        f"would corrupt vector search). Memories will be findable via text search only."
                    )
            except Exception as e:
                logger.warning(f"batch_store: embedding generation failed: {e}")

        ids = []
        # Hold the lock for the entire batch to avoid per-item lock
        # acquisition overhead (RLock allows store() to re-enter).
        with self._lock:
            for item in items:
                node_id = self.store(
                    content=item["content"],
                    session_id=item.get("session_id"),
                    metadata=item.get("metadata"),
                    embedding=item.get("embedding"),
                    dependencies=item.get("dependencies"),
                    ttl_seconds=item.get("ttl_seconds"),
                )
                ids.append(node_id)
        # Single cache invalidation after all inserts
        self._invalidate_query_cache()

        return ids

    def mark_superseded(self, node_id: str, superseded_by: str) -> bool:
        """Mark a memory as superseded by a newer memory.

        Sets metadata.superseded=True and metadata.superseded_by on the target,
        and invalidates the query cache.

        Returns True if the node was found and updated.
        """
        self._invalidate_query_cache()
        with self._lock:
            row = self._conn.execute(
                "SELECT metadata, content, event_type FROM memories WHERE node_id = ?",
                (node_id,),
            ).fetchone()
            if not row:
                return False
            meta = json.loads(row[0]) if row[0] else {}
            meta["superseded"] = True
            meta["superseded_by"] = superseded_by
            meta["superseded_at"] = datetime.now(timezone.utc).isoformat()
            self._conn.execute(
                "UPDATE memories SET metadata = ? WHERE node_id = ?",
                (json.dumps(meta), node_id),
            )
            # Bi-temporal: set valid_until when superseding
            now_str = meta["superseded_at"]
            self._conn.execute(
                "UPDATE memories SET valid_until = ?, status = 'superseded' WHERE node_id = ?",
                (now_str, node_id),
            )
            self._log_forgetting(
                node_id, row[1] or "", row[2] or "",
                "ingest_superseded", {"superseded_by": superseded_by},
            )
            self._commit()
        return True

    # ------------------------------------------------------------------
    # Contradiction detection
    # ------------------------------------------------------------------

    _CONTRADICTION_CANDIDATE_LIMIT = 10
    _CONTRADICTION_CONFIDENCE_THRESHOLD = 0.4
    _TEMPORAL_SUPERSESSION_THRESHOLD = 0.75
    _TEMPORAL_SUPERSESSION_TYPES = frozenset({
        "decision", "user_preference", "lesson_learned", "error_pattern",
    })

    def _check_contradictions(
        self, new_node_id: str, new_content: str, embedding: List[float]
    ) -> list:
        """Check if the newly stored memory contradicts existing ones.

        First applies temporal supersession: if a candidate has the same
        event_type, high embedding similarity, and is older, it is marked
        superseded (no signal words required).

        Then runs contradiction detection heuristics on remaining candidates
        and annotates metadata on both sides.
        Never raises — all errors are logged and swallowed.

        Returns:
            List of dicts with keys: node_id, confidence, reason, content_preview.
            Empty list if no contradictions found.
        """
        from omega.contradictions import detect_contradictions

        # Find similar existing memories (exclude the one we just stored)
        similar = self._vec_query(embedding, limit=self._CONTRADICTION_CANDIDATE_LIMIT + 1)
        if not similar:
            return []

        candidate_ids = []
        candidate_contents = []
        candidate_similarities = []
        # Batch fetch all candidate metadata instead of N individual SELECTs
        rowids = [rowid for rowid, _ in similar]
        distances = {rowid: distance for rowid, distance in similar}
        if rowids:
            placeholders = ",".join("?" * len(rowids))
            rows = self._conn.execute(
                f"SELECT id, node_id, content FROM memories WHERE id IN ({placeholders})",
                rowids,
            ).fetchall()
            row_map = {r[0]: (r[1], r[2]) for r in rows}
            for rowid in rowids:
                if rowid not in row_map:
                    continue
                node_id_val, content_val = row_map[rowid]
                if node_id_val == new_node_id:
                    continue
                candidate_ids.append(node_id_val)
                candidate_contents.append(content_val)
                candidate_similarities.append(1.0 - distances[rowid])

        if not candidate_contents:
            return []

        # --- Temporal supersession (runs before contradiction detection) ---
        # If a candidate has the same event_type AND high similarity AND is
        # older, the older memory is superseded by the new one.
        new_row = self._conn.execute(
            "SELECT event_type, created_at FROM memories WHERE node_id = ?",
            (new_node_id,),
        ).fetchone()
        new_event_type = new_row[0] if new_row else None
        new_created_at = self._parse_dt(new_row[1]) if new_row else None

        superseded_indices: set = set()
        if (
            new_event_type
            and new_event_type in self._TEMPORAL_SUPERSESSION_TYPES
            and new_created_at
        ):
            # Batch fetch event_type and created_at for all candidates
            cand_placeholders = ",".join("?" * len(candidate_ids))
            cand_rows = self._conn.execute(
                f"SELECT node_id, event_type, created_at FROM memories WHERE node_id IN ({cand_placeholders})",
                candidate_ids,
            ).fetchall()
            cand_meta = {r[0]: (r[1], r[2]) for r in cand_rows}
            for i, cand_id in enumerate(candidate_ids):
                if candidate_similarities[i] < self._TEMPORAL_SUPERSESSION_THRESHOLD:
                    continue
                cand_info = cand_meta.get(cand_id)
                if not cand_info or cand_info[0] != new_event_type:
                    continue
                cand_created = self._parse_dt(cand_info[1])
                if cand_created and cand_created < new_created_at:
                    self.mark_superseded(cand_id, new_node_id)
                    superseded_indices.add(i)
                    logger.info(
                        "Temporal supersession: %s superseded by %s "
                        "(type=%s, similarity=%.3f)",
                        cand_id, new_node_id, new_event_type,
                        candidate_similarities[i],
                    )

            if superseded_indices:
                self.stats.setdefault("temporal_supersessions", 0)
                self.stats["temporal_supersessions"] += len(superseded_indices)

        # Remove superseded candidates before contradiction detection
        if superseded_indices:
            candidate_ids = [
                v for i, v in enumerate(candidate_ids) if i not in superseded_indices
            ]
            candidate_contents = [
                v for i, v in enumerate(candidate_contents) if i not in superseded_indices
            ]

        if not candidate_contents:
            return []

        results = detect_contradictions(
            new_content,
            candidate_contents,
            contradiction_threshold=self._CONTRADICTION_CONFIDENCE_THRESHOLD,
        )

        if not results:
            return []

        with self._lock:
            for r in results:
                old_node_id = candidate_ids[r.candidate_index]

                # Annotate the NEW memory: what it contradicts
                new_row = self._conn.execute(
                    "SELECT metadata FROM memories WHERE node_id = ?",
                    (new_node_id,),
                ).fetchone()
                if new_row:
                    new_meta = json.loads(new_row[0]) if new_row[0] else {}
                    contradicts = new_meta.get("contradicts", [])
                    contradicts.append({
                        "node_id": old_node_id,
                        "confidence": r.confidence,
                        "reason": r.reason,
                    })
                    new_meta["contradicts"] = contradicts
                    self._conn.execute(
                        "UPDATE memories SET metadata = ? WHERE node_id = ?",
                        (json.dumps(new_meta), new_node_id),
                    )

                # Annotate the OLD memory: mark as potentially superseded
                old_row = self._conn.execute(
                    "SELECT metadata FROM memories WHERE node_id = ?",
                    (old_node_id,),
                ).fetchone()
                if old_row:
                    old_meta = json.loads(old_row[0]) if old_row[0] else {}
                    contradicted_by = old_meta.get("contradicted_by", [])
                    contradicted_by.append({
                        "node_id": new_node_id,
                        "confidence": r.confidence,
                        "reason": r.reason,
                    })
                    old_meta["contradicted_by"] = contradicted_by
                    self._conn.execute(
                        "UPDATE memories SET metadata = ? WHERE node_id = ?",
                        (json.dumps(old_meta), old_node_id),
                    )

                # Add a "contradicts" edge between the two memories
                now = datetime.now(timezone.utc).isoformat()
                self._conn.execute(
                    """INSERT OR IGNORE INTO edges
                       (source_id, target_id, edge_type, weight, created_at)
                       VALUES (?, ?, 'contradicts', ?, ?)""",
                    (new_node_id, old_node_id, r.confidence, now),
                )

            self._commit()

        self.stats.setdefault("contradictions_found", 0)
        self.stats["contradictions_found"] += len(results)
        logger.info(
            "Contradiction check: %d contradiction(s) found for %s",
            len(results), new_node_id,
        )

        # Build surfaced results for caller visibility
        surfaced = []
        for r in results:
            old_nid = candidate_ids[r.candidate_index]
            surfaced.append({
                "node_id": old_nid,
                "confidence": round(r.confidence, 3),
                "reason": r.reason,
                "content_preview": candidate_contents[r.candidate_index][:80],
            })
        self._last_contradiction_results = surfaced
        return surfaced
