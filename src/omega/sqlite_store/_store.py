"""CRUD operations mixin for SQLiteStore."""

import hashlib
import logging
import os
import time as _time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from omega import json_compat as json
from omega.exceptions import StorageError
from ._types import (
    EMBEDDING_DIM,
    MemoryResult,
    SupersessionRecord,
    _canonicalize,
    _serialize_f32,
    coerce_priority,
)

logger = logging.getLogger("omega.sqlite_store")

_PRIORITY_EDIT_HISTORY_LIMIT = 20


@dataclass(frozen=True)
class _Neighbour:
    """An active same-scope memory near a newly stored one."""

    node_id: str
    content: str
    similarity: float
    event_type: Optional[str]
    created_at: Optional[datetime]


def _check_embedding_dim(embedding: List[float]) -> None:
    """Raise if ``embedding`` does not match the configured vector dimension.

    Checked before the insert so the failure names both dimensions. sqlite-vec
    reports its own mismatch, but the store is the only layer that knows the
    configured dimension came from ``OMEGA_EMBEDDING_DIM``, which is what the
    operator actually has to change.
    """
    if len(embedding) != EMBEDDING_DIM:
        raise StorageError(
            f"Embedding dimension mismatch: got {len(embedding)}, "
            f"store expects {EMBEDDING_DIM}. The vector tables were built at "
            f"{EMBEDDING_DIM} dimensions; set OMEGA_EMBEDDING_DIM to match the "
            f"embedding model and re-embed the store if the model changed."
        )


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
        sensitivity: Optional[str] = None,
        allow_supersession: bool = True,
    ) -> str:
        """Store a memory. Returns the node ID.

        ``allow_supersession=False`` stops this write from retiring any older
        memory; would-be retirements are recorded as candidates instead. Hook
        captures pass it: text captured from a prompt or transcript is not an
        explicit statement that an older memory is obsolete.
        """
        _t0_agency = _time.monotonic()
        self._last_store_deduped = False
        self._last_contradiction_results = []
        self._last_supersession_results = []
        self._total_write_count += 1
        if not content:
            raise StorageError("content must be a non-empty string")
        if len(content) > self._MAX_CONTENT_SIZE:
            raise StorageError(
                f"Content size ({len(content):,} bytes) exceeds limit ({self._MAX_CONTENT_SIZE:,} bytes). "
                "Override with OMEGA_MAX_CONTENT_SIZE env var."
            )
        meta = dict(metadata or {})
        if sensitivity:
            meta["sensitivity"] = sensitivity
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

        # NOTE: embedding-similarity dedup used to run here. It was removed
        # after measurement showed it had no true-positive yield: it runs
        # *after* canonical-hash and content-hash dedup, so everything reaching
        # it is textually novel by construction. Measured over 3,000 real
        # memories, the 0.88 threshold discarded 334 writes of which only 2
        # were byte-identical — and those two content-hash dedup already
        # catches. Sentence embeddings are near-blind to digits, so records
        # differing only in their numbers (successive benchmark runs, version
        # counts, metrics) scored ~0.96 and collapsed into each other. No
        # threshold separates duplicates from distinct content: the most
        # similar non-identical pair scored 0.9845. Each false positive
        # silently discarded a write while store() returned the existing node
        # ID, which callers reported as a successful save.
        # Do not reintroduce similarity-based dedup without a mechanism that
        # preserves the incoming content.

        with self._lock:
            # Capacity check (inside lock — queries shared connection).
            # _MAX_NODES re-reads extension capabilities per access via the
            # property on SQLiteStoreBase. Core never trusts a local license
            # function to unlock the Free cap.
            #
            # ONE COUNT(*) query — the count is reused for both the
            # capacity check and the grandfather computation, so we do not
            # leave two prepared statements in flight on older SQLite.
            #
            # Defensive coercion of the COUNT row: on Python 3.12 + older
            # SQLite (Ubuntu CI 3.40-class), this cursor occasionally
            # yields a None first column even though COUNT(*) cannot
            # legitimately return NULL. Treating the result as 0 fails
            # safe: the capacity check below skips, no write block, and
            # the next store() retries the probe. A latent failure that
            # only surfaced after Sprint 0 (PR #59) unblocked CI past
            # the prior import_from_file stop point.
            self._capacity_warning = None
            base_max = self._MAX_NODES
            if base_max > 0:
                row = self._exec("SELECT COUNT(*) FROM memories").fetchone()
                count = row[0] if row and row[0] is not None else 0
                max_nodes = self._apply_grandfather(base_max, count)
                unlimited_memory = False
                try:
                    from omega.plugins import has_capability
                    unlimited_memory = has_capability("unlimited_memory")
                except Exception as e:
                    logger.debug("Capability check failed in capacity: %s", e)
                if count >= max_nodes:
                    if unlimited_memory:
                        raise StorageError(
                            f"Node count ({count:,}) has reached the limit ({max_nodes:,}). "
                            "Run omega_consolidate to prune, or raise OMEGA_MAX_NODES env var."
                        )
                    raise StorageError(
                        f"Free tier write cap reached ({count:,}/{max_nodes:,} memories). "
                        "Existing memories remain queryable. "
                        "Upgrade to Pro for unlimited memories + full retrieval quality: "
                        "https://omegamax.co/pro?ref=core-hard-cap"
                    )
                if count >= int(max_nodes * 0.9):
                    if unlimited_memory:
                        self._capacity_warning = (
                            f"Memory store is at {count:,}/{max_nodes:,} "
                            f"({count*100//max_nodes}% capacity). "
                            "Consider running omega_consolidate or omega_compact to free space."
                        )
                    else:
                        self._capacity_warning = (
                            f"Free tier near write cap ({count:,}/{max_nodes:,}). "
                            "Upgrade to Pro to remove the cap: "
                            "https://omegamax.co/pro?ref=core-cap-warning"
                        )
                    logger.warning(self._capacity_warning)
            self._invalidate_query_cache(new_content=content)

            project = meta.get("project") or os.getcwd()
            # Wire entity_id from metadata if not passed directly
            effective_entity_id = entity_id or meta.get("entity_id")

            # Dedup only against a live memory in the same project and entity.
            # Collapsing into another scope's row hid the write from this
            # scope's queries, and collapsing into a retired row returned a
            # memory that queries no longer show (audit finding B4).
            dedup_scope = """
                   AND project IS ? AND entity_id IS ?
                   AND COALESCE(status, 'active') != 'superseded'
                   AND COALESCE(json_extract(metadata, '$.superseded'), 0) = 0
                   AND (ttl_seconds IS NULL
                        OR datetime(created_at, '+' || ttl_seconds || ' seconds') > datetime('now'))"""

            # Canonical dedup (#6): catch reformatted duplicates
            canonical_existing = self._exec(
                "SELECT node_id, id FROM memories WHERE canonical_hash = ?"
                + dedup_scope + " LIMIT 1",
                (canonical_hash, project, effective_entity_id),
            ).fetchone()
            if canonical_existing:
                self.stats.setdefault("dedup_canonical", 0)
                self.stats["dedup_canonical"] += 1
                self._last_store_deduped = True
                self._record_timing("write", (_time.monotonic() - _t0_agency) * 1000)
                return canonical_existing[0]

            # Exact-match dedup via content hash
            existing = self._exec(
                "SELECT node_id, id FROM memories WHERE content_hash = ?"
                + dedup_scope + " LIMIT 1",
                (content_hash, project, effective_entity_id),
            ).fetchone()
            if existing:
                self.stats.setdefault("dedup_exact", 0)
                self.stats["dedup_exact"] += 1
                self._last_store_deduped = True
                self._record_timing("write", (_time.monotonic() - _t0_agency) * 1000)
                return existing[0]

            # Generate node ID
            node_id = f"mem-{uuid.uuid4().hex[:12]}"

            event_type = meta.get("event_type") or meta.get("type")
            now = datetime.now(timezone.utc).isoformat()

            # Determine priority from metadata or event type default. Coerce
            # untrusted metadata values (agents can pass "high", tuples, etc.)
            # so the stored value and downstream scoring stay numeric (issue #66).
            _raw_priority = meta.get("priority")
            if _raw_priority is not None:
                priority = coerce_priority(_raw_priority)
                meta["priority"] = priority  # persist normalized value
            else:
                priority = self._DEFAULT_PRIORITY.get(event_type, 3)
            referenced_date = meta.get("referenced_date")

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
                    updated_at, ttl_seconds, session_id, event_type, project, content_hash,
                    priority, referenced_date, entity_id, agent_type, canonical_hash,
                    extracted_keywords, memory_type, valid_from,
                    derived_from, source_uri, status)
                   VALUES (?, ?, ?, ?, 0, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    node_id,
                    content,
                    json.dumps(meta),
                    now,
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
                # A failure here used to be logged at DEBUG and swallowed, so
                # the memory row committed with no vector and store() still
                # returned an id: the write reported success and was
                # unretrievable by every later semantic search. Roll the whole
                # store back and raise instead — a rejected write the caller
                # can retry beats a silently vectorless one.
                try:
                    _check_embedding_dim(embedding)
                    self._exec(
                        "INSERT INTO memories_vec (rowid, embedding) VALUES (?, ?)", (rowid, _serialize_f32(embedding))
                    )
                except StorageError:
                    # Dimension mismatch: the message already says what to fix.
                    self._conn.rollback()
                    raise
                except Exception as e:
                    self._conn.rollback()
                    logger.error("Vec insert failed; rolled back store: %s", e, exc_info=True)
                    raise StorageError(f"Failed to store embedding vector: {e}") from e

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
                    node_id, content, embedding, allow_supersession=allow_supersession
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

    def get_last_supersession_results(self) -> List[SupersessionRecord]:
        """Older memories the most recent store() retired or flagged. Consume-once."""
        results = self._last_supersession_results
        self._last_supersession_results = []
        return results

    def get_last_store_deduped(self) -> bool:
        """Whether the most recent store() collapsed into an existing memory. Consume-once.

        store() returns a node ID whether it inserted or deduped, so callers
        that report the outcome to a user must check this to avoid claiming a
        write that never happened.
        """
        deduped = self._last_store_deduped
        self._last_store_deduped = False
        return deduped

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
                self.record_memory_access(node_id)

            return self._row_to_result(row)

    def record_memory_access(self, node_id: str) -> bool:
        """Record an explicit direct retrieval or final context injection.

        The audit counter remains truthful and is never rewritten.  Ranking
        consumes only the calibrated capped portion, so extreme historical
        values cannot create an unbounded feedback loop.
        """
        return self.record_memory_accesses([node_id]) > 0

    def record_memory_accesses(self, node_ids: List[str]) -> int:
        """Best-effort batch accounting for unique rendered/direct memory IDs.

        Persisted counts remain an unbounded audit trail. Ranking consumes only
        the separately capped contribution, and accounting failures never turn
        an otherwise successful read into an error.
        """
        unique_ids = list(dict.fromkeys(
            node_id for node_id in node_ids if isinstance(node_id, str) and node_id
        ))
        if not unique_ids:
            return 0

        self._invalidate_query_cache()
        try:
            with self._lock:
                now = datetime.now(timezone.utc).isoformat()
                placeholders = ",".join("?" for _ in unique_ids)
                cursor = self._conn.execute(
                    f"""UPDATE memories
                        SET access_count = access_count + 1, last_accessed = ?
                        WHERE node_id IN ({placeholders})""",
                    (now, *unique_ids),
                )
                updated = max(cursor.rowcount, 0)
                self._commit()
                return updated
        except Exception as exc:
            logger.warning("Memory access accounting skipped: %s", exc)
            try:
                with self._lock:
                    self._conn.rollback()
            except Exception:
                logger.debug("Access accounting rollback failed", exc_info=True)
            return 0

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
        priority: Optional[int] = None,
        record_edit: bool = False,
    ) -> bool:
        """Update fields on an existing node."""
        if priority is not None and (
            isinstance(priority, bool)
            or not isinstance(priority, int)
            or not 1 <= priority <= 5
        ):
            raise ValueError("priority must be an integer from 1 to 5")

        self._invalidate_query_cache()
        new_embedding = None
        if content is not None:
            # Re-embed to keep vec table in sync (CPU-bound, done outside lock)
            if self._vec_available:
                try:
                    from omega.embedding import generate_embedding, get_active_backend, is_embedding_degraded

                    new_embedding = generate_embedding(content)
                    if is_embedding_degraded() and get_active_backend() is not None:
                        new_embedding = None  # Hash fallback from real backend — don't store
                except Exception as e:
                    logger.debug("update_node: re-embed failed: %s", e)

        with self._lock:
            try:
                self._exec("BEGIN IMMEDIATE")
                sets = []
                params = []
                edit_timestamp = datetime.now(timezone.utc).isoformat()
                current = None

                if priority is not None or metadata is not None or record_edit:
                    current = self._exec(
                        """SELECT metadata, priority, event_type, session_id, project
                           FROM memories WHERE node_id = ?""",
                        (node_id,),
                    ).fetchone()
                    if current is None:
                        self._conn.rollback()
                        return False

                current_metadata = (
                    json.loads(current[0]) if current and current[0] else {}
                )
                if record_edit:
                    effective_metadata = current_metadata
                    if metadata is not None:
                        effective_metadata.update(metadata)
                    effective_metadata["edited_at"] = edit_timestamp
                    effective_metadata["edit_count"] = (
                        current_metadata.get("edit_count", 0) + 1
                    )
                else:
                    effective_metadata = (
                        dict(metadata) if metadata is not None else None
                    )

                if priority is not None:
                    if effective_metadata is None:
                        effective_metadata = current_metadata
                    history = effective_metadata.get("priority_edit_history", [])
                    if not isinstance(history, list):
                        history = []
                    else:
                        history = list(history)
                    if current[1] != priority:
                        history.append(
                            {
                                "old_priority": current[1],
                                "new_priority": priority,
                                "edited_at": edit_timestamp,
                            }
                        )
                    effective_metadata["priority_edit_history"] = history[
                        -_PRIORITY_EDIT_HISTORY_LIMIT:
                    ]
                    effective_metadata["priority"] = priority
                    sets.append("priority = ?")
                    params.append(priority)

                if content is not None:
                    sets.extend(
                        ("content = ?", "content_hash = ?", "canonical_hash = ?")
                    )
                    params.extend(
                        (
                            content,
                            hashlib.sha256(content.encode()).hexdigest(),
                            hashlib.sha256(_canonicalize(content).encode()).hexdigest(),
                        )
                    )

                if effective_metadata is not None:
                    sets.append("metadata = ?")
                    params.append(json.dumps(effective_metadata))
                    # Update denormalized columns without clearing omitted scope.
                    sets.extend(("event_type = ?", "session_id = ?", "project = ?"))
                    params.extend(
                        (
                            effective_metadata.get("event_type")
                            or effective_metadata.get("type")
                            or current[2],
                            (
                                effective_metadata["session_id"]
                                if "session_id" in effective_metadata
                                else current[3]
                            ),
                            (
                                effective_metadata["project"]
                                if "project" in effective_metadata
                                else current[4]
                            ),
                        )
                    )
                if access_count is not None:
                    sets.append("access_count = ?")
                    params.append(access_count)

                if (
                    content is not None
                    or effective_metadata is not None
                    or priority is not None
                ):
                    sets.append("updated_at = ?")
                    params.append(edit_timestamp)

                if not sets:
                    self._conn.rollback()
                    return False

                params.append(node_id)
                cursor = self._exec(
                    f"UPDATE memories SET {', '.join(sets)} WHERE node_id = ?", params
                )
                if cursor.rowcount == 0:
                    self._conn.rollback()
                    return False

                # Update vec embedding if content changed
                if new_embedding is not None:
                    row = self._exec(
                        "SELECT id FROM memories WHERE node_id = ?", (node_id,)
                    ).fetchone()
                    if row:
                        _check_embedding_dim(new_embedding)
                        self._exec("DELETE FROM memories_vec WHERE rowid = ?", (row[0],))
                        self._exec(
                            "INSERT INTO memories_vec (rowid, embedding) VALUES (?, ?)",
                            (row[0], _serialize_f32(new_embedding)),
                        )
                self._commit()
            except StorageError:
                self._conn.rollback()
                raise
            except Exception as e:
                self._conn.rollback()
                logger.error("update_node failed; rolled back: %s", e, exc_info=True)
                raise StorageError(f"Failed to update memory: {e}") from e
        return True

    # ------------------------------------------------------------------
    # Batch operations
    # ------------------------------------------------------------------

    def batch_store(self, items: List[Dict[str, Any]]) -> List[str]:
        """Store multiple memories efficiently."""
        if not items:
            return []

        # Work on shallow copies so embedding generation and field
        # normalization never rewrite caller-owned item dictionaries.
        normalized_items = [dict(item) for item in items]

        # Batch-generate embeddings for items without them
        items_needing = [
            (i, item)
            for i, item in enumerate(normalized_items)
            if item.get("embedding") is None
        ]
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
            for item in normalized_items:
                # Batch items expose the same metadata-backed fields as a
                # direct store() call. Copy before merging so callers do not
                # see their metadata mappings rewritten in place.
                metadata = dict(item.get("metadata") or {})
                for field in (
                    "event_type",
                    "type",
                    "priority",
                    "project",
                    "project_id",
                    "referenced_date",
                    "entity_id",
                    "agent_type",
                    "derived_from",
                    "source_uri",
                    "status",
                    "sensitivity",
                ):
                    if field in item and item[field] is not None:
                        metadata[field] = item[field]

                node_id = self.store(
                    content=item["content"],
                    session_id=item.get("session_id"),
                    metadata=metadata,
                    embedding=item.get("embedding"),
                    dependencies=item.get("dependencies"),
                    ttl_seconds=item.get("ttl_seconds"),
                    graphs=item.get("graphs"),
                    skip_inference=item.get("skip_inference", False),
                    entity_id=item.get("entity_id"),
                    agent_type=item.get("agent_type"),
                    derived_from=item.get("derived_from"),
                    source_uri=item.get("source_uri"),
                    status=item.get("status"),
                    sensitivity=item.get("sensitivity"),
                )
                ids.append(node_id)
        # Single cache invalidation after all inserts
        self._invalidate_query_cache()

        return ids

    def mark_superseded(
        self, node_id: str, superseded_by: str, reason: Optional[str] = None
    ) -> bool:
        """Mark a memory as superseded by a newer memory.

        Sets metadata.superseded=True and metadata.superseded_by on the target
        (plus metadata.superseded_reason when ``reason`` is given), and
        invalidates the query cache.

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
            if reason:
                meta["superseded_reason"] = reason
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
            details = {"superseded_by": superseded_by}
            if reason:
                details["reason"] = reason
            self._log_forgetting(
                node_id, row[1] or "", row[2] or "", "ingest_superseded", details,
            )
            self._commit()
        return True

    def supersede_with_replacement(
        self,
        old_id: str,
        replacement_id: Optional[str] = None,
        reason: str = "manual supersession",
        *,
        expected_entity_id: Optional[str] = None,
        expected_project: Optional[str] = None,
        expected_session_id: Optional[str] = None,
    ) -> tuple[bool, Optional[str]]:
        """Retire ``old_id`` and optionally link its replacement atomically.

        Both records must have identical entity, project, and session ownership.
        The lineage direction is always replacement -> supersedes -> old.
        """
        if not old_id:
            return False, "old memory ID is required"
        if replacement_id == old_id:
            return False, "replacement must be a different memory"

        now = datetime.now(timezone.utc).isoformat()
        with self._lock:
            try:
                self._exec("BEGIN IMMEDIATE")
                old_row = self._exec(
                    """SELECT metadata, content, event_type, entity_id, project,
                              session_id, status
                       FROM memories WHERE node_id = ?""",
                    (old_id,),
                ).fetchone()
                if old_row is None:
                    self._conn.rollback()
                    return False, f"Memory {old_id} not found"

                old_meta = json.loads(old_row[0]) if old_row[0] else {}
                if old_meta.get("superseded") or old_row[6] == "superseded":
                    self._conn.rollback()
                    return False, f"Memory {old_id} is already superseded"

                caller_scope_fields = (
                    ("entity", expected_entity_id, old_row[3]),
                    ("project", expected_project, old_row[4]),
                    ("session", expected_session_id, old_row[5]),
                )
                unauthorized_scope = next(
                    (
                        name
                        for name, expected_value, stored_value in caller_scope_fields
                        if expected_value is not None and expected_value != stored_value
                    ),
                    None,
                )
                if unauthorized_scope:
                    self._conn.rollback()
                    return (
                        False,
                        f"Caller authorization failed: {unauthorized_scope} scope differs",
                    )

                replacement_row = None
                if replacement_id:
                    replacement_row = self._exec(
                        """SELECT metadata, content, event_type, entity_id, project,
                                  session_id, status
                           FROM memories WHERE node_id = ?""",
                        (replacement_id,),
                    ).fetchone()
                    if replacement_row is None:
                        self._conn.rollback()
                        return False, f"Replacement memory {replacement_id} not found"

                    replacement_meta = (
                        json.loads(replacement_row[0]) if replacement_row[0] else {}
                    )
                    sql_status = replacement_row[6] or "active"
                    metadata_status = replacement_meta.get("status")
                    if (
                        metadata_status is not None
                        and metadata_status != sql_status
                    ):
                        self._conn.rollback()
                        return (
                            False,
                            f"Replacement memory {replacement_id} has conflicting status",
                        )
                    if replacement_meta.get("superseded") or sql_status != "active":
                        self._conn.rollback()
                        return (
                            False,
                            f"Replacement memory {replacement_id} is not active",
                        )

                    ownership_fields = (
                        ("entity", old_row[3], replacement_row[3]),
                        ("project", old_row[4], replacement_row[4]),
                        ("session", old_row[5], replacement_row[5]),
                    )
                    mismatch = next(
                        (name for name, old_value, new_value in ownership_fields
                         if old_value != new_value),
                        None,
                    )
                    if mismatch:
                        self._conn.rollback()
                        return False, f"Ownership check failed: {mismatch} scope differs"

                old_meta["superseded"] = True
                old_meta["superseded_by"] = replacement_id or f"manual: {reason}"
                old_meta["superseded_at"] = now
                old_meta["superseded_reason"] = reason
                self._exec(
                    """UPDATE memories
                       SET metadata = ?, valid_until = ?, status = 'superseded'
                       WHERE node_id = ?""",
                    (json.dumps(old_meta), now, old_id),
                )

                if replacement_id:
                    self._exec(
                        """INSERT OR IGNORE INTO edges
                           (source_id, target_id, edge_type, weight, created_at)
                           VALUES (?, ?, 'supersedes', 1.0, ?)""",
                        (replacement_id, old_id, now),
                    )

                self._log_forgetting(
                    old_id,
                    old_row[1] or "",
                    old_row[2] or "",
                    "manual_superseded",
                    {"superseded_by": replacement_id, "reason": reason},
                )
                self._commit()
            except Exception as exc:
                self._conn.rollback()
                logger.error(
                    "Atomic supersede failed for %s: %s", old_id, exc, exc_info=True
                )
                return False, str(exc)

        self._invalidate_query_cache()
        return True, None

    # ------------------------------------------------------------------
    # Contradiction detection
    # ------------------------------------------------------------------

    _CONTRADICTION_CANDIDATE_LIMIT = 10
    _CONTRADICTION_CONFIDENCE_THRESHOLD = 0.4

    # Cosine at which an older memory is considered for supersession at all.
    # Similarity only nominates: retirement also needs a shared scope and an
    # explicit update signal (_settle_supersession). Similarity alone used to
    # retire at this threshold and took 11 of 30 related-but-distinct pairs,
    # across projects and clients (audit finding B1, 2026-09-29). The lowest
    # genuine update in that audit ("Stop using pytest-xdist in CI ...")
    # scored 0.755, so the gate stays here rather than rising.
    _SUPERSESSION_SIMILARITY_THRESHOLD = 0.75
    _SUPERSESSION_TYPES = frozenset({
        "decision", "user_preference", "user_fact", "lesson_learned", "error_pattern",
    })
    # A newer memory of the key type may also supersede these older types
    # ("stop suggesting HN" retires "post Show HN on Tuesday"). One-way.
    _CROSS_TYPE_SUPERSESSION = {"user_preference": frozenset({"decision"})}
    _MAX_SUPERSESSION_CANDIDATES = 20

    def _check_contradictions(
        self,
        new_node_id: str,
        new_content: str,
        embedding: List[float],
        allow_supersession: bool = True,
    ) -> list:
        """Settle supersession, then annotate contradictions, for a new memory.

        Only older memories in the same project and entity are considered;
        a store never changes a memory that belongs to another scope. Each
        eligible older memory is either retired (it carries an explicit update
        signal) or recorded as a supersession candidate on the new memory.
        The outcome is left for get_last_supersession_results().

        Then runs contradiction detection heuristics on the remaining
        same-scope memories and annotates metadata on both sides.

        Returns:
            List of dicts with keys: node_id, confidence, reason, content_preview.
            Empty list if no contradictions found.
        """
        from omega.contradictions import detect_contradictions

        new_row = self._conn.execute(
            "SELECT event_type, created_at, project, entity_id FROM memories WHERE node_id = ?",
            (new_node_id,),
        ).fetchone()
        if not new_row:
            return []
        new_event_type, new_created_raw, new_project, new_entity = new_row
        new_created_at = self._parse_dt(new_created_raw)

        similar = self._vec_query(embedding, limit=self._CONTRADICTION_CANDIDATE_LIMIT + 1)
        if not similar:
            return []
        distances = {rowid: distance for rowid, distance in similar}
        placeholders = ",".join("?" * len(distances))
        rows = self._conn.execute(
            f"""SELECT id, node_id, content, event_type, created_at, status,
                       json_extract(metadata, '$.superseded')
                FROM memories
                WHERE id IN ({placeholders}) AND node_id != ?
                  AND project IS ? AND entity_id IS ?""",
            (*distances, new_node_id, new_project, new_entity),
        ).fetchall()
        row_map = {r[0]: r[1:] for r in rows}

        neighbours: List[_Neighbour] = []
        for rowid, _ in similar:  # keep nearest-first order
            if rowid not in row_map:
                continue
            node_id_val, content_val, event_type, created_raw, status, superseded = row_map[rowid]
            if status == "superseded" or superseded:
                continue
            neighbours.append(_Neighbour(
                node_id=node_id_val,
                content=content_val,
                similarity=1.0 - distances[rowid],
                event_type=event_type,
                created_at=self._parse_dt(created_raw),
            ))

        retired = self._settle_supersession(
            new_node_id, new_content, new_event_type, new_created_at,
            neighbours, allow_supersession,
        )
        remaining = [n for n in neighbours if n.node_id not in retired]
        if not remaining:
            return []
        candidate_ids = [n.node_id for n in remaining]
        candidate_contents = [n.content for n in remaining]

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

    def _settle_supersession(
        self,
        new_node_id: str,
        new_content: str,
        new_event_type: Optional[str],
        new_created_at: Optional[datetime],
        neighbours: List[_Neighbour],
        allow_supersession: bool,
    ) -> set:
        """Retire or flag the older neighbours a new memory may replace.

        ``neighbours`` are already limited to active memories in the new
        memory's project and entity. An older one of an eligible type at or
        above the similarity gate is retired when the new text carries an
        explicit update signal and ``allow_supersession`` is set; otherwise it
        is recorded as a candidate on the new memory and left active.

        Returns the node IDs retired.
        """
        from omega.contradictions import detect_update_signal

        if new_event_type not in self._SUPERSESSION_TYPES or not new_created_at:
            return set()
        replaceable_types = {new_event_type} | self._CROSS_TYPE_SUPERSESSION.get(
            new_event_type, frozenset()
        )

        retired: set = set()
        candidates = []
        report: List[SupersessionRecord] = []
        for n in neighbours:
            if n.similarity < self._SUPERSESSION_SIMILARITY_THRESHOLD:
                continue
            if n.event_type not in replaceable_types:
                continue
            if not n.created_at or n.created_at >= new_created_at:
                continue
            signal = detect_update_signal(new_content, n.content)
            if (
                signal
                and allow_supersession
                and self.mark_superseded(n.node_id, new_node_id, reason=signal)
            ):
                self.add_edge(new_node_id, n.node_id, "supersedes", n.similarity)
                retired.add(n.node_id)
                action = "retired"
                logger.info(
                    "Superseded %s by %s (type=%s, signal=%s, similarity=%.3f)",
                    n.node_id, new_node_id, n.event_type, signal, n.similarity,
                )
            else:
                candidates.append({
                    "target_id": n.node_id,
                    "similarity": round(n.similarity, 3),
                    "detector": "store_similarity",
                    "reason": signal or "same scope and type, high similarity, no update signal",
                    "target_event_type": n.event_type,
                })
                action = "candidate"
            report.append(SupersessionRecord(
                node_id=n.node_id,
                action=action,
                signal=signal,
                similarity=round(n.similarity, 3),
                content_preview=n.content[:80],
            ))

        if retired:
            self.stats.setdefault("temporal_supersessions", 0)
            self.stats["temporal_supersessions"] += len(retired)
        if candidates:
            self._record_supersession_candidates(new_node_id, candidates)
            self.stats.setdefault("supersession_candidates", 0)
            self.stats["supersession_candidates"] += len(candidates)
        self._last_supersession_results = report
        return retired

    def _record_supersession_candidates(self, node_id: str, candidates: List[dict]) -> None:
        """Append bounded, non-authoritative replacement proposals to a memory."""
        with self._lock:
            row = self._conn.execute(
                "SELECT metadata FROM memories WHERE node_id = ?", (node_id,)
            ).fetchone()
            if not row:
                return
            meta = json.loads(row[0]) if row[0] else {}
            targets = {c["target_id"] for c in candidates}
            kept = [
                c for c in meta.get("supersession_candidates", [])
                if c.get("target_id") not in targets
            ]
            meta["supersession_candidates"] = (kept + candidates)[-self._MAX_SUPERSESSION_CANDIDATES:]
            self._conn.execute(
                "UPDATE memories SET metadata = ? WHERE node_id = ?",
                (json.dumps(meta), node_id),
            )
            self._commit()
