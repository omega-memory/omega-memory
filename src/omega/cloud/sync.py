"""OMEGA Cloud Sync -- Bidirectional sync between local SQLite and Supabase.

Sync strategy:
  - Local is source of truth for memories
  - Cloud enables mobile read access
  - Sync runs periodically (configurable) or on-demand
  - Conflict resolution: local wins for memories, cloud wins for shared docs

Dependencies: supabase-py (pip install supabase)
"""

import json
import logging
import os
import stat
import struct
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger("omega.cloud.sync")


def _omega_home() -> Path:
    return Path(os.environ.get("OMEGA_HOME", str(Path.home() / ".omega")))


def _load_supabase_config() -> dict:
    """Load Supabase configuration from secrets file."""
    secrets_path = _omega_home() / "secrets.json"
    if not secrets_path.exists():
        raise FileNotFoundError(
            f"Supabase config not found at {secrets_path}. "
            "Run 'omega cloud setup' to configure."
        )

    # Enforce secure file permissions (like SSH does for keys)
    current_mode = secrets_path.stat().st_mode
    if current_mode & (stat.S_IRWXG | stat.S_IRWXO):
        logger.warning(
            "secrets.json has insecure permissions %o, fixing to 0600",
            stat.S_IMODE(current_mode),
        )
        os.chmod(str(secrets_path), 0o600)

    with open(secrets_path) as f:
        config = json.load(f)

    url = config.get("supabase_url")
    key = config.get("supabase_key")
    if not url or not key:
        raise ValueError("supabase_url and supabase_key required in secrets.json")

    return {"url": url, "key": key}


def _get_supabase_client():
    """Create a Supabase client with timeout configuration."""
    try:
        from supabase import create_client
        from supabase.lib.client_options import SyncClientOptions
    except ImportError:
        raise ImportError(
            "supabase package required for cloud sync. "
            "Install with: pip install omega-memory[cloud]"
        )
    config = _load_supabase_config()
    options = SyncClientOptions(
        postgrest_client_timeout=30,  # 30s per REST call (default 120s is too long)
        storage_client_timeout=30,
    )
    return create_client(config["url"], config["key"], options=options)


def _is_transient_error(exc: Exception) -> bool:
    """Check if an exception is a transient network error worth retrying."""
    msg = str(exc).lower()
    transient_markers = (
        "server disconnected",
        "connection reset",
        "connection refused",
        "timed out",
        "timeout",
        "nodename nor servname",
        "name resolution",
        "broken pipe",
        "eof occurred",
    )
    return any(marker in msg for marker in transient_markers)


class CloudSync:
    """Manages bidirectional sync between local OMEGA and Supabase."""

    def __init__(self, local_db_path: Optional[Path] = None):
        self._local_db_path = local_db_path or (_omega_home() / "omega.db")
        self._supabase = None  # Lazy
        self._lock = threading.Lock()
        self._last_sync: Optional[float] = None
        self._user_id = self._load_user_id()

    @staticmethod
    def _load_user_id() -> str:
        """Load user_id from ~/.omega/config.json, generating one if missing."""
        import uuid
        config_path = _omega_home() / "config.json"
        config = {}
        if config_path.exists():
            try:
                config = json.loads(config_path.read_text())
            except Exception:
                pass
        
        user_id = config.get("user_id")
        if not user_id:
            user_id = str(uuid.uuid4())
            config["user_id"] = user_id
            try:
                config_path.parent.mkdir(parents=True, exist_ok=True)
                config_path.write_text(json.dumps(config, indent=2))
                logger.info("Generated new user_id for cloud sync: %s", user_id)
            except Exception as e:
                logger.error("Failed to save generated user_id: %s", e)
        
        return user_id

    def _get_client(self):
        if self._supabase is None:
            self._supabase = _get_supabase_client()
        return self._supabase

    def _reset_client(self):
        """Reset cached client after connection errors."""
        self._supabase = None

    def _get_local_conn(self):
        from omega.crypto import secure_connect

        conn = secure_connect(self._local_db_path, timeout=30)
        conn.row_factory = __import__("sqlite3").Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=30000")
        return conn

    def sync_memories(self, batch_size: int = 100) -> dict:
        """Sync memories from local SQLite to Supabase."""
        client = self._get_client()
        conn = self._get_local_conn()

        try:
            # Get last synced local_id from Supabase
            result = (
                client.table("sync_state")
                .select("last_local_id")
                .eq("table_name", "memories")
                .eq("user_id", self._user_id)
                .execute()
            )
            last_id = result.data[0]["last_local_id"] if result.data else 0

            # Fetch new local memories
            rows = conn.execute(
                """
                SELECT id, content, event_type, priority, session_id, project,
                       metadata, created_at, entity_id, memory_type, access_count
                FROM memories
                WHERE id > ?
                ORDER BY id
                LIMIT ?
                """,
                (last_id, batch_size),
            ).fetchall()

            if not rows:
                return {"synced": 0, "status": "up_to_date"}

            # Prepare records for upsert
            records = []
            max_id = last_id
            for row in rows:
                meta = json.loads(row["metadata"]) if row["metadata"] else None
                tags = []
                if meta and "tags" in meta:
                    tags = meta.pop("tags", [])

                record = {
                    "local_id": row["id"],
                    "content": row["content"],
                    "event_type": row["event_type"],
                    "priority": row["priority"],
                    "session_id": row["session_id"],
                    "project": row["project"],
                    "tags": tags,
                    "metadata": meta,
                    "created_at": row["created_at"],
                    "entity_id": row["entity_id"],
                    "memory_type": row["memory_type"],
                    "access_count": row["access_count"] or 0,
                    "synced_at": datetime.now(timezone.utc).isoformat(),
                    "user_id": self._user_id,
                }
                records.append(record)
                max_id = max(max_id, row["id"])

            # Upsert to Supabase
            client.table("memories").upsert(
                records, on_conflict="local_id,user_id"
            ).execute()

            # Sync embeddings for these memories
            self._sync_memory_embeddings(conn, client, [r["local_id"] for r in records])

            # Update sync state
            sync_record = {
                "table_name": "memories",
                "last_local_id": max_id,
                "last_sync_at": datetime.now(timezone.utc).isoformat(),
                "sync_count": len(records),
                "user_id": self._user_id,
            }
            client.table("sync_state").upsert(
                sync_record, on_conflict="table_name,user_id"
            ).execute()

            return {"synced": len(records), "last_id": max_id, "status": "ok"}

        finally:
            conn.close()

    def _sync_memory_embeddings(self, conn, client, local_ids: list[int]) -> None:
        """Sync memory embeddings to Supabase pgvector.

        Per-item isolation: individual failures are logged and skipped.
        """
        if not local_ids:
            return

        # Batch-fetch cloud UUIDs for all local_ids in one query
        try:
            uuid_result = client.table("memories").select("id,local_id").in_("local_id", local_ids).execute()
            uuid_map = {row["local_id"]: row["id"] for row in (uuid_result.data or [])}
        except Exception as e:
            logger.debug("Batch UUID lookup failed: %s", e)
            return

        for local_id in local_ids:
            try:
                # Read embedding from local sqlite-vec
                try:
                    row = conn.execute(
                        "SELECT embedding FROM memories_vec WHERE rowid = ?",
                        (local_id,),
                    ).fetchone()
                except Exception:
                    continue  # Vec table might not exist

                if not row or not row[0]:
                    continue

                # Parse binary embedding
                emb_bytes = row[0]
                dim = len(emb_bytes) // 4
                embedding = list(struct.unpack(f"{dim}f", emb_bytes))

                # Get cloud memory UUID from pre-fetched map
                memory_uuid = uuid_map.get(local_id)
                if not memory_uuid:
                    continue

                # Upsert embedding
                client.table("memory_embeddings").upsert({
                    "memory_id": memory_uuid,
                    "embedding": embedding,
                }, on_conflict="memory_id").execute()
            except Exception as e:
                logger.debug("Embedding sync failed for local_id=%d: %s", local_id, e)

    def sync_deletions(self, batch_size: int = 100) -> dict:
        """Delete memories from Supabase that were pruned locally.

        Reads the cloud_delete_queue table, batch-deletes matching rows
        from Supabase by local_id, then clears processed queue entries.
        """
        client = self._get_client()
        conn = self._get_local_conn()

        try:
            rows = conn.execute(
                "SELECT id, local_id FROM cloud_delete_queue ORDER BY id LIMIT ?",
                (batch_size,),
            ).fetchall()

            if not rows:
                return {"deleted": 0, "status": "up_to_date"}

            local_ids = [r["local_id"] for r in rows]
            queue_ids = [r["id"] for r in rows]

            # Batch delete from Supabase (in chunks of 50 to avoid URL length limits)
            deleted = 0
            for i in range(0, len(local_ids), 50):
                batch = local_ids[i : i + 50]
                result = client.table("memories").delete().in_("local_id", batch).execute()
                deleted += len(result.data) if result.data else 0

                # Also clean up orphaned embeddings (CASCADE should handle this,
                # but be explicit for safety)
                cloud_rows = client.table("memories").select("id").in_("local_id", batch).execute()
                if cloud_rows.data:
                    mem_uuids = [r["id"] for r in cloud_rows.data]
                    client.table("memory_embeddings").delete().in_("memory_id", mem_uuids).execute()

            # Clear processed queue entries
            placeholders = ",".join("?" * len(queue_ids))
            conn.execute(f"DELETE FROM cloud_delete_queue WHERE id IN ({placeholders})", queue_ids)
            conn.commit()

            return {"deleted": deleted, "queued": len(rows), "status": "ok"}

        finally:
            conn.close()

    def reconcile_memories(self) -> dict:
        """Remove Supabase memories whose local_id no longer exists locally.

        Handles orphans that bypassed the cloud_delete_queue (e.g., after a
        local DB restore or crash during deletion). Uses the
        cleanup_orphaned_memories RPC for a single round-trip.
        """
        client = self._get_client()
        conn = self._get_local_conn()
        try:
            rows = conn.execute("SELECT id FROM memories ORDER BY id").fetchall()
            valid_ids = [r["id"] for r in rows]

            result = client.rpc(
                "cleanup_orphaned_memories",
                {"valid_local_ids": valid_ids},
            ).execute()

            deleted = result.data if isinstance(result.data, int) else 0
            logger.info("reconcile_memories: removed %d orphaned cloud records", deleted)
            return {"orphans_deleted": deleted, "local_count": len(valid_ids), "status": "ok"}
        except Exception as e:
            logger.warning("reconcile_memories failed: %s", e)
            return {"orphans_deleted": 0, "status": "error", "error": str(e)}
        finally:
            conn.close()

    def sync_documents(self, batch_size: int = 50) -> dict:
        """Sync documents and chunks from local to Supabase.

        Per-document error isolation: individual failures are logged and
        skipped so the rest of the batch can still sync.
        """
        client = self._get_client()
        conn = self._get_local_conn()

        try:
            result = client.table("sync_state").select("last_local_id").eq("table_name", "documents").execute()
            last_id = result.data[0]["last_local_id"] if result.data else 0

            rows = conn.execute(
                """
                SELECT id, source_path, source_type, title, checksum, chunk_count,
                       created_at, updated_at
                FROM documents
                WHERE id > ?
                ORDER BY id
                LIMIT ?
                """,
                (last_id, batch_size),
            ).fetchall()

            if not rows:
                return {"synced": 0, "status": "up_to_date"}

            max_id = last_id
            synced = 0
            errors = 0
            total_chunks = 0

            for row in rows:
                try:
                    doc_record = {
                        "local_id": row["id"],
                        "source_path": row["source_path"],
                        "source_type": row["source_type"],
                        "title": row["title"],
                        "checksum": row["checksum"],
                        "chunk_count": row["chunk_count"],
                        "created_at": row["created_at"],
                        "updated_at": row["updated_at"],
                        "synced_at": datetime.now(timezone.utc).isoformat(),
                    }
                    if self._user_id:
                        doc_record["user_id"] = self._user_id
                    result = client.table("documents").upsert(doc_record, on_conflict="source_path,user_id").execute()
                    doc_uuid = result.data[0]["id"] if result.data else None

                    if doc_uuid:
                        chunks_synced = self._sync_document_chunks(conn, client, row["id"], doc_uuid)
                        total_chunks += chunks_synced

                    synced += 1
                    max_id = max(max_id, row["id"])
                except Exception as e:
                    errors += 1
                    logger.warning("Document sync failed for id=%d (%s): %s", row["id"], row["source_path"], e)
                    if _is_transient_error(e):
                        self._reset_client()
                        client = self._get_client()

            # Only advance sync cursor if we synced at least one document
            if synced > 0:
                try:
                    doc_sync_record = {
                        "table_name": "documents",
                        "last_local_id": max_id,
                        "last_sync_at": datetime.now(timezone.utc).isoformat(),
                        "sync_count": synced,
                    }
                    if self._user_id:
                        doc_sync_record["user_id"] = self._user_id
                    doc_conflict_key = "table_name,user_id" if self._user_id else "table_name"
                    client.table("sync_state").upsert(
                        doc_sync_record, on_conflict=doc_conflict_key
                    ).execute()
                except Exception as e:
                    logger.warning("Failed to update document sync state: %s", e)

            return {"synced": synced, "errors": errors, "chunks": total_chunks, "status": "ok" if errors == 0 else "partial"}

        finally:
            conn.close()

    def _sync_document_chunks(self, conn, client, local_doc_id: int, cloud_doc_uuid: str) -> int:
        """Sync chunks for a single document."""
        chunks = conn.execute(
            """
            SELECT id, chunk_index, content, chunk_type, page_number, token_count, created_at
            FROM document_chunks
            WHERE document_id = ?
            ORDER BY chunk_index
            """,
            (local_doc_id,),
        ).fetchall()

        if not chunks:
            return 0

        records = []
        for chunk in chunks:
            record = {
                "document_id": cloud_doc_uuid,
                "local_id": chunk["id"],
                "chunk_index": chunk["chunk_index"],
                "content": chunk["content"],
                "chunk_type": chunk["chunk_type"],
                "page_number": chunk["page_number"],
                "token_count": chunk["token_count"],
                "created_at": chunk["created_at"],
            }

            # Read embedding from local vec table
            try:
                row = conn.execute(
                    "SELECT embedding FROM document_chunks_vec WHERE rowid = ?",
                    (chunk["id"],),
                ).fetchone()
                if row and row[0]:
                    emb_bytes = row[0]
                    dim = len(emb_bytes) // 4
                    record["embedding"] = list(struct.unpack(f"{dim}f", emb_bytes))
            except Exception:
                pass

            records.append(record)

        if self._user_id:
            for r in records:
                r["user_id"] = self._user_id

        # Batch insert chunks
        for i in range(0, len(records), 50):
            batch = records[i : i + 50]
            client.table("document_chunks").upsert(batch, on_conflict="local_id").execute()

        return len(records)

    def sync_profile(self) -> dict:
        """Sync encrypted profile to Supabase (ciphertext only)."""
        client = self._get_client()
        conn = self._get_local_conn()

        try:
            rows = conn.execute(
                "SELECT id, category, field_name, field_value_encrypted, metadata, created_at, updated_at FROM secure_profile"
            ).fetchall()

            if not rows:
                return {"synced": 0, "status": "up_to_date"}

            records = []
            for row in rows:
                rec = {
                    "local_id": row["id"],
                    "category": row["category"],
                    "field_name": row["field_name"],
                    "field_value_encrypted": row["field_value_encrypted"],
                    "metadata": json.loads(row["metadata"]) if row["metadata"] else None,
                    "created_at": row["created_at"],
                    "updated_at": row["updated_at"],
                    "synced_at": datetime.now(timezone.utc).isoformat(),
                }
                if self._user_id:
                    rec["user_id"] = self._user_id
                records.append(rec)

            client.table("secure_profile").upsert(records, on_conflict="category,field_name").execute()

            return {"synced": len(records), "status": "ok"}

        finally:
            conn.close()

    def sync_all(self) -> dict:
        """Run a full sync of all tables."""
        results = {}
        with self._lock:
            try:
                results["memories"] = self.sync_memories()
            except Exception as e:
                logger.error("Memory sync failed: %s", e)
                results["memories"] = {"status": "error", "error": str(e)}

            try:
                results["deletions"] = self.sync_deletions()
            except Exception as e:
                logger.error("Deletion sync failed: %s", e)
                results["deletions"] = {"status": "error", "error": str(e)}

            try:
                results["documents"] = self.sync_documents()
            except Exception as e:
                logger.error("Document sync failed: %s", e)
                results["documents"] = {"status": "error", "error": str(e)}

            try:
                results["edges"] = self.sync_edges()
            except Exception as e:
                logger.error("Edge sync failed: %s", e)
                results["edges"] = {"status": "error", "error": str(e)}

            # Secure profile is excluded from cloud sync — decryption key
            # lives in the local macOS Keychain and encrypted blobs plus
            # field names / metadata should never leave the machine.
            results["profile"] = {"status": "excluded", "reason": "sensitive"}

            self._last_sync = time.monotonic()

        return results

    def pull_memories(self, batch_size: int = 100) -> dict:
        """Pull memories from Supabase cloud to local SQLite.

        Conflict resolution: skip if content_hash already exists locally.
        Embeddings are NOT transferred (pgvector → sqlite-vec format mismatch).
        Run `omega doctor` to regenerate embeddings after pull.
        """
        import hashlib
        import uuid

        client = self._get_client()
        conn = self._get_local_conn()

        try:
            pulled = 0
            skipped = 0
            offset = 0

            while True:
                result = (
                    client.table("memories")
                    .select("*")
                    .order("created_at")
                    .range(offset, offset + batch_size - 1)
                    .execute()
                )
                rows = result.data
                if not rows:
                    break

                for row in rows:
                    content = row.get("content", "")
                    content_hash = hashlib.sha256(content.encode()).hexdigest()

                    # Skip if content_hash already exists locally
                    existing = conn.execute(
                        "SELECT id FROM memories WHERE content_hash = ?",
                        (content_hash,),
                    ).fetchone()
                    if existing:
                        skipped += 1
                        continue

                    node_id = f"mem-{uuid.uuid4().hex[:12]}"
                    meta = row.get("metadata")
                    tags = row.get("tags", [])
                    if tags and isinstance(meta, dict):
                        meta["tags"] = tags
                    elif tags:
                        meta = {"tags": tags}
                    metadata_json = json.dumps(meta) if meta else None
                    now = datetime.now(timezone.utc).isoformat()

                    conn.execute(
                        """INSERT INTO memories
                           (node_id, content, metadata, created_at, last_accessed,
                            access_count, session_id, event_type, project,
                            content_hash, priority)
                           VALUES (?, ?, ?, ?, ?, 0, ?, ?, ?, ?, ?)""",
                        (
                            node_id,
                            content,
                            metadata_json,
                            row.get("created_at", now),
                            now,
                            row.get("session_id"),
                            row.get("event_type"),
                            row.get("project"),
                            content_hash,
                            row.get("priority", 3),
                        ),
                    )
                    pulled += 1

                conn.commit()
                offset += batch_size

                if len(rows) < batch_size:
                    break

            return {"pulled": pulled, "skipped": skipped, "status": "ok"}

        finally:
            conn.close()

    def _pull_document_chunks(self, conn, client, cloud_doc_uuid: str, local_doc_id: int) -> int:
        """Pull chunks for a single document from Supabase."""
        result = (
            client.table("document_chunks")
            .select("*")
            .eq("document_id", cloud_doc_uuid)
            .order("chunk_index")
            .execute()
        )
        chunks = result.data
        if not chunks:
            return 0

        now = datetime.now(timezone.utc).isoformat()
        count = 0
        for chunk in chunks:
            conn.execute(
                """INSERT INTO document_chunks
                   (document_id, chunk_index, content, chunk_type,
                    page_number, token_count, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    local_doc_id,
                    chunk.get("chunk_index", 0),
                    chunk.get("content", ""),
                    chunk.get("chunk_type"),
                    chunk.get("page_number"),
                    chunk.get("token_count"),
                    chunk.get("created_at", now),
                ),
            )
            count += 1
        return count

    def pull_documents(self, batch_size: int = 50) -> dict:
        """Pull documents and chunks from Supabase cloud to local SQLite.

        Conflict resolution: skip if source_path already exists locally.
        Embeddings are NOT transferred — run `omega doctor` to regenerate.
        """
        client = self._get_client()
        conn = self._get_local_conn()

        try:
            pulled = 0
            skipped = 0
            total_chunks = 0
            offset = 0

            while True:
                result = (
                    client.table("documents")
                    .select("*")
                    .order("created_at")
                    .range(offset, offset + batch_size - 1)
                    .execute()
                )
                rows = result.data
                if not rows:
                    break

                for row in rows:
                    source_path = row.get("source_path", "")

                    # Skip if source_path already exists locally
                    existing = conn.execute(
                        "SELECT id FROM documents WHERE source_path = ?",
                        (source_path,),
                    ).fetchone()
                    if existing:
                        skipped += 1
                        continue

                    now = datetime.now(timezone.utc).isoformat()
                    cursor = conn.execute(
                        """INSERT INTO documents
                           (source_path, source_type, title, checksum,
                            chunk_count, created_at, updated_at)
                           VALUES (?, ?, ?, ?, ?, ?, ?)""",
                        (
                            source_path,
                            row.get("source_type", "unknown"),
                            row.get("title"),
                            row.get("checksum"),
                            row.get("chunk_count", 0),
                            row.get("created_at", now),
                            row.get("updated_at", now),
                        ),
                    )
                    local_doc_id = cursor.lastrowid

                    # Pull chunks for this document
                    cloud_doc_uuid = row.get("id")
                    if cloud_doc_uuid and local_doc_id:
                        chunks_pulled = self._pull_document_chunks(
                            conn, client, cloud_doc_uuid, local_doc_id
                        )
                        total_chunks += chunks_pulled

                    pulled += 1

                conn.commit()
                offset += batch_size

                if len(rows) < batch_size:
                    break

            return {"pulled": pulled, "skipped": skipped, "chunks": total_chunks, "status": "ok"}

        finally:
            conn.close()

    def pull_all(self) -> dict:
        """Pull all tables from Supabase cloud to local SQLite."""
        results = {}
        with self._lock:
            try:
                results["memories"] = self.pull_memories()
            except Exception as e:
                logger.error("Memory pull failed: %s", e)
                results["memories"] = {"status": "error", "error": str(e)}

            try:
                results["documents"] = self.pull_documents()
            except Exception as e:
                logger.error("Document pull failed: %s", e)
                results["documents"] = {"status": "error", "error": str(e)}

            # Profile is excluded — decryption key lives in local Keychain
            results["profile"] = {"status": "excluded", "reason": "sensitive"}

        return results

    # ------------------------------------------------------------------
    # Coordination dual-writes (fire-and-forget)
    # ------------------------------------------------------------------

    def upsert_session(
        self,
        session_id: str,
        project: Optional[str],
        status: str,
        task: Optional[str],
        last_heartbeat: str,
        started_at: str,
        pid: Optional[int] = None,
        capabilities: Optional[str] = None,
        metadata: Optional[str] = None,
        agent_type: Optional[str] = None,
    ) -> None:
        """Upsert a coordination session to Supabase. Fire-and-forget."""
        try:
            client = self._get_client()
            row: dict = {
                "session_id": session_id,
                "project": project,
                "status": status,
                "task": task,
                "last_heartbeat": last_heartbeat,
                "started_at": started_at,
            }
            if pid is not None:
                row["pid"] = pid
            if capabilities is not None:
                row["capabilities"] = capabilities
            if metadata is not None:
                row["metadata"] = metadata
            if agent_type is not None:
                row["agent_type"] = agent_type
            client.table("coord_sessions").upsert(
                row,
                on_conflict="session_id",
            ).execute()
        except Exception:
            logger.debug("cloud upsert_session failed", exc_info=True)

    def delete_session_claims(self, session_id: str) -> None:
        """Delete all file claims for a session from Supabase. Fire-and-forget."""
        try:
            client = self._get_client()
            client.table("coord_file_claims").delete().eq(
                "session_id", session_id
            ).execute()
        except Exception:
            logger.debug("cloud delete_session_claims failed", exc_info=True)

    def upsert_file_claim(
        self, file_path: str, session_id: str, task: Optional[str], claimed_at: str
    ) -> None:
        """Upsert a file claim to Supabase. Fire-and-forget."""
        try:
            client = self._get_client()
            client.table("coord_file_claims").upsert(
                {
                    "file_path": file_path,
                    "session_id": session_id,
                    "task": task,
                    "claimed_at": claimed_at,
                    "last_activity": claimed_at,
                },
                on_conflict="file_path",
            ).execute()
        except Exception:
            logger.debug("cloud upsert_file_claim failed", exc_info=True)

    def upsert_file_read(self, session_id: str, file_path: str, first_read_at: str, read_count: int = 1) -> None:
        """Upsert a file read to Supabase. Fire-and-forget."""
        try:
            client = self._get_client()
            client.table("coord_file_reads").upsert(
                {
                    "session_id": session_id,
                    "file_path": file_path,
                    "first_read_at": first_read_at,
                    "read_count": read_count,
                },
                on_conflict="session_id,file_path",
            ).execute()
        except Exception:
            logger.debug("cloud upsert_file_read failed", exc_info=True)

    def delete_session_file_reads(self, session_id: str) -> None:
        """Delete all file reads for a session from Supabase. Fire-and-forget."""
        try:
            client = self._get_client()
            client.table("coord_file_reads").delete().eq(
                "session_id", session_id
            ).execute()
        except Exception:
            logger.debug("cloud delete_session_file_reads failed", exc_info=True)

    def update_session_heartbeat(self, session_id: str, last_heartbeat: str) -> None:
        """Update heartbeat timestamp, re-creating the session if missing. Fire-and-forget."""
        try:
            client = self._get_client()
            client.table("coord_sessions").upsert(
                {"session_id": session_id, "last_heartbeat": last_heartbeat, "started_at": last_heartbeat},
                on_conflict="session_id",
            ).execute()
        except Exception:
            logger.debug("cloud update_session_heartbeat failed", exc_info=True)

    def update_session_fields(
        self, session_id: str, project: Optional[str], task: Optional[str], last_heartbeat: str
    ) -> None:
        """Update project, task, and heartbeat, re-creating session if missing. Fire-and-forget."""
        try:
            client = self._get_client()
            fields: dict = {"session_id": session_id, "last_heartbeat": last_heartbeat, "started_at": last_heartbeat}
            if project is not None:
                fields["project"] = project
            if task is not None:
                fields["task"] = task
            client.table("coord_sessions").upsert(
                fields, on_conflict="session_id",
            ).execute()
        except Exception:
            logger.debug("cloud update_session_fields failed", exc_info=True)

    def update_session_status(self, session_id: str, status: str, last_heartbeat: str) -> None:
        """Update session status and heartbeat, re-creating session if missing. Fire-and-forget."""
        try:
            client = self._get_client()
            client.table("coord_sessions").upsert(
                {"session_id": session_id, "status": status, "last_heartbeat": last_heartbeat, "started_at": last_heartbeat},
                on_conflict="session_id",
            ).execute()
        except Exception:
            logger.debug("cloud update_session_status failed", exc_info=True)

    def delete_file_claim(self, file_path: str, session_id: str) -> None:
        """Delete a file claim from Supabase. Fire-and-forget."""
        try:
            client = self._get_client()
            client.table("coord_file_claims").delete().eq(
                "file_path", file_path
            ).eq("session_id", session_id).execute()
        except Exception:
            logger.debug("cloud delete_file_claim failed", exc_info=True)

    def delete_session(self, session_id: str) -> None:
        """Delete a session and all its related data from Supabase. Fire-and-forget."""
        try:
            client = self._get_client()
            # Delete related data first, then the session
            for table in ("coord_file_claims", "coord_file_reads", "coord_audit", "coord_metrics", "coord_intents"):
                client.table(table).delete().eq("session_id", session_id).execute()
            client.table("coord_sessions").delete().eq("session_id", session_id).execute()
        except Exception:
            logger.debug("cloud delete_session failed", exc_info=True)

    def clean_stale_cloud_sessions(self, stale_ids: list) -> None:
        """Mark stale sessions as stopped and clean their claims in Supabase. Fire-and-forget."""
        try:
            client = self._get_client()
            for sid in stale_ids:
                client.table("coord_file_claims").delete().eq("session_id", sid).execute()
                client.table("coord_file_reads").delete().eq("session_id", sid).execute()
                client.table("coord_sessions").update({"status": "stopped"}).eq("session_id", sid).execute()
        except Exception:
            logger.debug("cloud clean_stale_cloud_sessions failed", exc_info=True)

    def reconcile_sessions(self, active_sessions: list) -> None:
        """Full-state reconciliation: ensure Supabase matches local SQLite exactly.

        1. Upsert all locally-active sessions to Supabase.
        2. Mark any Supabase sessions not in the local set as 'stopped'.
        Fire-and-forget.
        """
        try:
            client = self._get_client()
            local_ids = set()
            for s in active_sessions:
                local_ids.add(s["session_id"])
                client.table("coord_sessions").upsert(
                    {
                        "session_id": s["session_id"],
                        "pid": s.get("pid"),
                        "project": s.get("project"),
                        "task": s.get("task") or "",
                        "status": "active",
                        "last_heartbeat": s.get("last_heartbeat"),
                        "started_at": s.get("started_at"),
                    },
                    on_conflict="session_id",
                ).execute()
            # Mark orphaned Supabase sessions as stopped
            resp = client.table("coord_sessions").select("session_id").not_(
                "status", "in_", "(ended,stopped)"
            ).execute()
            for row in resp.data or []:
                if row["session_id"] not in local_ids:
                    client.table("coord_sessions").update(
                        {"status": "stopped"}
                    ).eq("session_id", row["session_id"]).execute()
                    client.table("coord_file_claims").delete().eq(
                        "session_id", row["session_id"]
                    ).execute()
        except Exception:
            logger.debug("cloud reconcile_sessions failed", exc_info=True)

    def insert_audit(
        self,
        session_id: str,
        tool_name: str,
        result_summary: Optional[str],
        created_at: str,
        call_index: Optional[int],
        result_status: str,
        input_size: Optional[int],
        latency_ms: Optional[int] = None,
    ) -> None:
        """Insert an audit entry to Supabase. Fire-and-forget."""
        try:
            client = self._get_client()
            row = {
                "session_id": session_id,
                "tool_name": tool_name,
                "result_summary": result_summary,
                "created_at": created_at,
                "call_index": call_index,
                "result_status": result_status,
                "input_size": input_size,
            }
            if latency_ms is not None:
                row["latency_ms"] = latency_ms
            client.table("coord_audit").insert(row).execute()
        except Exception:
            logger.debug("cloud insert_audit failed", exc_info=True)

    def upsert_task(
        self,
        local_id: int,
        title: str,
        project: Optional[str],
        session_id: Optional[str],
        status: str,
        priority: int,
        created_by: str,
        created_at: str,
        claimed_at: Optional[str],
        completed_at: Optional[str],
        progress: int,
        result: Optional[str],
    ) -> None:
        """Upsert a coordination task to Supabase. Fire-and-forget."""
        try:
            client = self._get_client()
            client.table("coord_tasks").upsert(
                {
                    "local_id": local_id,
                    "title": title,
                    "project": project,
                    "session_id": session_id,
                    "status": status,
                    "priority": priority,
                    "created_by": created_by,
                    "created_at": created_at,
                    "claimed_at": claimed_at,
                    "completed_at": completed_at,
                    "progress": progress,
                    "result": result,
                },
                on_conflict="local_id",
            ).execute()
        except Exception:
            logger.debug("cloud upsert_task failed", exc_info=True)

    # ── New coordination table sync ─────────────────────────────────

    def insert_message(
        self,
        from_session: str,
        to_session: Optional[str],
        project: Optional[str],
        msg_type: str,
        context_id: Optional[str],
        subject: str,
        body: Optional[str],
        ref_task_id: Optional[int],
        created_at: str,
        expires_at: Optional[str],
    ) -> None:
        """Insert a coordination message to Supabase. Fire-and-forget."""
        try:
            client = self._get_client()
            client.table("coord_messages").insert(
                {
                    "from_session": from_session,
                    "to_session": to_session,
                    "project": project,
                    "msg_type": msg_type,
                    "context_id": context_id,
                    "subject": subject,
                    "body": body,
                    "ref_task_id": ref_task_id,
                    "created_at": created_at,
                    "expires_at": expires_at,
                }
            ).execute()
        except Exception:
            logger.debug("cloud insert_message failed", exc_info=True)

    def insert_handoff(
        self,
        session_id: str,
        project: Optional[str],
        completed_tasks: str,
        blocked_items: str,
        key_context: Optional[str],
        next_steps: str,
        files_modified: str,
        decisions_made: str,
        git_branch: Optional[str],
        git_dirty_files: str,
        created_at: str,
    ) -> None:
        """Insert a handoff to Supabase. Fire-and-forget."""
        try:
            client = self._get_client()
            client.table("coord_handoffs").insert(
                {
                    "session_id": session_id,
                    "project": project,
                    "completed_tasks": completed_tasks,
                    "blocked_items": blocked_items,
                    "key_context": key_context,
                    "next_steps": next_steps,
                    "files_modified": files_modified,
                    "decisions_made": decisions_made,
                    "git_branch": git_branch,
                    "git_dirty_files": git_dirty_files,
                    "created_at": created_at,
                }
            ).execute()
        except Exception:
            logger.debug("cloud insert_handoff failed", exc_info=True)

    def insert_git_event(
        self,
        session_id: Optional[str],
        project: str,
        event_type: str,
        commit_hash: Optional[str],
        branch: Optional[str],
        message: Optional[str],
        created_at: str,
    ) -> None:
        """Insert a git event to Supabase. Fire-and-forget."""
        try:
            client = self._get_client()
            client.table("coord_git_events").insert(
                {
                    "session_id": session_id,
                    "project": project,
                    "event_type": event_type,
                    "commit_hash": commit_hash,
                    "branch": branch,
                    "message": message,
                    "created_at": created_at,
                }
            ).execute()
        except Exception:
            logger.debug("cloud insert_git_event failed", exc_info=True)

    def insert_intent(
        self,
        session_id: str,
        intent_type: str,
        description: str,
        target_files: Optional[str],
        target_branch: Optional[str],
        created_at: str,
        expires_at: str,
    ) -> None:
        """Insert an intent to Supabase. Fire-and-forget."""
        try:
            client = self._get_client()
            client.table("coord_intents").insert(
                {
                    "session_id": session_id,
                    "intent_type": intent_type,
                    "description": description,
                    "target_files": target_files,
                    "target_branch": target_branch,
                    "created_at": created_at,
                    "expires_at": expires_at,
                }
            ).execute()
        except Exception:
            logger.debug("cloud insert_intent failed", exc_info=True)

    def insert_decision(
        self,
        domain: str,
        project: str,
        decision: str,
        rationale: Optional[str],
        decided_by: str,
        goal_id: Optional[int],
        status: str,
        created_at: str,
        metadata: Optional[str],
    ) -> None:
        """Insert a decision to Supabase. Fire-and-forget."""
        try:
            client = self._get_client()
            client.table("coord_decisions").insert(
                {
                    "domain": domain,
                    "project": project,
                    "decision": decision,
                    "rationale": rationale,
                    "decided_by": decided_by,
                    "goal_id": goal_id,
                    "status": status,
                    "created_at": created_at,
                    "metadata": metadata,
                }
            ).execute()
        except Exception:
            logger.debug("cloud insert_decision failed", exc_info=True)

    def insert_metric(
        self,
        metric_name: str,
        metric_value: int,
        session_id: Optional[str],
        project: Optional[str],
        metadata: Optional[str],
        created_at: str,
    ) -> None:
        """Insert a metric to Supabase. Fire-and-forget."""
        try:
            client = self._get_client()
            client.table("coord_metrics").insert(
                {
                    "metric_name": metric_name,
                    "metric_value": metric_value,
                    "session_id": session_id,
                    "project": project,
                    "metadata": metadata,
                    "created_at": created_at,
                }
            ).execute()
        except Exception:
            logger.debug("cloud insert_metric failed", exc_info=True)

    def sync_edges(self, batch_size: int = 200) -> dict:
        """Sync edges from local SQLite to Supabase memory_edges table.

        Translation: local node_id → local integer id → Supabase UUID.
        Per-item isolation: individual edge failures are logged and skipped.
        """
        client = self._get_client()
        conn = self._get_local_conn()

        try:
            # Read all local edges
            edges = conn.execute(
                "SELECT source_id, target_id, edge_type, weight FROM edges"
            ).fetchall()

            if not edges:
                return {"synced": 0, "status": "up_to_date"}

            # Collect all unique node_ids referenced by edges
            node_ids = set()
            for e in edges:
                node_ids.add(e["source_id"])
                node_ids.add(e["target_id"])

            # Build node_id → local integer id mapping
            placeholders = ",".join("?" for _ in node_ids)
            rows = conn.execute(
                f"SELECT id, node_id FROM memories WHERE node_id IN ({placeholders})",
                list(node_ids),
            ).fetchall()
            node_to_local = {r["node_id"]: r["id"] for r in rows}

            # Build local_id → Supabase UUID mapping (batch query)
            local_ids = list(node_to_local.values())
            uuid_map: dict[int, str] = {}
            for i in range(0, len(local_ids), batch_size):
                batch = local_ids[i : i + batch_size]
                result = (
                    client.table("memories")
                    .select("id, local_id")
                    .in_("local_id", batch)
                    .execute()
                )
                for r in result.data:
                    uuid_map[r["local_id"]] = r["id"]

            # Translate and upsert edges
            synced = 0
            skipped = 0
            records = []
            for e in edges:
                src_local = node_to_local.get(e["source_id"])
                tgt_local = node_to_local.get(e["target_id"])
                if not src_local or not tgt_local:
                    skipped += 1
                    continue
                src_uuid = uuid_map.get(src_local)
                tgt_uuid = uuid_map.get(tgt_local)
                if not src_uuid or not tgt_uuid:
                    skipped += 1
                    continue

                # Normalize edge_type to allowed set
                etype = e["edge_type"]
                allowed = {
                    "related", "contains_fact", "same_entity",
                    "temporal_cluster", "evolution", "contradicts", "supersedes",
                }
                if etype not in allowed:
                    etype = "related"  # fallback

                records.append({
                    "source_memory_id": src_uuid,
                    "target_memory_id": tgt_uuid,
                    "edge_type": etype,
                    "weight": e["weight"] or 1.0,
                })

            # Batch upsert
            for i in range(0, len(records), batch_size):
                batch = records[i : i + batch_size]
                try:
                    client.table("memory_edges").upsert(
                        batch,
                        on_conflict="source_memory_id,target_memory_id,edge_type",
                    ).execute()
                    synced += len(batch)
                except Exception as e_err:
                    logger.warning("Edge batch upsert failed: %s", e_err)
                    # Per-item fallback
                    for rec in batch:
                        try:
                            client.table("memory_edges").upsert(
                                rec,
                                on_conflict="source_memory_id,target_memory_id,edge_type",
                            ).execute()
                            synced += 1
                        except Exception as item_err:
                            logger.debug("Edge sync failed: %s", item_err)
                            skipped += 1

            return {"synced": synced, "skipped": skipped, "status": "ok"}

        finally:
            conn.close()

    def sync_access_counts(self, batch_size: int = 200) -> dict:
        """Bulk-update access_count in Supabase from local SQLite.

        Local SQLite has accurate access counts (up to 6115); cloud may
        have stale values. This reads all local counts and batch-updates.
        """
        client = self._get_client()
        conn = self._get_local_conn()

        try:
            rows = conn.execute(
                "SELECT id, access_count FROM memories WHERE access_count > 0"
            ).fetchall()

            if not rows:
                return {"updated": 0, "status": "up_to_date"}

            updated = 0
            for i in range(0, len(rows), batch_size):
                batch = rows[i : i + batch_size]
                local_ids = [r["id"] for r in batch]
                count_map = {r["id"]: r["access_count"] for r in batch}

                # Look up Supabase UUIDs
                result = (
                    client.table("memories")
                    .select("id, local_id")
                    .in_("local_id", local_ids)
                    .execute()
                )
                for r in result.data:
                    local_count = count_map.get(r["local_id"])
                    if local_count is not None:
                        try:
                            client.table("memories").update(
                                {"access_count": local_count}
                            ).eq("id", r["id"]).execute()
                            updated += 1
                        except Exception as e:
                            logger.debug(
                                "Access count update failed for %s: %s", r["id"], e
                            )

            return {"updated": updated, "total_local": len(rows), "status": "ok"}

        finally:
            conn.close()

    def status(self) -> str:
        """Get sync status summary."""
        try:
            client = self._get_client()
            result = client.table("sync_state").select("*").execute()
            if not result.data:
                return "No sync state found. Run `omega cloud sync` to start."

            lines = ["## Cloud Sync Status\n"]
            for row in result.data:
                lines.append(
                    f"- **{row['table_name']}**: last synced {row.get('last_sync_at', 'never')}, "
                    f"{row.get('sync_count', 0)} records"
                )
            return "\n".join(lines)
        except Exception as e:
            return f"Cloud sync not configured: {e}"


# Singleton
_sync_instance: Optional[CloudSync] = None
_sync_lock = threading.Lock()


def get_sync() -> CloudSync:
    global _sync_instance
    if _sync_instance is not None:
        return _sync_instance
    with _sync_lock:
        if _sync_instance is not None:
            return _sync_instance
        _sync_instance = CloudSync()
    return _sync_instance
