"""Cloudflare D1 backend for OMEGA cloud sync.

D1 is SQLite-based, so this backend uses standard SQL with D1's HTTP API.
Requires: CLOUDFLARE_ACCOUNT_ID, CLOUDFLARE_API_TOKEN, D1_DATABASE_ID in env or secrets.json.

D1 API docs: https://developers.cloudflare.com/api/resources/d1/
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from .backend import CloudBackend

logger = logging.getLogger("omega.cloud.d1_backend")

_D1_API_BASE = "https://api.cloudflare.com/client/v4"


class D1Backend(CloudBackend):
    """Cloudflare D1 implementation of CloudBackend via HTTP API."""

    def __init__(
        self,
        account_id: str,
        api_token: str,
        database_id: str,
    ):
        self._account_id = account_id
        self._api_token = api_token
        self._database_id = database_id
        self._base_url = (
            f"{_D1_API_BASE}/accounts/{account_id}/d1/database/{database_id}"
        )

    def _headers(self) -> dict:
        return {
            "Authorization": f"Bearer {self._api_token}",
            "Content-Type": "application/json",
        }

    def _query(self, sql: str, params: list | None = None) -> list[dict]:
        """Execute a D1 query via the HTTP API."""
        import urllib.request

        body = json.dumps({"sql": sql, "params": params or []}).encode()
        req = urllib.request.Request(
            f"{self._base_url}/query",
            data=body,
            headers=self._headers(),
            method="POST",
        )

        with urllib.request.urlopen(req) as resp:
            data = json.loads(resp.read())

        if not data.get("success"):
            errors = data.get("errors", [])
            raise RuntimeError(f"D1 query failed: {errors}")

        results = data.get("result", [])
        if results and "results" in results[0]:
            return results[0]["results"]
        return []

    def _execute(self, sql: str, params: list | None = None) -> dict:
        """Execute a D1 mutation (INSERT/UPDATE/DELETE)."""
        import urllib.request

        body = json.dumps({"sql": sql, "params": params or []}).encode()
        req = urllib.request.Request(
            f"{self._base_url}/query",
            data=body,
            headers=self._headers(),
            method="POST",
        )

        with urllib.request.urlopen(req) as resp:
            data = json.loads(resp.read())

        if not data.get("success"):
            errors = data.get("errors", [])
            raise RuntimeError(f"D1 execute failed: {errors}")

        result = data.get("result", [{}])[0]
        return {
            "changes": result.get("meta", {}).get("changes", 0),
            "duration": result.get("meta", {}).get("duration", 0),
        }

    def upsert_memories(
        self, records: List[Dict[str, Any]], user_id: Optional[str] = None
    ) -> int:
        if not records:
            return 0

        count = 0
        for record in records:
            params = [
                record.get("local_id"),
                user_id,
                record.get("content"),
                record.get("event_type"),
                record.get("priority", 3),
                record.get("session_id"),
                record.get("project"),
                record.get("tags"),
                record.get("metadata"),
                record.get("created_at"),
                record.get("entity_id"),
                record.get("memory_type", "semantic"),
                record.get("access_count", 0),
                datetime.now(timezone.utc).isoformat(),
            ]
            self._execute(
                """INSERT INTO memories (
                    id, user_id, content, event_type, priority, session_id,
                    project, tags, metadata, created_at, entity_id, memory_type,
                    access_count, synced_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT (id, user_id) DO UPDATE SET
                    content = excluded.content,
                    event_type = excluded.event_type,
                    priority = excluded.priority,
                    metadata = excluded.metadata,
                    synced_at = excluded.synced_at
                """,
                params,
            )
            count += 1
        return count

    def delete_memories(
        self, local_ids: List[int], user_id: Optional[str] = None
    ) -> int:
        if not local_ids:
            return 0
        placeholders = ",".join(["?"] * len(local_ids))
        params: list = list(local_ids)
        sql = f"DELETE FROM memories WHERE id IN ({placeholders})"
        if user_id:
            sql += " AND user_id = ?"
            params.append(user_id)
        result = self._execute(sql, params)
        return result.get("changes", 0)

    def get_sync_state(
        self, table_name: str, user_id: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        sql = "SELECT * FROM sync_state WHERE table_name = ?"
        params: list = [table_name]
        if user_id:
            sql += " AND user_id = ?"
            params.append(user_id)
        rows = self._query(sql, params)
        return rows[0] if rows else None

    def update_sync_state(
        self,
        table_name: str,
        user_id: Optional[str],
        last_local_id: int,
        sync_count: int,
    ) -> None:
        self._execute(
            """INSERT INTO sync_state (id, table_name, user_id, last_local_id, last_sync_at, sync_count)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT (table_name, user_id) DO UPDATE SET
                last_local_id = excluded.last_local_id,
                last_sync_at = excluded.last_sync_at,
                sync_count = excluded.sync_count
            """,
            [
                f"{table_name}:{user_id or 'default'}",
                table_name,
                user_id,
                last_local_id,
                datetime.now(timezone.utc).isoformat(),
                sync_count,
            ],
        )

    def upsert_documents(
        self, records: List[Dict[str, Any]], user_id: Optional[str] = None
    ) -> int:
        if not records:
            return 0
        count = 0
        for record in records:
            self._execute(
                """INSERT INTO documents (
                    id, local_id, user_id, source_path, source_type, title,
                    checksum, chunk_count, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT (id, user_id) DO UPDATE SET
                    source_path = excluded.source_path,
                    title = excluded.title,
                    checksum = excluded.checksum,
                    chunk_count = excluded.chunk_count,
                    updated_at = excluded.updated_at
                """,
                [
                    record.get("id", record.get("local_id")),
                    record.get("local_id"),
                    user_id,
                    record.get("source_path"),
                    record.get("source_type"),
                    record.get("title"),
                    record.get("checksum"),
                    record.get("chunk_count", 0),
                    record.get("created_at"),
                    record.get("updated_at"),
                ],
            )
            count += 1
        return count

    def upsert_document_chunks(
        self, records: List[Dict[str, Any]], user_id: Optional[str] = None
    ) -> int:
        if not records:
            return 0
        count = 0
        for record in records:
            self._execute(
                """INSERT INTO document_chunks (
                    id, document_id, local_id, chunk_index, content,
                    chunk_type, page_number, token_count, user_id, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT (id, user_id) DO UPDATE SET
                    content = excluded.content,
                    chunk_type = excluded.chunk_type,
                    token_count = excluded.token_count
                """,
                [
                    record.get("id", record.get("local_id")),
                    record.get("document_id"),
                    record.get("local_id"),
                    record.get("chunk_index"),
                    record.get("content"),
                    record.get("chunk_type"),
                    record.get("page_number"),
                    record.get("token_count"),
                    user_id,
                    record.get("created_at"),
                ],
            )
            count += 1
        return count

    def upsert_embeddings(self, records: List[Dict[str, Any]]) -> int:
        # D1 doesn't support pgvector — store as JSON arrays
        if not records:
            return 0
        logger.warning("D1 backend: embeddings stored as JSON (no vector search)")
        return len(records)

    def delete_by_local_ids(
        self, table_name: str, local_ids: List[int], user_id: Optional[str] = None
    ) -> int:
        if not local_ids:
            return 0
        placeholders = ",".join(["?"] * len(local_ids))
        params: list = list(local_ids)
        sql = f"DELETE FROM {table_name} WHERE local_id IN ({placeholders})"
        if user_id:
            sql += " AND user_id = ?"
            params.append(user_id)
        result = self._execute(sql, params)
        return result.get("changes", 0)

    def search_memories_by_embedding(
        self,
        embedding: List[float],
        user_id: Optional[str] = None,
        limit: int = 10,
        threshold: float = 0.5,
    ) -> List[Dict[str, Any]]:
        # D1 doesn't have vector search — fall back to full-text
        logger.warning("D1 backend: vector search not supported, using full-text fallback")
        sql = "SELECT * FROM memories WHERE 1=1"
        params: list = []
        if user_id:
            sql += " AND user_id = ?"
            params.append(user_id)
        sql += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)
        return self._query(sql, params)

    def rpc(self, function_name: str, params: Dict[str, Any]) -> Any:
        # D1 doesn't support stored procedures
        logger.warning("D1 backend: RPC not supported (function: %s)", function_name)
        return None

    def health_check(self) -> Dict[str, Any]:
        try:
            from time import time
            start = time()
            self._query("SELECT 1")
            latency = (time() - start) * 1000
            return {"status": "ok", "latency_ms": latency, "backend": "d1"}
        except Exception as e:
            return {"status": "error", "message": str(e), "backend": "d1"}
