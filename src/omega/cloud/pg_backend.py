import logging
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional

from .backend import CloudBackend

logger = logging.getLogger("omega.cloud.pg_backend")


class PostgresBackend(CloudBackend):
    """PostgreSQL implementation of CloudBackend using SQLAlchemy."""

    def __init__(self, dsn: str):
        self._dsn = dsn
        try:
            from sqlalchemy import create_engine, text
            self._engine = create_engine(dsn)
            self._text = text
        except ImportError:
            raise ImportError(
                "sqlalchemy package required. Install with: pip install sqlalchemy"
            )

    def _execute(self, query: str, params: Dict[str, Any]) -> Any:
        with self._engine.connect() as conn:
            result = conn.execute(self._text(query), params)
            if query.strip().upper().startswith(("INSERT", "UPDATE", "DELETE")):
                conn.commit()
            return result

    def upsert_memories(self, records: List[Dict[str, Any]], user_id: Optional[str]) -> int:
        if not records:
            return 0
        
        # Build multi-row INSERT ... ON CONFLICT
        # This is a bit complex with raw SQL but let's try a simple version or loop
        # For efficiency, we should batch, but here we'll do one by one or a simple batch loop
        count = 0
        query = """
            INSERT INTO memories (
                local_id, user_id, content, event_type, priority, session_id, 
                project, tags, metadata, created_at, entity_id, memory_type, 
                access_count, synced_at
            ) VALUES (
                :local_id, :user_id, :content, :event_type, :priority, :session_id, 
                :project, :tags, :metadata, :created_at, :entity_id, :memory_type, 
                :access_count, :synced_at
            )
            ON CONFLICT (local_id, user_id) DO UPDATE SET
                content = EXCLUDED.content,
                event_type = EXCLUDED.event_type,
                priority = EXCLUDED.priority,
                session_id = EXCLUDED.session_id,
                project = EXCLUDED.project,
                tags = EXCLUDED.tags,
                metadata = EXCLUDED.metadata,
                created_at = EXCLUDED.created_at,
                entity_id = EXCLUDED.entity_id,
                memory_type = EXCLUDED.memory_type,
                access_count = EXCLUDED.access_count,
                synced_at = EXCLUDED.synced_at
        """
        with self._engine.connect() as conn:
            for record in records:
                params = record.copy()
                params["user_id"] = user_id
                conn.execute(self._text(query), params)
            conn.commit()
            count = len(records)
        return count

    def delete_memories(self, local_ids: List[int], user_id: Optional[str]) -> int:
        if not local_ids:
            return 0
        query = "DELETE FROM memories WHERE local_id IN :ids"
        params = {"ids": tuple(local_ids)}
        if user_id:
            query += " AND user_id = :uid"
            params["uid"] = user_id
        
        result = self._execute(query, params)
        return result.rowcount

    def get_sync_state(self, table_name: str, user_id: Optional[str]) -> Optional[Dict[str, Any]]:
        query = "SELECT * FROM sync_state WHERE table_name = :table"
        params = {"table": table_name}
        if user_id:
            query += " AND user_id = :uid"
            params["uid"] = user_id
        
        result = self._execute(query, params).fetchone()
        return dict(result._mapping) if result else None

    def update_sync_state(
        self,
        table_name: str,
        user_id: Optional[str],
        last_local_id: int,
        sync_count: int,
    ) -> None:
        query = """
            INSERT INTO sync_state (table_name, user_id, last_local_id, last_sync_at, sync_count)
            VALUES (:table, :uid, :last_id, :last_sync, :count)
            ON CONFLICT (table_name, user_id) DO UPDATE SET
                last_local_id = EXCLUDED.last_local_id,
                last_sync_at = EXCLUDED.last_sync_at,
                sync_count = EXCLUDED.sync_count
        """
        params = {
            "table": table_name,
            "uid": user_id,
            "last_id": last_local_id,
            "last_sync": datetime.now(timezone.utc).isoformat(),
            "count": sync_count,
        }
        self._execute(query, params)

    def upsert_documents(self, records: List[Dict[str, Any]], user_id: Optional[str]) -> int:
        if not records:
            return 0
        query = """
            INSERT INTO documents (
                local_id, user_id, source_path, source_type, title, checksum,
                chunk_count, created_at, updated_at
            ) VALUES (
                :local_id, :user_id, :source_path, :source_type, :title, :checksum,
                :chunk_count, :created_at, :updated_at
            )
            ON CONFLICT (local_id, user_id) DO UPDATE SET
                source_path = EXCLUDED.source_path,
                source_type = EXCLUDED.source_type,
                title = EXCLUDED.title,
                checksum = EXCLUDED.checksum,
                chunk_count = EXCLUDED.chunk_count,
                created_at = EXCLUDED.created_at,
                updated_at = EXCLUDED.updated_at
        """
        with self._engine.connect() as conn:
            for record in records:
                params = record.copy()
                params["user_id"] = user_id
                conn.execute(self._text(query), params)
            conn.commit()
        return len(records)

    def upsert_document_chunks(self, records: List[Dict[str, Any]], user_id: Optional[str]) -> int:
        if not records:
            return 0
        query = """
            INSERT INTO document_chunks (
                local_id, user_id, document_id, chunk_index, content, 
                chunk_type, page_number, token_count, created_at
            ) VALUES (
                :local_id, :user_id, :document_id, :chunk_index, :content, 
                :chunk_type, :page_number, :token_count, :created_at
            )
            ON CONFLICT (local_id, user_id) DO UPDATE SET
                document_id = EXCLUDED.document_id,
                chunk_index = EXCLUDED.chunk_index,
                content = EXCLUDED.content,
                chunk_type = EXCLUDED.chunk_type,
                page_number = EXCLUDED.page_number,
                token_count = EXCLUDED.token_count,
                created_at = EXCLUDED.created_at
        """
        with self._engine.connect() as conn:
            for record in records:
                params = record.copy()
                params["user_id"] = user_id
                conn.execute(self._text(query), params)
            conn.commit()
        return len(records)

    def upsert_embeddings(self, records: List[Dict[str, Any]]) -> int:
        if not records:
            return 0
        # Records expected to have memory_id or document_chunk_id, and embedding
        # assuming embedding is a list of floats, pgvector handles it if cast to vector
        query = """
            INSERT INTO embeddings (memory_id, document_chunk_id, embedding)
            VALUES (:memory_id, :document_chunk_id, :embedding)
            ON CONFLICT (memory_id) DO UPDATE SET
                embedding = EXCLUDED.embedding
            -- and similar for document_chunk_id if needed
        """
        # Note: real implementation would need to handle which ID is present
        with self._engine.connect() as conn:
            for record in records:
                conn.execute(self._text(query), record)
            conn.commit()
        return len(records)

    def delete_by_local_ids(self, table_name: str, local_ids: List[int], user_id: Optional[str]) -> int:
        if not local_ids:
            return 0
        query = f"DELETE FROM {table_name} WHERE local_id IN :ids"
        params = {"ids": tuple(local_ids)}
        if user_id:
            query += " AND user_id = :uid"
            params["uid"] = user_id
        
        result = self._execute(query, params)
        return result.rowcount

    def search_memories_by_embedding(
        self,
        embedding: List[float],
        user_id: Optional[str],
        limit: int = 10,
        threshold: float = 0.5,
    ) -> List[Dict[str, Any]]:
        # Using pgvector operator <=> (cosine distance)
        # 1 - distance = cosine similarity
        query = """
            SELECT m.*, (1 - (e.embedding <=> :emb)) as similarity
            FROM memories m
            JOIN embeddings e ON m.id = e.memory_id
            WHERE (1 - (e.embedding <=> :emb)) > :threshold
        """
        params = {"emb": embedding, "threshold": threshold, "limit": limit}
        if user_id:
            query += " AND m.user_id = :uid"
            params["uid"] = user_id
        
        query += " ORDER BY similarity DESC LIMIT :limit"
        
        result = self._execute(query, params).fetchall()
        return [dict(r._mapping) for r in result]

    def rpc(self, function_name: str, params: Dict[str, Any]) -> Any:
        # PostgreSQL doesn't have a direct 'rpc' like Supabase, but we can call functions
        param_placeholders = ", ".join([f":{k}" for k in params.keys()])
        query = f"SELECT * FROM {function_name}({param_placeholders})"
        result = self._execute(query, params)
        try:
            return [dict(r._mapping) for r in result.fetchall()]
        except Exception:
            return None

    def health_check(self) -> Dict[str, Any]:
        try:
            start = datetime.now()
            self._execute("SELECT 1", {})
            latency = (datetime.now() - start).total_seconds() * 1000
            return {"status": "ok", "latency_ms": latency, "backend": "postgres"}
        except Exception as e:
            return {"status": "error", "message": str(e), "backend": "postgres"}
