import logging
import os
import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Union
from pathlib import Path

from .backend import CloudBackend

logger = logging.getLogger("omega.cloud.supabase")


def _omega_home() -> Path:
    return Path(os.environ.get("OMEGA_HOME", str(Path.home() / ".omega")))


class SupabaseBackend(CloudBackend):
    """Supabase implementation of CloudBackend using supabase-py REST client."""

    def __init__(
        self,
        url: str,
        key: str,
        service_key: Optional[str] = None,
        timeout: int = 30,
    ):
        try:
            from supabase import create_client
            from supabase.lib.client_options import SyncClientOptions
        except ImportError:
            raise ImportError(
                "supabase package required for SupabaseBackend. "
                "Install with: pip install supabase"
            )

        self.url = url
        self.key = service_key or key
        self.options = SyncClientOptions(
            postgrest_client_timeout=timeout,
            storage_client_timeout=timeout,
        )
        self.client = create_client(self.url, self.key, options=self.options)
        self.user_id = self._load_user_id()

    @staticmethod
    def _load_user_id() -> Optional[str]:
        """Load user_id from ~/.omega/config.json for cloud sync attribution."""
        config_path = _omega_home() / "config.json"
        if config_path.exists():
            try:
                config = json.loads(config_path.read_text())
                return config.get("user_id")
            except Exception:
                pass
        return None

    def upsert_memories(self, records: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Upsert memory records to the cloud."""
        if self.user_id:
            for record in records:
                record["user_id"] = self.user_id
        
        result = self.client.table("memories").upsert(
            records, on_conflict="local_id,user_id" if self.user_id else "local_id"
        ).execute()
        return {"data": result.data, "status": "ok"}

    def delete_memories(self, memory_ids: List[str]) -> Dict[str, Any]:
        """Delete memories by their cloud UUIDs."""
        query = self.client.table("memories").delete().in_("id", memory_ids)
        if self.user_id:
            query = query.eq("user_id", self.user_id)
        
        result = query.execute()
        return {"data": result.data, "status": "ok"}

    def get_sync_state(self, table_name: str) -> Optional[Dict[str, Any]]:
        """Get the sync state for a specific table."""
        query = self.client.table("sync_state").select("*").eq("table_name", table_name)
        if self.user_id:
            query = query.eq("user_id", self.user_id)
        
        result = query.execute()
        return result.data[0] if result.data else None

    def update_sync_state(self, table_name: str, state: Dict[str, Any]) -> Dict[str, Any]:
        """Update the sync state for a specific table."""
        state["table_name"] = table_name
        if self.user_id:
            state["user_id"] = self.user_id
        
        conflict_key = "table_name,user_id" if self.user_id else "table_name"
        result = self.client.table("sync_state").upsert(
            state, on_conflict=conflict_key
        ).execute()
        return {"data": result.data, "status": "ok"}

    def upsert_documents(self, records: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Upsert document records to the cloud."""
        if self.user_id:
            for record in records:
                record["user_id"] = self.user_id
        
        result = self.client.table("documents").upsert(
            records, on_conflict="checksum,user_id" if self.user_id else "checksum"
        ).execute()
        return {"data": result.data, "status": "ok"}

    def upsert_document_chunks(self, records: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Upsert document chunk records to the cloud."""
        # document_chunks usually relate back to documents which have user_id
        # if the table has user_id, we add it.
        if self.user_id:
            for record in records:
                record["user_id"] = self.user_id
        
        # document_chunks might not have a simple unique constraint other than ID or (document_id, chunk_index)
        result = self.client.table("document_chunks").upsert(
            records, on_conflict="id" 
        ).execute()
        return {"data": result.data, "status": "ok"}

    def upsert_embeddings(
        self, table_name: str, records: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Upsert embedding records to the cloud."""
        if self.user_id:
            for record in records:
                record["user_id"] = self.user_id
        
        # Assuming embeddings table has a local_id or similar
        result = self.client.table(table_name).upsert(
            records, on_conflict="local_id,user_id" if self.user_id else "local_id"
        ).execute()
        return {"data": result.data, "status": "ok"}

    def delete_by_local_ids(
        self, table_name: str, local_ids: List[int]
    ) -> Dict[str, Any]:
        """Delete records from a table by their local integer IDs."""
        query = self.client.table(table_name).delete().in_("local_id", local_ids)
        if self.user_id:
            query = query.eq("user_id", self.user_id)
        
        result = query.execute()
        return {"data": result.data, "status": "ok"}

    def search_memories_by_embedding(
        self, embedding: List[float], limit: int = 10, threshold: float = 0.5
    ) -> List[Dict[str, Any]]:
        """Search for memories using vector similarity."""
        params = {
            "query_embedding": embedding,
            "match_threshold": threshold,
            "match_count": limit,
        }
        if self.user_id:
            params["user_id_param"] = self.user_id
            
        result = self.client.rpc("match_memories", params).execute()
        return result.data

    def rpc(self, function_name: str, params: Dict[str, Any]) -> Any:
        """Call a remote procedure/function."""
        if self.user_id and "user_id" not in params:
            params["user_id"] = self.user_id
            
        result = self.client.rpc(function_name, params).execute()
        return result.data

    def health_check(self) -> bool:
        """Check if the backend is reachable and healthy."""
        try:
            # Simple query to check connectivity
            self.client.table("sync_state").select("count", count="exact").limit(1).execute()
            return True
        except Exception:
            return False
