from typing import Any, Dict, List, Optional, Protocol, Union


class CloudBackend(Protocol):
    """Protocol for cloud backends (Supabase, PostgreSQL, etc.)."""

    def upsert_memories(self, records: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Upsert memory records to the cloud."""
        ...

    def delete_memories(self, memory_ids: List[str]) -> Dict[str, Any]:
        """Delete memories by their cloud UUIDs."""
        ...

    def get_sync_state(self, table_name: str) -> Optional[Dict[str, Any]]:
        """Get the sync state for a specific table."""
        ...

    def update_sync_state(self, table_name: str, state: Dict[str, Any]) -> Dict[str, Any]:
        """Update the sync state for a specific table."""
        ...

    def upsert_documents(self, records: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Upsert document records to the cloud."""
        ...

    def upsert_document_chunks(self, records: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Upsert document chunk records to the cloud."""
        ...

    def upsert_embeddings(
        self, table_name: str, records: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Upsert embedding records to the cloud."""
        ...

    def delete_by_local_ids(
        self, table_name: str, local_ids: List[int]
    ) -> Dict[str, Any]:
        """Delete records from a table by their local integer IDs."""
        ...

    def search_memories_by_embedding(
        self, embedding: List[float], limit: int = 10, threshold: float = 0.5
    ) -> List[Dict[str, Any]]:
        """Search for memories using vector similarity."""
        ...

    def rpc(self, function_name: str, params: Dict[str, Any]) -> Any:
        """Call a remote procedure/function."""
        ...

    def health_check(self) -> bool:
        """Check if the backend is reachable and healthy."""
        ...
