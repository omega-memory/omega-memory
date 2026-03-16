"""Pydantic v2 schemas for OMEGA API request/response models."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


# ===================================================================
# Memory
# ===================================================================

class MemoryCreate(BaseModel):
    """Payload for creating a new memory."""
    model_config = ConfigDict(from_attributes=True)

    content: str
    event_type: Optional[str] = None
    priority: int = 3
    session_id: Optional[str] = None
    project: Optional[str] = None
    entity_id: Optional[str] = None
    agent_type: Optional[str] = None
    memory_type: str = "semantic"
    referenced_date: Optional[str] = None
    end_date: Optional[str] = None
    ttl_seconds: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = None
    derived_from: Optional[str] = None
    source_uri: Optional[str] = None


class MemoryRead(BaseModel):
    """Full memory representation returned by queries."""
    model_config = ConfigDict(from_attributes=True)

    id: int
    node_id: str
    content: str
    metadata: Optional[str] = None
    created_at: str
    last_accessed: Optional[str] = None
    access_count: int = 0
    ttl_seconds: Optional[int] = None
    session_id: Optional[str] = None
    event_type: Optional[str] = None
    project: Optional[str] = None
    content_hash: Optional[str] = None
    priority: int = 3
    referenced_date: Optional[str] = None
    entity_id: Optional[str] = None
    agent_type: Optional[str] = None
    canonical_hash: Optional[str] = None
    end_date: Optional[str] = None
    extracted_keywords: Optional[str] = None
    retrieval_count: int = 0
    memory_type: str = "semantic"
    valid_from: Optional[str] = None
    valid_until: Optional[str] = None
    derived_from: Optional[str] = None
    source_uri: Optional[str] = None
    status: str = "active"


class MemorySync(BaseModel):
    """Memory payload used for cloud synchronisation."""
    model_config = ConfigDict(from_attributes=True)

    local_id: int
    node_id: str
    content: str
    metadata: Optional[str] = None
    created_at: str
    event_type: Optional[str] = None
    project: Optional[str] = None
    priority: int = 3
    memory_type: str = "semantic"
    user_id: Optional[str] = None

    def to_cloud_dict(self, user_id: str) -> Dict[str, Any]:
        """Return a dict suitable for cloud upsert, injecting *user_id*."""
        d = self.model_dump(exclude_none=True)
        d["user_id"] = user_id
        return d


# ===================================================================
# Edge
# ===================================================================

class EdgeCreate(BaseModel):
    """Payload for creating a graph edge."""
    model_config = ConfigDict(from_attributes=True)

    source_id: str
    target_id: str
    edge_type: str
    weight: float = 1.0
    metadata: Optional[str] = None


# ===================================================================
# Document
# ===================================================================

class DocumentCreate(BaseModel):
    """Payload for ingesting a document."""
    model_config = ConfigDict(from_attributes=True)

    source_path: str
    source_type: str
    title: Optional[str] = None
    checksum: Optional[str] = None
    user_id: Optional[str] = None

    def to_cloud_dict(self, user_id: str) -> Dict[str, Any]:
        d = self.model_dump(exclude_none=True)
        d["user_id"] = user_id
        return d


# ===================================================================
# SyncState
# ===================================================================

class SyncStateRecord(BaseModel):
    """Sync cursor for a given table and user."""
    model_config = ConfigDict(from_attributes=True)

    id: str
    table_name: str
    user_id: Optional[str] = None
    last_local_id: int = 0
    last_sync_at: Optional[str] = None
    sync_count: int = 0

    def to_cloud_dict(self, user_id: str) -> Dict[str, Any]:
        d = self.model_dump(exclude_none=True)
        d["user_id"] = user_id
        return d


# ===================================================================
# Auth — User
# ===================================================================

class UserCreate(BaseModel):
    """Payload for registering a new user."""
    model_config = ConfigDict(from_attributes=True)

    email: str
    password: str
    display_name: Optional[str] = None


class UserRead(BaseModel):
    """Public user representation (no password hash)."""
    model_config = ConfigDict(from_attributes=True)

    id: str
    email: str
    display_name: Optional[str] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    is_active: int = 1


# ===================================================================
# Auth — Token
# ===================================================================

class TokenCreate(BaseModel):
    """Payload for issuing an API token."""
    model_config = ConfigDict(from_attributes=True)

    name: str
    scopes: Optional[List[str]] = None
    expires_at: Optional[str] = None


# ===================================================================
# Auth — Certificate
# ===================================================================

class CertificateInfo(BaseModel):
    """Read-only certificate metadata."""
    model_config = ConfigDict(from_attributes=True)

    id: str
    agent_name: str
    fingerprint: str
    serial_number: str
    issued_at: Optional[str] = None
    expires_at: str
    is_revoked: int = 0


# ===================================================================
# Workspace
# ===================================================================

class WorkspaceCreate(BaseModel):
    """Payload for creating a workspace."""
    model_config = ConfigDict(from_attributes=True)

    name: str
    slug: str
