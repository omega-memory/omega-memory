"""SQLAlchemy 2.0 declarative models for OMEGA.

Every column type intentionally uses TEXT for UUIDs and datetimes
to stay compatible with SQLite while remaining portable to PostgreSQL.
"""

from __future__ import annotations

from sqlalchemy import (
    Float,
    Integer,
    LargeBinary,
    Text,
    UniqueConstraint,
    ForeignKey,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


# ---------------------------------------------------------------------------
# Base
# ---------------------------------------------------------------------------

class Base(DeclarativeBase):
    """Shared declarative base for all OMEGA models."""
    pass


# ===================================================================
# Existing tables (schema v14)
# ===================================================================

class Memory(Base):
    __tablename__ = "memories"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    node_id: Mapped[str] = mapped_column(Text, unique=True, nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    meta_: Mapped[str | None] = mapped_column("metadata", Text)
    created_at: Mapped[str] = mapped_column(Text, nullable=False)
    last_accessed: Mapped[str | None] = mapped_column(Text)
    access_count: Mapped[int] = mapped_column(Integer, default=0)
    ttl_seconds: Mapped[int | None] = mapped_column(Integer)
    session_id: Mapped[str | None] = mapped_column(Text, index=True)
    event_type: Mapped[str | None] = mapped_column(Text, index=True)
    project: Mapped[str | None] = mapped_column(Text, index=True)
    content_hash: Mapped[str | None] = mapped_column(Text, index=True)
    priority: Mapped[int] = mapped_column(Integer, default=3)
    referenced_date: Mapped[str | None] = mapped_column(Text, index=True)
    entity_id: Mapped[str | None] = mapped_column(Text, index=True)
    agent_type: Mapped[str | None] = mapped_column(Text, index=True)
    canonical_hash: Mapped[str | None] = mapped_column(Text, index=True)
    end_date: Mapped[str | None] = mapped_column(Text, index=True)
    extracted_keywords: Mapped[str | None] = mapped_column(Text)
    retrieval_count: Mapped[int] = mapped_column(Integer, default=0)
    memory_type: Mapped[str] = mapped_column(Text, default="semantic", index=True)
    valid_from: Mapped[str | None] = mapped_column(Text, index=True)
    valid_until: Mapped[str | None] = mapped_column(Text, index=True)
    derived_from: Mapped[str | None] = mapped_column(Text, index=True)
    source_uri: Mapped[str | None] = mapped_column(Text, index=True)
    status: Mapped[str] = mapped_column(Text, default="active", index=True)


class Edge(Base):
    __tablename__ = "edges"
    __table_args__ = (
        UniqueConstraint("source_id", "target_id", "edge_type"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    source_id: Mapped[str] = mapped_column(Text, nullable=False, index=True)
    target_id: Mapped[str] = mapped_column(Text, nullable=False, index=True)
    edge_type: Mapped[str] = mapped_column(Text, nullable=False, index=True)
    weight: Mapped[float] = mapped_column(Float, default=1.0)
    meta_: Mapped[str | None] = mapped_column("metadata", Text)
    created_at: Mapped[str] = mapped_column(Text, server_default="(datetime('now'))")


class ForgettingLog(Base):
    __tablename__ = "forgetting_log"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    node_id: Mapped[str] = mapped_column(Text, nullable=False)
    content_preview: Mapped[str | None] = mapped_column(Text)
    event_type: Mapped[str | None] = mapped_column(Text)
    reason: Mapped[str] = mapped_column(Text, nullable=False, index=True)
    deleted_at: Mapped[str] = mapped_column(Text, nullable=False, index=True)
    meta_: Mapped[str | None] = mapped_column("metadata", Text)


class CloudDeleteQueue(Base):
    __tablename__ = "cloud_delete_queue"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    local_id: Mapped[int] = mapped_column(Integer, nullable=False)
    deleted_at: Mapped[str] = mapped_column(Text, nullable=False)


class EntityIndex(Base):
    __tablename__ = "entity_index"

    entity_name: Mapped[str] = mapped_column(Text, primary_key=True)
    entity_type: Mapped[str] = mapped_column(Text, default="person")
    statement_count: Mapped[int] = mapped_column(Integer, default=0)
    outcome_count: Mapped[int] = mapped_column(Integer, default=0)
    contradiction_score: Mapped[float] = mapped_column(Float, default=0.0)
    follow_through_rate: Mapped[float] = mapped_column(Float, default=0.0)
    first_seen: Mapped[str] = mapped_column(Text, nullable=False)
    last_updated: Mapped[str] = mapped_column(Text, nullable=False)
    meta_: Mapped[str | None] = mapped_column("metadata", Text)


class MemoryCluster(Base):
    __tablename__ = "memory_clusters"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    cluster_id: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    label: Mapped[str] = mapped_column(Text, nullable=False)
    member_count: Mapped[int] = mapped_column(Integer, nullable=False)
    centroid: Mapped[bytes | None] = mapped_column(LargeBinary)
    representative_keywords: Mapped[str | None] = mapped_column(Text)
    representative_memory_ids: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[str] = mapped_column(Text, nullable=False)
    updated_at: Mapped[str] = mapped_column(Text, nullable=False)
    superseded: Mapped[int] = mapped_column(Integer, default=0, index=True)


class ThompsonArm(Base):
    __tablename__ = "thompson_arms"

    arm_id: Mapped[str] = mapped_column(Text, primary_key=True)
    arm_type: Mapped[str] = mapped_column(Text, nullable=False, index=True)
    alpha: Mapped[float] = mapped_column(Float, default=1.0)
    beta: Mapped[float] = mapped_column(Float, default=1.0)
    total_trials: Mapped[int] = mapped_column(Integer, default=0)
    total_successes: Mapped[int] = mapped_column(Integer, default=0)
    last_updated: Mapped[str] = mapped_column(Text, nullable=False)
    context: Mapped[str | None] = mapped_column(Text)


class MaintenanceDLQ(Base):
    __tablename__ = "maintenance_dlq"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    stage_name: Mapped[str] = mapped_column(Text, nullable=False)
    error_class: Mapped[str] = mapped_column(Text, default="transient")
    error_message: Mapped[str | None] = mapped_column(Text)
    remediation_attempts: Mapped[int] = mapped_column(Integer, default=0)
    max_remediation: Mapped[int] = mapped_column(Integer, default=3)
    status: Mapped[str] = mapped_column(Text, default="pending", index=True)
    next_retry_at: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[str] = mapped_column(Text, nullable=False)
    updated_at: Mapped[str] = mapped_column(Text, nullable=False)


# ===================================================================
# New auth / multi-tenant tables
# ===================================================================

class User(Base):
    __tablename__ = "users"

    id: Mapped[str] = mapped_column(Text, primary_key=True)  # uuid
    email: Mapped[str] = mapped_column(Text, unique=True, nullable=False)
    password_hash: Mapped[str | None] = mapped_column(Text)
    display_name: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[str | None] = mapped_column(Text)
    updated_at: Mapped[str | None] = mapped_column(Text)
    is_active: Mapped[int] = mapped_column(Integer, default=1)


class APIToken(Base):
    __tablename__ = "api_tokens"

    id: Mapped[str] = mapped_column(Text, primary_key=True)
    user_id: Mapped[str] = mapped_column(Text, ForeignKey("users.id"), nullable=False)
    token_hash: Mapped[str] = mapped_column(Text, nullable=False)
    name: Mapped[str] = mapped_column(Text, nullable=False)
    scopes: Mapped[str | None] = mapped_column(Text)  # JSON
    expires_at: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[str | None] = mapped_column(Text)
    last_used_at: Mapped[str | None] = mapped_column(Text)
    is_revoked: Mapped[int] = mapped_column(Integer, default=0)


class AgentCertificate(Base):
    __tablename__ = "agent_certificates"

    id: Mapped[str] = mapped_column(Text, primary_key=True)
    user_id: Mapped[str | None] = mapped_column(Text, ForeignKey("users.id"))
    agent_name: Mapped[str] = mapped_column(Text, nullable=False)
    fingerprint: Mapped[str] = mapped_column(Text, unique=True, nullable=False)
    cert_pem: Mapped[str] = mapped_column(Text, nullable=False)
    serial_number: Mapped[str] = mapped_column(Text, unique=True, nullable=False)
    issued_at: Mapped[str | None] = mapped_column(Text)
    expires_at: Mapped[str] = mapped_column(Text, nullable=False)
    revoked_at: Mapped[str | None] = mapped_column(Text)
    is_revoked: Mapped[int] = mapped_column(Integer, default=0)


class UserSession(Base):
    __tablename__ = "user_sessions"

    id: Mapped[str] = mapped_column(Text, primary_key=True)
    user_id: Mapped[str | None] = mapped_column(Text, ForeignKey("users.id"))
    token_hash: Mapped[str] = mapped_column(Text, nullable=False)
    device_info: Mapped[str | None] = mapped_column(Text)
    ip_address: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[str | None] = mapped_column(Text)
    expires_at: Mapped[str] = mapped_column(Text, nullable=False)
    is_revoked: Mapped[int] = mapped_column(Integer, default=0)


class OIDCProvider(Base):
    __tablename__ = "oidc_providers"

    id: Mapped[str] = mapped_column(Text, primary_key=True)
    name: Mapped[str] = mapped_column(Text, unique=True, nullable=False)
    discovery_url: Mapped[str] = mapped_column(Text, nullable=False)
    client_id: Mapped[str] = mapped_column(Text, nullable=False)
    client_secret_encrypted: Mapped[str | None] = mapped_column(Text)
    is_enabled: Mapped[int] = mapped_column(Integer, default=1)
    created_at: Mapped[str | None] = mapped_column(Text)


class Workspace(Base):
    __tablename__ = "workspaces"

    id: Mapped[str] = mapped_column(Text, primary_key=True)
    name: Mapped[str] = mapped_column(Text, nullable=False)
    slug: Mapped[str] = mapped_column(Text, unique=True, nullable=False)
    owner_id: Mapped[str] = mapped_column(Text, ForeignKey("users.id"), nullable=False)
    created_at: Mapped[str | None] = mapped_column(Text)


class WorkspaceMember(Base):
    __tablename__ = "workspace_members"
    __table_args__ = (
        UniqueConstraint("workspace_id", "user_id"),
    )

    id: Mapped[str] = mapped_column(Text, primary_key=True)
    workspace_id: Mapped[str] = mapped_column(Text, ForeignKey("workspaces.id"), nullable=False)
    user_id: Mapped[str] = mapped_column(Text, ForeignKey("users.id"), nullable=False)
    role: Mapped[str] = mapped_column(Text, default="member")
    joined_at: Mapped[str | None] = mapped_column(Text)


# ===================================================================
# Cloud-synced tables
# ===================================================================

class SyncState(Base):
    __tablename__ = "sync_state"
    __table_args__ = (
        UniqueConstraint("table_name", "user_id"),
    )

    id: Mapped[str] = mapped_column(Text, primary_key=True)
    table_name: Mapped[str] = mapped_column(Text, nullable=False)
    user_id: Mapped[str | None] = mapped_column(Text, ForeignKey("users.id"))
    last_local_id: Mapped[int] = mapped_column(Integer, default=0)
    last_sync_at: Mapped[str | None] = mapped_column(Text)
    sync_count: Mapped[int] = mapped_column(Integer, default=0)


class Document(Base):
    __tablename__ = "documents"

    id: Mapped[str] = mapped_column(Text, primary_key=True)
    local_id: Mapped[int | None] = mapped_column(Integer)
    source_path: Mapped[str] = mapped_column(Text, nullable=False)
    source_type: Mapped[str] = mapped_column(Text, nullable=False)
    title: Mapped[str | None] = mapped_column(Text)
    checksum: Mapped[str | None] = mapped_column(Text)
    chunk_count: Mapped[int] = mapped_column(Integer, default=0)
    user_id: Mapped[str | None] = mapped_column(Text, ForeignKey("users.id"))
    created_at: Mapped[str | None] = mapped_column(Text)
    updated_at: Mapped[str | None] = mapped_column(Text)
    synced_at: Mapped[str | None] = mapped_column(Text)


class DocumentChunk(Base):
    __tablename__ = "document_chunks"

    id: Mapped[str] = mapped_column(Text, primary_key=True)
    document_id: Mapped[str | None] = mapped_column(Text, ForeignKey("documents.id"))
    local_id: Mapped[int | None] = mapped_column(Integer)
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    chunk_type: Mapped[str | None] = mapped_column(Text)
    page_number: Mapped[int | None] = mapped_column(Integer)
    token_count: Mapped[int | None] = mapped_column(Integer)
    user_id: Mapped[str | None] = mapped_column(Text, ForeignKey("users.id"))
    created_at: Mapped[str | None] = mapped_column(Text)


class SecureProfile(Base):
    __tablename__ = "secure_profiles"

    id: Mapped[str] = mapped_column(Text, primary_key=True)
    local_id: Mapped[int | None] = mapped_column(Integer)
    category: Mapped[str] = mapped_column(Text, nullable=False)
    field_name: Mapped[str] = mapped_column(Text, nullable=False)
    field_value_encrypted: Mapped[str] = mapped_column(Text, nullable=False)
    meta_: Mapped[str | None] = mapped_column("metadata", Text)
    user_id: Mapped[str | None] = mapped_column(Text, ForeignKey("users.id"))
    created_at: Mapped[str | None] = mapped_column(Text)
    updated_at: Mapped[str | None] = mapped_column(Text)
    synced_at: Mapped[str | None] = mapped_column(Text)
