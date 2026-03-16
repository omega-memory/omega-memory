"""Add auth and multi-tenant tables.

Revision ID: 0002
Revises: 0001
Create Date: 2026-03-16
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "0002"
down_revision: Union[str, None] = "0001"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # ------------------------------------------------------------------
    # users
    # ------------------------------------------------------------------
    op.create_table(
        "users",
        sa.Column("id", sa.Text, primary_key=True),
        sa.Column("email", sa.Text, unique=True, nullable=False),
        sa.Column("password_hash", sa.Text),
        sa.Column("display_name", sa.Text),
        sa.Column("created_at", sa.Text),
        sa.Column("updated_at", sa.Text),
        sa.Column("is_active", sa.Integer, server_default="1"),
    )

    # ------------------------------------------------------------------
    # api_tokens
    # ------------------------------------------------------------------
    op.create_table(
        "api_tokens",
        sa.Column("id", sa.Text, primary_key=True),
        sa.Column("user_id", sa.Text, sa.ForeignKey("users.id"), nullable=False),
        sa.Column("token_hash", sa.Text, nullable=False),
        sa.Column("name", sa.Text, nullable=False),
        sa.Column("scopes", sa.Text),
        sa.Column("expires_at", sa.Text),
        sa.Column("created_at", sa.Text),
        sa.Column("last_used_at", sa.Text),
        sa.Column("is_revoked", sa.Integer, server_default="0"),
    )
    op.create_index("idx_api_tokens_user_id", "api_tokens", ["user_id"])
    op.create_index("idx_api_tokens_token_hash", "api_tokens", ["token_hash"])

    # ------------------------------------------------------------------
    # agent_certificates
    # ------------------------------------------------------------------
    op.create_table(
        "agent_certificates",
        sa.Column("id", sa.Text, primary_key=True),
        sa.Column("user_id", sa.Text, sa.ForeignKey("users.id")),
        sa.Column("agent_name", sa.Text, nullable=False),
        sa.Column("fingerprint", sa.Text, unique=True, nullable=False),
        sa.Column("cert_pem", sa.Text, nullable=False),
        sa.Column("serial_number", sa.Text, unique=True, nullable=False),
        sa.Column("issued_at", sa.Text),
        sa.Column("expires_at", sa.Text, nullable=False),
        sa.Column("revoked_at", sa.Text),
        sa.Column("is_revoked", sa.Integer, server_default="0"),
    )

    # ------------------------------------------------------------------
    # user_sessions
    # ------------------------------------------------------------------
    op.create_table(
        "user_sessions",
        sa.Column("id", sa.Text, primary_key=True),
        sa.Column("user_id", sa.Text, sa.ForeignKey("users.id")),
        sa.Column("token_hash", sa.Text, nullable=False),
        sa.Column("device_info", sa.Text),
        sa.Column("ip_address", sa.Text),
        sa.Column("created_at", sa.Text),
        sa.Column("expires_at", sa.Text, nullable=False),
        sa.Column("is_revoked", sa.Integer, server_default="0"),
    )
    op.create_index("idx_user_sessions_token_hash", "user_sessions", ["token_hash"])
    op.create_index("idx_user_sessions_user_id", "user_sessions", ["user_id"])

    # ------------------------------------------------------------------
    # oidc_providers
    # ------------------------------------------------------------------
    op.create_table(
        "oidc_providers",
        sa.Column("id", sa.Text, primary_key=True),
        sa.Column("name", sa.Text, unique=True, nullable=False),
        sa.Column("discovery_url", sa.Text, nullable=False),
        sa.Column("client_id", sa.Text, nullable=False),
        sa.Column("client_secret_encrypted", sa.Text),
        sa.Column("is_enabled", sa.Integer, server_default="1"),
        sa.Column("created_at", sa.Text),
    )

    # ------------------------------------------------------------------
    # workspaces
    # ------------------------------------------------------------------
    op.create_table(
        "workspaces",
        sa.Column("id", sa.Text, primary_key=True),
        sa.Column("name", sa.Text, nullable=False),
        sa.Column("slug", sa.Text, unique=True, nullable=False),
        sa.Column("owner_id", sa.Text, sa.ForeignKey("users.id"), nullable=False),
        sa.Column("created_at", sa.Text),
    )

    # ------------------------------------------------------------------
    # workspace_members
    # ------------------------------------------------------------------
    op.create_table(
        "workspace_members",
        sa.Column("id", sa.Text, primary_key=True),
        sa.Column("workspace_id", sa.Text, sa.ForeignKey("workspaces.id"), nullable=False),
        sa.Column("user_id", sa.Text, sa.ForeignKey("users.id"), nullable=False),
        sa.Column("role", sa.Text, server_default="'member'"),
        sa.Column("joined_at", sa.Text),
        sa.UniqueConstraint("workspace_id", "user_id"),
    )

    # ------------------------------------------------------------------
    # sync_state
    # ------------------------------------------------------------------
    op.create_table(
        "sync_state",
        sa.Column("id", sa.Text, primary_key=True),
        sa.Column("table_name", sa.Text, nullable=False),
        sa.Column("user_id", sa.Text, sa.ForeignKey("users.id")),
        sa.Column("last_local_id", sa.Integer, server_default="0"),
        sa.Column("last_sync_at", sa.Text),
        sa.Column("sync_count", sa.Integer, server_default="0"),
        sa.UniqueConstraint("table_name", "user_id"),
    )

    # ------------------------------------------------------------------
    # documents
    # ------------------------------------------------------------------
    op.create_table(
        "documents",
        sa.Column("id", sa.Text, primary_key=True),
        sa.Column("local_id", sa.Integer),
        sa.Column("source_path", sa.Text, nullable=False),
        sa.Column("source_type", sa.Text, nullable=False),
        sa.Column("title", sa.Text),
        sa.Column("checksum", sa.Text),
        sa.Column("chunk_count", sa.Integer, server_default="0"),
        sa.Column("user_id", sa.Text, sa.ForeignKey("users.id")),
        sa.Column("created_at", sa.Text),
        sa.Column("updated_at", sa.Text),
        sa.Column("synced_at", sa.Text),
    )
    op.create_index("idx_documents_user_id", "documents", ["user_id"])

    # ------------------------------------------------------------------
    # document_chunks
    # ------------------------------------------------------------------
    op.create_table(
        "document_chunks",
        sa.Column("id", sa.Text, primary_key=True),
        sa.Column("document_id", sa.Text, sa.ForeignKey("documents.id")),
        sa.Column("local_id", sa.Integer),
        sa.Column("chunk_index", sa.Integer, nullable=False),
        sa.Column("content", sa.Text, nullable=False),
        sa.Column("chunk_type", sa.Text),
        sa.Column("page_number", sa.Integer),
        sa.Column("token_count", sa.Integer),
        sa.Column("user_id", sa.Text, sa.ForeignKey("users.id")),
        sa.Column("created_at", sa.Text),
    )
    op.create_index("idx_document_chunks_user_id", "document_chunks", ["user_id"])

    # ------------------------------------------------------------------
    # secure_profiles
    # ------------------------------------------------------------------
    op.create_table(
        "secure_profiles",
        sa.Column("id", sa.Text, primary_key=True),
        sa.Column("local_id", sa.Integer),
        sa.Column("category", sa.Text, nullable=False),
        sa.Column("field_name", sa.Text, nullable=False),
        sa.Column("field_value_encrypted", sa.Text, nullable=False),
        sa.Column("metadata", sa.Text),
        sa.Column("user_id", sa.Text, sa.ForeignKey("users.id")),
        sa.Column("created_at", sa.Text),
        sa.Column("updated_at", sa.Text),
        sa.Column("synced_at", sa.Text),
    )
    op.create_index("idx_secure_profiles_user_id", "secure_profiles", ["user_id"])


def downgrade() -> None:
    op.drop_table("secure_profiles")
    op.drop_table("document_chunks")
    op.drop_table("documents")
    op.drop_table("sync_state")
    op.drop_table("workspace_members")
    op.drop_table("workspaces")
    op.drop_table("oidc_providers")
    op.drop_table("user_sessions")
    op.drop_table("agent_certificates")
    op.drop_table("api_tokens")
    op.drop_table("users")
