"""Baseline schema v14 — all existing OMEGA tables.

Revision ID: 0001
Revises: None
Create Date: 2026-03-15
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "0001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # ------------------------------------------------------------------
    # schema_version (bookkeeping for legacy migration path)
    # ------------------------------------------------------------------
    op.create_table(
        "schema_version",
        sa.Column("version", sa.Integer, nullable=False),
    )
    op.execute("INSERT INTO schema_version (version) VALUES (14)")

    # ------------------------------------------------------------------
    # memories
    # ------------------------------------------------------------------
    op.create_table(
        "memories",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("node_id", sa.Text, unique=True, nullable=False),
        sa.Column("content", sa.Text, nullable=False),
        sa.Column("metadata", sa.Text),
        sa.Column("created_at", sa.Text, nullable=False),
        sa.Column("last_accessed", sa.Text),
        sa.Column("access_count", sa.Integer, server_default="0"),
        sa.Column("ttl_seconds", sa.Integer),
        sa.Column("session_id", sa.Text),
        sa.Column("event_type", sa.Text),
        sa.Column("project", sa.Text),
        sa.Column("content_hash", sa.Text),
        sa.Column("priority", sa.Integer, server_default="3"),
        sa.Column("referenced_date", sa.Text),
        sa.Column("entity_id", sa.Text),
        sa.Column("agent_type", sa.Text),
        sa.Column("canonical_hash", sa.Text),
        sa.Column("end_date", sa.Text),
        sa.Column("extracted_keywords", sa.Text),
        sa.Column("retrieval_count", sa.Integer, server_default="0"),
        sa.Column("memory_type", sa.Text, server_default="'semantic'"),
        sa.Column("valid_from", sa.Text),
        sa.Column("valid_until", sa.Text),
        sa.Column("derived_from", sa.Text),
        sa.Column("source_uri", sa.Text),
        sa.Column("status", sa.Text, server_default="'active'"),
    )

    for col in (
        "node_id", "event_type", "session_id", "project", "created_at",
        "content_hash", "priority", "referenced_date", "entity_id",
        "agent_type", "canonical_hash", "end_date", "last_accessed",
        "ttl_seconds", "memory_type", "valid_from", "valid_until",
        "derived_from", "source_uri", "status",
    ):
        op.create_index(f"idx_memories_{col}", "memories", [col])

    op.create_index("idx_memories_event_access", "memories", ["event_type", "access_count"])

    # ------------------------------------------------------------------
    # edges
    # ------------------------------------------------------------------
    op.create_table(
        "edges",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("source_id", sa.Text, nullable=False),
        sa.Column("target_id", sa.Text, nullable=False),
        sa.Column("edge_type", sa.Text, nullable=False),
        sa.Column("weight", sa.Float, server_default="1.0"),
        sa.Column("metadata", sa.Text),
        sa.Column("created_at", sa.Text, server_default="(datetime('now'))"),
        sa.UniqueConstraint("source_id", "target_id", "edge_type"),
    )
    for col in ("source_id", "target_id", "edge_type"):
        op.create_index(f"idx_edges_{col}", "edges", [col])

    # ------------------------------------------------------------------
    # forgetting_log
    # ------------------------------------------------------------------
    op.create_table(
        "forgetting_log",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("node_id", sa.Text, nullable=False),
        sa.Column("content_preview", sa.Text),
        sa.Column("event_type", sa.Text),
        sa.Column("reason", sa.Text, nullable=False),
        sa.Column("deleted_at", sa.Text, nullable=False),
        sa.Column("metadata", sa.Text),
    )
    op.create_index("idx_forgetting_log_deleted_at", "forgetting_log", ["deleted_at"])
    op.create_index("idx_forgetting_log_reason", "forgetting_log", ["reason"])
    op.create_index(
        "idx_forgetting_log_node_reason", "forgetting_log",
        ["node_id", "reason"], unique=True,
    )

    # ------------------------------------------------------------------
    # cloud_delete_queue
    # ------------------------------------------------------------------
    op.create_table(
        "cloud_delete_queue",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("local_id", sa.Integer, nullable=False),
        sa.Column("deleted_at", sa.Text, nullable=False),
    )

    # ------------------------------------------------------------------
    # entity_index
    # ------------------------------------------------------------------
    op.create_table(
        "entity_index",
        sa.Column("entity_name", sa.Text, primary_key=True),
        sa.Column("entity_type", sa.Text, server_default="'person'"),
        sa.Column("statement_count", sa.Integer, server_default="0"),
        sa.Column("outcome_count", sa.Integer, server_default="0"),
        sa.Column("contradiction_score", sa.Float, server_default="0.0"),
        sa.Column("follow_through_rate", sa.Float, server_default="0.0"),
        sa.Column("first_seen", sa.Text, nullable=False),
        sa.Column("last_updated", sa.Text, nullable=False),
        sa.Column("metadata", sa.Text),
    )
    op.create_index("idx_entity_index_type", "entity_index", ["entity_type"])
    op.create_index("idx_entity_index_score", "entity_index", ["contradiction_score"])
    op.create_index("idx_entity_index_updated", "entity_index", ["last_updated"])

    # ------------------------------------------------------------------
    # memory_clusters
    # ------------------------------------------------------------------
    op.create_table(
        "memory_clusters",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("cluster_id", sa.Integer, nullable=False),
        sa.Column("label", sa.Text, nullable=False),
        sa.Column("member_count", sa.Integer, nullable=False),
        sa.Column("centroid", sa.LargeBinary),
        sa.Column("representative_keywords", sa.Text),
        sa.Column("representative_memory_ids", sa.Text),
        sa.Column("created_at", sa.Text, nullable=False),
        sa.Column("updated_at", sa.Text, nullable=False),
        sa.Column("superseded", sa.Integer, server_default="0"),
    )
    op.create_index("idx_memory_clusters_superseded", "memory_clusters", ["superseded"])
    op.create_index("idx_memory_clusters_cluster_id", "memory_clusters", ["cluster_id"])

    # ------------------------------------------------------------------
    # thompson_arms
    # ------------------------------------------------------------------
    op.create_table(
        "thompson_arms",
        sa.Column("arm_id", sa.Text, primary_key=True),
        sa.Column("arm_type", sa.Text, nullable=False),
        sa.Column("alpha", sa.Float, server_default="1.0"),
        sa.Column("beta", sa.Float, server_default="1.0"),
        sa.Column("total_trials", sa.Integer, server_default="0"),
        sa.Column("total_successes", sa.Integer, server_default="0"),
        sa.Column("last_updated", sa.Text, nullable=False),
        sa.Column("context", sa.Text),
    )
    op.create_index("idx_thompson_arms_type", "thompson_arms", ["arm_type"])

    # ------------------------------------------------------------------
    # maintenance_dlq
    # ------------------------------------------------------------------
    op.create_table(
        "maintenance_dlq",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("stage_name", sa.Text, nullable=False),
        sa.Column("error_class", sa.Text, server_default="'transient'"),
        sa.Column("error_message", sa.Text),
        sa.Column("remediation_attempts", sa.Integer, server_default="0"),
        sa.Column("max_remediation", sa.Integer, server_default="3"),
        sa.Column("status", sa.Text, server_default="'pending'"),
        sa.Column("next_retry_at", sa.Text),
        sa.Column("created_at", sa.Text, nullable=False),
        sa.Column("updated_at", sa.Text, nullable=False),
    )
    op.create_index("idx_maintenance_dlq_status", "maintenance_dlq", ["status"])


def downgrade() -> None:
    op.drop_table("maintenance_dlq")
    op.drop_table("thompson_arms")
    op.drop_table("memory_clusters")
    op.drop_table("entity_index")
    op.drop_table("cloud_delete_queue")
    op.drop_table("forgetting_log")
    op.drop_table("edges")
    op.drop_table("memories")
    op.drop_table("schema_version")
