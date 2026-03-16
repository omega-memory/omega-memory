"""Add user_id column to core tables for multi-tenant support.

Revision ID: 0004
Revises: 0003
Create Date: 2026-03-16

The baseline schema v14 tables (memories, edges, etc.) were created without
user_id. This migration adds user_id to enable multi-tenant data isolation.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "0004"
down_revision: Union[str, None] = "0003"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Add user_id to memories
    op.add_column("memories", sa.Column("user_id", sa.Text))

    # Add user_id to edges
    op.add_column("edges", sa.Column("user_id", sa.Text))

    # Create indexes
    op.create_index("idx_memories_user_id", "memories", ["user_id"])
    op.create_index("idx_edges_user_id", "edges", ["user_id"])


def downgrade() -> None:
    op.drop_index("idx_edges_user_id", "edges")
    op.drop_index("idx_memories_user_id", "memories")
    op.drop_column("edges", "user_id")
    op.drop_column("memories", "user_id")
