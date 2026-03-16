"""Add dashboard, coordination, licensing, and scheduling tables.

Revision ID: 0003
Revises: 0002
Create Date: 2026-03-16

Aligned with website/src/lib/db/schema.ts (Drizzle schema).
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "0003"
down_revision: Union[str, None] = "0002"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # ------------------------------------------------------------------
    # licenses (pro license keys) — matches Drizzle: licenses
    # ------------------------------------------------------------------
    op.create_table(
        "licenses",
        sa.Column("id", sa.Text, primary_key=True),
        sa.Column("key", sa.Text, unique=True, nullable=False),
        sa.Column("email", sa.Text, nullable=False),
        sa.Column("status", sa.Text, server_default="'active'"),
        sa.Column("stripe_customer_id", sa.Text),
        sa.Column("stripe_subscription_id", sa.Text),
        sa.Column("created_at", sa.Text),
        sa.Column("last_validated_at", sa.Text),
    )

    # ------------------------------------------------------------------
    # pro_releases — matches Drizzle: proReleases
    # ------------------------------------------------------------------
    op.create_table(
        "pro_releases",
        sa.Column("id", sa.Text, primary_key=True),
        sa.Column("version", sa.Text, nullable=False),
        sa.Column("url", sa.Text, nullable=False),
        sa.Column("filename", sa.Text, nullable=False),
        sa.Column("file_size", sa.Integer),
        sa.Column("changelog", sa.Text),
        sa.Column("created_at", sa.Text),
    )

    # ------------------------------------------------------------------
    # admin_alerts — matches Drizzle: adminAlerts
    # ------------------------------------------------------------------
    op.create_table(
        "admin_alerts",
        sa.Column("id", sa.Text, primary_key=True),
        sa.Column("type", sa.Text, nullable=False),
        sa.Column("severity", sa.Text, server_default="'warning'"),
        sa.Column("message", sa.Text, nullable=False),
        sa.Column("metadata", sa.Text),
        sa.Column("status", sa.Text, server_default="'active'"),
        sa.Column("recurrence_count", sa.Integer, server_default="0"),
        sa.Column("created_at", sa.Text),
        sa.Column("resolved_at", sa.Text),
    )

    # ------------------------------------------------------------------
    # coord_sessions — matches Drizzle: coordSessions
    # ------------------------------------------------------------------
    op.create_table(
        "coord_sessions",
        sa.Column("id", sa.Text, primary_key=True),
        sa.Column("user_id", sa.Text, sa.ForeignKey("users.id")),
        sa.Column("agent_type", sa.Text),
        sa.Column("started_at", sa.Text),
        sa.Column("ended_at", sa.Text),
        sa.Column("status", sa.Text, server_default="'active'"),
        sa.Column("metadata", sa.Text),
    )

    # ------------------------------------------------------------------
    # coord_file_claims — matches Drizzle: coordFileClaims
    # ------------------------------------------------------------------
    op.create_table(
        "coord_file_claims",
        sa.Column("id", sa.Text, primary_key=True),
        sa.Column("session_id", sa.Text, sa.ForeignKey("coord_sessions.id")),
        sa.Column("file_path", sa.Text, nullable=False),
        sa.Column("claim_type", sa.Text, server_default="'write'"),
        sa.Column("claimed_at", sa.Text),
        sa.Column("released_at", sa.Text),
    )

    # ------------------------------------------------------------------
    # schedules — matches Drizzle: schedules
    # ------------------------------------------------------------------
    op.create_table(
        "schedules",
        sa.Column("id", sa.Text, primary_key=True),
        sa.Column("name", sa.Text, nullable=False),
        sa.Column("cron_expr", sa.Text),
        sa.Column("handler", sa.Text, nullable=False),
        sa.Column("enabled", sa.Integer, server_default="1"),
        sa.Column("last_run_at", sa.Text),
        sa.Column("next_run_at", sa.Text),
        sa.Column("metadata", sa.Text),
    )

    # ------------------------------------------------------------------
    # schedule_runs — matches Drizzle: scheduleRuns
    # ------------------------------------------------------------------
    op.create_table(
        "schedule_runs",
        sa.Column("id", sa.Text, primary_key=True),
        sa.Column("schedule_id", sa.Text, sa.ForeignKey("schedules.id")),
        sa.Column("status", sa.Text, server_default="'running'"),
        sa.Column("started_at", sa.Text),
        sa.Column("finished_at", sa.Text),
        sa.Column("error", sa.Text),
        sa.Column("metadata", sa.Text),
    )


def downgrade() -> None:
    op.drop_table("schedule_runs")
    op.drop_table("schedules")
    op.drop_table("coord_file_claims")
    op.drop_table("coord_sessions")
    op.drop_table("admin_alerts")
    op.drop_table("pro_releases")
    op.drop_table("licenses")
