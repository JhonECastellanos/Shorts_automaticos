"""Migración inicial: tablas jobs y job_events.

Revision ID: 001_initial
Revises: 
Create Date: 2026-04-14

"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "001_initial"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "jobs",
        sa.Column("id", sa.String(8), primary_key=True),
        sa.Column("project", sa.String(100), nullable=False),
        sa.Column("episode", sa.String(100), nullable=False),
        sa.Column("url", sa.Text, nullable=True),
        sa.Column("local_file", sa.Text, nullable=True),
        sa.Column("ai_model", sa.String(100), nullable=True),
        sa.Column("diarization_provider", sa.String(50), nullable=True),
        sa.Column("min_duration", sa.Integer, nullable=True),
        sa.Column("max_duration", sa.Integer, nullable=True),
        sa.Column("steps", sa.JSON, nullable=False, server_default="[]"),
        sa.Column("steps_completed", sa.JSON, nullable=False, server_default="[]"),
        sa.Column("step_durations", sa.JSON, nullable=False, server_default="{}"),
        sa.Column("status", sa.String(20), nullable=False, server_default="pending"),
        sa.Column("current_step", sa.String(50), nullable=True),
        sa.Column("progress", sa.Float, nullable=True),
        sa.Column("error", sa.Text, nullable=True),
        sa.Column("elapsed_seconds", sa.Float, nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index("ix_jobs_project", "jobs", ["project"])
    op.create_index("ix_jobs_episode", "jobs", ["episode"])
    op.create_index("ix_jobs_status", "jobs", ["status"])
    op.create_index("ix_jobs_project_episode", "jobs", ["project", "episode"])

    op.create_table(
        "job_events",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column(
            "job_id",
            sa.String(8),
            sa.ForeignKey("jobs.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("step", sa.String(50), nullable=False),
        sa.Column("message", sa.Text, nullable=False),
        sa.Column("percent", sa.Float, nullable=True),
        sa.Column("occurred_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_job_events_job_id", "job_events", ["job_id"])


def downgrade() -> None:
    op.drop_table("job_events")
    op.drop_table("jobs")
