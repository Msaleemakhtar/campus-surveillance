"""initial schema — 7 tables

Revision ID: 0001
Revises:
Create Date: 2026-04-05
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from pgvector.sqlalchemy import Vector

revision: str = "0001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    op.create_table(
        "registered_faces",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("person_id", sa.String(64), unique=True, nullable=False),
        sa.Column("name", sa.String(128), nullable=False),
        sa.Column("role", sa.String(32), nullable=False),
        sa.Column("photo_path", sa.Text, nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )

    op.create_table(
        "incidents",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("camera_id", sa.String(16), nullable=False),
        sa.Column("incident_type", sa.String(64), nullable=False),
        sa.Column("severity", sa.String(8), nullable=False),
        sa.Column("track_id", sa.Integer, nullable=True),
        sa.Column("frame_path", sa.Text, nullable=True),
        sa.Column("timestamp", sa.DateTime(timezone=True), nullable=False),
        sa.Column("resolved", sa.Boolean, default=False),
        sa.Column("assigned_to", sa.String(128), nullable=True),
        sa.Column("notes", sa.Text, nullable=True),
    )

    op.create_table(
        "detected_persons",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("track_id", sa.Integer, nullable=False),
        sa.Column("camera_id", sa.String(16), nullable=False),
        sa.Column("timestamp", sa.DateTime(timezone=True), nullable=False),
        sa.Column("bbox_x", sa.Integer, nullable=False),
        sa.Column("bbox_y", sa.Integer, nullable=False),
        sa.Column("bbox_w", sa.Integer, nullable=False),
        sa.Column("bbox_h", sa.Integer, nullable=False),
        sa.Column("is_authorized", sa.Boolean, nullable=False, default=False),
        sa.Column("matched_person_id", sa.String(64), sa.ForeignKey("registered_faces.person_id"), nullable=True),
        sa.Column("confidence", sa.Float, nullable=False, default=0.0),
        sa.Column("behavior", sa.String(32), nullable=True),
    )

    op.create_table(
        "zone_counts",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("camera_id", sa.String(16), nullable=False),
        sa.Column("zone_name", sa.String(64), nullable=False),
        sa.Column("person_count", sa.Integer, nullable=False),
        sa.Column("congestion_score", sa.Float, nullable=False),
        sa.Column("recorded_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )

    op.create_table(
        "clip_embeddings",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("camera_id", sa.String(16), nullable=False),
        sa.Column("timestamp", sa.DateTime(timezone=True), nullable=False),
        sa.Column("frame_path", sa.Text, nullable=True),
        sa.Column("embedding", Vector(512), nullable=False),
    )

    op.create_table(
        "attendance_logs",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("person_id", sa.String(64), sa.ForeignKey("registered_faces.person_id"), nullable=False),
        sa.Column("camera_id", sa.String(16), nullable=False),
        sa.Column("entry_time", sa.DateTime(timezone=True), nullable=False),
        sa.Column("exit_time", sa.DateTime(timezone=True), nullable=True),
        sa.Column("zone_name", sa.String(64), nullable=True),
    )

    op.create_table(
        "responders",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("name", sa.String(128), nullable=False),
        sa.Column("status", sa.String(16), nullable=False, default="ON_DUTY"),
        sa.Column("assigned_incident_id", sa.Integer, sa.ForeignKey("incidents.id"), nullable=True),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )


def downgrade() -> None:
    op.drop_table("responders")
    op.drop_table("attendance_logs")
    op.drop_table("clip_embeddings")
    op.drop_table("zone_counts")
    op.drop_table("detected_persons")
    op.drop_table("incidents")
    op.drop_table("registered_faces")
    op.execute("DROP EXTENSION IF EXISTS vector")
