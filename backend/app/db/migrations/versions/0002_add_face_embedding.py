"""add face_embedding to registered_faces, make photo_path nullable

Revision ID: 0002
Revises: 0001
Create Date: 2026-04-07
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from pgvector.sqlalchemy import Vector

revision: str = "0002"
down_revision: Union[str, None] = "0001"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Store pre-computed InsightFace normed_embedding (512-float) directly in DB.
    # This removes the local filesystem dependency: no photo_path needed for inference.
    op.add_column(
        "registered_faces",
        sa.Column("face_embedding", Vector(512), nullable=True),
    )
    # photo_path is now optional (kept for reference/UI only, not required for inference)
    op.alter_column("registered_faces", "photo_path", nullable=True)


def downgrade() -> None:
    op.drop_column("registered_faces", "face_embedding")
    op.alter_column("registered_faces", "photo_path", nullable=False)
