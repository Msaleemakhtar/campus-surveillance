from datetime import datetime

import sqlalchemy as sa
from pgvector.sqlalchemy import Vector
from sqlalchemy.orm import Mapped, mapped_column

from app.db.postgres import Base


class RegisteredFace(Base):
    __tablename__ = "registered_faces"

    id: Mapped[int] = mapped_column(primary_key=True)
    person_id: Mapped[str] = mapped_column(sa.String(64), unique=True, nullable=False)
    name: Mapped[str] = mapped_column(sa.String(128), nullable=False)
    role: Mapped[str] = mapped_column(sa.String(32), nullable=False)   # "student" | "staff"
    photo_path: Mapped[str] = mapped_column(sa.Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(server_default=sa.func.now())


class DetectedPerson(Base):
    __tablename__ = "detected_persons"

    id: Mapped[int] = mapped_column(primary_key=True)
    track_id: Mapped[int] = mapped_column(nullable=False)
    camera_id: Mapped[str] = mapped_column(sa.String(16), nullable=False)
    timestamp: Mapped[datetime] = mapped_column(nullable=False)
    bbox_x: Mapped[int] = mapped_column(nullable=False)
    bbox_y: Mapped[int] = mapped_column(nullable=False)
    bbox_w: Mapped[int] = mapped_column(nullable=False)
    bbox_h: Mapped[int] = mapped_column(nullable=False)
    is_authorized: Mapped[bool] = mapped_column(nullable=False, default=False)
    matched_person_id: Mapped[str | None] = mapped_column(sa.String(64), sa.ForeignKey("registered_faces.person_id"), nullable=True)
    confidence: Mapped[float] = mapped_column(nullable=False, default=0.0)
    behavior: Mapped[str | None] = mapped_column(sa.String(32), nullable=True)


class Incident(Base):
    __tablename__ = "incidents"

    id: Mapped[int] = mapped_column(primary_key=True)
    camera_id: Mapped[str] = mapped_column(sa.String(16), nullable=False)
    incident_type: Mapped[str] = mapped_column(sa.String(64), nullable=False)
    severity: Mapped[str] = mapped_column(sa.String(8), nullable=False)   # LOW | MEDIUM | HIGH
    track_id: Mapped[int | None] = mapped_column(nullable=True)
    frame_path: Mapped[str | None] = mapped_column(sa.Text, nullable=True)
    timestamp: Mapped[datetime] = mapped_column(nullable=False)
    resolved: Mapped[bool] = mapped_column(default=False)
    assigned_to: Mapped[str | None] = mapped_column(sa.String(128), nullable=True)
    notes: Mapped[str | None] = mapped_column(sa.Text, nullable=True)


class ZoneCount(Base):
    __tablename__ = "zone_counts"

    id: Mapped[int] = mapped_column(primary_key=True)
    camera_id: Mapped[str] = mapped_column(sa.String(16), nullable=False)
    zone_name: Mapped[str] = mapped_column(sa.String(64), nullable=False)
    person_count: Mapped[int] = mapped_column(nullable=False)
    congestion_score: Mapped[float] = mapped_column(nullable=False)
    recorded_at: Mapped[datetime] = mapped_column(server_default=sa.func.now())


class ClipEmbedding(Base):
    __tablename__ = "clip_embeddings"

    id: Mapped[int] = mapped_column(primary_key=True)
    camera_id: Mapped[str] = mapped_column(sa.String(16), nullable=False)
    timestamp: Mapped[datetime] = mapped_column(nullable=False)
    frame_path: Mapped[str | None] = mapped_column(sa.Text, nullable=True)
    embedding: Mapped[list[float]] = mapped_column(Vector(512), nullable=False)


class AttendanceLog(Base):
    __tablename__ = "attendance_logs"

    id: Mapped[int] = mapped_column(primary_key=True)
    person_id: Mapped[str] = mapped_column(sa.String(64), sa.ForeignKey("registered_faces.person_id"), nullable=False)
    camera_id: Mapped[str] = mapped_column(sa.String(16), nullable=False)
    entry_time: Mapped[datetime] = mapped_column(nullable=False)
    exit_time: Mapped[datetime | None] = mapped_column(nullable=True)
    zone_name: Mapped[str | None] = mapped_column(sa.String(64), nullable=True)


class Responder(Base):
    __tablename__ = "responders"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(sa.String(128), nullable=False)
    status: Mapped[str] = mapped_column(sa.String(16), nullable=False, default="ON_DUTY")  # ON_DUTY | RESPONDING | ON_CALL
    assigned_incident_id: Mapped[int | None] = mapped_column(sa.ForeignKey("incidents.id"), nullable=True)
    updated_at: Mapped[datetime] = mapped_column(server_default=sa.func.now(), onupdate=sa.func.now())
