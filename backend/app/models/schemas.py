from __future__ import annotations

from typing import Optional
from pydantic import BaseModel


class Track(BaseModel):
    track_id: int
    bbox: list[int]          # [x, y, w, h]
    status: str              # "AUTHORIZED" | "UNAUTHORIZED" | "UNKNOWN"
    person_name: Optional[str] = None
    zone: Optional[str] = None
    confidence: float = 0.0
    behavior: Optional[str] = None   # "loitering" | "running" | "fighting"


class Alert(BaseModel):
    type: str                # "UNAUTHORIZED_VISITOR" | "FIGHTING" | etc.
    severity: str            # "LOW" | "MEDIUM" | "HIGH"
    track_id: Optional[int] = None
    message: str


class WSMessage(BaseModel):
    cam_id: str
    timestamp: str
    frame: str               # base64 JPEG
    tracks: list[Track] = []
    alerts: list[Alert] = []
    zone_counts: dict[str, int] = {}
    heatmap: Optional[str] = None   # base64 PNG or null


class InferRequest(BaseModel):
    cam_id: str
    frame: str               # base64 JPEG
    timestamp: str


class InferResponse(BaseModel):
    tracks: list[Track] = []
    alerts: list[Alert] = []
    zone_counts: dict[str, int] = {}


class EmbedRequest(BaseModel):
    cam_id: str
    frame: str               # base64 JPEG
    timestamp: str


class EmbedResponse(BaseModel):
    embedding: list[float]   # 512-dim CLIP vector
