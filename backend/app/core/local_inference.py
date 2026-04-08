"""
Phase 5: InsightFace face recognition + behaviour-based alerts.

One dedicated daemon thread runs YOLO in a tight loop.
Each camera submits its latest frame to a per-camera slot (overwrite
semantics — worker always sees the freshest frame, no queue backlog).

After YOLO detection:
  1. _CentroidTracker assigns stable IDs (Hungarian algorithm).
  2. _FaceRecognizer identifies persons against a registered face DB.
     Face recognition is expensive (~40–80 ms per call on CPU), so results
     are cached per-track and re-run only every _FACE_CACHE_EVERY cycles.
  3. Alerts are generated: AFTER_HOURS_PRESENCE, UNAUTHORIZED_VISITOR,
     LOITERING (>120 s dwell), RUNNING (centroid speed >180 px/s at 416 px
     wide).

Earlier phases:
  Phase 2 — CentroidTracker (Hungarian assignment, edge expiry)
  Phase 3 — TrackExtrapolator (smooth motion between results) [in pipeline.py]
  Phase 4 — Subscriber registry (demand-driven inference)

Hardware reality on i5-8250U:
  YOLO imgsz=416: avg 291ms (faster than 320 on this CPU)
  10 cameras × 291ms = 2.91s cycle per camera
  → accept ~0.3 Hz per camera, design everything around that
"""
from __future__ import annotations

import logging
import math
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from ultralytics import YOLO

from app.core.config import settings
from app.models.schemas import Alert, InferResponse, Track

log = logging.getLogger(__name__)

# ── Model singletons ──────────────────────────────────────────────────────────

_yolo: YOLO | None = None

# ── Pending / result slots (per camera) ──────────────────────────────────────

_pending: dict[str, tuple[np.ndarray, bool, int]] = {}
_pending_lock = threading.Lock()

_results: dict[str, tuple[InferResponse, int]] = {}
_results_lock = threading.Lock()

# ── Re-ID queue (two-tier priority, overwrite semantics per (cam_id, tid)) ────
# CLIP is decoupled from the YOLO worker thread entirely.
# _infer() submits crops here; _reid_worker_loop() consumes them asynchronously
# and writes face_status/name/confidence back to the _ActiveTrack object.
#
# Two dicts separate first-seen crops (priority — new person, needs ID fast)
# from periodic refresh crops (background — can wait behind priority entries).
_reid_priority: dict[tuple[str, int], np.ndarray] = {}  # first detection of a track
_reid_refresh: dict[tuple[str, int], np.ndarray] = {}   # periodic re-ID refresh
_reid_lock = threading.Lock()

_NUM_WORKERS = 1   # serial is faster than concurrent on CPU

# ── Subscriber registry ───────────────────────────────────────────────────────
# Tracks how many WebSocket clients are watching each camera.
# Pipeline checks this before submitting frames to YOLO — zero subscribers
# means no inference, saving ~291 ms of CPU per skipped cycle.

_subscriber_counts: dict[str, int] = {}
_sub_lock = threading.Lock()

# ── Alert history (rolling buffer for analytics) ──────────────────────────────
# Stores recent alerts enriched with cam_id + UTC timestamp for the dashboard.
# maxlen=200 keeps ~several minutes of multi-camera activity without unbounded growth.

_alert_history: deque[dict] = deque(maxlen=200)
_alert_history_lock = threading.Lock()

# ── Frame embed slots (Phase 8 — CLIP search) ─────────────────────────────────
# One slot per camera (overwrite semantics — worker always sees freshest frame).
_frame_embed_slots: dict[str, tuple[str, np.ndarray]] = {}   # cam_id → (timestamp, frame)
_frame_embed_lock = threading.Lock()
_frame_embed_event = threading.Event()


def add_subscriber(cam_id: str) -> None:
    with _sub_lock:
        prev = _subscriber_counts.get(cam_id, 0)
        _subscriber_counts[cam_id] = prev + 1
    if prev == 0:
        log.info("Camera %s → active (first subscriber)", cam_id)


def remove_subscriber(cam_id: str) -> None:
    with _sub_lock:
        count = _subscriber_counts.get(cam_id, 0)
        if count > 1:
            _subscriber_counts[cam_id] = count - 1
        else:
            _subscriber_counts.pop(cam_id, None)
    if count <= 1:
        log.info("Camera %s → idle (no subscribers)", cam_id)


_analytics_active_until: float = 0.0   # monotonic — set by analytics heartbeat
_ANALYTICS_TTL = 10.0                  # seconds after last poll before going idle


def analytics_heartbeat() -> None:
    """Called by analytics REST endpoints to keep inference running while dashboard is open."""
    global _analytics_active_until
    _analytics_active_until = time.monotonic() + _ANALYTICS_TTL


def has_subscribers(cam_id: str) -> bool:
    if time.monotonic() < _analytics_active_until:
        return True
    with _sub_lock:
        return _subscriber_counts.get(cam_id, 0) > 0


def subscriber_counts() -> dict[str, int]:
    with _sub_lock:
        return dict(_subscriber_counts)


# ── Centroid-distance tracker (worker-thread-only) ────────────────────────────

@dataclass
class _ActiveTrack:
    track_id: int
    cx: float          # centroid x (pixels)
    cy: float          # centroid y (pixels)
    last_seen_frame: int
    first_seen_time: float  # time.monotonic() — for loitering
    speed_px_f: float = 0.0   # pixels/frame between last two inference results
    vx_f: float = 0.0         # horizontal velocity (px/frame)
    vy_f: float = 0.0         # vertical velocity (px/frame)

    # Face recognition cache — updated every _FACE_CACHE_EVERY cycles
    face_status: str = "UNKNOWN"       # "AUTHORIZED" | "UNKNOWN"
    face_name: str | None = None
    face_confidence: float = 0.0
    face_infer_count: int = 0          # incremented each time track appears in infer result

    # Alert deduplication — tracks last fire time per alert type (monotonic seconds)
    last_alert_times: dict[str, float] = field(default_factory=dict)


class _CentroidTracker:
    """
    Match detections to existing tracks using the Hungarian algorithm.
    Uses frame indices for motion modelling to ensure perfect sync with the
    video clock, regardless of CPU/OS scheduling jitter.

    Cost: distance to predicted centroid (cx + vx_f * elapsed_frames).
    """

    _INF = 1e9

    def __init__(
        self,
        speed_px_per_frame: float = 20.0,
        max_distance: float = 400.0,
        max_age_frames: int = 60,
    ) -> None:
        self._tracks: dict[int, _ActiveTrack] = {}
        self._next_id = 1
        self._speed = speed_px_per_frame
        self._max_dist = max_distance
        self._max_age = max_age_frames

    def update(
        self,
        bboxes: list[list[int]],
        frame_idx: int,
        frame_w: int = 0,
        frame_h: int = 0,
    ) -> list[int]:
        now_mono = time.monotonic()

        # Prune stale tracks — shorter lifetime for edge-adjacent tracks
        edge_margin = int(min(frame_w, frame_h) * 0.08) if frame_w and frame_h else 0
        stale = []
        for tid, t in self._tracks.items():
            near_edge = edge_margin > 0 and (
                t.cx < edge_margin
                or t.cy < edge_margin
                or t.cx > frame_w - edge_margin
                or t.cy > frame_h - edge_margin
            )
            # Edge-pruning must be shorter than max_age but still longer than the
            # worst-case YOLO cycle (~3s * 10fps = 30 frames).  40 < max_age=60 ✓
            effective_max_age = 40 if near_edge else self._max_age
            if frame_idx - t.last_seen_frame > effective_max_age:
                stale.append(tid)
        for tid in stale:
            del self._tracks[tid]

        if not bboxes:
            return []

        n_dets = len(bboxes)
        det_cx = [x + w / 2.0 for x, y, w, h in bboxes]
        det_cy = [y + h / 2.0 for x, y, w, h in bboxes]

        track_ids = list(self._tracks.keys())
        n_tracks = len(track_ids)

        assigned: list[int | None] = [None] * n_dets

        if n_tracks > 0:
            # Build cost matrix (n_dets × n_tracks)
            cost = np.full((n_dets, n_tracks), self._INF)
            for i, (dcx, dcy) in enumerate(zip(det_cx, det_cy)):
                for j, tid in enumerate(track_ids):
                    track = self._tracks[tid]
                    df = frame_idx - track.last_seen_frame
                    # Predict current position using last known frame-velocity
                    pred_cx = track.cx + track.vx_f * df
                    pred_cy = track.cy + track.vy_f * df

                    radius = min(self._speed * df, self._max_dist)
                    d = math.hypot(dcx - pred_cx, dcy - pred_cy)
                    if d < radius:
                        cost[i, j] = d

            # Optimal assignment
            row_ind, col_ind = linear_sum_assignment(cost)
            for r, c in zip(row_ind, col_ind):
                if cost[r, c] < self._INF:
                    tid = track_ids[c]
                    t = self._tracks[tid]
                    df = frame_idx - t.last_seen_frame
                    if df > 0:
                        new_vx_f = (det_cx[r] - t.cx) / df
                        new_vy_f = (det_cy[r] - t.cy) / df
                        # Cap at 20 px/f (running speed at 10 FPS)
                        new_vx_f = max(-20.0, min(new_vx_f, 20.0))
                        new_vy_f = max(-20.0, min(new_vy_f, 20.0))

                        # alpha=0.9: fast adaptation so the box tracks the
                        # person closely after the first inference cycle.
                        # 0.7 caused a 30% velocity lag → 27px trail at v=3px/f.
                        alpha = 0.9
                        t.vx_f = alpha * new_vx_f + (1 - alpha) * t.vx_f
                        t.vy_f = alpha * new_vy_f + (1 - alpha) * t.vy_f
                        # Zero out noise-floor jitter (YOLO localization noise is
                        # ±3-5 px; at df≈23 frames that's ~0.2 px/frame — raise
                        # floor to 1.5 so slow centroid drift doesn't feed the
                        # extrapolator with false velocity).
                        if abs(t.vx_f) < 1.5:
                            t.vx_f = 0.0
                        if abs(t.vy_f) < 1.5:
                            t.vy_f = 0.0
                        t.speed_px_f = math.hypot(t.vx_f, t.vy_f)

                    t.cx = det_cx[r]
                    t.cy = det_cy[r]
                    t.last_seen_frame = frame_idx
                    assigned[r] = tid

        # Unmatched detections start new tracks
        for i in range(n_dets):
            if assigned[i] is None:
                new_id = self._next_id
                self._next_id += 1
                self._tracks[new_id] = _ActiveTrack(
                    track_id=new_id,
                    cx=det_cx[i],
                    cy=det_cy[i],
                    last_seen_frame=frame_idx,
                    first_seen_time=now_mono,
                )
                assigned[i] = new_id

        return assigned  # type: ignore[return-value]


# Per-camera tracker parameters.
_TRACKER_PARAMS: dict[str, dict] = {
    "cam11": {"speed_px_per_frame": 25.0, "max_distance": 600.0},
    "cam12": {"speed_px_per_frame": 25.0, "max_distance": 600.0},
}

# One tracker per camera — created lazily, only touched by the worker thread
_trackers: dict[str, _CentroidTracker] = {}


# ── Face recognizer ───────────────────────────────────────────────────────────

# How many inference cycles before re-submitting a track for re-ID.
# The re-ID runs asynchronously in a separate thread — this only controls
# how often we enqueue a new crop, not how long it takes to process.
_REID_CACHE_EVERY = 5

_LOITERING_S = 120.0    # seconds of dwell before loitering alert
_RUNNING_PX_S = 180.0   # pixels/second at imgsz=416 that counts as "running"
_RUNNING_PX_F = _RUNNING_PX_S / 10.0  # px/frame at TARGET_FPS=10 → 18.0


class _PersonReIDMatcher:
    """
    Person re-identification using OpenCLIP's ViT-B/32 visual encoder.

    Takes full-body YOLO bounding box crops — no face detection needed.
    Works on the same person crops that YOLO already produces, so it handles
    the 50–200 px person heights typical of surveillance footage where faces
    are too small for reliable face recognition.

    Embeddings are 512-dim (ViT-B/32 image encoder output), L2-normalised
    for cosine similarity matching.

    Threshold 0.80: CLIP similarity scores for same-person crops from the
    same video cluster near 0.85–0.95; different persons typically fall
    below 0.75.  Adjust down if enrolled persons are missed; up if strangers
    are incorrectly matched.

    DB is loaded from the `registered_faces` table in Neon.  Embeddings are
    stored in the `face_embedding Vector(512)` column and were computed once
    during enrollment — no model call needed at startup for DB loading.
    """

    _THRESHOLD = 0.75   # cosine similarity threshold for a positive match

    def __init__(self) -> None:
        import open_clip
        log.info("Loading CLIP ViT-B/32 for person re-ID…")
        model, _, preprocess = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained="openai"
        )
        model.eval()
        self._model = model            # full model — needed for text encoding (Phase 8)
        self._encoder = model.visual   # image encoder shortcut for re-ID crops
        self._preprocess = preprocess
        self._tokenizer = open_clip.get_tokenizer("ViT-B-32")
        self._db: list[tuple[str, np.ndarray]] = []
        self._lock = threading.Lock()
        self._build_db()

    # ── DB loading ─────────────────────────────────────────────────────────

    @staticmethod
    def _fetch_rows() -> list[tuple[str, np.ndarray]]:
        """
        Synchronously fetch (name, face_embedding) from registered_faces.
        Uses asyncio.run() in a daemon worker thread — WorkerAsyncSessionLocal
        uses NullPool so connections are not bound to the FastAPI event loop.
        """
        import asyncio
        from sqlalchemy import select
        from app.db.models import RegisteredFace
        from app.db.postgres import WorkerAsyncSessionLocal

        async def _q() -> list[tuple[str, np.ndarray]]:
            async with WorkerAsyncSessionLocal() as session:
                result = await session.execute(
                    select(RegisteredFace).where(
                        RegisteredFace.face_embedding.is_not(None)
                    )
                )
                return [
                    (r.name, np.array(r.face_embedding, dtype=np.float32))
                    for r in result.scalars()
                ]

        return asyncio.run(_q())

    def _build_db(self) -> None:
        """Load pre-computed CLIP embeddings from Neon into memory."""
        try:
            rows = self._fetch_rows()
        except Exception as exc:
            log.error("Person DB: could not query registered_faces: %s", exc)
            return

        db: list[tuple[str, np.ndarray]] = []
        for name, emb in rows:
            existing = next((i for i, (n, _) in enumerate(db) if n == name), None)
            if existing is not None:
                avg = (db[existing][1] + emb) / 2.0
                norm = np.linalg.norm(avg)
                if norm > 0:
                    avg /= norm
                db[existing] = (name, avg)
            else:
                db.append((name, emb))
            log.info("Person DB: loaded '%s' from Neon", name)

        with self._lock:
            self._db = db
        log.info("Person DB ready: %d person(s) registered", len(db))

    def reload(self) -> None:
        """Re-query Neon and rebuild in-memory lookup. Thread-safe."""
        log.info("Person DB: reloading from registered_faces…")
        self._build_db()

    # ── Embedding ──────────────────────────────────────────────────────────

    def embed(self, crop: np.ndarray) -> np.ndarray | None:
        """
        Compute a normalised 512-dim CLIP embedding for a BGR person crop.
        Returns None if the crop is too small to process.
        """
        if crop.shape[0] < 20 or crop.shape[1] < 10:
            return None
        from PIL import Image
        img = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        tensor = self._preprocess(img).unsqueeze(0)
        with torch.no_grad():
            feat = self._encoder(tensor)
            feat = feat / feat.norm(dim=-1, keepdim=True)
        return feat.squeeze(0).cpu().numpy()

    def embed_frame(self, frame: np.ndarray) -> np.ndarray | None:
        """Compute a normalised 512-dim CLIP embedding for a full BGR frame (Phase 8)."""
        from PIL import Image
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        tensor = self._preprocess(img).unsqueeze(0)
        with torch.no_grad():
            feat = self._encoder(tensor)
            feat = feat / feat.norm(dim=-1, keepdim=True)
        return feat.squeeze(0).cpu().numpy()

    def encode_text(self, query: str) -> np.ndarray:
        """Encode a text query into a normalised 512-dim CLIP embedding (Phase 8)."""
        tokens = self._tokenizer([query])
        with torch.no_grad():
            feat = self._model.encode_text(tokens)
            feat = feat / feat.norm(dim=-1, keepdim=True)
        return feat.squeeze(0).cpu().numpy()

    # ── Matching ───────────────────────────────────────────────────────────

    def identify(self, crop: np.ndarray) -> tuple[str, str | None, float]:
        """
        Returns (status, name, confidence).
        status: "AUTHORIZED" | "UNKNOWN"
        """
        with self._lock:
            db = list(self._db)
        if not db:
            return "UNKNOWN", None, 0.0

        emb = self.embed(crop)
        if emb is None:
            return "UNKNOWN", None, 0.0

        best_score, best_name = 0.0, None
        for name, db_emb in db:
            score = float(np.dot(emb, db_emb))
            if score > best_score:
                best_score, best_name = score, name

        if best_score >= self._THRESHOLD:
            return "AUTHORIZED", best_name, best_score
        return "UNKNOWN", None, best_score

    @property
    def has_persons(self) -> bool:
        with self._lock:
            return bool(self._db)


_face_recognizer: _PersonReIDMatcher | None = None


# ── Core inference ────────────────────────────────────────────────────────────

def _deduplicate_boxes(boxes: list[list[int]]) -> list[list[int]]:
    """Remove partial-body detections whose centroid falls inside a larger box.

    At conf=0.20, YOLO frequently emits both a full-body box and a torso-only
    or legs-only box for the same person. Their IoU can be below 0.45 so YOLO's
    own NMS doesn't remove them, yet the smaller box's centroid always falls
    inside the larger box. Removing these here prevents the tracker from
    creating a second track for the same person.
    """
    if len(boxes) <= 1:
        return boxes
    # Sort largest area first so we always keep the full-body detection
    boxes_sorted = sorted(boxes, key=lambda b: b[2] * b[3], reverse=True)
    kept: list[list[int]] = []
    for bx, by, bw, bh in boxes_sorted:
        cx = bx + bw / 2.0
        cy = by + bh / 2.0
        suppressed = False
        for kx, ky, kw, kh in kept:
            # Only suppress if the kept box is meaningfully larger (1.4×)
            if (kw * kh) >= (bw * bh) * 1.4:
                if kx <= cx <= kx + kw and ky <= cy <= ky + kh:
                    suppressed = True
                    break
        if not suppressed:
            kept.append([bx, by, bw, bh])
    return kept


def _infer(cam_id: str, frame: np.ndarray, after_hours: bool, frame_idx: int) -> InferResponse:
    if _yolo is None:
        return InferResponse()

    t0 = time.monotonic()
    yolo_results = _yolo(frame, imgsz=416, classes=[0], conf=0.20, iou=0.35, verbose=False)
    ms = (time.monotonic() - t0) * 1000

    fh, fw = frame.shape[:2]
    raw_boxes: list[list[int]] = []

    if yolo_results and yolo_results[0].boxes is not None and len(yolo_results[0].boxes) > 0:
        for box in yolo_results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(fw - 1, x2), min(fh - 1, y2)
            w, h = x2 - x1, y2 - y1
            if w > 0 and h > 0:
                raw_boxes.append([x1, y1, w, h])

    raw_boxes = _deduplicate_boxes(raw_boxes)

    if cam_id not in _trackers:
        _trackers[cam_id] = _CentroidTracker(**_TRACKER_PARAMS.get(cam_id, {}))
    track_ids = _trackers[cam_id].update(raw_boxes, frame_idx, frame_w=fw, frame_h=fh)

    now_mono = time.monotonic()
    tracks: list[Track] = []
    alerts: list[Alert] = []

    for tid, bbox in zip(track_ids, raw_boxes):
        active = _trackers[cam_id]._tracks.get(tid)

        if active is not None:
            active.face_infer_count += 1
            should_reid = (active.face_infer_count == 1 or
                           active.face_infer_count % _REID_CACHE_EVERY == 0)
            if should_reid and _face_recognizer is not None and _face_recognizer.has_persons:
                x, y, w, h = bbox
                crop = frame[y: y + h, x: x + w]
                if (crop.size > 0
                        and min(w, h) >= settings.REID_MIN_CROP_PX
                        and (h == 0 or w / h <= settings.REID_MAX_ASPECT_RATIO)):
                    _submit_reid(cam_id, tid, crop, is_first=active.face_infer_count == 1)
            status = active.face_status
            name = active.face_name
            conf = active.face_confidence
        else:
            status, name, conf = "UNKNOWN", None, 0.0

        behavior: str | None = None
        if active is not None:
            duration = now_mono - active.first_seen_time
            if active.speed_px_f > _RUNNING_PX_F:
                behavior = "running"
            elif duration > _LOITERING_S:
                behavior = "loitering"

        tracks.append(Track(
            track_id=tid,
            bbox=bbox,
            status=status,
            person_name=name,
            confidence=conf,
            behavior=behavior,
            vx=active.vx_f if active else 0.0,
            vy=active.vy_f if active else 0.0,
        ))

        if behavior == "loitering" and active is not None:
            if now_mono - active.last_alert_times.get("LOITERING", 0) >= settings.ALERT_COOLDOWN_LOITERING:
                duration = now_mono - active.first_seen_time
                alerts.append(Alert(
                    type="LOITERING",
                    severity="MEDIUM",
                    track_id=tid,
                    message=f"Person #{tid} loitering for {int(duration)}s on {cam_id}",
                ))
                active.last_alert_times["LOITERING"] = now_mono

        if behavior == "running" and active is not None:
            if now_mono - active.last_alert_times.get("RUNNING", 0) >= settings.ALERT_COOLDOWN_RUNNING:
                alerts.append(Alert(
                    type="RUNNING",
                    severity="LOW",
                    track_id=tid,
                    message=f"Person #{tid} running on {cam_id}",
                ))
                active.last_alert_times["RUNNING"] = now_mono

        if after_hours and active is not None:
            if status == "UNKNOWN":
                if now_mono - active.last_alert_times.get("UNAUTHORIZED_VISITOR", 0) >= settings.ALERT_COOLDOWN_UNAUTHORIZED:
                    alerts.append(Alert(
                        type="UNAUTHORIZED_VISITOR",
                        severity="HIGH",
                        track_id=tid,
                        message=f"Unknown person #{tid} detected after hours on {cam_id}",
                    ))
                    active.last_alert_times["UNAUTHORIZED_VISITOR"] = now_mono
            else:
                if now_mono - active.last_alert_times.get("AFTER_HOURS_PRESENCE", 0) >= settings.ALERT_COOLDOWN_AFTER_HOURS:
                    alerts.append(Alert(
                        type="AFTER_HOURS_PRESENCE",
                        severity="LOW",
                        track_id=tid,
                        message=f"{name} (#{tid}) present after hours on {cam_id}",
                    ))
                    active.last_alert_times["AFTER_HOURS_PRESENCE"] = now_mono

    log.info("infer %s: %d tracks, %d alerts (%.0f ms)", cam_id, len(tracks), len(alerts), ms)
    return InferResponse(frame_idx=frame_idx, tracks=tracks, alerts=alerts)


# ── Re-ID helpers ─────────────────────────────────────────────────────────────

def _submit_reid(cam_id: str, tid: int, crop: np.ndarray, is_first: bool = False) -> None:
    """
    Non-blocking. One pending slot per (camera, track).
    First-seen crops go to the priority queue so new persons are identified
    before periodic refresh crops from established tracks.
    """
    with _reid_lock:
        key = (cam_id, tid)
        if is_first:
            _reid_priority[key] = crop.copy()
        else:
            _reid_refresh[key] = crop.copy()


def _reid_worker_loop() -> None:
    """
    Dedicated daemon thread for CLIP re-ID.
    Drains _reid_priority before _reid_refresh so new persons are identified
    within ~1 YOLO cycle even when the refresh queue has a backlog.
    Never blocks the YOLO worker thread.
    """
    while True:
        with _reid_lock:
            if _reid_priority:
                (cam_id, tid), crop = next(iter(_reid_priority.items()))
                del _reid_priority[(cam_id, tid)]
                item = (cam_id, crop, tid)
            elif _reid_refresh:
                (cam_id, tid), crop = next(iter(_reid_refresh.items()))
                del _reid_refresh[(cam_id, tid)]
                item = (cam_id, crop, tid)
            else:
                item = None
        if item is None:
            time.sleep(0.005)
            continue
        cam_id, crop, tid = item
        if _face_recognizer is None:
            continue
        try:
            status, name, conf = _face_recognizer.identify(crop)
        except Exception as exc:
            log.debug("re-ID error cam=%s tid=%d: %s", cam_id, tid, exc)
            continue
        # Write back to tracker — GIL protects individual attribute writes
        tracker = _trackers.get(cam_id)
        if tracker:
            track = tracker._tracks.get(tid)
            if track is not None:
                track.face_status = status
                track.face_name = name
                track.face_confidence = conf


# ── Worker thread ─────────────────────────────────────────────────────────────

def _worker_loop() -> None:
    while True:
        item = _pop_next()
        if item is None:
            time.sleep(0.005)
            continue
        cam_id, frame, after_hours, frame_idx = item
        try:
            result = _infer(cam_id, frame, after_hours, frame_idx)
        except Exception as exc:
            log.error("Inference error cam=%s frame=%d: %s", cam_id, frame_idx, exc, exc_info=True)
            continue
        with _results_lock:
            _results[cam_id] = (result, frame_idx)
        if result.alerts:
            ts = datetime.now(timezone.utc).isoformat()
            with _alert_history_lock:
                for alert in result.alerts:
                    _alert_history.append({
                        "timestamp": ts,
                        "cam_id": cam_id,
                        "type": alert.type,
                        "severity": alert.severity,
                        "track_id": alert.track_id,
                        "message": alert.message,
                    })


def _pop_next() -> tuple[str, np.ndarray, bool, int] | None:
    with _pending_lock:
        if not _pending:
            return None
        cam_id, (frame, after_hours, frame_idx) = next(iter(_pending.items()))
        del _pending[cam_id]
        return cam_id, frame, after_hours, frame_idx


# ── Public API ────────────────────────────────────────────────────────────────

def submit_frame(cam_id: str, frame: np.ndarray,
                 after_hours: bool, frame_idx: int) -> None:
    """Overwrite pending slot — worker always gets the latest frame."""
    with _pending_lock:
        _pending[cam_id] = (frame, after_hours, frame_idx)


def get_latest_result(cam_id: str) -> tuple[InferResponse, int] | None:
    """Non-blocking: return latest (result, frame_idx) or None."""
    with _results_lock:
        return _results.get(cam_id)


def reload_faces() -> int:
    """Re-query registered_faces from DB and rebuild embeddings. Returns new count."""
    if _face_recognizer is None:
        return 0
    _face_recognizer.reload()
    return len(_face_recognizer._db)


# ── Startup ───────────────────────────────────────────────────────────────────

def load_models() -> None:
    global _yolo, _face_recognizer

    # 4 threads measured optimal on i5-8250U — leaving 4 cores for OS + asyncio
    torch.set_num_threads(4)
    log.info("PyTorch using 4 threads for YOLO")

    log.info("Loading YOLOv8n...")
    _yolo = YOLO("yolov8n.pt")
    _yolo.fuse()
    # Warm up so first real call is not slow
    _yolo(np.zeros((416, 416, 3), dtype=np.uint8), imgsz=416, verbose=False)
    log.info("YOLOv8n ready")

    log.info("Loading CLIP ViT-B/32 (person re-ID) + person DB from Neon…")
    try:
        _face_recognizer = _PersonReIDMatcher()
        log.info("Person re-ID ready (%d person(s) in DB)", len(_face_recognizer._db))
    except Exception as exc:
        log.error("CLIP re-ID failed to load: %s — running without person recognition", exc)
        _face_recognizer = None

    for i in range(_NUM_WORKERS):
        t = threading.Thread(target=_worker_loop, daemon=True, name=f"yolo-{i}")
        t.start()
    log.info("Started %d inference worker(s)", _NUM_WORKERS)

    reid_thread = threading.Thread(target=_reid_worker_loop, daemon=True, name="reid-0")
    reid_thread.start()
    log.info("Started re-ID worker thread (CLIP decoupled from YOLO)")

    embed_thread = threading.Thread(target=_frame_embed_worker_loop, daemon=True, name="embed-0")
    embed_thread.start()
    log.info("Started frame embed worker thread (Phase 8 CLIP search)")


def submit_frame_for_embedding(cam_id: str, frame: np.ndarray, timestamp: str) -> None:
    """Queue a frame for CLIP embedding (Phase 8). Overwrite semantics — no backlog."""
    with _frame_embed_lock:
        _frame_embed_slots[cam_id] = (timestamp, frame)
    _frame_embed_event.set()


def encode_text_query(query: str) -> np.ndarray | None:
    """Encode a NL query with the CLIP text encoder. Returns None if model not ready."""
    if _face_recognizer is None:
        return None
    return _face_recognizer.encode_text(query)


def _frame_embed_worker_loop() -> None:
    """
    Daemon thread: drains _frame_embed_slots, runs CLIP image encoder on each frame,
    saves JPEG to disk, writes ClipEmbedding row to Postgres.
    Runs independently from YOLO and re-ID workers.
    """
    import asyncio
    from pathlib import Path
    from app.db.models import ClipEmbedding
    from app.db.postgres import WorkerAsyncSessionLocal

    async def _write(cam_id: str, ts_naive: datetime, frame_path: str, emb: list[float]) -> None:
        async with WorkerAsyncSessionLocal() as session:
            async with session.begin():
                session.add(ClipEmbedding(
                    camera_id=cam_id,
                    timestamp=ts_naive,
                    frame_path=frame_path,
                    embedding=emb,
                ))

    while True:
        _frame_embed_event.wait(timeout=5.0)
        _frame_embed_event.clear()

        with _frame_embed_lock:
            slots = list(_frame_embed_slots.items())
            _frame_embed_slots.clear()

        for cam_id, (timestamp_str, frame) in slots:
            if _face_recognizer is None:
                continue
            emb = _face_recognizer.embed_frame(frame)
            if emb is None:
                continue

            # Save frame JPEG
            frame_dir = Path(settings.FRAME_STORE) / cam_id
            frame_dir.mkdir(parents=True, exist_ok=True)
            safe_ts = timestamp_str.replace(":", "-").replace(".", "-").replace("+", "")
            frame_path = frame_dir / f"{safe_ts}.jpg"
            cv2.imwrite(str(frame_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 75])

            # Write embedding to DB
            try:
                ts = datetime.fromisoformat(timestamp_str).replace(tzinfo=None)
            except ValueError:
                ts = datetime.now(timezone.utc).replace(tzinfo=None)
            try:
                asyncio.run(_write(cam_id, ts, str(frame_path), emb.tolist()))
            except Exception as exc:
                log.warning("Frame embed DB write failed for %s: %s", cam_id, exc)


def models_ready() -> bool:
    return _yolo is not None


def get_analytics_summary() -> dict:
    """
    Aggregate live track data across all cameras for the analytics dashboard.
    Returns visitor_count, authorized/unknown breakdown, and per-camera centroid list.
    Thread-safe: reads from _results under lock.
    """
    visitor_count = 0
    authorized = 0
    unknown = 0
    camera_tracks: dict[str, list[dict]] = {}

    with _results_lock:
        snapshot = list(_results.items())

    for cam_id, (result, _) in snapshot:
        cam_centroids = []
        for t in result.tracks:
            visitor_count += 1
            if t.status == "AUTHORIZED":
                authorized += 1
            else:
                unknown += 1
            x, y, w, h = t.bbox
            cam_centroids.append({
                "track_id": t.track_id,
                "cx": x + w // 2,
                "cy": y + h // 2,
                "status": t.status,
            })
        camera_tracks[cam_id] = cam_centroids

    return {
        "visitor_count": visitor_count,
        "authorized": authorized,
        "unknown": unknown,
        "cameras_active": len(snapshot),
        "camera_tracks": camera_tracks,
    }


def get_recent_incidents(limit: int = 20) -> list[dict]:
    """Return the most recent `limit` alerts from the rolling history buffer."""
    with _alert_history_lock:
        items = list(_alert_history)
    return items[-limit:][::-1]  # most recent first
