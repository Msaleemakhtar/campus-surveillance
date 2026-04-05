"""
Phase 1: Raw YOLO detection only.

One dedicated daemon thread runs YOLO in a tight loop.
Each camera submits its latest frame to a per-camera slot (overwrite
semantics — worker always sees the freshest frame, no queue backlog).

Later phases add:
  Phase 2 — ByteTrack (stable track IDs)
  Phase 3 — TrackExtrapolator (smooth motion between results)
  Phase 4 — Subscriber registry (demand-driven inference)
  Phase 5 — InsightFace face recognition + alerts

Hardware reality on i5-8250U:
  YOLO imgsz=416: avg 291ms (faster than 320 on this CPU)
  10 cameras × 291ms = 2.91s cycle per camera
  → accept ~1 Hz per camera, design everything around that
"""
from __future__ import annotations

import logging
import threading
import time

import numpy as np
import torch
from ultralytics import YOLO

from app.models.schemas import InferResponse, Track

log = logging.getLogger(__name__)

# ── Model singleton ────────────────────────────────────────────────────────────

_yolo: YOLO | None = None

# ── Pending / result slots (per camera) ───────────────────────────────────────

_pending: dict[str, tuple[np.ndarray, bool, float]] = {}
_pending_lock = threading.Lock()

_results: dict[str, tuple[InferResponse, float]] = {}
_results_lock = threading.Lock()

_NUM_WORKERS = 1   # serial is faster than concurrent on CPU


def submit_frame(cam_id: str, frame: np.ndarray,
                 after_hours: bool, capture_time: float) -> None:
    """Overwrite pending slot — worker always gets the latest frame."""
    with _pending_lock:
        _pending[cam_id] = (frame, after_hours, capture_time)


def get_latest_result(cam_id: str) -> tuple[InferResponse, float] | None:
    """Non-blocking: return latest (result, capture_time) or None."""
    with _results_lock:
        return _results.get(cam_id)


def _pop_next() -> tuple[str, np.ndarray, bool, float] | None:
    with _pending_lock:
        if not _pending:
            return None
        cam_id, (frame, after_hours, capture_time) = next(iter(_pending.items()))
        del _pending[cam_id]
        return cam_id, frame, after_hours, capture_time


# ── Core inference ─────────────────────────────────────────────────────────────

def _infer(cam_id: str, frame: np.ndarray) -> InferResponse:
    if _yolo is None:
        return InferResponse()

    t0 = time.monotonic()
    # imgsz=416 is measurably faster than 320 on this CPU (291ms vs 352ms avg)
    results = _yolo(frame, imgsz=416, classes=[0], conf=0.20, verbose=False)
    ms = (time.monotonic() - t0) * 1000

    if not results or results[0].boxes is None or len(results[0].boxes) == 0:
        log.debug("infer %s: 0 detections (%.0f ms)", cam_id, ms)
        return InferResponse()

    fh, fw = frame.shape[:2]
    tracks: list[Track] = []

    for i, box in enumerate(results[0].boxes):
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(fw - 1, x2), min(fh - 1, y2)
        w, h = x2 - x1, y2 - y1
        if w <= 0 or h <= 0:
            continue
        tracks.append(Track(
            track_id=i,       # raw index — stable IDs come in Phase 2 (ByteTrack)
            bbox=[x1, y1, w, h],
            status="UNKNOWN",
        ))

    log.info("infer %s: %d detections (%.0f ms)", cam_id, len(tracks), ms)
    return InferResponse(tracks=tracks)


# ── Worker thread ──────────────────────────────────────────────────────────────

def _worker_loop() -> None:
    while True:
        item = _pop_next()
        if item is None:
            time.sleep(0.005)
            continue
        cam_id, frame, after_hours, capture_time = item
        result = _infer(cam_id, frame)
        with _results_lock:
            _results[cam_id] = (result, capture_time)


# ── Startup ────────────────────────────────────────────────────────────────────

def load_models() -> None:
    global _yolo
    # 4 threads measured optimal on i5-8250U — leaving 4 cores for OS + asyncio
    torch.set_num_threads(4)
    log.info("PyTorch using 4 threads for YOLO")

    log.info("Loading YOLOv8n...")
    _yolo = YOLO("yolov8n.pt")
    _yolo.fuse()
    # Warm up so first real call is not slow
    _yolo(np.zeros((416, 416, 3), dtype=np.uint8), imgsz=416, verbose=False)
    log.info("YOLOv8n ready")

    for i in range(_NUM_WORKERS):
        t = threading.Thread(target=_worker_loop, daemon=True, name=f"yolo-{i}")
        t.start()
    log.info("Started %d inference worker(s)", _NUM_WORKERS)


def models_ready() -> bool:
    return _yolo is not None
