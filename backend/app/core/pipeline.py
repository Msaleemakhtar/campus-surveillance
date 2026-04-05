"""
Phase 1: Camera pipeline — video read → YOLO → annotate → stream.

One asyncio.Task per camera. Frames are submitted to the shared YOLO
worker thread (local_inference.py). Results are consumed on every render
frame via a non-blocking dict lookup. Last known boxes are drawn on every
frame — no extrapolation yet (Phase 3).

Later phases add:
  Phase 2 — stable track IDs via ByteTrack (in local_inference.py)
  Phase 3 — linear extrapolation for smooth motion
  Phase 4 — demand-driven inference (subscriber registry)
  Phase 5 — face recognition labels + alerts
  Phase 6 — DB writes
"""
from __future__ import annotations

import asyncio
import base64
import itertools
import logging
import random
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

from app.core import annotator, local_inference
from app.core.config import settings
from app.db.redis_client import get_redis
from app.models.schemas import InferResponse, Track, WSMessage

log = logging.getLogger(__name__)

# ── Camera → video file mapping ───────────────────────────────────────────────

_VIDEOS_BASE = Path(settings.VIDEOS_BASE)

_CAM_SINGLE: dict[str, Path] = {
    "cam01": _VIDEOS_BASE / "normal" / "cam01.mp4",
    "cam02": _VIDEOS_BASE / "normal" / "cam02.mp4",
    "cam03": _VIDEOS_BASE / "normal" / "cam03.mp4",
    "cam04": _VIDEOS_BASE / "normal" / "cam04.mp4",
    "cam05": _VIDEOS_BASE / "normal" / "cam05.mp4",
    "cam11": _VIDEOS_BASE / "outdoor" / "cam11.avi",
    "cam12": _VIDEOS_BASE / "outdoor" / "cam12.avi",
    "cam15": _VIDEOS_BASE / "normal" / "cam15.mp4",
}

_CAM_PLAYLIST: dict[str, list[Path]] = {
    "cam19": (
        sorted((_VIDEOS_BASE / "anomaly" / "Fighting").glob("*.mp4"))
        + sorted((_VIDEOS_BASE / "anomaly" / "Assault").glob("*.mp4"))
        + sorted((_VIDEOS_BASE / "anomaly" / "Shoplifting").glob("*.mp4"))
    ),
    "cam20": (
        sorted((_VIDEOS_BASE / "anomaly" / "Vandalism").glob("*.mp4"))
        + sorted((_VIDEOS_BASE / "anomaly" / "Burglary").glob("*.mp4"))
        + sorted((_VIDEOS_BASE / "anomaly" / "Stealing").glob("*.mp4"))
    ),
}


def _is_after_hours() -> bool:
    hour = datetime.now(timezone.utc).hour
    return hour >= settings.AFTER_HOURS_START or hour < settings.AFTER_HOURS_END


def _encode_frame(frame: np.ndarray, quality: int) -> str:
    """JPEG-encode and base64. Runs in thread pool."""
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return base64.b64encode(buf.tobytes()).decode()


# ── Camera pipeline ────────────────────────────────────────────────────────────

class CameraPipeline:
    def __init__(self, cam_id: str) -> None:
        self.cam_id = cam_id
        self._task: asyncio.Task | None = None
        self._last_result: InferResponse = InferResponse()
        self._last_result_time: float = 0.0
        self._running = False
        self._frame_offset = random.randint(0, max(1, settings.INFER_EVERY_N_FRAMES - 1))

    def start(self) -> None:
        self._running = True
        self._task = asyncio.create_task(self._run(), name=f"pipeline-{self.cam_id}")

    async def stop(self) -> None:
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    async def _run(self) -> None:
        if self.cam_id in _CAM_SINGLE:
            await self._run_single(_CAM_SINGLE[self.cam_id])
        elif self.cam_id in _CAM_PLAYLIST:
            await self._run_playlist(_CAM_PLAYLIST[self.cam_id])
        else:
            log.error("Unknown cam_id: %s", self.cam_id)

    # ── Video loops ────────────────────────────────────────────────────────────

    async def _run_single(self, video_path: Path) -> None:
        cap: cv2.VideoCapture | None = None
        try:
            cap = await asyncio.to_thread(cv2.VideoCapture, str(video_path))
            if not cap.isOpened():
                log.error("Cannot open %s", video_path)
                return
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total > 1:
                await asyncio.to_thread(
                    cap.set, cv2.CAP_PROP_POS_FRAMES, random.randint(0, total - 1)
                )
            frame_count = 0
            frame_interval = 1.0 / settings.TARGET_FPS
            while self._running:
                t0 = asyncio.get_event_loop().time()
                ok, frame = await asyncio.to_thread(cap.read)
                if not ok or frame is None:
                    await asyncio.to_thread(cap.set, cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                await self._process_frame(frame, frame_count)
                frame_count += 1
                sleep = max(0.0, frame_interval - (asyncio.get_event_loop().time() - t0))
                if sleep:
                    await asyncio.sleep(sleep)
        finally:
            if cap:
                await asyncio.to_thread(cap.release)

    async def _run_playlist(self, clips: list[Path]) -> None:
        if not clips:
            log.error("No clips for %s", self.cam_id)
            return
        playlist = itertools.cycle(clips)
        frame_count = 0
        frame_interval = 1.0 / settings.TARGET_FPS
        while self._running:
            clip = next(playlist)
            cap: cv2.VideoCapture | None = None
            try:
                cap = await asyncio.to_thread(cv2.VideoCapture, str(clip))
                if not cap.isOpened():
                    continue
                while self._running:
                    t0 = asyncio.get_event_loop().time()
                    ok, frame = await asyncio.to_thread(cap.read)
                    if not ok or frame is None:
                        break
                    await self._process_frame(frame, frame_count)
                    frame_count += 1
                    sleep = max(0.0, frame_interval - (asyncio.get_event_loop().time() - t0))
                    if sleep:
                        await asyncio.sleep(sleep)
            finally:
                if cap:
                    await asyncio.to_thread(cap.release)

    # ── Core per-frame logic ───────────────────────────────────────────────────

    async def _process_frame(self, frame: np.ndarray, frame_count: int) -> None:
        now_mono = time.monotonic()
        timestamp = datetime.now(timezone.utc).isoformat()
        after_hours = _is_after_hours()

        # ── 1. Submit frame for inference (fire-and-forget, overwrite semantics) ─
        if (frame_count + self._frame_offset) % settings.INFER_EVERY_N_FRAMES == 0:
            local_inference.submit_frame(self.cam_id, frame.copy(), after_hours, now_mono)

        # ── 2. Consume latest result (non-blocking) ───────────────────────────
        result_data = local_inference.get_latest_result(self.cam_id)
        if result_data is not None:
            result, capture_time = result_data
            if capture_time > self._last_result_time:
                self._last_result = result
                self._last_result_time = capture_time

        # ── 3. Render — draw last known boxes on current frame ────────────────
        display = frame.copy()
        if after_hours:
            display = cv2.convertScaleAbs(display, alpha=0.3, beta=0)
        annotated = annotator.annotate(display, self._last_result.tracks)

        # ── 4. Encode + publish ───────────────────────────────────────────────
        final_b64 = await asyncio.to_thread(
            _encode_frame, annotated, settings.JPEG_QUALITY
        )

        msg = WSMessage(
            cam_id=self.cam_id,
            timestamp=timestamp,
            frame=final_b64,
            tracks=self._last_result.tracks,
            alerts=self._last_result.alerts,
            zone_counts={},
            heatmap=None,
        )
        await self._publish(msg)

    async def _publish(self, msg: WSMessage) -> None:
        try:
            redis = await get_redis()
            await redis.publish(f"cam:{self.cam_id}", msg.model_dump_json())
        except Exception as exc:
            log.debug("Redis publish error %s: %s", self.cam_id, exc)


# ── Registry ───────────────────────────────────────────────────────────────────

ALL_CAMERAS = list(_CAM_SINGLE.keys()) + list(_CAM_PLAYLIST.keys())
_pipelines: dict[str, CameraPipeline] = {}


def start_pipelines(cameras: list[str] = ALL_CAMERAS) -> None:
    for cam_id in cameras:
        p = CameraPipeline(cam_id)
        p.start()
        _pipelines[cam_id] = p
        log.info("Started pipeline: %s", cam_id)


async def stop_pipelines() -> None:
    await asyncio.gather(*[p.stop() for p in _pipelines.values()])
    _pipelines.clear()
