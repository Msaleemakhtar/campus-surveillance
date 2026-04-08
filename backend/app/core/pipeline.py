"""
Phase 3: Linear extrapolation for smooth box motion.

One asyncio.Task per camera. Frames are submitted to the shared YOLO
worker thread (local_inference.py). Results are consumed on every render
frame via a non-blocking dict lookup.

Between inference updates (~3 s apart on YOLO-only path), each track's
bounding box is extrapolated forward using its last measured velocity so
the box glides with the person instead of jumping on each new result.

Later phases add:
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
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

from app.core import annotator, local_inference
from app.core.config import settings
from app.db.models import Incident
from app.db.postgres import AsyncSessionLocal
from app.db.redis_client import get_redis
from app.models.schemas import Alert, InferResponse, Track, WSMessage

log = logging.getLogger(__name__)

# Max frames to project forward from the inference centroid.
# Beyond this the box freezes (stops moving) rather than drifting off.
# 20 frames = 2s — covers the 8-camera YOLO cycle (8 × 291ms ≈ 2.3s).
_MAX_EXTRAP_DF = 20

# Frames without a result before a track is removed from the display.
# 25 frames = 2.5s — just over one full 8-camera cycle (2.3s) with margin.
_STALE_FRAMES = 25

# Active prune threshold: ghost tracks absent from a new result for longer
# than this are immediately removed in _update_extrap.  Must be < _STALE_FRAMES.
_ACTIVE_PRUNE_FRAMES = _STALE_FRAMES // 2

# Velocity cap (±20 px/frame) is enforced in local_inference._CentroidTracker.update()


@dataclass
class _TrackState:
    cx: float        # centroid x at last inference (px)
    cy: float        # centroid y at last inference (px)
    w: int           # bbox width
    h: int           # bbox height
    vx_f: float      # horizontal velocity (px/frame)
    vy_f: float      # vertical velocity   (px/frame)
    updated_at_frame: int  # frame_idx of the inference result
    last_seen_frame: int   # frame_idx of the latest result update
    # Face recognition fields
    status: str = "UNKNOWN"
    person_name: str | None = None
    confidence: float = 0.0
    behavior: str | None = None


# ── Camera → video file mapping ───────────────────────────────────────────────

_VIDEOS_BASE = Path(settings.VIDEOS_BASE)

_CAM_SINGLE: dict[str, Path] = {
    "cam01": _VIDEOS_BASE / "normal" / "cam01.mp4",
    "cam02": _VIDEOS_BASE / "normal" / "cam02.mp4",
    "cam03": _VIDEOS_BASE / "normal" / "cam03.mp4",
    "cam04": _VIDEOS_BASE / "normal" / "cam04.mp4",
    "cam11": _VIDEOS_BASE / "outdoor" / "cam11.avi",
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
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok or buf is None:
        raise RuntimeError("JPEG encode failed")
    return base64.b64encode(buf.tobytes()).decode()


# ── Camera pipeline ────────────────────────────────────────────────────────────

class CameraPipeline:
    def __init__(self, cam_id: str, startup_delay: float = 0.0) -> None:
        self.cam_id = cam_id
        self._startup_delay = startup_delay
        self._task: asyncio.Task | None = None
        self._last_result: InferResponse = InferResponse()
        self._last_result_time: float = 0.0
        self._running = False
        self._frame_offset = random.randint(0, max(1, settings.INFER_EVERY_N_FRAMES - 1))
        # Phase 3: per-track extrapolation state
        self._extrap: dict[int, _TrackState] = {}

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
        if self._startup_delay:
            await asyncio.sleep(self._startup_delay)
        while self._running:
            try:
                if self.cam_id in _CAM_SINGLE:
                    await self._run_single(_CAM_SINGLE[self.cam_id])
                elif self.cam_id in _CAM_PLAYLIST:
                    await self._run_playlist(_CAM_PLAYLIST[self.cam_id])
                else:
                    log.error("Unknown cam_id: %s", self.cam_id)
                    break
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                log.error("Pipeline %s crashed: %s — restarting in 2s", self.cam_id, exc)
                await asyncio.sleep(2.0)

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
                # Skip decode entirely when nobody is watching — H.264 decode
                # is the main CPU cost even without YOLO.
                if not local_inference.has_subscribers(self.cam_id):
                    await asyncio.sleep(0.5)
                    continue
                t0 = asyncio.get_event_loop().time()
                ok, frame = await asyncio.to_thread(cap.read)
                if not ok or frame is None:
                    await asyncio.to_thread(cap.set, cv2.CAP_PROP_POS_FRAMES, 0)
                    await asyncio.sleep(0.1)
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
            if not local_inference.has_subscribers(self.cam_id):
                await asyncio.sleep(0.5)
                continue
            clip = next(playlist)
            cap: cv2.VideoCapture | None = None
            try:
                cap = await asyncio.to_thread(cv2.VideoCapture, str(clip))
                if not cap.isOpened():
                    continue
                while self._running:
                    if not local_inference.has_subscribers(self.cam_id):
                        await asyncio.sleep(0.5)
                        break  # back to outer loop so we re-check before opening next clip
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
        # ── 0. Skip all work when nobody is watching ──────────────────────────
        if not local_inference.has_subscribers(self.cam_id):
            return

        timestamp = datetime.now(timezone.utc).isoformat()
        after_hours = _is_after_hours()

        # ── 1. Submit frame for inference ─────────────────────────────────────
        if (frame_count + self._frame_offset) % settings.INFER_EVERY_N_FRAMES == 0:
            local_inference.submit_frame(self.cam_id, frame.copy(), after_hours, frame_count)

        # ── 2. Consume latest result and update extrapolation state ───────────
        result_data = local_inference.get_latest_result(self.cam_id)
        if result_data is not None:
            result, result_frame_idx = result_data
            if result_frame_idx > self._last_result_time:
                self._update_extrap(result, result_frame_idx, frame_count)
                self._last_result = result
                self._last_result_time = float(result_frame_idx)
                await self._persist_alerts(result.alerts)

        # ── 3. Render — extrapolate each track to the current moment ──────────
        extrap_tracks = self._extrapolate(frame.shape, frame_count)
        annotated = annotator.annotate(frame, extrap_tracks)

        # ── 4. Encode + publish ───────────────────────────────────────────────
        final_b64 = await asyncio.to_thread(
            _encode_frame, annotated, settings.JPEG_QUALITY
        )

        msg = WSMessage(
            cam_id=self.cam_id,
            timestamp=timestamp,
            frame_idx=frame_count,
            frame=final_b64,
            tracks=extrap_tracks,
            alerts=self._last_result.alerts,
            zone_counts={},
            heatmap=None,
        )
        await self._publish(msg)

    def _update_extrap(self, result: InferResponse, result_frame_idx: int, current_frame_idx: int) -> None:
        """Update per-track extrapolation state using frame-based velocity."""
        live_tids = {t.track_id for t in result.tracks}

        # Actively prune ghost tracks that dropped out of this result and are
        # stale (not seen for > _STALE_FRAMES / 2 frames).  Without this, a
        # person who leaves and re-enters with a new track ID would show both
        # the old ghost box and the new real box for up to _STALE_FRAMES.
        stale_limit = _ACTIVE_PRUNE_FRAMES
        ghosts = [
            tid for tid, s in self._extrap.items()
            if tid not in live_tids and current_frame_idx - s.last_seen_frame > stale_limit
        ]
        for tid in ghosts:
            del self._extrap[tid]

        for track in result.tracks:
            tid = track.track_id
            x, y, w, h = track.bbox
            cx = x + w / 2.0
            cy = y + h / 2.0

            # Use px/frame velocity directly from the inference result
            self._extrap[tid] = _TrackState(
                cx=cx, cy=cy, w=w, h=h,
                vx_f=track.vx, vy_f=track.vy,
                updated_at_frame=result_frame_idx,
                last_seen_frame=current_frame_idx,
                status=track.status,
                person_name=track.person_name,
                confidence=track.confidence,
                behavior=track.behavior,
            )

    def _extrapolate(self, shape: tuple, current_frame_idx: int) -> list[Track]:
        """Return track list with bboxes projected forward to current_frame_idx."""
        fh, fw = shape[:2]
        tracks: list[Track] = []
        expired_ids: list[int] = []

        for tid, s in self._extrap.items():
            # Track not seen in a result for > _STALE_FRAMES → truly gone
            if current_frame_idx - s.last_seen_frame > _STALE_FRAMES:
                expired_ids.append(tid)
                continue

            # Extrapolate from the frame where the result was captured.
            # Velocity decays linearly from vx_f at df=0 to zero at df=_MAX_EXTRAP_DF
            # using the exact integral: displacement = vx * df * (1 - df/2M).
            # This prevents boxes floating when a person stops or exits between
            # YOLO updates — max displacement = vx * M/2 instead of vx * M.
            df = min(current_frame_idx - s.updated_at_frame, _MAX_EXTRAP_DF)
            decay_scale = 1.0 - df / (2.0 * _MAX_EXTRAP_DF)

            ecx = s.cx + s.vx_f * df * decay_scale
            ecy = s.cy + s.vy_f * df * decay_scale

            # If centroid has drifted outside the frame, the person has exited — expire now
            if ecx < 0 or ecx > fw or ecy < 0 or ecy > fh:
                expired_ids.append(tid)
                continue

            x = int(ecx - s.w / 2)
            y = int(ecy - s.h / 2)

            # Clamp to frame bounds
            x = max(0, min(x, fw - s.w))
            y = max(0, min(y, fh - s.h))

            tracks.append(Track(
                track_id=tid,
                bbox=[x, y, s.w, s.h],
                status=s.status,
                person_name=s.person_name,
                confidence=s.confidence,
                behavior=s.behavior,
                vx=s.vx_f,
                vy=s.vy_f,
            ))

        for tid in expired_ids:
            del self._extrap[tid]

        return tracks

    async def _persist_alerts(self, alerts: list[Alert]) -> None:
        """Write fired alerts to the incidents table. One row per alert."""
        if not alerts:
            return
        now = datetime.now(timezone.utc).replace(tzinfo=None)
        try:
            async with AsyncSessionLocal() as session:
                async with session.begin():
                    for alert in alerts:
                        session.add(Incident(
                            camera_id=self.cam_id,
                            incident_type=alert.type,
                            severity=alert.severity,
                            track_id=alert.track_id,
                            timestamp=now,
                        ))
            log.debug("Persisted %d alert(s) for %s", len(alerts), self.cam_id)
        except Exception as exc:
            log.warning("Failed to persist alerts for %s: %s", self.cam_id, exc)

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
    for i, cam_id in enumerate(cameras):
        p = CameraPipeline(cam_id, startup_delay=i * 0.3)
        p.start()
        _pipelines[cam_id] = p
        log.info("Started pipeline: %s", cam_id)


async def stop_pipelines() -> None:
    await asyncio.gather(*[p.stop() for p in _pipelines.values()])
    _pipelines.clear()
