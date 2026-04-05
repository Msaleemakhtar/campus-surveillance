"""
heatmap.py —
verlay.

Each CameraPipeline calls update() per frame with track center points.
Every 5 minutes the accumulator resets so historical data doesn't dominate.
get_overlay() returns a base64 PNG alpha-blended on top of the camera frame.
"""

from __future__ import annotations

import base64
import time
from collections import defaultdict

import cv2
import numpy as np

_RESET_INTERVAL = 300  # seconds — reset heatmap every 5 minutes
_FRAME_W = 640
_FRAME_H = 480
_BLUR_KERNEL = 31  # Gaussian blur radius — larger = softer heat blobs
_ALPHA = 0.45  # overlay transparency


class HeatmapGenerator:
    """Per-camera heatmap accumulator."""

    def __init__(self, width: int = _FRAME_W, height: int = _FRAME_H) -> None:
        self.width = width
        self.height = height
        self._accum = np.zeros((height, width), dtype=np.float32)
        self._last_reset = time.monotonic()

    def update(self, centers: list[tuple[int, int]]) -> None:
        """Add foot-traffic center points to the accumulator."""
        now = time.monotonic()
        if now - self._last_reset >= _RESET_INTERVAL:
            self._accum[:] = 0
            self._last_reset = now

        for cx, cy in centers:
            cx = max(0, min(cx, self.width - 1))
            cy = max(0, min(cy, self.height - 1))
            self._accum[cy, cx] += 1.0

    def get_overlay(self, frame: np.ndarray) -> np.ndarray:
        """Return frame with JET heatmap blended in. Input: BGR frame."""
        if self._accum.max() == 0:
            return frame

        # Normalize → 0–255 uint8
        norm = (self._accum / self._accum.max() * 255).astype(np.uint8)

        # Gaussian blur for smooth blobs
        blurred = cv2.GaussianBlur(norm, (_BLUR_KERNEL, _BLUR_KERNEL), 0)

        # Apply JET colormap
        colored = cv2.applyColorMap(blurred, cv2.COLORMAP_JET)

        # Resize to match frame if needed
        if colored.shape[:2] != frame.shape[:2]:
            colored = cv2.resize(colored, (frame.shape[1], frame.shape[0]))

        # Alpha blend over frame
        result = cv2.addWeighted(frame, 1.0 - _ALPHA, colored, _ALPHA, 0)
        return result

    def get_overlay_b64(self, frame: np.ndarray) -> str | None:
        """Return base64 JPEG of heatmap overlay, or None if no data."""
        if self._accum.max() == 0:
            return None
        overlay = self.get_overlay(frame)
        _, buf = cv2.imencode(".jpg", overlay, [cv2.IMWRITE_JPEG_QUALITY, 75])
        return base64.b64encode(buf.tobytes()).decode()


# ── Global registry (one HeatmapGenerator per camera) ────────────────────────

_generators: dict[str, HeatmapGenerator] = defaultdict(HeatmapGenerator)


def get_generator(cam_id: str) -> HeatmapGenerator:
    return _generators[cam_id]
