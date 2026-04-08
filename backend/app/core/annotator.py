import cv2
import numpy as np

from app.models.schemas import Track

_STATUS_COLOR: dict[str, tuple[int, int, int]] = {
    "AUTHORIZED":   (87, 197, 34),    # green
    "UNAUTHORIZED": (68, 68, 239),    # red
    "UNKNOWN":      (0, 100, 255),     # orange-red
}
_DEFAULT_COLOR = (200, 200, 200)


def annotate(frame: np.ndarray, tracks: list[Track]) -> np.ndarray:
    """Draw bounding boxes and labels. Clamps coords to frame bounds."""
    fh, fw = frame.shape[:2]

    for track in tracks:
        if not track.bbox or len(track.bbox) != 4:
            continue

        x, y, w, h = track.bbox
        # Clamp to frame — extrapolated boxes can go off-edge
        x = max(0, min(x, fw - 1))
        y = max(0, min(y, fh - 1))
        w = max(1, min(w, fw - x))
        h = max(1, min(h, fh - y))

        color = _STATUS_COLOR.get(track.status, _DEFAULT_COLOR)
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

        label_parts = [f"#{track.track_id}"]
        if track.person_name:
            label_parts.append(track.person_name)
        if track.behavior:
            label_parts.append(track.behavior)
        label = " | ".join(label_parts)

        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        label_y = max(y, th + 6)   # prevent text going above frame top
        cv2.rectangle(frame, (x, label_y - th - 6), (x + tw + 4, label_y), color, -1)
        cv2.putText(frame, label, (x + 2, label_y - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

    return frame
