"""
Enroll a person into the registered_faces table by extracting their best
face crop from a video file.

Usage:
    uv run python scripts/enroll_faces.py \
        --video /path/to/video.mp4 \
        --name  "Alice" \
        --role  student \
        --sample-every 15        # check every N frames (default 15)
        --max-candidates 200     # stop after this many face candidates

The script:
  1. Samples frames from the video.
  2. Runs InsightFace (buffalo_sc) to detect faces in each frame.
  3. Scores each detected face by size × detection confidence.
  4. Saves the single best-quality crop to:
         datasets/mock_faces/<person_id>.jpg
  5. Inserts (or updates) a row in registered_faces.

After running, call  POST /faces/reload  to make the backend pick it up
without restarting.
"""
from __future__ import annotations

import argparse
import asyncio
import sys
import uuid
from pathlib import Path

import cv2
import numpy as np

# ── Paths ─────────────────────────────────────────────────────────────────────
BACKEND_DIR = Path(__file__).resolve().parent.parent
FACES_DIR   = BACKEND_DIR / "datasets" / "mock_faces"
FACES_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(BACKEND_DIR))   # make app.* importable


# ── Face extraction ───────────────────────────────────────────────────────────

def extract_best_face(
    video_path: str,
    sample_every: int = 15,
    max_candidates: int = 200,
) -> np.ndarray | None:
    """
    Scan video frames, return the single highest-quality face crop found.
    Quality = face_bbox_area × detection_score.
    Returns None if no face is detected.
    """
    from insightface.app import FaceAnalysis

    fa = FaceAnalysis(name="buffalo_sc", providers=["CPUExecutionProvider"])
    fa.prepare(ctx_id=-1, det_size=(320, 320))   # larger det_size for enrollment quality

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video: {video_path}")
        return None

    best_crop: np.ndarray | None = None
    best_score: float = 0.0
    candidates = 0
    frame_idx = 0

    print(f"Scanning {video_path} (every {sample_every} frames) …")

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1
        if frame_idx % sample_every != 0:
            continue

        faces = fa.get(frame)
        for face in faces:
            x1, y1, x2, y2 = map(int, face.bbox)
            x1, y1 = max(0, x1), max(0, y1)
            x2 = min(frame.shape[1], x2)
            y2 = min(frame.shape[0], y2)
            if x2 <= x1 or y2 <= y1:
                continue

            area = (x2 - x1) * (y2 - y1)
            score = area * float(face.det_score)
            if score > best_score:
                best_score = score
                # Add a small margin for a natural-looking crop
                pad_x = int((x2 - x1) * 0.15)
                pad_y = int((y2 - y1) * 0.15)
                cx1 = max(0, x1 - pad_x)
                cy1 = max(0, y1 - pad_y)
                cx2 = min(frame.shape[1], x2 + pad_x)
                cy2 = min(frame.shape[0], y2 + pad_y)
                best_crop = frame[cy1:cy2, cx1:cx2].copy()

        candidates += len(faces)
        if candidates >= max_candidates:
            print(f"  Reached {max_candidates} face candidates at frame {frame_idx} — stopping early.")
            break

    cap.release()

    if best_crop is None:
        print("[WARN] No faces detected in the video.")
    else:
        h, w = best_crop.shape[:2]
        print(f"  Best face: {w}×{h} px, quality score {best_score:.0f}")

    return best_crop


# ── DB insertion ──────────────────────────────────────────────────────────────

async def upsert_face(person_id: str, name: str, role: str, photo_path: str) -> None:
    from sqlalchemy import select, delete
    from app.db.models import RegisteredFace
    from app.db.postgres import AsyncSessionLocal

    async with AsyncSessionLocal() as session:
        async with session.begin():
            # Remove any existing row for this name to avoid duplicates
            await session.execute(
                delete(RegisteredFace).where(RegisteredFace.name == name)
            )
            session.add(RegisteredFace(
                person_id=person_id,
                name=name,
                role=role,
                photo_path=photo_path,
            ))
    print(f"  DB: upserted '{name}' ({person_id}) role={role}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Enroll a person from a video clip")
    parser.add_argument("--video",          required=True,  help="Path to source video")
    parser.add_argument("--name",           required=True,  help="Person's display name")
    parser.add_argument("--role",           default="student",
                        choices=["student", "staff"],   help="Role (default: student)")
    parser.add_argument("--sample-every",   type=int, default=15,
                        help="Sample one frame every N frames (default: 15)")
    parser.add_argument("--max-candidates", type=int, default=200,
                        help="Stop after this many face detections (default: 200)")
    args = parser.parse_args()

    crop = extract_best_face(
        args.video,
        sample_every=args.sample_every,
        max_candidates=args.max_candidates,
    )
    if crop is None:
        print("[ERROR] Enrollment failed — no face found.")
        sys.exit(1)

    # Save crop
    person_id  = str(uuid.uuid4())[:8]
    photo_path = FACES_DIR / f"{person_id}.jpg"
    cv2.imwrite(str(photo_path), crop)
    print(f"  Saved face crop: {photo_path}")

    # Insert into DB
    asyncio.run(upsert_face(person_id, args.name, args.role, str(photo_path)))

    print(f"\nDone. Now call:  curl -X POST http://localhost:8000/faces/reload")


if __name__ == "__main__":
    main()
