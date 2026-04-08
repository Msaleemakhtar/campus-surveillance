"""
Enroll persons into registered_faces using YOLO person detection + CLIP re-ID.

Works on SHANGHAI_Test frame directories OR any image directory.
No face detection needed — uses full-body YOLO crops, which are reliable
at the 50–200 px person heights typical of surveillance footage.

Two modes:

  --auto (scan all "has people" sequences from SHANGHAI metadata):
    uv run python scripts/enroll_from_frames.py \\
        --auto \\
        --metadata  /path/to/SHANGHAI_Test/SHANGHAI_test.txt \\
        --frames-base /path/to/SHANGHAI_Test/frames \\
        --names "Alice,Bob,Charlie" \\
        --roles "student,staff,student" \\
        --sample-every 5

  Single directory:
    uv run python scripts/enroll_from_frames.py \\
        --frames-dir /path/to/frames/01_0015 \\
        --start 86 --end 373 \\
        --name "Alice" --role student

After enrollment:
    curl -X POST http://localhost:8000/faces/reload
"""
from __future__ import annotations

import argparse
import asyncio
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch

BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))


# ── Models (loaded once) ──────────────────────────────────────────────────────

def _load_models():
    from ultralytics import YOLO
    import open_clip

    print("Loading YOLOv8n for person detection…")
    yolo = YOLO("yolov8n.pt")

    print("Loading CLIP ViT-B/32 for re-ID embedding…")
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="openai"
    )
    model.eval()
    encoder = model.visual
    return yolo, encoder, preprocess


def _embed(crop: np.ndarray, encoder, preprocess) -> np.ndarray | None:
    """Compute normalised 512-dim CLIP embedding from a BGR crop."""
    if crop.shape[0] < 20 or crop.shape[1] < 10:
        return None
    from PIL import Image
    img = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
    tensor = preprocess(img).unsqueeze(0)
    with torch.no_grad():
        feat = encoder(tensor)
        feat = feat / feat.norm(dim=-1, keepdim=True)
    return feat.squeeze(0).cpu().numpy()


# ── Metadata parsing ──────────────────────────────────────────────────────────

@dataclass
class _SeqInfo:
    name: str
    has_people: bool
    start: int
    end: int


def parse_metadata(txt_path: str) -> list[_SeqInfo]:
    """Parse SHANGHAI_test.txt (5 columns: path total has_people start end)."""
    seqs: list[_SeqInfo] = []
    with open(txt_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            seqs.append(_SeqInfo(
                name=Path(parts[0]).name,
                has_people=int(parts[2]) == 1,
                start=int(parts[3]),
                end=int(parts[4]),
            ))
    return seqs


# ── Frame scanning ────────────────────────────────────────────────────────────

@dataclass
class _RawCandidate:
    """YOLO-only candidate — no CLIP yet."""
    score: float
    crop: np.ndarray
    seq: str
    frame: int


@dataclass
class _Candidate:
    score: float
    crop: np.ndarray
    embedding: np.ndarray
    seq: str
    frame: int


def scan_sequence_yolo(
    frames_dir: Path,
    start: int,
    end: int,
    yolo,
    sample_every: int,
) -> list[_RawCandidate]:
    """
    Phase 1: YOLO-only scan. Fast — no CLIP.
    Returns all person detections scored by area × confidence.
    """
    candidates: list[_RawCandidate] = []
    frame_files = sorted(frames_dir.glob("*.jpg"))

    for img_path in frame_files:
        try:
            fnum = int(img_path.stem)
        except ValueError:
            continue
        if fnum < start or fnum > end:
            continue
        if (fnum - start) % sample_every != 0:
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            continue

        results = yolo(img, imgsz=416, classes=[0], conf=0.20, verbose=False)
        if not results or results[0].boxes is None:
            continue

        fh, fw = img.shape[:2]
        for box in results[0].boxes:
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(fw, x2), min(fh, y2)
            w, h = x2 - x1, y2 - y1
            if w < 10 or h < 20:
                continue
            candidates.append(_RawCandidate(
                score=w * h * conf,
                crop=img[y1:y2, x1:x2].copy(),
                seq=frames_dir.name,
                frame=fnum,
            ))

    return candidates


def embed_top(
    raw: list[_RawCandidate],
    n: int,
    encoder,
    preprocess,
) -> list[_Candidate]:
    """
    Phase 2: run CLIP only on the top-n by YOLO score.
    Much faster than embedding every detection.
    """
    top = sorted(raw, key=lambda c: c.score, reverse=True)[:n]
    result = []
    for c in top:
        emb = _embed(c.crop, encoder, preprocess)
        if emb is not None:
            result.append(_Candidate(
                score=c.score, crop=c.crop, embedding=emb,
                seq=c.seq, frame=c.frame,
            ))
    return result


# ── DB insertion ──────────────────────────────────────────────────────────────

async def upsert_all(persons: list[tuple[str, str, str, np.ndarray]]) -> None:
    from sqlalchemy import delete
    from app.db.models import RegisteredFace
    from app.db.postgres import AsyncSessionLocal

    async with AsyncSessionLocal() as session:
        async with session.begin():
            for person_id, name, role, embedding in persons:
                await session.execute(
                    delete(RegisteredFace).where(RegisteredFace.name == name)
                )
                session.add(RegisteredFace(
                    person_id=person_id,
                    name=name,
                    role=role,
                    photo_path=None,
                    face_embedding=embedding.tolist(),
                ))
                print(f"  DB: upserted '{name}' ({person_id}) role={role}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Enroll persons from frame directories using YOLO + CLIP"
    )
    parser.add_argument("--auto", action="store_true")
    parser.add_argument("--metadata")
    parser.add_argument("--frames-base")
    parser.add_argument("--names")
    parser.add_argument("--roles")

    parser.add_argument("--frames-dir")
    parser.add_argument("--start", type=int, default=1)
    parser.add_argument("--end",   type=int, default=99999)
    parser.add_argument("--name")
    parser.add_argument("--role",  default="student", choices=["student", "staff"])

    parser.add_argument("--sample-every", type=int, default=5)

    args = parser.parse_args()

    yolo, encoder, preprocess = _load_models()

    if args.auto:
        if not args.metadata or not args.frames_base or not args.names:
            print("[ERROR] --auto requires --metadata, --frames-base, and --names")
            sys.exit(1)

        names = [n.strip() for n in args.names.split(",")]
        roles_raw = args.roles.split(",") if args.roles else []
        roles = [r.strip() for r in roles_raw] if roles_raw else ["student"] * len(names)
        if len(roles) < len(names):
            roles += ["student"] * (len(names) - len(roles))

        seqs = parse_metadata(args.metadata)
        people_seqs = [s for s in seqs if s.has_people]
        print(f"Found {len(people_seqs)} sequences with people.")

        frames_base = Path(args.frames_base)
        all_raw: list[_RawCandidate] = []

        # Phase 1: YOLO scan all sequences (fast)
        for seq in people_seqs:
            seq_dir = frames_base / seq.name
            if not seq_dir.exists():
                continue
            print(f"  Scanning {seq.name} frames {seq.start}–{seq.end}…", end=" ", flush=True)
            raw = scan_sequence_yolo(seq_dir, seq.start, seq.end, yolo, args.sample_every)
            print(f"{len(raw)} detections")
            all_raw.extend(raw)

        if not all_raw:
            print("[ERROR] No persons detected.")
            sys.exit(1)

        print(f"\nTotal YOLO detections: {len(all_raw)}. "
              f"Running CLIP on top {len(names) * 10} per sequence…")

        # Phase 2: CLIP only on top candidates per sequence (fast)
        # Group by sequence, take best per sequence, then pick N distinct
        from collections import defaultdict
        by_seq: dict[str, list[_RawCandidate]] = defaultdict(list)
        for c in all_raw:
            by_seq[c.seq].append(c)

        all_candidates: list[_Candidate] = []
        for seq_name, raw_list in by_seq.items():
            embedded = embed_top(raw_list, n=5, encoder=encoder, preprocess=preprocess)
            all_candidates.extend(embedded)

        all_candidates.sort(key=lambda c: c.score, reverse=True)
        print(f"CLIP-embedded: {len(all_candidates)} candidates.")

        # Pick top candidates spread across distinct sequences
        picked: list[_Candidate] = []
        used_seqs: set[str] = set()
        for cand in all_candidates:
            if len(picked) >= len(names):
                break
            if cand.seq not in used_seqs:
                picked.append(cand)
                used_seqs.add(cand.seq)
        # Fill remaining if not enough distinct sequences
        for cand in all_candidates:
            if len(picked) >= len(names):
                break
            if cand not in picked:
                picked.append(cand)

        to_insert = []
        for cand, name, role in zip(picked, names, roles):
            h, w = cand.crop.shape[:2]
            print(f"\nEnrolling '{name}' (role={role})")
            print(f"  Source: {cand.seq}/frame {cand.frame:04d}, "
                  f"crop {w}×{h}px, YOLO score {cand.score:.0f}")
            to_insert.append((str(uuid.uuid4())[:8], name, role, cand.embedding))

        asyncio.run(upsert_all(to_insert))

    else:
        if not args.frames_dir or not args.name:
            print("[ERROR] Provide --frames-dir and --name, or use --auto.")
            sys.exit(1)

        frames_dir = Path(args.frames_dir)
        print(f"Scanning {frames_dir.name} frames {args.start}–{args.end}…")
        raw = scan_sequence_yolo(frames_dir, args.start, args.end, yolo, args.sample_every)
        if not raw:
            print("[ERROR] No persons detected.")
            sys.exit(1)
        candidates = embed_top(raw, n=10, encoder=encoder, preprocess=preprocess)
        if not candidates:
            print("[ERROR] CLIP embedding failed on all candidates.")
            sys.exit(1)

        best = max(candidates, key=lambda c: c.score)
        h, w = best.crop.shape[:2]
        print(f"Best crop: {w}×{h}px, score {best.score:.0f}, frame {best.frame:04d}")

        person_id = str(uuid.uuid4())[:8]
        asyncio.run(upsert_all([(person_id, args.name, args.role, best.embedding)]))

    print("\nDone. Reload the live backend with:")
    print("  curl -X POST http://localhost:8000/faces/reload")


if __name__ == "__main__":
    main()
