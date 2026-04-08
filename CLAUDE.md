# CLAUDE.md

Guidance for Claude Code when working in this repository.

---

## Architecture

All ML runs **locally on CPU**. No Colab, no cloud GPU, no ngrok, no external inference server.

```
Browser (Next.js)
      ↕  WebSocket
FastAPI (uvicorn, asyncio event loop)
      ↕  Redis pub/sub
CameraPipeline  (one asyncio.Task per camera)
      ↕  submit_frame() / get_latest_result()  [non-blocking dict ops]
local_inference.py  ←  1 daemon YOLO worker thread
      ↕
YOLOv8n  (CPU, in-process)
```

---

## Hardware Constraints

| Fact | Value |
|------|-------|
| CPU | Intel i5-8250U (4 cores / 8 threads) |
| YOLO latency (single call) | ~291 ms avg at imgsz=416 |
| YOLO latency at imgsz=320 | ~352 ms — **slower** than 416 on this CPU |
| Optimal PyTorch threads | 4 (6 causes thermal throttling) |
| Max parallel YOLO calls | 1 (serial is faster on CPU — no contention) |

These numbers were measured. Do not assume lower latency.

---

## Current Phase: Phase 5 — CLIP Re-ID + Alerts

Phases 1–4 are complete and verified.

---

## Completed Phases

### Phase 1 — Raw YOLO ✓
- 8 cameras stream annotated frames via WebSocket (cam01–04, cam11, cam15, cam19, cam20)
- Every visible person has a bounding box; boxes update every ~2.3 s (8-camera YOLO cycle)
- No crashes after 60 s of streaming (verified)

### Phase 2 — Stable Track IDs ✓
- Per-camera `_CentroidTracker` in `local_inference.py` assigns stable IDs
- Uses **Hungarian algorithm** (`scipy.optimize.linear_sum_assignment`) for globally optimal detection→track assignment — greedy nearest-neighbour was discarded because it swaps IDs when two people walk near each other's previous positions
- **ByteTrack was evaluated and dropped**: its Kalman filter assumes 25–30 fps; at our real ~0.3 Hz per-camera rate its predicted positions diverge and even stationary persons lose their ID
- Time-scaled match radius: `min(speed_px_per_frame × df, max_distance)` — grows with inference gap so walking persons still match after a long cycle
- **Edge-proximity expiry**: tracks whose centroid is within 8% of any frame boundary get `max_age=40 frames` instead of 60, preventing an exiting person's ID being assigned to someone entering from the same edge
- Default `speed_px_per_frame=20.0`, `max_age_frames=60`; outdoor cams cam11/12 use `speed=25.0`
- Velocity EMA `alpha=0.9` for fast adaptation; noise floor `1.5 px/frame` — zeroes sub-noise velocity so stationary persons don't feed the extrapolator false drift

### Phase 3 — Extrapolation with Velocity Decay ✓
- `CameraPipeline` maintains `_extrap: dict[int, _TrackState]` per track
- On each new YOLO result: velocity `(vx_f, vy_f)` in px/frame is stored from the tracker's EMA
- On each display frame (10 FPS): centroid is projected forward using **integral velocity decay**:
  `displacement = vx * df * (1 - df / (2 × _MAX_EXTRAP_DF))`
  Velocity decays linearly to zero at `_MAX_EXTRAP_DF` — boxes decelerate to a stop rather than floating off when a person stops or exits between YOLO updates
- `_MAX_EXTRAP_DF = 20` frames (2 s) — covers 8-camera YOLO cycle (2.3 s); box freezes for ~0.3 s at most before next update
- `_STALE_FRAMES = 25` — ghost track expires 2.5 s after last YOLO sighting
- `_ACTIVE_PRUNE_FRAMES = 12` — ghost tracks absent from a new result for > 12 frames are pruned immediately on the next YOLO update
- **`_deduplicate_boxes()`**: removes partial-body detections (torso-only, legs-only) whose centroid falls inside a larger detection. YOLO at conf=0.20 emits these with IoU < 0.45 so its own NMS misses them; without deduplication each partial becomes a separate track.
- YOLO called with `iou=0.35` (vs default 0.45) for more aggressive overlap NMS before deduplication

### Phase 4 — Demand-Driven Inference ✓
- Subscriber registry in `local_inference.py`: `add_subscriber` / `remove_subscriber` / `has_subscribers` (thread-safe, protected by `_sub_lock`)
- `stream.py` calls `add_subscriber` on WebSocket accept, `remove_subscriber` in `finally` on disconnect
- Pipeline `_run_single` / `_run_playlist`: checks `has_subscribers` at the top of each loop iteration — skips H.264 decode entirely (not just YOLO) when no subscribers
- `_process_frame` also returns early if no subscribers as a second gate
- CPU drops from ~50% → ~2% when all browser tabs are closed (verified)
- Health endpoint (`GET /health`) exposes live subscriber counts: `{"subscribers": {"cam01": 1, ...}}`
- **`visibilitychange` was tried and dropped**: disconnect + immediate reconnect creates a race where count never reaches 0; tab switching is not a meaningful "unwatched" signal for a surveillance feed — only actual tab close matters
- Idle check interval: 0.5 s (pipeline sleeps this long between subscriber checks when idle)

---

## Phase Plan (overview)

| Phase | Feature | Stop Condition | Status |
|-------|---------|----------------|--------|
| 1 | Raw YOLO BBs on all cameras | Green boxes on all persons, no crashes | ✓ done |
| 2 | Stable track IDs | Same person keeps same ID across frames | ✓ done |
| 3 | Extrapolation with velocity decay | Boxes move smoothly; stop floating when person stops/exits | ✓ done |
| 4 | Demand-driven inference | Unwatched cameras skip YOLO; CPU usage drops | ✓ done |
| 5 | CLIP Re-ID + alerts | Known/unknown labels; after-hours alert fires | in progress |
| 6 | DB writes | Incidents + tracks saved to PostgreSQL | |
| 7 | Dashboard UI | Grid view, analytics, incident timeline | |
| 8 | CLIP search | Natural language video search via pgvector | |

---

## Locked-In Decisions (do NOT change these)

| Decision | Value | Reason |
|----------|-------|--------|
| `torch.set_num_threads` | 4 | Optimal on i5-8250U; 6 causes thermal throttle |
| YOLO `imgsz` | 416 | Faster than 320 on this CPU (measured) |
| YOLO `conf` | 0.20 | Low threshold needed for distant persons |
| YOLO `iou` | 0.35 | More aggressive NMS to suppress partial-body duplicates |
| `INFER_EVERY_N_FRAMES` | 1 | Overwrite semantics make higher values pointless — worker always gets the latest frame |
| `TARGET_FPS` | 10 | Display rate; inference runs at ~0.3 Hz |
| `_NUM_WORKERS` | 1 | One YOLO thread; serial is faster on CPU |
| Thread pool `max_workers` | 4 | Set inside `lifespan` via `get_running_loop()` |
| Annotation | Server-side (burn into JPEG) | Eliminates client-side coordinate space bugs |
| Tracker | `_CentroidTracker` (Hungarian) | ByteTrack Kalman fails at 0.3 Hz; centroid+scipy works |
| Tracker assignment | Hungarian (`linear_sum_assignment`) | Greedy NN causes ID swaps when people pass near each other |
| Tracker speed default | `speed_px_per_frame=20.0` | At df≈3: radius=60px covers walking (30px) + YOLO noise (10px) |
| Tracker max age | `max_age_frames=60` | 6 s = 2 full YOLO cycles; prevents stale tracks false-matching new entrants |
| Edge expiry | `40 frames` within 8% of frame boundary | < max_age=60; clears exiting persons without affecting interior tracks |
| Velocity EMA | `alpha=0.9` | Fast adaptation; 0.7 caused 27px visible trail |
| Velocity noise floor | `1.5 px/frame` | Below this is YOLO localization noise; zeroed to prevent stationary drift |
| Extrapolation | Integral velocity decay | `disp = vx * df * (1 - df/2M)`; boxes decelerate to stop instead of floating |
| `_MAX_EXTRAP_DF` | 20 frames (2 s) | Covers 8-camera cycle (2.3 s); box freezes ≤ 0.3 s before next update |
| `_STALE_FRAMES` | 25 frames (2.5 s) | Ghost track lifetime after last YOLO sighting |
| Cameras active | 8 (cam01–04, cam11, cam15, cam19, cam20) | cam05 removed (fast motion); cam12 removed |
| YOLO cycle | 8 × 291 ms ≈ 2.3 s per camera | Measured; used to size all timeout constants above |

---

## Active Cameras

| Camera | Label | Source |
|--------|-------|--------|
| cam01 | Main Entrance | `videos/normal/cam01.mp4` |
| cam02 | Hallway A | `videos/normal/cam02.mp4` |
| cam03 | Hallway B | `videos/normal/cam03.mp4` |
| cam04 | Library | `videos/normal/cam04.mp4` |
| cam11 | Main Gate | `videos/outdoor/cam11.avi` |
| cam15 | Sports Ground | `videos/normal/cam15.mp4` |
| cam19 | THREAT FEED | playlist: `videos/anomaly/Fighting + Assault + Shoplifting` |
| cam20 | INTRUSION FEED | playlist: `videos/anomaly/Vandalism + Burglary + Stealing` |

---

## Phase 5 — CLIP Re-ID Details

Re-ID uses CLIP ViT-B/32 (512-dim cosine similarity), not face detection — full-body crops work reliably at the 50–200px person heights typical of surveillance footage.

**How it works:**
- `_PersonReIDMatcher` holds enrolled embeddings loaded from `registered_faces.face_embedding` (PostgreSQL pgvector)
- On each YOLO result: `_submit_reid(cam_id, tid, crop)` queues a per-`(cam_id, tid)` crop — each track has its own slot, so 5 visible persons all get queued (old per-camera slot overwrote all but the last)
- Background `_reid_worker_loop` pops one entry, runs CLIP, updates `active.face_status` / `face_name` / `face_confidence`
- Re-ID triggered on first detection and every `_REID_CACHE_EVERY=5` YOLO cycles (~14.5 s)
- Match threshold: `_THRESHOLD=0.75` (same-person crops score 0.85–0.95; different persons < 0.75)

**Enrollment (required before re-ID works):**
```bash
cd /home/salim/Desktop/campus-surveillance/backend

# Option A — from SHANGHAI_Test dataset
uv run python scripts/enroll_from_frames.py \
    --auto \
    --metadata /home/salim/Desktop/campus-surveillance/SHANGHAI_Test/SHANGHAI_test.txt \
    --frames-base /home/salim/Desktop/campus-surveillance/SHANGHAI_Test/frames \
    --names "Alice,Bob,Charlie,David" \
    --roles "staff,student,student,staff" \
    --sample-every 5

# Option B — from a live camera video
mkdir -p /tmp/cam01_frames
ffmpeg -i /home/salim/Desktop/campus-surveillance/videos/normal/cam01.mp4 \
       -vf fps=1 /tmp/cam01_frames/%04d.jpg
uv run python scripts/enroll_from_frames.py \
    --frames-dir /tmp/cam01_frames --name "Alice" --role staff

# Reload without restart
curl -X POST http://localhost:8000/faces/reload
```

**Phase 5 stop condition:**
1. Enrolled persons appear with GREEN box + name within ~15 s of first appearing
2. Un-enrolled persons appear with ORANGE (UNKNOWN)
3. After-hours (18:00–06:00 UTC): UNKNOWN persons trigger `UNAUTHORIZED_VISITOR` HIGH alert

---

## Key Files

| File | Role |
|------|------|
| `backend/app/core/local_inference.py` | YOLO worker thread, `_CentroidTracker` (Hungarian, edge expiry, EMA velocity), CLIP re-ID, alert generation |
| `backend/app/core/pipeline.py` | Per-camera asyncio.Task: read → infer → extrapolate (decay) → annotate → publish |
| `backend/app/core/annotator.py` | OpenCV BB drawing with status colour coding and coord clamping |
| `backend/app/api/stream.py` | WebSocket `/ws/stream/{cam_id}` — subscriber registry add/remove |
| `backend/app/api/faces.py` | `POST /faces/reload` — hot-reloads CLIP embeddings from DB |
| `backend/app/main.py` | FastAPI lifespan: load models → start pipelines |
| `backend/app/core/config.py` | Settings via pydantic-settings + `.env` |
| `backend/scripts/enroll_from_frames.py` | YOLO+CLIP enrollment tool (see Phase 5 section) |
| `backend/.env` | Runtime config (VIDEOS_BASE, REDIS_URL, etc.) |

**Files that exist but are NOT yet active** (keep, do not delete):
- `heatmap.py`, `zones.py` — used in later phases
- `app/db/` — migrations, models, postgres, redis_client (redis IS active)

---

## Startup

```bash
# Terminal 1 — infrastructure
cd /home/salim/Desktop/campus-surveillance/backend
docker compose up -d redis

# Terminal 2 — backend
cd /home/salim/Desktop/campus-surveillance/backend
uv run uvicorn app.main:app --host 0.0.0.0 --port 8000

# Terminal 3 — frontend
cd /home/salim/Desktop/campus-surveillance/frontend
npm run dev
```

Health check: `curl http://localhost:8000/health` → `{"status":"ok","models_ready":true,"subscribers":{"cam01":1,...}}`

---

## Dependency Management

All Python deps via UV only:
```bash
uv add <package>
uv remove <package>
uv sync
uv run uvicorn app.main:app ...
```
Never use pip, conda, or poetry in the backend directory.

---

## Rules That Must Not Be Violated

1. **Never advance a phase without verifying the stop condition in the browser.**
2. **Never introduce Colab/cloud GPU** — all inference is local CPU, period.
3. **Never run more than 1 concurrent YOLO call** — `_NUM_WORKERS = 1`, serial is faster.
4. **Never set thread pool executor at module level** — must be inside `lifespan` via `asyncio.get_running_loop()`.
5. **Never create summary.md, notes.md, or duplicate scripts.**
6. **Never use pip/conda** in the backend environment.
7. **Never assume YOLO is fast** — plan around 291ms/call.
8. **Never skip phase stop condition verification** — complexity stacked on a broken base caused the bounding box crisis that required a full rebuild.

---

## .env Reference

```env
DATABASE_URL=postgresql+asyncpg://...
REDIS_URL=redis://localhost:6379
INFER_EVERY_N_FRAMES=1
JPEG_QUALITY=75
TARGET_FPS=10
AFTER_HOURS_START=18
AFTER_HOURS_END=6
VIDEOS_BASE=/home/salim/Desktop/campus-surveillance/videos
```
