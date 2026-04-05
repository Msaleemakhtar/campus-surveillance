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

## Current Phase: Phase 1 — Raw YOLO

**What Phase 1 does:**
- Read video frames from disk for all 10 cameras
- Submit every frame to 1 YOLO daemon thread (overwrite semantics — no queue)
- Get latest YOLO result (non-blocking), draw boxes on frame, JPEG encode, publish to Redis
- WebSocket clients receive annotated frames with raw YOLO bounding boxes
- No ByteTrack, no InsightFace, no extrapolation, no alerts, no DB writes

**Phase 1 stop condition (verify before advancing):**
> Open cam01 in the browser. Every visible person has a green box. Boxes update roughly every 2–3 seconds. No crashes after 60 seconds of streaming.

**Do not implement Phase 2 until this is confirmed in the browser.**

---

## Phase Plan (overview)

| Phase | Feature | Stop Condition |
|-------|---------|----------------|
| 1 | Raw YOLO BBs on all cameras | Green boxes on all persons, no crashes |
| 2 | ByteTrack stable IDs | Same person keeps same ID across frames |
| 3 | Linear extrapolation | Boxes move smoothly between inference updates |
| 4 | Demand-driven inference | Unwatched cameras skip YOLO; CPU usage drops |
| 5 | InsightFace + alerts | Known/unknown labels; after-hours alert fires |
| 6 | DB writes | Incidents + tracks saved to PostgreSQL |
| 7 | Dashboard UI | Grid view, analytics, incident timeline |
| 8 | CLIP search | Natural language video search via pgvector |

---

## Locked-In Decisions (do NOT change these)

| Decision | Value | Reason |
|----------|-------|--------|
| `torch.set_num_threads` | 4 | Optimal on i5-8250U; 6 causes thermal throttle |
| YOLO `imgsz` | 416 | Faster than 320 on this CPU (measured) |
| YOLO `conf` | 0.20 | Low threshold needed for distant persons |
| `INFER_EVERY_N_FRAMES` | 1 (Phase 1) | Submit every frame; tune in Phase 4 |
| `TARGET_FPS` | 10 | Display rate; inference runs at ~0.3 Hz |
| `_NUM_WORKERS` | 1 | One YOLO thread; serial is faster on CPU |
| Thread pool `max_workers` | 4 | Set inside `lifespan` via `get_running_loop()` |
| Annotation | Server-side (burn into JPEG) | Eliminates client-side coordinate space bugs |

---

## Key Files

| File | Role |
|------|------|
| `backend/app/core/local_inference.py` | YOLO worker thread — only active module in Phase 1 |
| `backend/app/core/pipeline.py` | Per-camera asyncio.Task: read → infer → annotate → publish |
| `backend/app/core/annotator.py` | OpenCV BB drawing with coord clamping |
| `backend/app/api/stream.py` | WebSocket `/ws/stream/{cam_id}` via Redis pub/sub |
| `backend/app/main.py` | FastAPI lifespan: load models → start pipelines |
| `backend/app/core/config.py` | Settings via pydantic-settings + `.env` |
| `backend/.env` | Runtime config (VIDEOS_BASE, REDIS_URL, etc.) |

**Files that exist but are NOT active in Phase 1** (keep, do not delete):
- `annotator.py`, `heatmap.py`, `zones.py` — used in later phases
- `app/db/` — migrations, models, postgres, redis_client (redis IS active)

---

## Startup

```bash
# Terminal 1 — infrastructure
cd /home/salim/Desktop/campus-survelliance/backend
docker compose up -d redis

# Terminal 2 — backend
cd /home/salim/Desktop/campus-survelliance/backend
uv run uvicorn app.main:app --host 0.0.0.0 --port 8000

# Terminal 3 — frontend (optional for Phase 1)
cd /home/salim/Desktop/campus-survelliance/frontend
npm run dev
```

Health check: `curl http://localhost:8000/health` → `{"status":"ok","models_ready":true}`

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
VIDEOS_BASE=/home/salim/Desktop/campus-survelliance/videos
```
