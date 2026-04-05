# Campus Surveillance AI

Real-time AI video intelligence for a school campus — 10 simulated cameras, person detection,
face recognition, behavior alerts, and natural language video search. Built in phases.

---

## Architecture

Everything runs locally on CPU. No Colab, no cloud GPU, no ngrok.

```
Browser (Next.js)
      ↕  WebSocket
FastAPI + uvicorn  (asyncio)
      ↕  Redis pub/sub
CameraPipeline  (one asyncio.Task per camera)
      ↕  non-blocking dict ops
local_inference.py  ←  1 YOLO daemon thread  (CPU, in-process)
```

---

## Build Status

| Phase | Feature | Status |
|-------|---------|--------|
| 0 | Dataset preparation | ✅ Done |
| 1 | Raw YOLO bounding boxes, all cameras streaming | **Active** |
| 2 | ByteTrack stable IDs | Queued |
| 3 | Linear extrapolation (smooth box motion) | Queued |
| 4 | Demand-driven inference (subscriber registry) | Queued |
| 5 | InsightFace face recognition + alerts | Queued |
| 6 | DB writes — incidents, tracks, PostgreSQL | Queued |
| 7 | Dashboard UI — grid, analytics, incident timeline | Queued |
| 8 | CLIP natural language video search | Queued |

---

## Dataset

| Source | Content | Cameras |
|--------|---------|---------|
| ShanghaiTech Campus | Indoor normal scenes | cam01–cam05, cam15 |
| CUHK Avenue | Outdoor gate / parking | cam11, cam12 |
| UCF-Crime (subset) | Anomaly simulation | cam19 (Fighting/Assault/Shoplifting), cam20 (Vandalism/Burglary/Stealing) |

All videos are pre-converted and stored in `videos/`:
```
videos/normal/    cam01–cam05, cam15 MP4s
videos/outdoor/   cam11, cam12 AVIs
videos/anomaly/   66 MP4s across 6 categories
```

---

## Prerequisites

- Python 3.11+, [uv](https://docs.astral.sh/uv/)
- Docker + Docker Compose (Redis only)
- Node.js 20+ (frontend)
- `ffmpeg` — `sudo apt install ffmpeg`

No Google account, no ngrok, no GPU required.

---

## Quick Start

```bash
# 1. Start Redis
cd backend
docker compose up -d redis

# 2. Start backend
uv run uvicorn app.main:app --host 0.0.0.0 --port 8000

# 3. (Optional) Start frontend
cd ../frontend
npm install && npm run dev
# → http://localhost:3000
```

Health check:
```bash
curl http://localhost:8000/health
# {"status":"ok","models_ready":true}
```

The backend downloads `yolov8n.pt` automatically on first run.

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Backend | FastAPI, Python 3.11, uv |
| ML inference | YOLOv8n (CPU, in-process) |
| Tracking (Phase 2+) | ByteTrack via supervision |
| Face recognition (Phase 5+) | InsightFace buffalo_s |
| Video search (Phase 8) | CLIP ViT-B-32 + pgvector |
| Video reading | OpenCV |
| Message broker | Redis 7 pub/sub |
| Database (Phase 6+) | PostgreSQL 16 + pgvector |
| ORM / migrations | SQLAlchemy async + Alembic |
| Frontend | Next.js 14 App Router, TypeScript |
| Styling | Tailwind CSS |
| Containers | Docker Compose (Redis; DB in later phases) |
