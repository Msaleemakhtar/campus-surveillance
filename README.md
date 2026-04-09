# Campus Surveillance AI

Real-time AI video intelligence for a school campus. Eight simulated cameras stream annotated video through a local CPU-only pipeline — person detection, stable tracking, CLIP-based re-identification, behaviour alerts, an analytics dashboard, and natural-language video search. Built in eight incremental phases, all verified against a stop condition before the next phase began.

No cloud GPU. No ngrok. No external inference server. Everything runs on an Intel i5-8250U.

---

## Demo

![Demo](demo.gif)

> Full video: [Download v3.mp4](https://github.com/Msaleemakhtar/campus-surveillance/releases/download/v1.0.0/v3.mp4)

---

## System Architecture

### Full Data-Flow Diagram

```mermaid
flowchart TD
    %% ── Video sources ──────────────────────────────────────────────────────
    subgraph SRC["Video Sources  (disk)"]
        direction LR
        V1["cam01–04 · cam15\nnormal/ MP4s"]
        V2["cam11\noutdoor/ AVI"]
        V3["cam19 · cam20\nanomaly/ playlists\n(Fighting · Vandalism …)"]
    end

    %% ── asyncio event loop ─────────────────────────────────────────────────
    subgraph LOOP["asyncio event loop  —  uvicorn main thread"]
        direction TB

        subgraph PL["CameraPipeline  ×8  (one asyncio.Task per camera)"]
            direction TB
            PL1["① Read frame\nOpenCV VideoCapture\n→ asyncio.to_thread"]
            PL2{"② has_subscribers?\n_subscriber_counts dict"}
            PL3["③ submit_frame\nwrite _pending[cam_id]\n(overwrite — always freshest)"]
            PL4["④ get_latest_result\nread _results[cam_id]\n(non-blocking dict read)"]
            PL5["⑤ _update_extrap\nstore track velocity state"]
            PL6["⑥ _extrapolate\nproject bbox forward\nintegral velocity decay"]
            PL7["⑦ annotator.annotate\nOpenCV server-side burn-in\n(green/orange/red boxes)"]
            PL8["⑧ zone assignment\n+ heatmap accumulator\nheatmap_mod.get_generator"]
            PL9{"⑨ 2 s elapsed\n+ tracks visible?"}
            PL10["⑩ submit_frame_for_embedding\nwrite _frame_embed_slots[cam_id]"]
            PL11["⑪ _encode_frame\nJPEG → base64\nasynco.to_thread"]
            PL12["⑫ redis.publish\ncam:{cam_id} channel\nWSMessage JSON"]
            PL_DB1[["fire-and-forget\n_persist_alerts\n→ incidents table"]]
            PL_DB2[["fire-and-forget\n_persist_detected_persons\n→ detected_persons table"]]
            PL_DB3[["throttled 30 s\n_persist_zone_counts\n→ zone_counts table"]]

            PL1 --> PL2
            PL2 -- "no → sleep 0.5 s" --> PL1
            PL2 -- yes --> PL3
            PL3 -.->|"async, non-blocking"| PL4
            PL4 --> PL5
            PL5 --> PL_DB1
            PL5 --> PL_DB2
            PL5 --> PL6
            PL6 --> PL7
            PL7 --> PL8
            PL8 --> PL_DB3
            PL8 --> PL9
            PL9 -- yes --> PL10
            PL9 -- no --> PL11
            PL10 --> PL11
            PL11 --> PL12
        end

        subgraph API["FastAPI  REST  +  WebSocket"]
            direction TB
            WS["WS  /ws/stream/{cam_id}\nsubscriber add/remove\nstream annotated JPEGs"]
            AN["GET  /analytics/summary\nreads in-memory _trackers\n+ _alert_history deque"]
            INC["GET  /analytics/incidents\nreads _alert_history deque\n(maxlen=200 rolling buffer)"]
            SR["GET  /search?query=\nCLIP text encode\n→ pgvector cosine distance"]
            FC["POST  /faces/reload\nhot-reload CLIP embeddings\nfrom registered_faces table"]
            HE["GET  /health\nmodels_ready · subscriber counts"]
            SM["GET  /frames/{cam}/{file}\nstatic JPEG mount\nfor search thumbnails"]
        end
    end

    %% ── daemon threads ─────────────────────────────────────────────────────
    subgraph THR["Daemon Threads  (local_inference.py)"]
        direction TB

        subgraph YW["YOLO Worker Thread  ×1  (_worker_loop)"]
            direction TB
            Y1["_pop_next\npop _pending dict\n(one camera at a time)"]
            Y2["YOLOv8n\nimgsz=416  conf=0.20  iou=0.35\nclasses=[0]  ~291 ms/call"]
            Y3["_deduplicate_boxes\nremove partial-body detections\n(centroid-inside-larger check)"]
            Y4["_CentroidTracker.update\nHungarian assignment\nEMA velocity  α=0.9\nedge expiry  max_age=40/60"]
            Y5["behavior detection\nLOITERING > 120 s\nRUNNING > 18 px/frame"]
            Y6["alert generation\nafter-hours check\ncooldown dedup"]
            Y7["write _results[cam_id]\n(InferResponse, frame_idx)"]
            Y8["append _alert_history\nrolling deque maxlen=200"]
            Y9["_submit_reid\nwrite _reid_priority or\n_reid_refresh dict"]

            Y1 --> Y2 --> Y3 --> Y4 --> Y5 --> Y6 --> Y7
            Y6 --> Y8
            Y4 --> Y9
        end

        subgraph RW["CLIP Re-ID Worker Thread  ×1  (_reid_worker_loop)"]
            direction TB
            R1["drain _reid_priority first\nthen _reid_refresh\n(new persons get fast ID)"]
            R2["_face_recognizer.identify\nCLIP ViT-B/32 image encode\n512-dim crop embedding"]
            R3["cosine similarity\nvs registered_faces DB\nthreshold = 0.75"]
            R4["write back to _ActiveTrack\nface_status · face_name\nface_confidence  (GIL)"]

            R1 --> R2 --> R3 --> R4
        end

        subgraph EW["Frame Embed Worker Thread  ×1  (_frame_embed_worker_loop)"]
            direction TB
            E1["drain _frame_embed_slots\n(overwrite slot per camera)"]
            E2["_face_recognizer.embed_frame\nCLIP ViT-B/32 full-frame\n512-dim embedding"]
            E3["save JPEG\nbackend/frame_store/{cam_id}/\ntimestamp.jpg"]
            E4["INSERT clip_embeddings\ncam_id · timestamp\nframe_path · embedding vector"]

            E1 --> E2 --> E3 --> E4
        end
    end

    %% ── storage ────────────────────────────────────────────────────────────
    subgraph STORE["Persistence"]
        RD[("Redis 7\npub/sub\ncam:{cam_id} channels")]
        PG[("PostgreSQL 16 + pgvector\nregistered_faces\ndetected_persons\nincidents\nzone_counts\nclip_embeddings")]
        FS["Frame Store\nbackend/frame_store/\n{cam_id}/{ts}.jpg"]
    end

    %% ── frontend ───────────────────────────────────────────────────────────
    subgraph FE["Next.js 14  Frontend"]
        direction LR
        F1["/ — Live Feed\nCameraFeed per cam\nWebSocket consumer\n10 FPS annotated stream"]
        F2["/analytics — Dashboard\nKPI cards · heatmap canvas\nincident timeline\nresponse team roster"]
        F3["/search — NL Search\ntext input → /search API\nframe thumbnail grid\n+ camera · timestamp"]
    end

    %% ── cross-subgraph edges ───────────────────────────────────────────────
    SRC --> PL1
    PL12 --> RD
    RD -->|"subscribe cam:{cam_id}"| WS
    WS -->|"annotated JPEG frames\nWSMessage JSON"| F1
    WS -->|"add_subscriber on connect\nremove_subscriber on disconnect"| PL2

    PL3 -->|"_pending dict\n(overwrite slot)"| Y1
    Y7 -->|"_results dict"| PL4
    Y9 -->|"_reid_priority / _reid_refresh"| R1
    R4 -.->|"writes _ActiveTrack fields\n→ picked up on next YOLO result"| Y4
    PL10 -->|"_frame_embed_slots dict"| E1

    AN -->|"in-memory _trackers state\n+ heatmap data"| F2
    INC -->|"_alert_history deque"| F2
    SR -->|"CLIP text encode\n→ pgvector <=> query"| PG
    SR --> F3
    SM -->|"static JPEG files"| F3

    PL_DB1 & PL_DB2 & PL_DB3 -->|"async SQLAlchemy\nfire-and-forget"| PG
    E3 --> FS
    E4 --> PG

    FC -->|"reload_faces\nre-query registered_faces"| PG
```

### Shared-State & Thread Interaction

```mermaid
flowchart LR
    %% ── asyncio tasks ──────────────────────────────────────────────────────
    subgraph EV["asyncio event loop  (uvicorn main thread)"]
        direction TB
        CAM["CameraPipeline Tasks\ncam01 · cam02 · … · cam20\n(8 concurrent asyncio.Tasks)"]
        WSH["WebSocket Handlers\n/ws/stream/{cam_id}"]
        RST["REST Handlers\n/analytics  /search  /faces/reload"]
        AH["analytics_heartbeat\nkeeps inference alive\nwhile dashboard is open\n(10 s TTL)"]
    end

    %% ── shared state ────────────────────────────────────────────────────────
    subgraph SS["Shared State  (thread-safe dicts  +  GIL)"]
        direction TB
        PEND["_pending\ndict[cam_id → frame]\nprotected by _pending_lock\n⚡ overwrite semantics"]
        RES["_results\ndict[cam_id → InferResponse]\nprotected by _results_lock\n⚡ overwrite semantics"]
        SUBSC["_subscriber_counts\ndict[cam_id → int]\nprotected by _sub_lock\nalso gated by analytics TTL"]
        RPRI["_reid_priority\ndict[(cam_id, tid) → crop]\nnew persons — processed first"]
        RREF["_reid_refresh\ndict[(cam_id, tid) → crop]\nperiodic refresh — lower priority"]
        FEMB["_frame_embed_slots\ndict[cam_id → (ts, frame)]\nprotected by _frame_embed_lock\n⚡ overwrite semantics"]
        AHI["_alert_history\ndeque[dict]  maxlen=200\nrolling alert buffer\nfor analytics API"]
        TRK["_trackers\ndict[cam_id → _CentroidTracker]\n_ActiveTrack objects\nface_status updated in-place"]
    end

    %% ── daemon threads ──────────────────────────────────────────────────────
    subgraph DT["Daemon Threads"]
        direction TB
        YT["YOLO Worker\n_worker_loop\n① pop _pending\n② run YOLOv8n ~291 ms\n③ CentroidTracker.update\n④ write _results\n⑤ submit crops to re-ID"]
        RT["CLIP Re-ID Worker\n_reid_worker_loop\n① drain priority then refresh\n② CLIP.identify crop\n③ write back to _ActiveTrack"]
        ET["Frame Embed Worker\n_frame_embed_worker_loop\n① drain _frame_embed_slots\n② CLIP.embed_frame\n③ save JPEG + write DB"]
    end

    %% ── edges ───────────────────────────────────────────────────────────────
    CAM -->|"submit_frame\n(overwrite)"| PEND
    PEND -->|"_pop_next"| YT
    YT -->|"write result"| RES
    RES -->|"get_latest_result\n(non-blocking)"| CAM
    YT -->|"update active tracks"| TRK
    YT -->|"_submit_reid\nis_first → priority\nperiodic → refresh"| RPRI
    YT -->|"_submit_reid\nperiodic refresh"| RREF
    YT -->|"append alert dicts"| AHI
    RPRI & RREF -->|"consumed by"| RT
    RT -->|"write face_status\nface_name · confidence"| TRK
    CAM -->|"submit_frame_for_embedding\n(overwrite, every 2 s)"| FEMB
    FEMB -->|"consumed by"| ET
    WSH -->|"add_subscriber\nremove_subscriber"| SUBSC
    CAM -->|"has_subscribers check\nbefore every frame"| SUBSC
    AH -->|"extend analytics TTL"| SUBSC
    RST -->|"analytics_heartbeat"| AH
    RST -->|"get_analytics_summary\nreads _trackers in-memory"| TRK
    RST -->|"get_recent_incidents\nreads rolling buffer"| AHI
```

### Database Schema

```mermaid
erDiagram
    registered_faces {
        int     id PK
        string  person_id UK
        string  name
        string  role
        text    photo_path
        vector  face_embedding "512-dim CLIP"
        datetime created_at
    }

    detected_persons {
        int     id PK
        int     track_id
        string  camera_id
        datetime timestamp
        int     bbox_x
        int     bbox_y
        int     bbox_w
        int     bbox_h
        bool    is_authorized
        string  matched_person_id FK
        float   confidence
        string  behavior
    }

    incidents {
        int     id PK
        string  camera_id
        string  incident_type
        string  severity "LOW|MEDIUM|HIGH"
        int     track_id
        text    frame_path
        datetime timestamp
        bool    resolved
        string  assigned_to
        text    notes
    }

    zone_counts {
        int     id PK
        string  camera_id
        string  zone_name
        int     person_count
        float   congestion_score
        datetime recorded_at
    }

    clip_embeddings {
        int     id PK
        string  camera_id
        datetime timestamp
        text    frame_path
        vector  embedding "512-dim CLIP"
    }

    attendance_logs {
        int     id PK
        string  person_id FK
        string  camera_id
        datetime entry_time
        datetime exit_time
        string  zone_name
    }

    responders {
        int     id PK
        string  name
        string  status "ON_DUTY|RESPONDING|ON_CALL"
        int     assigned_incident_id FK
        datetime updated_at
    }

    registered_faces ||--o{ detected_persons : "matched_person_id"
    registered_faces ||--o{ attendance_logs  : "person_id"
    incidents        ||--o| responders       : "assigned_incident_id"
```

---

## Features

| Feature | Detail |
|---------|--------|
| **Person detection** | YOLOv8n, `conf=0.20`, `iou=0.35`, `imgsz=416` — optimised for distant persons at surveillance resolution |
| **Stable track IDs** | `_CentroidTracker` with Hungarian assignment (`scipy.optimize.linear_sum_assignment`); EMA velocity smoothing; edge-proximity expiry |
| **Smooth motion** | Integral velocity decay extrapolation between YOLO updates — boxes decelerate to a stop instead of teleporting |
| **Demand-driven CPU** | Subscriber registry: YOLO and H.264 decode are skipped entirely when no browser tab is watching — idle CPU drops from ~50 % to ~2 % |
| **Person re-ID** | CLIP ViT-B/32 full-body crop cosine similarity (threshold 0.75); no face detection required; per-`(cam_id, track_id)` priority queue |
| **Behaviour alerts** | LOITERING (> 120 s stationary), RUNNING (> 180 px/s), AFTER_HOURS_PRESENCE (18:00–06:00 UTC), UNAUTHORIZED_VISITOR |
| **Analytics dashboard** | Live KPI cards, inferno-colourmap heatmap from track centroids, incident timeline, response-team roster |
| **NL video search** | CLIP text encoder → pgvector cosine distance → frame thumbnails; one frame stored per camera every 2 s |
| **DB persistence** | PostgreSQL 16 + pgvector; async fire-and-forget writes; Alembic migrations |

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Backend runtime | Python 3.12, FastAPI, uvicorn (asyncio) |
| ML — detection | YOLOv8n via `ultralytics` |
| ML — re-ID + search | CLIP ViT-B/32 via `open-clip-torch` |
| Person tracking | Custom `_CentroidTracker` + `scipy` (Hungarian algorithm) |
| Video decoding | OpenCV (`cv2.VideoCapture`) |
| Message broker | Redis 7 pub/sub |
| Database | PostgreSQL 16 + pgvector extension |
| ORM / migrations | SQLAlchemy 2 async + Alembic |
| Dependency manager | [uv](https://docs.astral.sh/uv/) (never pip/conda) |
| Frontend | Next.js 14 App Router, TypeScript, Tailwind CSS |
| Containers | Docker Compose — Redis + PostgreSQL only |

---

## Hardware

Designed and measured on an **Intel i5-8250U** (4 cores / 8 threads, no GPU).

| Constant | Value | Reason |
|----------|-------|--------|
| `torch.set_num_threads` | 4 | Optimal; 6 causes thermal throttle |
| YOLO `imgsz` | 416 | Faster than 320 on this CPU (measured) |
| YOLO latency | ~291 ms/call | 8 cameras × 291 ms ≈ 2.3 s cycle |
| Max parallel YOLO calls | 1 | Serial is faster than concurrent on CPU |
| `TARGET_FPS` (display) | 10 | Independent of the ~0.3 Hz inference rate |

---

## Repository Layout

```
campus-surveillance/
├── backend/
│   ├── app/
│   │   ├── api/
│   │   │   ├── analytics.py      # /analytics/* — live summaries + incident feed
│   │   │   ├── faces.py          # /faces/reload — hot-reload CLIP embeddings from DB
│   │   │   ├── search.py         # /search — NL query → pgvector → frame thumbnails
│   │   │   └── stream.py         # /ws/stream/{cam_id} — WebSocket + subscriber registry
│   │   ├── core/
│   │   │   ├── annotator.py      # OpenCV bounding-box drawing + coord clamping
│   │   │   ├── config.py         # pydantic-settings (.env)
│   │   │   ├── heatmap.py        # Per-camera HeatmapGenerator accumulator
│   │   │   ├── local_inference.py# YOLO worker · _CentroidTracker · CLIP re-ID · alerts
│   │   │   ├── pipeline.py       # Per-camera asyncio.Task: read → infer → extrapolate → publish
│   │   │   └── zones.py          # Zone polygon definitions per camera
│   │   ├── db/
│   │   │   ├── migrations/       # Alembic revisions (0001 initial, 0002 pgvector embedding)
│   │   │   ├── models.py         # SQLAlchemy ORM models
│   │   │   ├── postgres.py       # Async engine + session factory
│   │   │   └── redis_client.py   # Async Redis connection
│   │   ├── models/
│   │   │   └── schemas.py        # Pydantic request/response schemas
│   │   └── main.py               # FastAPI app + lifespan (model load → pipelines start)
│   ├── scripts/
│   │   └── enroll_from_frames.py # YOLO+CLIP enrollment tool
│   ├── frame_store/              # Saved frame JPEGs (cam_id/timestamp.jpg)
│   ├── docker-compose.yml        # Redis + PostgreSQL services
│   ├── pyproject.toml
│   └── .env                      # Runtime config (see below)
├── frontend/
│   ├── app/
│   │   ├── page.tsx              # / — Live Feed (8-camera grid)
│   │   ├── analytics/page.tsx    # /analytics — Dashboard
│   │   └── search/page.tsx       # /search — Natural language search
│   ├── components/
│   │   ├── CameraGrid.tsx        # 2×4 grid layout
│   │   ├── CameraFeed.tsx        # Single camera WebSocket consumer
│   │   ├── AnalyticsDashboard.tsx# KPI cards, incident timeline, responders
│   │   ├── HeatmapCanvas.tsx     # Client-side inferno colourmap heatmap
│   │   └── NLSearchBar.tsx       # Search input + results grid
│   ├── hooks/
│   │   ├── useCameraFeed.ts      # WebSocket hook (single camera)
│   │   └── useAnalyticsFeed.ts   # Multi-camera WebSocket + incident persistence
│   └── lib/
│       ├── types.ts              # Shared TypeScript types
│       └── websocket.ts          # WebSocket reconnect logic
└── videos/
    ├── normal/                   # cam01–04, cam15 MP4s
    ├── outdoor/                  # cam11 AVI
    └── anomaly/                  # cam19, cam20 playlist sources
```

---

## Prerequisites

- Python 3.12+ and [uv](https://docs.astral.sh/uv/)
- Docker + Docker Compose
- Node.js 20+
- `ffmpeg` — `sudo apt install ffmpeg`

No GPU, no Google account, no cloud services required.

---

## Quick Start

### 1. Start infrastructure

```bash
cd backend
docker compose up -d
```

This starts PostgreSQL 16 (port 5432) and Redis 7 (port 6379).

### 2. Configure environment

```bash
cp .env.example .env   # or create .env manually — see reference below
```

### 3. Run database migrations

```bash
cd backend
uv run alembic upgrade head
```

### 4. Start the backend

```bash
cd backend
uv run uvicorn app.main:app --host 0.0.0.0 --port 8000
```

On first run, `yolov8n.pt` and the CLIP ViT-B/32 weights are downloaded automatically (~50 MB total).

Health check:
```bash
curl http://localhost:8000/health
# {"status":"ok","models_ready":true,"subscribers":{}}
```

### 5. Start the frontend

```bash
cd frontend
npm install
npm run dev
# → http://localhost:3000
```

---

## Environment Variables

Create `backend/.env`:

```env
# Database
DATABASE_URL=postgresql+asyncpg://postgres:postgres@localhost/campus_db

# Redis
REDIS_URL=redis://localhost:6379

# Video source directory (absolute path)
VIDEOS_BASE=/absolute/path/to/campus-surveillance/videos

# Frame store for CLIP search results
FRAME_STORE=/absolute/path/to/campus-surveillance/backend/frame_store

# Display frame rate (inference runs at ~0.3 Hz independently)
TARGET_FPS=10

# JPEG compression quality (1–100)
JPEG_QUALITY=75

# Demand-driven inference — submit every frame (leave at 1)
INFER_EVERY_N_FRAMES=1

# After-hours window (UTC hours, 24-hour clock)
AFTER_HOURS_START=18
AFTER_HOURS_END=6

# Alert cooldowns (seconds) per track
ALERT_COOLDOWN_LOITERING=60.0
ALERT_COOLDOWN_RUNNING=10.0
ALERT_COOLDOWN_UNAUTHORIZED=120.0
ALERT_COOLDOWN_AFTER_HOURS=300.0

# CLIP re-ID quality filters
REID_MIN_CROP_PX=40
REID_MAX_ASPECT_RATIO=2.0
```

---

## Active Cameras

| Camera ID | Label | Source file |
|-----------|-------|-------------|
| cam01 | Main Entrance | `videos/normal/cam01.mp4` |
| cam02 | Hallway A | `videos/normal/cam02.mp4` |
| cam03 | Hallway B | `videos/normal/cam03.mp4` |
| cam04 | Library | `videos/normal/cam04.mp4` |
| cam11 | Main Gate | `videos/outdoor/cam11.avi` |
| cam15 | Sports Ground | `videos/normal/cam15.mp4` |
| cam19 | THREAT FEED | playlist — Fighting · Assault · Shoplifting |
| cam20 | INTRUSION FEED | playlist — Vandalism · Burglary · Stealing |

---

## Person Enrollment (CLIP Re-ID)

Re-ID requires at least one enrolled person. Enrolment extracts a CLIP embedding from representative crops and stores it in `registered_faces`.

### Option A — from the SHANGHAI_Test dataset

```bash
cd backend
uv run python scripts/enroll_from_frames.py \
    --auto \
    --metadata /path/to/SHANGHAI_Test/SHANGHAI_test.txt \
    --frames-base /path/to/SHANGHAI_Test/frames \
    --names "Alice,Bob,Charlie,David" \
    --roles "staff,student,student,staff" \
    --sample-every 5
```

### Option B — from a camera video

```bash
mkdir -p /tmp/cam01_frames
ffmpeg -i videos/normal/cam01.mp4 -vf fps=1 /tmp/cam01_frames/%04d.jpg

cd backend
uv run python scripts/enroll_from_frames.py \
    --frames-dir /tmp/cam01_frames \
    --name "Alice" \
    --role staff
```

### Hot-reload without restart

```bash
curl -X POST http://localhost:8000/faces/reload
```

### Box colour coding

| Colour | Meaning |
|--------|---------|
| Green | Enrolled person — recognised |
| Orange | Unknown person |
| Red | Active alert (loitering, running, after-hours) |

---

## API Reference

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Model readiness + subscriber counts |
| `WS` | `/ws/stream/{cam_id}` | WebSocket — annotated JPEG frames at 10 FPS |
| `GET` | `/analytics/summary` | Live visitor count, badge breakdown, centroids |
| `GET` | `/analytics/incidents` | Recent alerts (newest first, up to 100) |
| `GET` | `/analytics/responders` | Response-team roster |
| `POST` | `/faces/reload` | Hot-reload CLIP embeddings from PostgreSQL |
| `GET` | `/search?query=&limit=` | Natural language frame search via CLIP + pgvector |
| `GET` | `/frames/{cam_id}/{file}` | Static JPEG frames for search results |

### WebSocket message schema

```json
{
  "cam_id": "cam01",
  "frame": "<base64-encoded JPEG>",
  "tracks": [
    {
      "track_id": 7,
      "bbox": [x, y, w, h],
      "face_status": "KNOWN",
      "face_name": "Alice",
      "face_confidence": 0.91,
      "behavior": null
    }
  ],
  "alerts": [
    {
      "type": "LOITERING",
      "severity": "MEDIUM",
      "track_id": 3,
      "cam_id": "cam01",
      "timestamp": "2026-04-08T14:32:00Z"
    }
  ],
  "zone_counts": { "entrance": 2, "corridor": 1 },
  "timestamp": "2026-04-08T14:32:00.123Z"
}
```

---

## Frontend Pages

### `/` — Live Feed

Eight camera streams in a 2×4 grid. Each tile connects independently via WebSocket; the subscriber registry ensures idle cameras consume no CPU.

### `/analytics` — Dashboard

- **KPI cards** — total visitors, authorised vs unknown count, active alerts
- **Heatmap** — inferno colourmap rendered on `<canvas>` from track centroids aggregated across all cameras
- **Incident timeline** — scrollable alert log with severity badges, persisted in `sessionStorage`
- **Response team** — live roster of security staff with assignment status

### `/search` — Natural Language Search

Type a description (e.g. *"person in red jacket near entrance"*) and the backend encodes it with CLIP, queries `clip_embeddings` by cosine distance, and returns matching frame thumbnails with camera label and timestamp.

---

## Build Phases

All eight phases were verified against a stop condition before the next phase began.

| Phase | Feature | Stop condition | Status |
|-------|---------|----------------|--------|
| 1 | Raw YOLO bounding boxes — all 8 cameras | Green boxes on all persons, no crashes after 60 s | ✅ Done |
| 2 | Stable track IDs (`_CentroidTracker` + Hungarian) | Same person keeps same ID across frames for 60 s | ✅ Done |
| 3 | Extrapolation with integral velocity decay | Boxes move smoothly; stop floating when person stops | ✅ Done |
| 4 | Demand-driven inference (subscriber registry) | CPU drops ~50 % → ~2 % when all tabs are closed | ✅ Done |
| 5 | CLIP Re-ID + behaviour alerts | Enrolled → GREEN + name; unknown → ORANGE; after-hours → HIGH alert | ✅ Done |
| 6 | Database persistence | `SELECT COUNT(*) FROM incidents` returns rows after 2 min | ✅ Done |
| 7 | Analytics dashboard | Dashboard updates with live data | ✅ Done |
| 8 | Natural language CLIP search | Query returns relevant frames with camera + timestamp | ✅ Done |

---

## Key Design Decisions

| Decision | Choice | Reason |
|----------|--------|--------|
| Tracker | `_CentroidTracker` + Hungarian | ByteTrack's Kalman filter assumes 25–30 fps; at ~0.3 Hz predicted positions diverge and stationary persons lose their ID |
| Tracker assignment | Hungarian (`linear_sum_assignment`) | Greedy nearest-neighbour causes ID swaps when two people pass near each other's previous positions |
| Extrapolation | Integral velocity decay | `disp = vx × df × (1 − df/2M)` — boxes decelerate to a stop instead of floating off-screen |
| Re-ID model | CLIP ViT-B/32 full-body crop | Same-person crops score 0.85–0.95; different persons < 0.75; no face detection needed at typical surveillance heights |
| Inference parallelism | 1 worker thread | Serial is faster than concurrent YOLO calls on a 4-core CPU — no contention, no context switching |
| Annotation | Server-side JPEG burn-in | Eliminates client-side coordinate-space bugs when the browser rescales the image |
| Idle behaviour | Skip H.264 decode entirely | Not just YOLO — the full decode+resize pipeline is gated behind the subscriber check |
| `imgsz` | 416 | Measured faster than 320 on i5-8250U; lower resolution is not always faster on CPU |

---

## Dataset

| Source | Content | Cameras |
|--------|---------|---------|
| ShanghaiTech Campus | Indoor normal scenes | cam01–04, cam15 |
| CUHK Avenue | Outdoor gate / parking | cam11 |
| UCF-Crime (subset) | Anomaly simulation | cam19 (Fighting · Assault · Shoplifting), cam20 (Vandalism · Burglary · Stealing) |

Videos are pre-converted and stored in `videos/`:

```
videos/
├── normal/    cam01–04, cam15 — MP4
├── outdoor/   cam11 — AVI
└── anomaly/   66 MP4s across 6 categories (looped as playlists for cam19 + cam20)
```
