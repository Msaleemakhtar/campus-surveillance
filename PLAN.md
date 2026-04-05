# Campus Surveillance AI — Robust Rebuild Plan

## Why Starting Fresh is the Right Call

Every iteration so far fixed symptoms. The root cause is a design/hardware mismatch baked in from Phase 1:

**Measured reality on i5-8250U:**
```
YOLO imgsz=320:  avg 352ms  (code assumed 24ms — 14× off)
YOLO imgsz=416:  avg 291ms  (actually FASTER than 320 on this CPU)
YOLO imgsz=640:  avg 534ms
cap.read():      avg 2.5ms  ✓
JPEG encode:     avg 2.1ms  ✓
```

**Consequence:**
- 10 cameras @ 291ms = 2.91s per full cycle
- Every extrapolation cap, ByteTrack parameter, and FPS setting was tuned against fictional numbers
- Each "fix" piled complexity onto a timing foundation that was never right

**What IS working (keep it):**
- FastAPI + uvicorn structure
- Redis pub/sub pipeline
- PostgreSQL schema (7 tables)
- Frontend: WebSocket + canvas rendering, all components correct
- Video dataset (all 640×480, verified)

**What to rebuild:**
- The inference + pipeline layer only — with correct timing assumptions from day one

---

## Hardware Constraints (Non-Negotiable)

| Metric | Value | Design Implication |
|--------|-------|-------------------|
| YOLO per call | 291ms avg | Max ~3.4 inferences/sec total |
| Cameras watched simultaneously | 3 → 0.87s cycle | ~1.1 Hz per camera |
| Acceptable display FPS | 10 fps | No inference every frame |
| BB update rate | ~1 Hz per camera | Show stale boxes between updates |
| PyTorch threads | 4 (not 6) | Leave 4 cores for OS + asyncio |

---

## Architecture (Final)

```
Video file  ──read──▶  CameraPipeline (asyncio.Task per camera)
                            │
                            ├── Frame display path (10 fps):
                            │     annotate(last_known_tracks) → JPEG → Redis → WS
                            │
                            └── Inference path (demand-driven, ~1 Hz):
                                  has_subscriber? → submit_frame() → YOLO worker thread
                                      │
                                  InferResponse → extrapolator.update() → _last_result
```

**Key decisions locked in:**
1. Server-side annotation (BBs burned into JPEG before sending — zero coordinate space bugs)
2. Demand-driven YOLO (only watched cameras; unwatched get 30s heartbeat)
3. Display rate (10 fps) decoupled from inference rate (~1 Hz)
4. Extrapolation only after raw detection is verified working (Phase 3, not Phase 1)
5. imgsz=416 (measurably faster than 320 on this CPU; more detections)
6. PyTorch 4 threads (not 6)

---

## Phase Plan

### Phase 0 — Baseline Verification ✅ DONE
**Goal:** Know exactly what the hardware can deliver before writing any inference code.

Done. Numbers above are authoritative. Never assume — always measure.

---

### Phase 1 — One Camera, Raw YOLO, Verified BBs ⬜
**Goal:** Single camera streaming with correct bounding boxes on ALL visible persons. No tracking, no face recognition, no extrapolation.

**What to build:**
- `local_inference.py`: YOLO only. `_infer()` returns raw detections. No ByteTrack, no InsightFace, no behavior.
- `pipeline.py`: Simplest possible loop — read frame → if inference slot ready → run YOLO → draw boxes → JPEG → Redis.
- Single endpoint verified: `ws://localhost:8000/ws/stream/cam01`

**What NOT to build yet:** ByteTrack, extrapolation, face recognition, multi-camera, alerts.

**Config:**
```
TARGET_FPS = 10
INFER_EVERY_N_FRAMES = 1
YOLO_THREADS = 4
imgsz = 416
conf = 0.20
```

**Pipeline (dead simple):**
```python
# Per frame:
result = yolo(frame, imgsz=416, conf=0.20, classes=[0])
boxes = result[0].boxes.xyxy.cpu().numpy()
for x1,y1,x2,y2 in boxes:
    cv2.rectangle(frame, (int(x1),int(y1)), (int(x2),int(y2)), (0,255,0), 2)
publish_frame(cam_id, frame)
```

**Stop condition:** Open cam01 in browser. See green boxes on EVERY visible person. No misses. No coordinate offset. Accept 2-3 fps update rate. ✓

---

### Phase 2 — Stable Track IDs (ByteTrack) ⬜
**Goal:** Each person gets a persistent ID that doesn't change while they're in frame.

**What to add:** ByteTrack tracker per camera, calibrated for actual inference rate.

**Key parameters:**
```python
sv.ByteTrack(
    track_activation_threshold=0.20,  # matches YOLO conf
    frame_rate=1,                     # Kalman for 1-second transitions
    lost_track_buffer=5,              # 5s tolerance at ~1Hz
    minimum_matching_threshold=0.15,  # lenient for 1s gaps + movement
)
```

**Stop condition:** IDs stable for 60s per person. No ID flickering. ✓

---

### Phase 3 — Smooth Motion (Extrapolation) ⬜
**Goal:** Boxes move smoothly between YOLO results instead of jumping.

**What to add:** `TrackExtrapolator` per camera.

**Parameters:**
```python
_MAX_EXTRAP_S = 5.0   # covers 3-camera cycle (3 × 291ms = 0.87s) with generous margin
velocity_cap = 600    # px/s — walking person at 640px wide is ~100px/s max
```

**How to verify:** Watch walking person — box continues moving in the same direction between YOLO fires. Does NOT freeze.

**Stop condition:** Single camera, walking person, box smoothly follows. No freeze. ✓

---

### Phase 4 — Multi-Camera with Demand-Driven Inference ⬜
**Goal:** All cameras visible in grid. Watched cameras get smooth tracking; unwatched use 30s heartbeat.

**What to add:**
- Subscriber registry in `local_inference.py`
- Pipeline checks `has_subscriber(cam_id)` before submitting
- `stream.py` registers/unregisters on WS connect/disconnect

**Verify:**
- 1 camera open → ~291ms cycle
- 3 cameras open → ~870ms cycle (~1.1 Hz each)
- 10 cameras open → ~2.9s cycle (within 5s extrap cap)
- Unwatched cameras → inference log silent except every 30s

**Stop condition:** 4×5 grid visible, 3 cameras simultaneously smooth, unwatched idle. ✓

---

### Phase 5 — Face Recognition + Alerts ⬜
**Goal:** Known persons show name; unauthorized triggers alert; after-hours and loitering detected.

**What to add:**
- InsightFace `det_size=(160,160)`, cache every 20 detections
- Face registry loaded from `datasets/mock_faces/`
- Alerts: AFTER_HOURS_PRESENCE, UNAUTHORIZED_VISITOR, LOITERING, RUNNING

**Stop condition:** 3 registered faces. Authorized → green + name. Unknown → red. Alert fires within 5s. ✓

---

### Phase 6 — Database Persistence ⬜
**Goal:** All incidents, zone counts, detected persons written to Postgres.

**Stop condition:** After 2 min running, `SELECT COUNT(*) FROM incidents` returns rows. ✓

---

### Phase 7 — Dashboard UI ⬜
**Goal:** Analytics page shows live incident counts, attendance, zone congestion.

**What to add:**
- `analytics.py` endpoints
- `events.py` CRUD
- Frontend: `AnalyticsDashboard.tsx`, `IncidentTimeline.tsx`, `QuickResponsePanel.tsx`

**Stop condition:** Dashboard updates every 5s with real data. ✓

---

### Phase 8 — CLIP Search + Campus Map ⬜
**Goal:** NL query → matching frame thumbnails from pgvector.

**What to add:**
- CLIP embeddings stored every 2s per camera
- `search.py`: cosine similarity top-k
- Frontend: `NLSearchBar.tsx`, `CampusMap.tsx`

**Stop condition:** Query returns relevant frames with camera + timestamp. ✓

---

## Implementation Rules (Never Violate)

1. **Never start next phase until stop condition of current phase is verified in browser**
2. **Measure inference time after every change** — run timing benchmark before claiming any number
3. **One inference worker thread** — two workers thrash CPU cache on this hardware
4. **PyTorch 4 threads** — `torch.set_num_threads(4)`, not `cpu_count - 2`
5. **imgsz=416** — measurably faster than 320 on this specific CPU
6. **conf=0.20 must match track_activation_threshold=0.20** — always keep these equal
7. **No extrapolation in Phase 1-2** — verify raw detection before adding motion prediction
8. **All videos are 640×480** — no coordinate scaling ever needed

---

## Files to Write (in phase order)

| Phase | File | Action |
|-------|------|--------|
| 1 | `app/core/local_inference.py` | Full rewrite: YOLO only, no ByteTrack |
| 1 | `app/core/pipeline.py` | Full rewrite: simple loop |
| 1 | `app/core/config.py` | Set correct defaults |
| 1 | `.env` | TARGET_FPS=10, INFER_EVERY_N_FRAMES=1 |
| 2 | `app/core/local_inference.py` | Add ByteTrack |
| 3 | `app/core/extrapolator.py` | Add/rewire extrapolator |
| 3 | `app/core/pipeline.py` | Wire extrapolator |
| 4 | `app/core/local_inference.py` | Add subscriber registry |
| 4 | `app/api/stream.py` | Register/unregister on connect/disconnect |
| 4 | `app/core/pipeline.py` | Demand-driven submit |
| 5 | `app/core/local_inference.py` | Add InsightFace + alerts |
| 6 | `app/core/pipeline.py` | DB writes |
| 7 | `app/api/analytics.py` | Dashboard endpoints |
| 8 | `app/api/search.py` | CLIP search |

**Keep unchanged:** `annotator.py`, `zones.py`, `heatmap.py`, `schemas.py`, `stream.py`, `main.py`, all frontend files, all DB migration files.
