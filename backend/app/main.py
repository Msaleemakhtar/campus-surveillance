from __future__ import annotations

import asyncio
import concurrent.futures
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.api import analytics, faces, search, stream
from app.core import local_inference
from app.core.config import settings
from app.core.pipeline import start_pipelines, stop_pipelines
from app.db.redis_client import close_redis

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("Starting campus surveillance backend")

    loop = asyncio.get_running_loop()
    loop.set_default_executor(
        concurrent.futures.ThreadPoolExecutor(max_workers=4, thread_name_prefix="io")
    )

    log.info("Loading YOLO model...")
    await asyncio.to_thread(local_inference.load_models)
    log.info("Model ready — starting camera pipelines")

    start_pipelines()

    yield

    log.info("Shutting down...")
    await stop_pipelines()
    await close_redis()
    log.info("Shutdown complete")


app = FastAPI(title="Campus Surveillance API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(stream.router)
app.include_router(faces.router)
app.include_router(analytics.router)
app.include_router(search.router)

# Serve saved frame JPEGs for the search results UI
_frame_store = Path(settings.FRAME_STORE)
_frame_store.mkdir(parents=True, exist_ok=True)
app.mount("/frames", StaticFiles(directory=str(_frame_store)), name="frames")


@app.get("/health")
async def health() -> dict:
    return {
        "status": "ok",
        "models_ready": local_inference.models_ready(),
        "subscribers": local_inference.subscriber_counts(),
    }
