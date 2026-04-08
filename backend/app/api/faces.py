import asyncio
import logging

from fastapi import APIRouter

from app.core import local_inference

log = logging.getLogger(__name__)
router = APIRouter(prefix="/faces", tags=["faces"])


@router.get("")
async def list_faces() -> dict:
    """Return names of all currently registered persons in the in-memory face DB."""
    fr = local_inference._face_recognizer
    if fr is None:
        return {"registered": [], "count": 0}
    with fr._lock:
        names = [name for name, _ in fr._db]
    return {"registered": names, "count": len(names)}


@router.post("/reload")
async def reload_faces() -> dict:
    """
    Re-query registered_faces from PostgreSQL and rebuild in-memory embeddings.
    Call this after enrolling or updating a person in the database.
    """
    log.info("POST /faces/reload — rebuilding face DB")
    count = await asyncio.to_thread(local_inference.reload_faces)
    return {"registered": count, "message": f"Face DB reloaded — {count} person(s) enrolled"}
