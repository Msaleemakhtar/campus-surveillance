from fastapi import APIRouter, Query

from app.core import local_inference

router = APIRouter(prefix="/analytics", tags=["analytics"])

_RESPONDERS = [
    {"initials": "SP", "name": "Sarah Powers", "status": "ON_DUTY",    "color": "#1e40af"},
    {"initials": "JH", "name": "John Harris",  "status": "RESPONDING", "color": "#92400e"},
    {"initials": "ML", "name": "Mike Lawson",  "status": "ON_CALL",    "color": "#1e293b"},
]


@router.get("/summary")
async def analytics_summary() -> dict:
    """
    Aggregate live track data across all cameras.
    Calling this endpoint keeps inference running (analytics heartbeat).
    Returns visitor count, badge breakdown, and per-camera centroid list
    (used by the frontend to render the combined heatmap).
    """
    local_inference.analytics_heartbeat()
    return local_inference.get_analytics_summary()


@router.get("/incidents")
async def analytics_incidents(limit: int = Query(default=20, ge=1, le=100)) -> dict:
    """Most recent alerts across all cameras, newest first."""
    return {"incidents": local_inference.get_recent_incidents(limit)}


@router.get("/responders")
async def analytics_responders() -> dict:
    """Quick-response team roster (static for Phase 7)."""
    return {"responders": _RESPONDERS}
