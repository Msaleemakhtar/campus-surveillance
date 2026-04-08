from fastapi import APIRouter, Query, HTTPException
from sqlalchemy import select

from app.core import local_inference
from app.db.models import ClipEmbedding
from app.db.postgres import AsyncSessionLocal

router = APIRouter(prefix="/search", tags=["search"])


@router.get("")
async def search(
    query: str = Query(..., min_length=1, description="Natural language search query"),
    limit: int = Query(default=10, ge=1, le=50),
) -> dict:
    """
    Natural language frame search via CLIP + pgvector cosine similarity.
    Returns top-k frames ranked by visual similarity to the text query.
    """
    emb = local_inference.encode_text_query(query)
    if emb is None:
        raise HTTPException(status_code=503, detail="CLIP model not ready")

    async with AsyncSessionLocal() as session:
        rows = (
            await session.execute(
                select(ClipEmbedding)
                .where(ClipEmbedding.frame_path.isnot(None))
                .order_by(ClipEmbedding.embedding.cosine_distance(emb.tolist()))
                .limit(limit)
            )
        ).scalars().all()

    return {
        "query": query,
        "results": [
            {
                "cam_id": r.camera_id,
                "timestamp": r.timestamp.isoformat(),
                "frame_url": f"/frames/{r.camera_id}/{r.frame_path.split('/')[-1]}" if r.frame_path else None,
            }
            for r in rows
        ],
    }
