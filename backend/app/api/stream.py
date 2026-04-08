import asyncio
import logging

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from starlette.websockets import WebSocketState

from app.core.local_inference import add_subscriber, remove_subscriber
from app.core.pipeline import ALL_CAMERAS
from app.db.redis_client import get_redis

log = logging.getLogger(__name__)
router = APIRouter()


@router.websocket("/ws/stream/{cam_id}")
async def stream(websocket: WebSocket, cam_id: str) -> None:
    if cam_id not in ALL_CAMERAS:
        await websocket.close(code=4004, reason=f"Unknown camera: {cam_id}")
        return

    await websocket.accept()
    add_subscriber(cam_id)
    log.info("WS connected: %s", cam_id)

    redis = await get_redis()
    pubsub = redis.pubsub()
    await pubsub.subscribe(f"cam:{cam_id}")

    listen_task = asyncio.create_task(_listen(pubsub, websocket, cam_id))

    try:
        while websocket.client_state == WebSocketState.CONNECTED:
            try:
                await asyncio.wait_for(websocket.receive_text(), timeout=5.0)
            except asyncio.TimeoutError:
                pass
    except WebSocketDisconnect:
        pass
    except Exception as exc:
        log.error("WS handler error %s: %s", cam_id, exc)
    finally:
        listen_task.cancel()
        try:
            await listen_task
        except asyncio.CancelledError:
            pass
        remove_subscriber(cam_id)
        await pubsub.unsubscribe(f"cam:{cam_id}")
        try:
            await pubsub.aclose()
        except Exception:
            pass
        log.info("WS disconnected: %s", cam_id)


async def _listen(pubsub, websocket: WebSocket, cam_id: str) -> None:
    try:
        async for raw in pubsub.listen():
            if raw["type"] != "message":
                continue
            if websocket.client_state != WebSocketState.CONNECTED:
                break
            data = raw["data"]
            if isinstance(data, bytes):
                data = data.decode("utf-8")
            try:
                await websocket.send_text(data)
            except Exception:
                break
    except asyncio.CancelledError:
        pass
    except Exception as exc:
        log.error("WS listener error %s: %s", cam_id, exc)
