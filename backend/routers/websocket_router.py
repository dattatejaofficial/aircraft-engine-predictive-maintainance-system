import asyncio
from fastapi import APIRouter
from fastapi import WebSocket
from fastapi import WebSocketDisconnect

from websocket.connection_manager import manager

router = APIRouter(prefix='/ws',tags=['WebSocket'])

@router.websocket('/')
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)

    try:
        while True:
            # await websocket.receive_text()
            await asyncio.sleep(60)
    
    except WebSocketDisconnect:
        manager.disconnect(websocket)