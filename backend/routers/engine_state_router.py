from fastapi import APIRouter
from fastapi import HTTPException

from utils.engine_state_store import engine_state_store

router = APIRouter(prefix="/engines", tags=["Engine States"])

@router.get("/")
async def get_all_engines():
    return {
        "total_engines" : len(engine_state_store.get_all()),
        "engines" : engine_state_store.get_all()
    }

@router.get("/{engine_id}")
async def get_engine(engine_id: int):
    if not engine_state_store.exists(engine_id):
        raise HTTPException(status_code=404, detail=f"Engine {engine_id} not found")
    
    return engine_state_store.get(engine_id)