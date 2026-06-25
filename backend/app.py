from fastapi import FastAPI
from contextlib import asynccontextmanager

from startup import initialize_backend

from routers.batch_prediction_router import router as batch_router
from routers.stream_prediction_router import router as stream_router
from routers.websocket_router import router as websocket_router
from routers.engine_state_router import router as engine_state_router

@asynccontextmanager
async def lifespan(app: FastAPI):
    await initialize_backend(app)
    yield

app = FastAPI(
    title="Aircraft Engine Predictive Maintainance API",
    description="Hybrid LSTM + XGBoost based Remaining Useful Life and Failure Prediction System",
    version="1.0.0",
    lifespan=lifespan
)

app.include_router(batch_router)
app.include_router(stream_router)
app.include_router(websocket_router)
app.include_router(engine_state_router)

@app.get("/")
async def root():
    return {
        "message" : "Aircraft Engine Predictive Maintainance API",
        "model_version" : app.state.metadata.get("version")
    }

@app.get("/health")
async def root():
    return {
        "status" : "healthy",
        "model_loaded" : hasattr(app.state,"hybrid_model"),
        "model_version" : app.state.metadata.get("version")
    }