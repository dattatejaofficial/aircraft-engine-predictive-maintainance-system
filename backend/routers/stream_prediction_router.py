from fastapi import APIRouter, Request, HTTPException

from schemas.stream_prediction_schema import StreamPredictionRequest
from services.stream_prediction_service import StreamPredictionService

router = APIRouter(prefix='/predict', tags=['Stream Prediction'])

@router.post('/stream')
async def predict_stream(payload: StreamPredictionRequest, request: Request):
    try:
        features = payload.model_dump()
        engine_id = features.pop("engine_id")

        result = await StreamPredictionService.predict(
            engine_id=engine_id,
            features=features,
            model=request.app.state.hybrid_model,
            scaler=request.app.state.feature_scaler,
            sequence_buffer=request.app.state.sequence_buffer
        )
        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))