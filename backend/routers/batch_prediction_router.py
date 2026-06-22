import os
import uuid
import pandas as pd

from fastapi import APIRouter, UploadFile, File, HTTPException, Request, BackgroundTasks
from fastapi.responses import FileResponse

from services.batch_prediction_service import BatchPredictionService

router = APIRouter(prefix='/predict', tags=['Batch Prediction'])

@router.post('/batch')
async def predict_batch(request : Request, background_tasks : BackgroundTasks, file: UploadFile = File(...)):
    try:
        if not file.filename.endswith('.csv'):
            raise HTTPException(status_code=400, detail='Only CSV files are supported')
        
        df = pd.read_csv(file.file)

        predictions = BatchPredictionService.predict(model=request.app.state.hybrid_model, scaler=request.app.state.feature_scaler, df=df)

        output_dir = 'temp'
        os.makedirs(output_dir, exist_ok=True)

        output_file = os.path.join(output_dir, f"predictions_{uuid.uuid4().hex}.csv")
        pd.DataFrame(predictions).to_csv(output_file, index=False)

        background_tasks.add_task(
            lambda: os.path.exists(output_file) and os.remove(output_file)
        )        

        return FileResponse(
            path=output_file,
            media_type='text/csv',
            filename='predictions.csv'
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))