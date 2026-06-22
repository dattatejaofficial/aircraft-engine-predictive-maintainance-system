import pandas as pd

from utils.feature_processor import FeatureProcessor

class BatchPredictionService:
    @staticmethod
    def predict(model, scaler, df : pd.DataFrame):
        prepared_data = FeatureProcessor.prepare_batch_data(df=df, scaler=scaler)
        
        engine_ids = prepared_data.pop('engine_ids')
        predictions = model.predict(prepared_data)
        predictions.insert(0, "engine_id", engine_ids)

        return predictions