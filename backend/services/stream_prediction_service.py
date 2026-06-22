from utils.feature_processor import FeatureProcessor
from utils.sequence_buffer import SequenceBuffer

class StreamPredictionService:
    @staticmethod
    def predict(*, engine_id: int, features: dict, model, scaler, sequence_buffer : SequenceBuffer) -> dict:
        row = {
            "engine_id" : engine_id,
            **features
        }
        sequence_buffer.add_record(engine_id=engine_id, row=row)

        if not sequence_buffer.is_ready(engine_id):
            return {
                "status" : "waiting",
                "engine_id" : engine_id,
                "current_sequence_size" : sequence_buffer.current_size(engine_id)
            }
        
        if not sequence_buffer.has_baseline_map(engine_id):
            baseline_map = FeatureProcessor.create_baseline_map(record=features, scaler=scaler)
            sequence_buffer.set_baseline_map(engine_id, baseline_map)
        
        sequence_df = sequence_buffer.get_sequence(engine_id)

        baseline_map = sequence_buffer.get_baseline_map(engine_id)
        prepared_data = FeatureProcessor.prepare_stream_data(df=sequence_df, scaler=scaler, baseline_map=baseline_map)

        prediction = model.predict(prepared_data)

        return {
            "status" : 'ready',
            "engine_id" : engine_id,
            **prediction.iloc[0].to_dict()
        }