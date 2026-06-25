from utils.feature_processor import FeatureProcessor
from utils.sequence_buffer import SequenceBuffer
from utils.alert_generator import AlertGenerator
from utils.engine_state_store import engine_state_store
from websocket.connection_manager import manager

class StreamPredictionService:
    @staticmethod
    async def predict(*, engine_id: int, features: dict, model, scaler, sequence_buffer : SequenceBuffer) -> dict:
        row = {
            "engine_id" : engine_id,
            **features
        }
        sequence_buffer.add_record(engine_id=engine_id, row=row)

        if not sequence_buffer.is_ready(engine_id):
            response =  {
                            "status" : "waiting",
                            "engine_id" : engine_id,
                            "current_sequence_size" : sequence_buffer.current_size(engine_id)
                        }

            engine_state_store.update(engine_id, response)

            return response
        
        if not sequence_buffer.has_baseline_map(engine_id):
            baseline_map = FeatureProcessor.create_baseline_map(record=features, scaler=scaler)
            sequence_buffer.set_baseline_map(engine_id, baseline_map)
        
        sequence_df = sequence_buffer.get_sequence(engine_id)
        cycle = sequence_buffer.current_size(engine_id)

        baseline_map = sequence_buffer.get_baseline_map(engine_id)
        
        prepared_data = FeatureProcessor.prepare_stream_data(df=sequence_df, scaler=scaler, baseline_map=baseline_map)
        
        prediction_df = model.predict(prepared_data)
        prediction = prediction_df.iloc[0].to_dict()
        
        predicted_rul = prediction['predicted_rul']
        classifier_probability = prediction['classifier_probability']

        alert = AlertGenerator.generate(rul = predicted_rul, failure_probability=classifier_probability)
        print("STEP 5")

        response = {
            "status" : 'ready',
            "engine_id" : engine_id,
            "cycle" : cycle,
            "rul" : predicted_rul,
            "failure_probabilty" : classifier_probability,
            "alert" : alert
        }

        engine_state_store.update(engine_id, response)

        await manager.broadcast({
            "event" : "engine_update",
            "data" : response
        })

        return response