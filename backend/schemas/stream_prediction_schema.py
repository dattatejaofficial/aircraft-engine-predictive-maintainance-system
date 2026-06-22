from pydantic import BaseModel
from typing import Optional

class StreamPredictionRequest(BaseModel):
    engine_id : int
    sensor_2 : float
    sensor_3 : float
    sensor_4 : float
    sensor_7 : float
    sensor_8 : float
    sensor_9 : float
    sensor_11 : float
    sensor_12 : float
    sensor_13 : float
    sensor_14 : float
    sensor_15 : float
    sensor_17 : float
    sensor_20 : float
    sensor_21 : float
    cycle : int

class StreamPredictionResponse(BaseModel):
    status: str
    engine_id: int

    current_sequence_size: Optional[int] = None
    required_sequence_size: Optional[int] = None
    remaining_records: Optional[int] = None

    predicted_rul: Optional[int] = None
    lstm_failure_prediction: Optional[int] = None
    classifier_prediction: Optional[int] = None
    classifier_probability: Optional[int] = None