import os
from dotenv import load_dotenv
from pydantic_settings import BaseSettings
load_dotenv()

class Settings(BaseSettings):
    AZURE_MODEL_ARTIFACTS_CONNECTION_STRING : str = os.getenv('AZURE_MODEL_ARTIFACTS_CONNECTION_STRING')
    AZURE_MODEL_ARTIFACTS_CONTAINER_NAME : str = os.getenv('AZURE_MODEL_ARTIFACTS_CONTAINER_NAME')

    MODEL_METADATA_BLOB : str = "production/deployment_metadata.json"
    LOCAL_MODEL_DIR : str = 'artifacts/current_model'

    SEQUENCE_LENGTH : int = 30

    EXCLUDE_COLUMNS : list = ['op_set_1','op_set_2','op_set_3','sensor_1','sensor_5','sensor_6','sensor_10','sensor_16','sensor_18','sensor_19']
    LSTM_FEATURES : list = ['sensor_2_norm', 'sensor_3_norm', 'sensor_4_norm', 'sensor_7_norm', 'sensor_8_norm', 'sensor_9_norm', 'sensor_11_norm', 'sensor_12_norm', 'sensor_13_norm', 'sensor_14_norm', 'sensor_15_norm', 'sensor_17_norm', 'sensor_20_norm', 'sensor_21_norm', 'health_index', 'cycle_count_norm']
    CLASSIFIER_EXCLUDE_FEATURES : list[str] = ['engine_id', 'cycle', 'cycle_count_norm']
    FEATURE_COLS : list[str] = ['sensor_2', 'sensor_3', 'sensor_4', 'sensor_7', 'sensor_8', 'sensor_9', 'sensor_11', 'sensor_12', 'sensor_13', 'sensor_14', 'sensor_15', 'sensor_17', 'sensor_20', 'sensor_21']
    DEGRADE_UP : list[str] = ['sensor_4_norm','sensor_11_norm','sensor_12_norm','sensor_15_norm','sensor_20_norm','sensor_21_norm']
    DEGRADE_DOWN : list[str] = ['sensor_2_norm','sensor_3_norm','sensor_7_norm','sensor_8_norm','sensor_9_norm','sensor_13_norm','sensor_14_norm','sensor_17_norm']

    
settings = Settings()