from typing import Any
import requests

from utils.constants import (
    BASE_API_URL,
    ROOT_ENDPOINT,
    HEALTH_ENDPOINT,
    ENGINES_ENDPOINT,
    STREAM_PREDICTION_ENDPOINT,
    BATCH_PREDICTION_ENDPOINT
)

class APIClient:
    def __init__(self):
        self.base_url = BASE_API_URL
        self.timeout = 30
    
    def _get(self, endpoint: str) -> dict:
        response = requests.get(self.base_url + endpoint, timeout=self.timeout)
        response.raise_for_status()

        return response.json()
    
    def _post(self, endpoint: str, *, json: dict | None = None, files: dict | None = None):
        response = requests.post(
            url=self.base_url + endpoint,
            json=json,
            files=files,
            timeout=self.timeout
        )

        response.raise_for_status()

        return response.json()
    
    def get_root(self):
        return self._get(ROOT_ENDPOINT)
    
    def get_health(self):
        return self._get(HEALTH_ENDPOINT)
    
    def get_all_engines(self):
        return self._get(ENGINES_ENDPOINT)
    
    def get_engine(self, engine_id: int):
        return self._get(f'{ENGINES_ENDPOINT}/{engine_id}')
    
    def get_engine_history(self, engine_id: int):
        return self._get(f'{ENGINES_ENDPOINT}/{engine_id}/history')
    
    def get_engine_details(self, engine_id: int):
        return self._get(f"{ENGINES_ENDPOINT}/{engine_id}/details")
    
    def batch_prediction(self, uploaded_file):
        file = {
            "file" : (uploaded_file.name, uploaded_file, "text/csv")
        }

        return self._post(BATCH_PREDICTION_ENDPOINT, files=file)
    
    def stream_prediction(self, payload: dict[str, Any]):
        return self._post(STREAM_PREDICTION_ENDPOINT, json=payload)
    
api_client = APIClient()