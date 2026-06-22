import os
import json
from pathlib import Path
import shutil

from azure.storage.blob import BlobServiceClient

from config import settings

class BlobLoader:
    def __init__(self):
        self.client = BlobServiceClient.from_connection_string(settings.AZURE_MODEL_ARTIFACTS_CONNECTION_STRING)
        self.container_client = self.client.get_container_client(settings.AZURE_MODEL_ARTIFACTS_CONTAINER_NAME)
    
    def load_metadata(self):
        blob = self.container_client.get_blob_client(settings.MODEL_METADATA_BLOB)
        content = blob.download_blob().readall().decode('utf-8')
    
        return json.loads(content)
    
    def download_model(self, model_blob_path: str):
        local_dir = Path(settings.LOCAL_MODEL_DIR)

        if local_dir.exists():
            shutil.rmtree(local_dir)
        
        local_dir.mkdir(parents=True, exist_ok=True)

        blobs = self.container_client.list_blobs(name_starts_with=model_blob_path)

        for blob in blobs:
            relative_path = blob.name.replace(model_blob_path, "").lstrip("/")
            local_file = local_dir / relative_path
            local_file.parent.mkdir(parents=True, exist_ok=True)

            blob_client = self.container_client.get_blob_client(blob.name)

            with open(local_file, 'wb') as file:
                data = blob_client.download_blob().readall()
                file.write(data)
        
        return local_dir
    