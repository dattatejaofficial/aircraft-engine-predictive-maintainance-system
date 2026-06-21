import os
import sys
import json
from dotenv import load_dotenv
from pathlib import Path
import shutil

load_dotenv()

from predictivesystem.logging.logger import logging
from predictivesystem.exception.exception import PredictiveMaintenanceException

from predictivesystem.entity.config_entity import ArtifactPublisherConfig
from predictivesystem.entity.artifact_entity import ModelFinalizerArtifact

from datetime import datetime, timezone

from mlflow.artifacts import download_artifacts

from azure.storage.blob import BlobServiceClient


class ArtifactPublisher:
    def __init__(self, model_finalizer_artifact : ModelFinalizerArtifact, artifact_publisher_config : ArtifactPublisherConfig):
        self.model_finalizer_artifact = model_finalizer_artifact
        self.artifact_publisher_config = artifact_publisher_config

        self.connection_string = os.getenv('AZURE_MODEL_ARTIFACTS_CONNECTION_STRING')
        if not self.connection_string:
            raise Exception('AZURE_MODEL_ARTIFACTS_CONNECTION_STRING is not set in Environment')
        
        self.model_blob_container = os.getenv('AZURE_MODEL_ARTIFACTS_CONTAINER_NAME')
        if not self.model_blob_container:
            raise Exception('AZURE_MODEL_ARTIFACTS_CONTAINER_NAME is not set in environment')
    
    def _upload_directory(self, blob_service_client : BlobServiceClient, local_dir : str, blob_prefix : str) -> None:
        try:
            container_client = blob_service_client.get_container_client(self.model_blob_container)

            for root, _, files in os.walk(local_dir):
                for file in files:
                    local_file_path = os.path.join(root, file)

                    relative_path = os.path.relpath(local_file_path, local_dir).replace('\\','/')

                    blob_path = f'{blob_prefix}/{relative_path}'

                    with open(local_file_path, 'rb') as data:
                        container_client.upload_blob(name=blob_path, data=data, overwrite=True)
            
            logging.info('Uploaded Artifacts into Blob')

        except Exception as e:
            raise PredictiveMaintenanceException(e, sys)
        
    def initiate_artifact_publishing(self) -> None:
        local_model_dir = None
        
        try:
            logging.info("Initiating Artifact Publishing")

            model_uri = self.model_finalizer_artifact.promoted_model_uri
            run_id = self.model_finalizer_artifact.promoted_run_id
            version = self.model_finalizer_artifact.promoted_model_version

            logging.info("Downloading MLFlow Artifacts")

            local_model_dir = download_artifacts(artifact_uri=model_uri)

            blob_service_client = BlobServiceClient.from_connection_string(self.connection_string)

            production_prefix = self.artifact_publisher_config.production_prefix
            model_blob_path = f'{production_prefix}/hybrid_model'

            logging.info("Uploading Model Artifacts to Azure Blob")

            self._upload_directory(blob_service_client=blob_service_client, local_dir=local_model_dir, blob_prefix=model_blob_path)

            deployment_metadata = {
                "run_id" : run_id,
                "model_uri" : model_uri,
                "model_path" : model_blob_path,
                "version" : version,
                "updated_at" : datetime.now(timezone.utc).isoformat()
            }

            container_client = blob_service_client.get_container_client(self.model_blob_container)
            container_client.upload_blob(name=f'{production_prefix}/deployment_metadata.json', data=json.dumps(deployment_metadata, indent=4), overwrite=True)
        
        except Exception as e:
            raise PredictiveMaintenanceException(e, sys)
        
        finally:
            if (local_model_dir and os.path.exists(local_model_dir)):
                shutil.rmtree(local_model_dir, ignore_errors=True)