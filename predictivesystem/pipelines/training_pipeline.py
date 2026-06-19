import sys

from predictivesystem.logging.logger import logging
from predictivesystem.exception.exception import PredicitiveMaintainanceException

from predictivesystem.entity.config_entity import (
    ConfigurationManager, 
    DatabaseConfig, 
    DataIngestionConfig
)
from predictivesystem.entity.artifact_entity import DataIngestionArtifact

from predictivesystem.components.ml.data_ingestion import DataIngestion

class TrainingPipeline:
    def __init__(self):
        self.config_manager = ConfigurationManager()
    
    def start_data_ingestion(self) -> DataIngestionArtifact:
        try:
            database_config = DatabaseConfig()
            data_ingestion_config = DataIngestionConfig(configuration_manager=self.config_manager, database_config=database_config)
            data_ingestion = DataIngestion(data_ingestion_config=data_ingestion_config)
            data_ingestion_artifact = data_ingestion.initiate_data_ingestion()

            return data_ingestion_artifact
        
        except Exception as e:
            raise PredicitiveMaintainanceException(e, sys)
    
    def run_pipeline(self):
        try:
            logging.info("Starting Training Pipeline")
            data_ingestion_artifact = self.start_data_ingestion()

        except Exception as e:
            raise PredicitiveMaintainanceException(e, sys)

if __name__ == '__main__':
    pipeline = TrainingPipeline()
    pipeline.run_pipeline()