import sys

from predictivesystem.logging.logger import logging
from predictivesystem.exception.exception import PredictiveMaintenanceException

from predictivesystem.entity.config_entity import (
    ConfigurationManager, 
    DatabaseConfig, 
    DataIngestionConfig,
    DataValidationConfig,
    ModelTrainingConfig,
    ModelFinalizingConfig
)
from predictivesystem.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact, ModelTrainerArtifact, ModelFinalizerArtifact

from predictivesystem.components.ml.data_ingestion import DataIngestion
from predictivesystem.components.ml.data_validation import DataValidation
from predictivesystem.components.ml.model_training import ModelTrainer
from predictivesystem.components.ml.model_finalizer import ModelFinalizer

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
            raise PredictiveMaintenanceException(e, sys)
    
    def start_data_validation(self, data_ingestion_artifact: DataIngestionArtifact) -> DataValidationArtifact:
        try:
            data_validation_config = DataValidationConfig(configuration_manager=self.config_manager)
            data_validation = DataValidation(data_ingestion_artifact=data_ingestion_artifact, data_validation_config=data_validation_config)
            data_validation_artifact = data_validation.initiate_data_validation()

            return data_validation_artifact
        
        except Exception as e:
            raise PredictiveMaintenanceException(e, sys)
    
    def start_model_training(self, data_validation_artifact: DataValidationArtifact) -> ModelTrainerArtifact:
        try:
            model_trainer_config = ModelTrainingConfig(configuration_manager=self.config_manager)
            model_trainer = ModelTrainer(data_validation_artifact=data_validation_artifact, model_training_config=model_trainer_config)
            model_trainer_artifact = model_trainer.initiate_model_training()

            return model_trainer_artifact

        except Exception as e:
            raise PredictiveMaintenanceException(e, sys)
    
    def start_model_finalization(self, model_trainer_artifact: ModelTrainerArtifact) -> ModelFinalizerArtifact:
        try:
            model_finalizer_config = ModelFinalizingConfig(configuration_manager=self.config_manager)
            model_finalizer = ModelFinalizer(model_trainer_artifact=model_trainer_artifact, model_finalizing_config=model_finalizer_config)
            model_finalizer_artifact = model_finalizer.initiate_model_finalizing()

            return model_finalizer_artifact

        except Exception as e:
            raise PredictiveMaintenanceException(e, sys)
    
    def run_pipeline(self):
        try:
            logging.info("Starting Training Pipeline")
            data_ingestion_artifact = self.start_data_ingestion()
            data_validation_artifact = self.start_data_validation(data_ingestion_artifact)
            model_trainer_artifact = self.start_model_training(data_validation_artifact)
            model_finalizer_artifact = self.start_model_finalization(model_trainer_artifact)

        except Exception as e:
            raise PredictiveMaintenanceException(e, sys)

if __name__ == '__main__':
    pipeline = TrainingPipeline()
    pipeline.run_pipeline()