import sys
from predictivesystem.logging.logger import logging
from predictivesystem.exception.exception import PredicitiveMaintainanceException

from predictivesystem.entity.config_entity import ConfigurationManager, DataExtractionConfig, DataTransformationConfig
from predictivesystem.entity.artifact_entity import DataExtractionArtifact, DataTransformationArtifact

from predictivesystem.components.etl.data_extraction import DataExtraction
from predictivesystem.components.etl.data_transformation import DataTransformation

class ETLPipeline:
    def __init__(self):
        self.config_manager = ConfigurationManager()
    
    def start_data_extraction(self) -> DataExtractionArtifact:
        try:
            data_extraction_config = DataExtractionConfig(configuration_manager = self.config_manager)
            data_extraction = DataExtraction(data_extraction_config = data_extraction_config)
            data_extraction_artifact = data_extraction.initiate_data_extraction()

            return data_extraction_artifact

        except Exception as e:
            raise PredicitiveMaintainanceException(e, sys)
    
    def start_data_transformation(self, data_extraction_artifact : DataExtractionArtifact) -> DataTransformationArtifact:
        try:
            data_transformation_config = DataTransformationConfig(configuration_manager = self.config_manager)
            data_transformation = DataTransformation(data_extraction_artifact = data_extraction_artifact, data_transformation_config = data_transformation_config)
            data_transformation_artifact = data_transformation.initiate_data_transformation()

            return data_transformation_artifact
        
        except Exception as e:
            raise PredicitiveMaintainanceException(e, sys)
    
    def run_pipeline(self):
        try:
            logging.info("Initiating ETL Process")
            data_extraction_artifact = self.start_data_extraction()
            data_transformation_artifact = self.start_data_transformation(data_extraction_artifact)
        
        except Exception as e:
            raise PredicitiveMaintainanceException(e,sys)

if __name__ == '__main__':
    pipeline = ETLPipeline()
    pipeline.run_pipeline()