import sys
from predictivesystem.logging.logger import logging
from predictivesystem.exception.exception import PredicitiveMaintainanceException

from predictivesystem.entity.config_entity import ConfigurationManager, DataExtractionConfig
from predictivesystem.entity.artifact_entity import DataExtractionArtifact

from predictivesystem.components.etl.data_extraction import DataExtraction

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
    
    def run_pipeline(self):
        try:
            logging.info("Initiating ETL Process")
            data_extraction_artifact = self.start_data_extraction()
        
        except Exception as e:
            raise PredicitiveMaintainanceException(e,sys)

if __name__ == '__main__':
    pipeline = ETLPipeline()
    pipeline.run_pipeline()