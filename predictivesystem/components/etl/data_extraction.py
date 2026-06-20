import os
import sys

from predictivesystem.logging.logger import logging
from predictivesystem.exception.exception import PredictiveMaintenanceException

from predictivesystem.entity.config_entity import DataExtractionConfig
from predictivesystem.entity.artifact_entity import DataExtractionArtifact

import pandas as pd

class DataExtraction:
    def __init__(self, data_extraction_config : DataExtractionConfig):
        self.extraction_config = data_extraction_config
    
    def _extract_data(self, path : str, columns : list[str]) -> pd.DataFrame:
        try:
            data = pd.read_csv(path, sep=r'\s+', header=None, names = columns)
            return data
        except Exception as e:
            raise PredictiveMaintenanceException(e, sys)
    
    def initiate_data_extraction(self) -> DataExtractionArtifact:
        try:
            logging.info("Initiating Data Extraction Process")

            raw_train_data_path = self.extraction_config.train_data_path
            raw_test_data_path = self.extraction_config.test_data_path
            raw_test_target_path = self.extraction_config.test_target_path

            features_columns = self.extraction_config.feature_columns
            target_column = self.extraction_config.target_column

            extracted_data_dir = self.extraction_config.extracted_data_dir
            os.makedirs(extracted_data_dir, exist_ok=True)

            extracted_train_data = self._extract_data(raw_train_data_path, features_columns)
            extracted_train_data_path = self.extraction_config.extracted_train_data_path
            extracted_train_data.to_csv(extracted_train_data_path, index=False)

            extracted_test_data = self._extract_data(raw_test_data_path, features_columns)
            extracted_test_data_path = self.extraction_config.extracted_test_data_path
            extracted_test_data.to_csv(extracted_test_data_path, index=False)

            extracted_test_target_data = self._extract_data(raw_test_target_path, target_column)
            extracted_test_target_path = self.extraction_config.extracted_test_target_path
            extracted_test_target_data.to_csv(extracted_test_target_path, index=False)

            logging.info("Saved the Extracted Data")

            data_extraction_artifact = DataExtractionArtifact(
                extracted_train_data_path = extracted_train_data_path,
                extracted_test_data_path = extracted_test_data_path,
                extracted_test_target_path = extracted_test_target_path 
            )

            return data_extraction_artifact

        except Exception as e:
            raise PredictiveMaintenanceException(e,sys)