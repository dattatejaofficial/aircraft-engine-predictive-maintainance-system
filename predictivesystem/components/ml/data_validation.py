import os
import sys
import json
from typing import Literal

from predictivesystem.logging.logger import logging
from predictivesystem.exception.exception import PredicitiveMaintainanceException

from predictivesystem.entity.config_entity import DataValidationConfig
from predictivesystem.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact

from predictivesystem.utils.main_utils import read_csv_file, read_yaml_file

import pandas as pd

class DataValidation:
    def __init__(self, data_ingestion_artifact : DataIngestionArtifact, data_validation_config : DataValidationConfig):
        self.data_ingestion_artifact = data_ingestion_artifact
        self.data_validation_config = data_validation_config
        self.schema = read_yaml_file(self.data_validation_config.data_schema_path)
        self.report = {
            "status" : "PASS",
            "train" : {
                "errors" : {},
                "warnings" : {}
            },
            "test" : {
                "errors" : {},
                "warnings" : {}
            },
            "test_target" : {
                "errors" : {},
                "warnings" : {}
            }
        }
    
    def _validate_schema(self, df : pd.DataFrame, dataset_name : Literal['train','test','test_target']) -> None:
        try:
            expected_columns = set(self.schema[dataset_name].keys())
            actual_columns = set(df.columns)

            missing_columns = expected_columns - actual_columns
            extra_columns = actual_columns - expected_columns

            if missing_columns:
                self.report[dataset_name]['errors']['missing_columns'] = list(missing_columns)
            
            if extra_columns:
                self.report[dataset_name]['warnings']['extra_columns'] = list(extra_columns)

        except Exception as e:
            raise PredicitiveMaintainanceException(e, sys)

    def _validate_dtypes(self, df : pd.DataFrame, dataset_name : Literal['train','test','test_target']) -> None:
        try:
            dtype_errors = {}

            for col, rules in self.schema[dataset_name].items():
                if col not in df.columns:
                    continue

                expected_dtype = rules.get('dtype')
                actual_dtype = str(df[col].dtype)

                if expected_dtype != actual_dtype:
                    dtype_errors[col] = {
                        'expected' : expected_dtype,
                        'actual' : actual_dtype
                    }
            
            if dtype_errors:
                self.report[dataset_name]['errors']['dtype_mismatch'] = dtype_errors

        except Exception as e:
            raise PredicitiveMaintainanceException(e, sys)
    
    def _validate_data(self, df : pd.DataFrame, dataset_name : Literal['train','test','test_target']) -> None:
        try:
            logging.info(f'Validating {dataset_name} data')
            self._validate_schema(df, dataset_name)

            if 'missing_columns' in self.report[dataset_name]['errors']:
                return
            
            self._validate_dtypes(df, dataset_name)

        except Exception as e:
            raise PredicitiveMaintainanceException(e, sys)

    def initiate_data_validation(self) -> DataValidationArtifact:
        try:
            logging.info('Initiating Data Validation')

            train_data = read_csv_file(self.data_ingestion_artifact.ingested_train_data_path)
            test_data = read_csv_file(self.data_ingestion_artifact.ingested_test_data_path)
            test_target_data = read_csv_file(self.data_ingestion_artifact.ingested_test_target_path)

            self._validate_data(train_data,'train')
            logging.info('Validated Train Data')

            self._validate_data(test_data,'test')
            logging.info('Validated Test Data')

            self._validate_data(test_target_data,'test_target')
            logging.info('Validated Test Target Data')

            if any([self.report['train']['errors'], self.report['test']['errors'], self.report['test_target']['errors']]):
                self.report['status'] = 'FAIL'
            
            if self.report['status'] != 'FAIL':
                os.makedirs(self.data_validation_config.validated_data_dir, exist_ok=True)

                train_data.to_csv(self.data_validation_config.validated_train_features_path, index=False)
                test_data.to_csv(self.data_validation_config.validated_test_features_path, index=False)
                test_target_data.to_csv(self.data_validation_config.validated_test_targets_path, index=False)

                logging.info('Saved the Validated Data')
            
            os.makedirs(self.data_validation_config.validation_reports_dir, exist_ok=True)
            validation_report_path = self.data_validation_config.validation_report_path

            with open(validation_report_path,'w') as file:
                json.dump(self.report, file, indent=4)
            
            logging.info('Saved the Validation Report')

            data_validation_artifact = DataValidationArtifact(
                validation_status=self.report['status'],
                validation_report_path=validation_report_path,
                validated_train_data_path=self.data_validation_config.validated_train_features_path,
                validated_test_data_path=self.data_validation_config.validated_test_features_path,
                validated_test_target_path=self.data_validation_config.validated_test_targets_path
            )

            return data_validation_artifact

        except Exception as e:
            raise PredicitiveMaintainanceException(e, sys)