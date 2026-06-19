import os
import sys
import json
from typing import Literal
import pickle

from predictivesystem.logging.logger import logging
from predictivesystem.exception.exception import PredicitiveMaintainanceException

from predictivesystem.entity.config_entity import DataTransformationConfig
from predictivesystem.entity.artifact_entity import DataTransformationArtifact, DataExtractionArtifact

from predictivesystem.utils.main_utils import read_csv_file, read_yaml_file
from predictivesystem.utils.feature_engineering import FeatureExtraction

import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler, MinMaxScaler

class DataTransformation:
    def __init__(self, data_extraction_artifact : DataExtractionArtifact, data_transformation_config : DataTransformationConfig):
        self.data_extraction_artifact = data_extraction_artifact
        self.data_transformation_config = data_transformation_config

        self.schema = read_yaml_file(self.data_transformation_config.data_schema_path)
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
            "target" : {
                "errors" : {},
                "warnings" : {}
            }
        }
    
    def _validate_schema(self, df: pd.DataFrame, dataset_name : Literal['train','test','target'], column_type : Literal['feature_columns','target_column']) -> None:
        try:
            expected_columns = set(self.schema[column_type].keys())
            actual_columns = set(df.columns)

            missing_columns = expected_columns - actual_columns
            extra_columns = actual_columns - expected_columns

            if missing_columns:
                self.report[dataset_name]['errors']['missing_columns'] = list(missing_columns)
            
            if extra_columns:
                self.report[dataset_name]['warnings']['extra_columns'] = list(extra_columns)

        except Exception as e:
            raise PredicitiveMaintainanceException(e, sys)
    
    def _validate_dtypes(self, df: pd.DataFrame, dataset_name : Literal['train','test','target'], column_type : Literal['feature_columns','target_column']) -> None:
        try:
            dtype_errors = {}

            for col, rules in self.schema[column_type].items():
                if col not in df.columns:
                    continue

                expected_dtype = rules.get('dtype')
                actual_dtype = str(df[col].dtype)

                if actual_dtype != expected_dtype:
                    dtype_errors[col] = {
                        'expected' : expected_dtype,
                        'actual' : actual_dtype
                    }
            
            if dtype_errors:
                self.report[dataset_name]['errors']['dtype_mismatch'] = dtype_errors

        except Exception as e:
            raise PredicitiveMaintainanceException(e, sys)
    
    def _validate_data(self, df: pd.DataFrame, dataset_name : Literal['train','test','target'], column_type : Literal['feature_columns','target_column']) -> None:
        try:
            self._validate_schema(df, dataset_name, column_type)

            if 'missing_columns' in self.report[dataset_name]['errors']:
                return
            
            self._validate_dtypes(df, dataset_name, column_type)
        
        except Exception as e:
            raise PredicitiveMaintainanceException(e, sys)
    
    def _clean_data(self, df : pd.DataFrame, drop_duplicates=True) -> pd.DataFrame:
        try:
            before = len(df)
            cleaned_df = df.copy().dropna()
            
            if drop_duplicates:
                cleaned_df = cleaned_df.drop_duplicates()

            after = len(cleaned_df)

            logging.info(f"Rows removed: {before - after}")

            return cleaned_df
        
        except Exception as e:
            raise PredicitiveMaintainanceException(e, sys)
    
    def _feature_engineering(self, df : pd.DataFrame, scaler : MinMaxScaler, fit : bool) -> pd.DataFrame:
        try:
            feature_extractor = FeatureExtraction(
                df = df,
                exclude_columns = self.data_transformation_config.exclude_columns,
                feature_columns = self.data_transformation_config.feature_columns,
                degrade_up = self.data_transformation_config.degrade_up_columns,
                degrade_down = self.data_transformation_config.degrade_down_columns,
                engine_col = self.data_transformation_config.engine_column,
                cycle_col = self.data_transformation_config.cycle_column,
                fit = fit,
                scaler = scaler
            )

            features_df = feature_extractor.initiate_feature_extraction()
            return features_df
        
        except Exception as e:
            raise PredicitiveMaintainanceException(e, sys)
    
    def _transform_target_data(self, train_df : pd.DataFrame, target_df : pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, StandardScaler]:
        try:
            engine_col = self.data_transformation_config.engine_column
            cycle_col = self.data_transformation_config.cycle_column
            target_col = self.data_transformation_config.target_column
            target_cap = self.data_transformation_config.target_cap
            target_cap_column = f'{target_col}_capped'

            target_scaler = StandardScaler()

            max_cycle = train_df.groupby(engine_col)[cycle_col].transform('max')
            train_df[target_col] = max_cycle - train_df[cycle_col]
            
            train_df[target_cap_column] = np.minimum(train_df[target_col], target_cap)

            train_df[f'{target_col}_scaled'] = target_scaler.fit_transform(train_df[[target_cap_column]])

            train_df = train_df.drop(columns=[target_col, engine_col, cycle_col])
            
            logging.info(f'Added {target_col} to the Training Data, Capped and Scaled')

            target_df[engine_col] = np.arange(1, len(target_df) + 1)
            target_df[target_cap_column] = np.minimum(target_df[target_col], target_cap)
            target_df[f'{target_col}_scaled'] = target_scaler.transform(target_df[[target_cap_column]])
            
            logging.info(f'Capped the target with {target_cap} for Testing Data and Scaled it')

            return train_df, target_df, target_scaler

        except Exception as e:
            raise PredicitiveMaintainanceException(e, sys)
    
    def initiate_data_transformation(self) -> DataTransformationArtifact:
        try:
            logging.info("Initiating Data Transformation")

            train_data = read_csv_file(self.data_extraction_artifact.extracted_train_data_path)
            test_data = read_csv_file(self.data_extraction_artifact.extracted_test_data_path)
            test_target = read_csv_file(self.data_extraction_artifact.extracted_test_target_path)

            self._validate_data(train_data, 'train' ,column_type = 'feature_columns')
            logging.info(f'Validated Train Data')

            self._validate_data(test_data, 'test' ,column_type = 'feature_columns')
            logging.info(f'Validated Test Data')

            self._validate_data(test_target, 'target' ,column_type = 'target_column')
            logging.info(f'Validated Test Target Data')

            if any([self.report['train']['errors'], self.report['test']['errors'], self.report['target']['errors']]):
                self.report['status'] = 'FAIL'

            if self.report['status'] != 'FAIL':
                train_data = self._clean_data(train_data, drop_duplicates=True)
                logging.info(f'Cleaned Train Data')

                test_data = self._clean_data(test_data, drop_duplicates=True)
                logging.info(f'Cleaned Test Data')

                test_target = self._clean_data(test_target, drop_duplicates=False)
                logging.info(f'Cleaned Test Target Data')

                feature_scaler = MinMaxScaler()
                transformed_train_data = self._feature_engineering(df = train_data, scaler = feature_scaler, fit = True)
                transformed_test_data = self._feature_engineering(df = test_data, scaler = feature_scaler, fit = False)

                logging.info("Performed Feature Extraction on Training and Testing Data")

                transformed_train_data, test_target_data, target_scaler = self._transform_target_data(transformed_train_data, test_target)

                os.makedirs(self.data_transformation_config.transformed_data_dir, exist_ok=True)

                transformed_train_data.to_csv(self.data_transformation_config.transformed_train_data_path, index=False)
                
                transformed_test_data = transformed_test_data.drop(columns = [self.data_transformation_config.engine_column, self.data_transformation_config.cycle_column])
                transformed_test_data.to_csv(self.data_transformation_config.transformed_test_data_path, index=False)

                test_target_data.to_csv(self.data_transformation_config.transformed_test_target_path, index=False)

                logging.info("Saved the data")

                os.makedirs(os.path.dirname(self.data_transformation_config.feature_scaler_path), exist_ok=True)

                with open(self.data_transformation_config.feature_scaler_path,'wb') as file:
                    pickle.dump(feature_scaler, file)
                
                with open(self.data_transformation_config.target_scaler_path, 'wb') as file:
                    pickle.dump(target_scaler, file)
                
                logging.info("Saved the Scalers")

            os.makedirs(self.data_transformation_config.validation_report_dir, exist_ok=True)
            validation_report_path = self.data_transformation_config.validation_report_path
            logging.info("Saved the Validation Report")

            with open(validation_report_path,'w') as file:
                json.dump(self.report, file, indent=4)
            
            data_transformation_artifact = DataTransformationArtifact(
                validation_status=self.report['status'],
                validation_report_path=validation_report_path,
                transformed_train_data_path=self.data_transformation_config.transformed_train_data_path,
                transformed_test_data_path=self.data_transformation_config.transformed_test_data_path,
                transformed_test_target_path=self.data_transformation_config.transformed_test_target_path
            )
            
            return data_transformation_artifact

        except Exception as e:
            raise PredicitiveMaintainanceException(e, sys)