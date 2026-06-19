import sys

from predictivesystem.logging.logger import logging
from predictivesystem.exception.exception import PredicitiveMaintainanceException

from predictivesystem.entity.config_entity import DataLoadingConfig
from predictivesystem.entity.artifact_entity import DataTransformationArtifact

from predictivesystem.database.database_manager import DatabaseManager

import pandas as pd

class DataLoader:
    def __init__(self, data_transformation_artifact : DataTransformationArtifact, data_loading_config : DataLoadingConfig):
        self.data_transformation_artifact = data_transformation_artifact
        self.data_loading_config = data_loading_config
        self.database_manager = DatabaseManager(config=self.data_loading_config.database_config)
    
    def _load_dataframe(self, df : pd.DataFrame, table_name : str) -> None:
        try:
            columns = df.columns.to_list()

            query = f"""
            INSERT IGNORE INTO {table_name} ({','.join(columns)}) VALUES ({','.join(['%s'] * len(columns))})
            """

            records = list(df.itertuples(index=False, name=None))

            self.database_manager.execute_many(query=query, params=records)

            logging.info(f'Loaded {len(records)} into {table_name}')
        
        except Exception as e:
            raise PredicitiveMaintainanceException(e, sys)
        
    def initiate_data_loading(self) -> None:
        try:
            if self.data_transformation_artifact.validation_status == "PASS":
                logging.info("Started Data Loading")

                self.database_manager.connect()

                train_features = pd.read_csv(self.data_transformation_artifact.transformed_train_data_path)
                test_features = pd.read_csv(self.data_transformation_artifact.transformed_test_data_path)
                test_targets = pd.read_csv(self.data_transformation_artifact.transformed_test_target_path)

                self._load_dataframe(df=train_features, table_name=self.data_loading_config.train_features_table_name)
                self._load_dataframe(df=test_features, table_name=self.data_loading_config.test_features_table_name)
                self._load_dataframe(df=test_targets, table_name=self.data_loading_config.test_targets_table_name)

                self.database_manager.commit()

        except Exception as e:
            self.database_manager.rollback()
            raise PredicitiveMaintainanceException(e, sys)
        
        finally:
            self.database_manager.close()
            logging.info("Closed Database")