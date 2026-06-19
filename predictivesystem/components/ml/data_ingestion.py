import os
import sys

from predictivesystem.logging.logger import logging
from predictivesystem.exception.exception import PredicitiveMaintainanceException

from predictivesystem.database.database_manager import DatabaseManager
from predictivesystem.entity.config_entity import DataIngestionConfig
from predictivesystem.entity.artifact_entity import DataIngestionArtifact

import pandas as pd

class DataIngestion:
    def __init__(self, data_ingestion_config : DataIngestionConfig):
        self.data_ingestion_config = data_ingestion_config
        self.database_manager = DatabaseManager(config = self.data_ingestion_config.database_config)

    
    def load_data_from_database(self, table_name : str) -> pd.DataFrame:
        try:
            logging.info(f"Ingesting data from table {table_name}")

            self.database_manager.connect()

            query = f'SELECT * FROM {table_name}'
            columns, rows = self.database_manager.fetch_all(query=query)
            df = pd.DataFrame(data=rows, columns=columns)
            df = df.drop(columns = self.data_ingestion_config.exclude_columns)

            logging.info(f'Loaded {len(df)} records from {table_name}')

            self.database_manager.close()

            return df

        except Exception as e:
            raise PredicitiveMaintainanceException(e, sys)
    
    def initiate_data_ingestion(self) -> DataIngestionArtifact:
        try:
            logging.info('Initiating Data Ingestion')

            train_features_df = self.load_data_from_database(self.data_ingestion_config.train_features_table)
            test_features_df = self.load_data_from_database(self.data_ingestion_config.test_features_table)
            test_targets_df = self.load_data_from_database(self.data_ingestion_config.test_targets_table)

            os.makedirs(self.data_ingestion_config.ingested_data_dir, exist_ok=True)

            train_features_df.to_csv(self.data_ingestion_config.ingested_train_data_path, index=False)
            test_features_df.to_csv(self.data_ingestion_config.ingested_test_data_path, index=False)
            test_targets_df.to_csv(self.data_ingestion_config.ingested_test_target_path, index=False)

            logging.info("Saved Ingested Data")

        except Exception as e:
            raise PredicitiveMaintainanceException(e, sys)