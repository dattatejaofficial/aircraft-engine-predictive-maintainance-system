import os
from dotenv import load_dotenv
import yaml
from datetime import datetime

load_dotenv()

class ConfigurationManager:
    def __init__(self, config_filepath = "configs/config.yaml", timestamp = datetime.now()):
        with open(config_filepath, 'r') as file:
            self.config = yaml.safe_load(file)
        
        self.artifact_config = self.config['artifacts']
        self.timestamp = timestamp.strftime(self.artifact_config['timestamp_format'])
        
        self.artifact_dir_name = self.artifact_config['artifact_dir']
        self.artifact_dir = os.path.join(self.artifact_dir_name, self.timestamp)
        
        self.etl_artifact_name = self.artifact_config['etl_artifact_dir']
        self.etl_artifact_dir = os.path.join(self.artifact_dir, self.etl_artifact_name)
    
    @property
    def logging_config(self):
        return self.config['logging']

class DatabaseConfig:
    def __init__(self, database_config_filepath = "configs/etl_config.yaml"):
        with open(database_config_filepath, 'r') as file:
            config = yaml.safe_load(file)    

        database_config = config['database_details']    

        self.host = database_config['host']
        self.port = database_config['port']
        self.database = database_config['database']
        self.username = os.getenv('MYSQL_USERNAME')
        self.password = os.getenv('MYSQL_PASSWORD')

class DataExtractionConfig:
    def __init__(self, configuration_manager : ConfigurationManager, data_extraction_config_filepath = 'configs/etl_config.yaml'):
        with open(data_extraction_config_filepath, 'r') as file:
            config = yaml.safe_load(file)
        
        data_extraction_config = config['data_extraction_details']

        self.raw_data_dir = data_extraction_config['raw_data_dir']
        self.train_data_path = os.path.join(self.raw_data_dir, data_extraction_config['train_data_path'])
        self.test_data_path = os.path.join(self.raw_data_dir, data_extraction_config['test_data_path'])
        self.test_target_path = os.path.join(self.raw_data_dir, data_extraction_config['test_target_path'])

        self.extracted_data_dir = os.path.join(
            configuration_manager.etl_artifact_dir,
            data_extraction_config['extracted_data_dir'])
        self.extracted_train_data_path = os.path.join(
            self.extracted_data_dir,
            data_extraction_config['extracted_train_data_path']
        )
        self.extracted_test_data_path = os.path.join(
            self.extracted_data_dir,
            data_extraction_config['extracted_test_data_path']
        )
        self.extracted_test_target_path = os.path.join(
            self.extracted_data_dir,
            data_extraction_config['extracted_test_target_path']
        )

        self.feature_columns = data_extraction_config['feature_columns']
        self.target_column = data_extraction_config['target_column']