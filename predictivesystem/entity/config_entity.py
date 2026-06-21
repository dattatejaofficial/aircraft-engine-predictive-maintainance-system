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

        self.feature_store_dir = self.artifact_config['feature_store_dir']
        
        self.etl_artifact_name = self.artifact_config['etl_artifact_dir']
        self.etl_artifact_dir = os.path.join(self.artifact_dir, self.etl_artifact_name)

        self.ml_artifact_dir_name = self.artifact_config['ml_artifact_dir']
        self.ml_artifact_dir = os.path.join(self.artifact_dir, self.ml_artifact_dir_name)
    
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
    def __init__(self, configuration_manager : ConfigurationManager, data_extraction_config_file_path = 'configs/etl_config.yaml'):
        with open(data_extraction_config_file_path, 'r') as file:
            config = yaml.safe_load(file)
        
        data_extraction_config = config['data_extraction_details']

        self.raw_data_dir = data_extraction_config['raw_data_dir']
        self.train_data_path = os.path.join(self.raw_data_dir, data_extraction_config['train_data_path'])
        self.test_data_path = os.path.join(self.raw_data_dir, data_extraction_config['test_data_path'])
        self.test_target_path = os.path.join(self.raw_data_dir, data_extraction_config['test_target_path'])

        self.data_extraction_dir = os.path.join(
            configuration_manager.etl_artifact_dir,
            data_extraction_config['data_extraction_dir']
        )
        self.extracted_data_dir = os.path.join(
            self.data_extraction_dir,
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

class DataTransformationConfig:
    def __init__(self, configuration_manager : ConfigurationManager, data_transformation_config_file_path = 'configs/etl_config.yaml'):
        with open(data_transformation_config_file_path,'r') as file:
            config = yaml.safe_load(file)

        data_transformation_config = config['data_transformation_details']

        self.data_schema_path = os.path.join(
            data_transformation_config['data_schema_dir'],
            data_transformation_config['data_schema_path']
        )

        self.data_transformation_dir = os.path.join(
            configuration_manager.etl_artifact_dir,
            data_transformation_config['data_transformation_dir']
        )

        self.transformed_data_dir = os.path.join(
            self.data_transformation_dir,
            data_transformation_config['transformed_data_dir']
        )
        self.transformed_train_data_path = os.path.join(
            self.transformed_data_dir,
            data_transformation_config['transformed_train_data_path']
        )
        self.transformed_test_data_path = os.path.join(
            self.transformed_data_dir,
            data_transformation_config['transformed_test_data_path']
        )
        self.transformed_test_target_path = os.path.join(
            self.transformed_data_dir,
            data_transformation_config['transformed_test_target_path']
        )

        self.validation_report_dir = os.path.join(
            self.data_transformation_dir,
            data_transformation_config['validation_report_dir']
        )
        self.validation_report_path = os.path.join(
            self.validation_report_dir,
            data_transformation_config['validation_report_path']
        )

        self.feature_scaler_path = os.path.join(
            configuration_manager.feature_store_dir,
            data_transformation_config['min_max_scaler_path']
        )
        self.target_scaler_path = os.path.join(
            configuration_manager.feature_store_dir,
            data_transformation_config['standard_scaler_path']
        )

        self.exclude_columns = data_transformation_config['exclude_columns']
        self.feature_columns = data_transformation_config['feature_columns']
        self.degrade_up_columns = data_transformation_config['degrade_up_columns']
        self.degrade_down_columns = data_transformation_config['degrade_down_columns']
        self.target_column = data_transformation_config['target_column']
        self.engine_column = data_transformation_config['engine_column']
        self.cycle_column = data_transformation_config['cycle_column']
        self.target_cap = data_transformation_config['target_cap']
        self.failure_window = data_transformation_config['failure_window']

class DataLoadingConfig:
    def __init__(self, database_config : DatabaseConfig, data_loader_config_file_path = "configs/etl_config.yaml"):
        with open(data_loader_config_file_path,'r') as file:
            config = yaml.safe_load(file)
        
        data_loading_config = config['data_loading_details']
        
        self.database_config = database_config
        self.train_features_table_name = data_loading_config['train_features_table_name']
        self.test_features_table_name = data_loading_config['test_features_table_name']
        self.test_targets_table_name = data_loading_config['test_targets_table_name']

class DataIngestionConfig:
    def __init__(self, configuration_manager : ConfigurationManager, database_config : DatabaseConfig, data_ingestion_config_file_path = "configs/ml_config.yaml"):
        with open(data_ingestion_config_file_path, 'r') as file:
            config = yaml.safe_load(file)
        
        data_ingestion_config = config['data_ingestion_details']

        self.database_config = database_config
        self.train_features_table = data_ingestion_config['train_features_table_name']
        self.test_features_table = data_ingestion_config['test_features_table_name']
        self.test_targets_table = data_ingestion_config['test_targets_table_name']

        self.data_ingestion_dir = os.path.join(
            configuration_manager.ml_artifact_dir,
            data_ingestion_config['data_ingestion_dir']
        )

        self.ingested_data_dir = os.path.join(
            self.data_ingestion_dir,
            data_ingestion_config['ingested_data_dir']
        )
        self.ingested_train_data_path = os.path.join(
            self.ingested_data_dir,
            data_ingestion_config['ingested_train_data_path']
        )
        self.ingested_test_data_path = os.path.join(
            self.ingested_data_dir,
            data_ingestion_config['ingested_test_data_path']
        )
        self.ingested_test_target_path = os.path.join(
            self.ingested_data_dir,
            data_ingestion_config['ingested_test_target_path']
        )

        self.exclude_columns = data_ingestion_config['exclude_columns']

class DataValidationConfig:
    def __init__(self, configuration_manager : ConfigurationManager, data_validation_config_file_path = 'configs/ml_config.yaml'):
        with open(data_validation_config_file_path,'r') as file:
            config = yaml.safe_load(file)
        
        validation_config = config['data_validation_details']

        self.data_schema_path = os.path.join(
            validation_config['data_schema_dir'],
            validation_config['data_schema_path']
        )

        self.data_validation_dir = os.path.join(
            configuration_manager.ml_artifact_dir,
            validation_config['data_validation_dir']
        )
        
        self.validated_data_dir = os.path.join(
            self.data_validation_dir,
            validation_config['validated_data_dir']
        )

        self.validation_reports_dir = os.path.join(
            self.data_validation_dir,
            validation_config['validation_report_dir']
        )

        self.validated_train_features_path = os.path.join(
            self.validated_data_dir,
            validation_config['validated_train_features_path']
        )
        self.validated_test_features_path = os.path.join(
            self.validated_data_dir,
            validation_config['validated_test_features_path']
        )
        self.validated_test_targets_path = os.path.join(
            self.validated_data_dir,
            validation_config['validated_test_targets_path']
        )

        self.validation_report_path = os.path.join(
            self.validation_reports_dir,
            validation_config['validation_report_path']
        )

class ModelTrainingConfig:
    def __init__(self, configuration_manager : ConfigurationManager, model_training_config_file_path = 'configs/ml_config.yaml'):
        with open(model_training_config_file_path, 'r') as file:
            config = yaml.safe_load(file)
        
        model_training_config = config['model_training_details']

        self.model_training_dir = os.path.join(
            configuration_manager.ml_artifact_dir,
            model_training_config['model_training_dir']
        )
        self.evaluation_report_dir = os.path.join(
            self.model_training_dir,
            model_training_config['evaluation_report_dir']
        )
        self.evaluation_report_path = os.path.join(
            self.model_training_dir,
            model_training_config['evaluation_report_path']
        )

        self.sequence_length = model_training_config['sequence_length']
        self.failure_threshold = model_training_config['failure_threshold']
        self.random_state = model_training_config['random_state']
        self.engine_id_column = model_training_config['engine_id_column']

        self.mlflow_config = model_training_config['mlflow']
        self.lstm_config = model_training_config['lstm']
        self.xgboost_config = model_training_config['xgboost']

        self.feature_store = model_training_config['feature_store']

class ModelFinalizingConfig:
    def __init__(self, configuration_manager : ConfigurationManager, model_finalizing_config_file_path = 'configs/ml_config.yaml'):
        with open(model_finalizing_config_file_path, 'r') as file:
            config = yaml.safe_load(file)
        
        model_finalizing_config = config['model_finalizer_details']

        self.model_finalizer_dir = os.path.join(
            configuration_manager.ml_artifact_dir,
            model_finalizing_config['model_finalizer_dir']
        )

        self.model_promotion_report_dir = os.path.join(
            self.model_finalizer_dir,
            model_finalizing_config['model_promotion_report_dir']
        )
        self.model_promotion_report_path = os.path.join(
            self.model_promotion_report_dir,
            model_finalizing_config['model_promotion_report_path']
        )

class ArtifactPublisherConfig:
    def __init__(self, artifact_publisher_config_file_path = 'configs/ml_config.yaml'):
        with open(artifact_publisher_config_file_path, 'r') as file:
            config = yaml.safe_load(file)
        
        artifact_publisher_config = config['artifact_publisher_details']
        
        self.production_prefix = artifact_publisher_config['production_prefix']