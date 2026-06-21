from dataclasses import dataclass

@dataclass
class DataExtractionArtifact:
    extracted_train_data_path : str
    extracted_test_data_path : str
    extracted_test_target_path : str

@dataclass
class DataTransformationArtifact:
    validation_status : str
    validation_report_path : str
    transformed_train_data_path : str
    transformed_test_data_path : str
    transformed_test_target_path : str

@dataclass
class DataIngestionArtifact:
    ingested_train_data_path : str
    ingested_test_data_path : str
    ingested_test_target_path : str

@dataclass
class DataValidationArtifact:
    validation_status : str
    validation_report_path : str
    validated_train_data_path : str
    validated_test_data_path : str
    validated_test_target_path : str

@dataclass
class ModelTrainerArtifact:
    evaluation_report_path : str
    run_id : str
    experiment_id : str
    registered_model_name : str

@dataclass
class ModelFinalizerArtifact:
    model_promotion_report_path : str
    promoted_model_uri : str
    promoted_run_id : str
    promoted_model_version : int