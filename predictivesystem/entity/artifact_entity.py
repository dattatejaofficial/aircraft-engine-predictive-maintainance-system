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