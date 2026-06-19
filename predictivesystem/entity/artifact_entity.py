from dataclasses import dataclass

@dataclass
class DataExtractionArtifact:
    extracted_train_data_path : str
    extracted_test_data_path : str
    extracted_test_target_path : str