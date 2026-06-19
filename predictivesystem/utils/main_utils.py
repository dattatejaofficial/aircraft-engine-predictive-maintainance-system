import sys
import yaml
import pandas as pd

from predictivesystem.exception.exception import PredicitiveMaintainanceException

def read_csv_file(path: str) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    
    except Exception as e:
        raise PredicitiveMaintainanceException(e, sys)

def read_yaml_file(path : str) -> dict:
    try:
        with open(path, 'r') as file:
            data = yaml.safe_load(file)
        
        return data
    
    except Exception as e:
        raise PredicitiveMaintainanceException(e, sys)