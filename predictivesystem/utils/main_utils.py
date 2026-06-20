import sys
import yaml
import pandas as pd
import numpy as np
import pickle

from predictivesystem.exception.exception import PredictiveMaintenanceException

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score

def read_csv_file(path: str) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    
    except Exception as e:
        raise PredictiveMaintenanceException(e, sys)

def read_yaml_file(path : str) -> dict:
    try:
        with open(path, 'r') as file:
            data = yaml.safe_load(file)
        
        return data
    
    except Exception as e:
        raise PredictiveMaintenanceException(e, sys)

def get_rul_scores(y_true_scaled, y_pred_scaled, scaler : StandardScaler) -> dict:
    try:
        y_true = scaler.inverse_transform(np.asarray(y_true_scaled).reshape(-1, 1)).flatten()
        y_pred = scaler.inverse_transform(np.asarray(y_pred_scaled).reshape(-1, 1)).flatten()

        return {
            'MSE' : mean_squared_error(y_true, y_pred),
            'MAE' : mean_absolute_error(y_true, y_pred),
            'RMSE' : root_mean_squared_error(y_true, y_pred),
            'R2' : r2_score(y_true, y_pred)
        }
    
    except Exception as e:
        raise PredictiveMaintenanceException(e, sys)

def get_classification_scores(y_true, y_pred) -> dict:
    try:
        return {
            'Accuracy' : accuracy_score(y_true, y_pred),
            'Precision' : precision_score(y_true, y_pred),
            'Recall' : recall_score(y_true, y_pred),
            'ROC-AUC Score' : roc_auc_score(y_true, y_pred)
        }
    
    except Exception as e:
        raise PredictiveMaintenanceException(e, sys)

def load_obj(path : str):
    try:
        with open(path,'rb') as file:
            obj = pickle.load(file)
        
        return obj
    
    except Exception as e:
        raise PredictiveMaintenanceException(e, sys)