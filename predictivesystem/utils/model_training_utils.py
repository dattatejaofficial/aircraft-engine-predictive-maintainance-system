import sys

from predictivesystem.exception.exception import PredictiveMaintenanceException

import pandas as pd
import numpy as np

import torch
from torch.utils.data import Dataset

from predictivesystem.utils.lstm_regressor import LSTMRegressor

def create_data_sequences(df: pd.DataFrame, seq_len : int, features: list[str], target: str, engine_col: str) -> tuple[np.array, np.array]:
    try:
        X, y = [], []

        for eid in df[engine_col].unique():
            engine = df[df[engine_col] == eid]

            if len(engine) < seq_len:
                continue

            data = engine[features].values
            rul = engine[target].values
        
            for i in range(len(engine) - seq_len):
                X.append(data[i:i+seq_len])
                y.append(rul[i+seq_len-1])
        
        return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)
    
    except Exception as e:
        raise PredictiveMaintenanceException(e, sys)

def create_test_data_sequences(df : pd.DataFrame, seq_len : int, features : list[str], engine_col: str) -> np.array:
    try:
        X_test = []

        for eid in df[engine_col].unique():
            engine = df[df[engine_col] == eid]

            if len(engine) < seq_len:
                continue

            data = engine[features].values
            X_test.append(data[-seq_len:])
        
        return np.array(X_test, dtype=np.float32)
    
    except Exception as e:
        raise PredictiveMaintenanceException(e, sys)

class AircraftDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X)
        self.y = torch.tensor(y)
    
    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return self.X[index], self.y[index]

def lstm_model_predictions(model : LSTMRegressor, X_test: np.array):
    model.eval()

    with torch.no_grad():
        y_pred = model(torch.tensor(X_test)).numpy()
    
    return y_pred

