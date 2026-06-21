import pandas as pd
import numpy as np

import torch

from mlflow.pyfunc.model import PythonModel

class HybridModel(PythonModel):
    def __init__(self, rul_model, clf_model, target_scaler, feature_scaler, failure_threshold : int):
        self.rul_model = rul_model
        self.classifier_model = clf_model
        self.target_scaler = target_scaler
        self.feature_scaler = feature_scaler
        self.failure_threshold = failure_threshold       

    def _predict_lstm(self, X: np.ndarray) -> np.ndarray:
        self.rul_model.eval()

        with torch.no_grad():
            X_tensor = torch.tensor(X, dtype=torch.float32)
            predictions = self.rul_model(X_tensor).cpu().numpy()
        
        return predictions
    
    def _inverse_scale_rul(self, scaled_rul : np.ndarray) -> np.ndarray:
        return self.target_scaler.inverse_transform(scaled_rul.reshape(-1, 1)).flatten()
    
    def _predict_failure_window(self, clf_X : np.ndarray) -> np.ndarray:
        return self.classifier_model.predict(clf_X)
    
    def _predict_failure_from_rul(self, rul_values : np.ndarray) -> np.ndarray:
        return (rul_values <= self.failure_threshold).astype(int)
    
    def _predict_failure_probability(self, clf_X : np.ndarray) -> np.ndarray:
        return self.classifier_model.predict_proba(clf_X)[:, 1]

    def predict(self, context, model_input):
        rul_X = model_input['rul_X']
        clf_X = model_input['clf_X']

        rul_scaled = self._predict_lstm(rul_X)
        predicted_rul = self._inverse_scale_rul(rul_scaled)

        lstm_failure_prediction = self._predict_failure_from_rul(predicted_rul)
        classifier_prediction = self._predict_failure_window(clf_X)
        classifier_probability = self._predict_failure_probability(clf_X)

        return pd.DataFrame({
            'predicted_rul' : predicted_rul.tolist(),
            'lstm_failure_prediction' : lstm_failure_prediction.tolist(),
            'classifier_prediction' : classifier_prediction.tolist(),
            'classifier_probability' : classifier_probability.tolist()
        })