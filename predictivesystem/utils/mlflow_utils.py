import sys

import pandas as pd
import numpy as np

import mlflow
import mlflow.pyfunc
from mlflow.pyfunc.model import PythonModel

import torch

from predictivesystem.logging.logger import logging
from predictivesystem.exception.exception import PredictiveMaintenanceException

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

class MLflowManager:
    def __init__(self, mlflow_config : dict, evaluation_report : dict):
        self.mlflow_config = mlflow_config
        self.evaluation_report = evaluation_report
    
    def _log_params(self, model_training_config, lstm_config, xgboost_config):
        try:
            params = {
                'sequence_length' : model_training_config.sequence_length,
                'random_state' : model_training_config.random_state,
                'failure_threshold' : model_training_config.failure_threshold
            }

            for key, value in lstm_config.items():
                params[f'lstm_{key}'] = value
            
            for key, value in xgboost_config.items():
                params[f'xgb_{key}'] = value
            
            mlflow.log_params(params)

        except Exception as e:
            raise PredictiveMaintenanceException(e, sys)
    
    def _log_metrics(self):
        try:
            lstm_metrics = self.evaluation_report['lstm_model']['metrics']
            mlflow.log_metrics({f'lstm_{k}' : float(v) for k, v in lstm_metrics.items()})

            classifier_metrics = self.evaluation_report['classifier']['metrics']
            mlflow.log_metrics({f'classifier_{k}' : float(v) for k, v in classifier_metrics.items()})

            hybrid_metrics = self.evaluation_report['hybrid_model']['metrics']
            mlflow.log_metric('hybrid_agreement_score',float(hybrid_metrics['agreement_score']))
            mlflow.log_metrics({f'hybrid_lstm_{k}' : float(v) for k, v in hybrid_metrics['lstm_failure_metrics'].items()})
            mlflow.log_metrics({f'hybrid_classifier_{k}' : float(v) for k, v in hybrid_metrics['classifier_failure_metrics'].items()})

        except Exception as e:
            raise PredictiveMaintenanceException(e, sys)
    
    def _log_hybrid_model(self, model_training_config, lstm_config, xgboost_config, rul_model, classifier_model, target_scaler, feature_scaler, failure_threshold : int, tracking_uri : str) -> tuple[str, str]:
        try:
            mlflow.set_tracking_uri(tracking_uri)
            mlflow.set_experiment(self.mlflow_config['experiment_name'])

            with mlflow.start_run() as run:
                logging.info('Started MLflow Run')

                self._log_params(
                    model_training_config=model_training_config,
                    lstm_config=lstm_config,
                    xgboost_config=xgboost_config
                )

                self._log_metrics()

                hybrid_model = HybridModel(
                    rul_model=rul_model,
                    clf_model=classifier_model,
                    target_scaler=target_scaler,
                    feature_scaler=feature_scaler,
                    failure_threshold=failure_threshold
                )

                mlflow.pyfunc.log_model(name='hybrid_model', python_model=hybrid_model)

                logging.info("Successfully logged Hybrid Model")

                model_uri = f'runs:/{run.info.run_id}/hybrid_model'
                registered_model = mlflow.register_model(model_uri=model_uri, name=self.mlflow_config['registered_model_name'])

                logging.info(f'Registered Model Version: {registered_model.version}')
            
            return (run.info.run_id, run.info.experiment_id)

        except Exception as e:
            raise PredictiveMaintenanceException(e, sys)