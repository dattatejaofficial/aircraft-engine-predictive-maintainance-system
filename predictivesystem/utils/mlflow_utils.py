import sys

import mlflow
import mlflow.pyfunc

from predictivesystem.utils.hybrid_model import HybridModel
from predictivesystem.logging.logger import logging
from predictivesystem.exception.exception import PredictiveMaintenanceException

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