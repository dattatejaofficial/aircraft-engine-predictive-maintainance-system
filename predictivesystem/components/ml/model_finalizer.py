import os
import sys
import json

from predictivesystem.logging.logger import logging
from predictivesystem.exception.exception import PredictiveMaintenanceException

from predictivesystem.entity.config_entity import ModelFinalizingConfig
from predictivesystem.entity.artifact_entity import ModelTrainerArtifact, ModelFinalizerArtifact

from mlflow.client import MlflowClient

class ModelFinalizer:
    def __init__(self, model_trainer_artifact : ModelTrainerArtifact, model_finalizing_config : ModelFinalizingConfig):
        self.model_trainer_artifact = model_trainer_artifact
        self.model_finalizing_config = model_finalizing_config
        self.client = MlflowClient()
    
    def _compute_weighted_score(self, r2_score : float, rmse_score : float, hybrid_recall : float, hybrid_roc_auc : float, hybrid_precision : float, classifier_recall : float, classifier_precision : float, classifier_roc_auc : float, agreement_score : float) -> float:
        try:
            logging.info("Computing Weighted Score")
            normalized_rmse = 1 / (1 + rmse_score)

            final_score = (0.35 * r2_score + 0.15 * normalized_rmse + 0.10 * hybrid_recall + 0.05 * hybrid_precision + 0.05 * hybrid_roc_auc + 0.10 * classifier_recall + 0.05 * classifier_precision + 0.05 * classifier_roc_auc + 0.10 * agreement_score)

            return final_score

        except Exception as e:
            raise PredictiveMaintenanceException(e, sys)
    
    def _calculate_candidate_score(self, report : dict) -> float:
        try:
            lstm_metrics = report['lstm_model']['metrics']
            classifier_metrics = report['classifier']['metrics']
            hybrid_metrics = report['hybrid_model']['metrics']
            hybrid_failure_metrics = hybrid_metrics['lstm_failure_metrics']

            return self._compute_weighted_score(
                r2_score=lstm_metrics['R2'],
                rmse_score=lstm_metrics['RMSE'],
                
                hybrid_recall=hybrid_failure_metrics['Recall'],
                hybrid_precision=hybrid_failure_metrics['Precision'],
                hybrid_roc_auc=hybrid_failure_metrics['ROC-AUC Score'],

                classifier_recall=classifier_metrics['Recall'],
                classifier_precision=classifier_metrics['Precision'],
                classifier_roc_auc=classifier_metrics['ROC-AUC Score'],

                agreement_score=hybrid_metrics['agreement_score']
            )
        except Exception as e:
            raise PredictiveMaintenanceException(e, sys)
    
    def _calculate_production_score(self, metrics : dict) -> float:
        try:
            return self._compute_weighted_score(
            r2_score=metrics["lstm_R2"],
            rmse_score=metrics["lstm_RMSE"],

            hybrid_recall=metrics["hybrid_lstm_Recall"],
            hybrid_precision=metrics["hybrid_lstm_Precision"],
            hybrid_roc_auc=metrics["hybrid_lstm_ROC-AUC Score"],

            classifier_recall=metrics["classifier_Recall"],
            classifier_precision=metrics["classifier_Precision"],
            classifier_roc_auc=metrics["classifier_ROC-AUC Score"],

            agreement_score=metrics["hybrid_agreement_score"]
        )

        except Exception as e:
            raise PredictiveMaintenanceException(e, sys)
    
    def _get_metrics(self, run_id: str) -> dict:
        try:
            run = self.client.get_run(run_id)
            return run.data.metrics
        
        except Exception as e:
            raise PredictiveMaintenanceException(e, sys)
    
    def _get_production_model(self):
        try:
            return self.client.get_model_version_by_alias(
                name=self.model_trainer_artifact.registered_model_name,
                alias='production'
            )
        
        except Exception:
            return None
    
    def _get_version_from_run_id(self, model_name : str, run_id : str) -> int:
        try:
            versions = self.client.search_model_versions(f"name='{model_name}'")

            for version in versions:
                if version.run_id == run_id:
                    return int(version.version)

        except Exception as e:
            raise PredictiveMaintenanceException(e, sys)
    
    def initiate_model_finalizing(self):
        try:
            logging.info('Initiating Model Finalizing')

            with open(self.model_trainer_artifact.evaluation_report_path,'r') as file:
                current_report = json.load(file)
            
            candidate_run_id = self.model_trainer_artifact.run_id
            candidate_model_uri = f'runs:/{candidate_run_id}/hybrid_model'
            model_name = self.model_trainer_artifact.registered_model_name

            candidate_score = self._calculate_candidate_score(current_report)

            production_model = self._get_production_model()

            if production_model is None:
                logging.info("No production model found.")
            
                candidate_version = self._get_version_from_run_id(model_name, candidate_run_id)
                
                self.client.set_registered_model_alias(name=model_name, alias='production', version=candidate_version)
                decision = 'promoted'

                selected_run_id = candidate_run_id
                selected_model_uri = candidate_model_uri
                selected_version = candidate_version

                production_score = None
            
            else:
                production_metrics = self._get_metrics(production_model.run_id)
                production_score = self._calculate_production_score(production_metrics)

                if candidate_score > production_score:
                    logging.info('Promoted Candidate')

                    candidate_version = self._get_version_from_run_id(model_name, candidate_run_id)

                    self.client.set_registered_model_alias(name=model_name, alias='archived', version=production_model.version)
                    self.client.set_registered_model_alias(name=model_name, alias='production', version=candidate_version)

                    decision='promoted'

                    selected_run_id = candidate_run_id
                    selected_model_uri = candidate_model_uri
                    selected_version = candidate_version
                
                else:
                    logging.info('Candidate Rejected')

                    decision = 'rejected'
                    selected_run_id = production_model.run_id
                    selected_model_uri = f'models:/{model_name}@production'
                    selected_version = int(production_model.version)
            
            report = {
                'decision' : decision,
                'candidate_run_id' : candidate_run_id,
                'candidate_score' : round(candidate_score, 6),
                'production_score' : round(production_score, 6) if production_score is not None else None,
                'selected_model_uri' : selected_model_uri,
                'selected_run_id' : selected_run_id,
                'score_difference' : round(candidate_score - production_score, 6) if production_score is not None else None
            }

            os.makedirs(self.model_finalizing_config.model_promotion_report_dir, exist_ok=True)

            with open(self.model_finalizing_config.model_promotion_report_path,'w') as file:
                json.dump(report, file, indent=4)
            
            model_finalizer_artifact = ModelFinalizerArtifact(
                model_promotion_report_path=self.model_finalizing_config.model_promotion_report_path,
                promoted_model_uri=selected_model_uri,
                promoted_run_id = selected_run_id,
                promoted_model_version = selected_version
            )

            return model_finalizer_artifact

        except Exception as e:
            raise PredictiveMaintenanceException(e, sys)