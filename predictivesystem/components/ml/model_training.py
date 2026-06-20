import os
import sys
from dotenv import load_dotenv
import json

load_dotenv()

from predictivesystem.logging.logger import logging
from predictivesystem.exception.exception import PredictiveMaintenanceException

from predictivesystem.entity.config_entity import ModelTrainingConfig
from predictivesystem.entity.artifact_entity import DataValidationArtifact, ModelTrainerArtifact

from predictivesystem.utils.main_utils import read_csv_file, get_rul_scores, get_classification_scores, load_obj
from predictivesystem.utils.model_training_utils import create_data_sequences, create_test_data_sequences, LSTMRegressor, AircraftDataset, lstm_model_predictions
from predictivesystem.utils.mlflow_utils import MLflowManager

import pandas as pd
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from xgboost import XGBClassifier

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

MLFLOW_TRACKING_URI = os.getenv('MLFLOW_TRACKING_URI')

class ModelTrainer:
    def __init__(self, data_validation_artifact : DataValidationArtifact, model_training_config : ModelTrainingConfig):
        self.data_validation_artifact = data_validation_artifact
        self.model_training_config = model_training_config

        self.lstm_config = self.model_training_config.lstm_config
        self.xgboost_config = self.model_training_config.xgboost_config
        self.mlflow_config = self.model_training_config.mlflow_config

        self.evaluation_report = {
            "lstm_model" : {},
            "classifier" : {},
            "hybrid_model" : {}
        }
    
    def _train_lstm_model(self, loader: DataLoader) -> LSTMRegressor:
        try:
            logging.info("Initializing LSTM model")

            model = LSTMRegressor(
                input_size=self.lstm_config.get('input_size'),
                hidden_size=self.lstm_config.get('hidden_size'),
                num_layers=self.lstm_config.get('num_layers'),
                dropout=self.lstm_config.get('dropout')
            )

            criterion = nn.MSELoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=self.lstm_config.get('learning_rate'))

            epochs = self.lstm_config.get('epochs')

            for _ in range(epochs):
                model.train()
                total_loss = 0

                for X_batch, y_batch in loader:
                    optimizer.zero_grad()
                    preds = model(X_batch)

                    loss = criterion(preds, y_batch)
                    loss.backward()

                    optimizer.step()

                    total_loss += loss.item()
            
            logging.info("LSTM training completed successfully")

            return model

        except Exception as e:
            raise PredictiveMaintenanceException(e, sys)
    
    def _train_classifier_model(self, X_train, y_train) -> XGBClassifier:
        try:
            logging.info("Initializing XGBoost classifier")

            classifier = XGBClassifier(
                n_estimators = self.xgboost_config['n_estimators'], 
                max_depth = self.xgboost_config['max_depth'],
                learning_rate = self.xgboost_config['learning_rate'],
                subsample = self.xgboost_config['subsample'],
                colsample_bytree = self.xgboost_config['colsample_bytree'],
                eval_metric = self.xgboost_config['eval_metric']
            )
            classifier.fit(X_train, y_train)

            logging.info("XGBoost training completed successfully")
            
            return classifier

        except Exception as e:
            raise PredictiveMaintenanceException(e, sys)
    
    def _evaluate_lstm_model(self, model: LSTMRegressor, X_test, y_test, target_scaler : StandardScaler) -> None:
        try:
            y_pred = lstm_model_predictions(model=model, X_test=X_test)
            metrics = get_rul_scores(y_test, y_pred, target_scaler)

            self.evaluation_report['lstm_model']['metrics'] = metrics

            logging.info("Completed evaluating LSTM model")

        except Exception as e:
            raise PredictiveMaintenanceException(e, sys)
    
    def _evaluate_classifier_model(self, model: XGBClassifier, X_test, y_test) -> None:
        try:
            y_pred = model.predict(X_test)
            metrics = get_classification_scores(y_test, y_pred)

            self.evaluation_report['classifier']['metrics'] = metrics

            logging.info("Completed evaluating XGBoost model")

        except Exception as e:
            raise PredictiveMaintenanceException(e, sys)
    
    def _evaluate_hybrid_model(self, lstm_model : LSTMRegressor, classifier_model : XGBClassifier, lstm_X_test : np.ndarray, clf_X_test, lstm_y_test : np.ndarray, target_scaler : StandardScaler) -> None:
        try:
            failure_threshold = self.model_training_config.failure_threshold

            lstm_pred_scaled = lstm_model_predictions(model=lstm_model, X_test=lstm_X_test)

            lstm_pred_rul = target_scaler.inverse_transform(lstm_pred_scaled.reshape(-1, 1)).flatten()
            actual_rul = target_scaler.inverse_transform(lstm_y_test.reshape(-1, 1)).flatten()

            y_true_failure = (actual_rul <= failure_threshold).astype(int)
            lstm_failure_pred = (lstm_pred_rul <= failure_threshold).astype(int)
            
            xgb_failure_pred = classifier_model.predict(clf_X_test)

            lstm_failure_metrics = get_classification_scores(y_true_failure, lstm_failure_pred)
            xgb_failure_metrics = get_classification_scores(y_true_failure, xgb_failure_pred)
            agreement_score = float(np.mean(lstm_failure_pred == xgb_failure_pred))

            self.evaluation_report['hybrid_model']['metrics'] = {
                'agreement_score' : agreement_score,
                'lstm_failure_metrics' : lstm_failure_metrics,
                'classifier_failure_metrics' : xgb_failure_metrics
            }

            logging.info("Completed evaluating the Hybrid model")

        except Exception as e:
            raise PredictiveMaintenanceException(e, sys)
    
    def _prepare_rul_data(self, train_data : pd.DataFrame, test_data : pd.DataFrame, test_target_data : pd.DataFrame, engine_col : str) -> tuple[np.array, np.array, np.array, np.array]:
        try:
            logging.info("Preparing LSTM sequence data")

            seq_len = self.model_training_config.sequence_length
            
            lstm_features = self.lstm_config['features']
            lstm_target = self.lstm_config['target']

            lstm_X_train_arr, lstm_y_train_arr = create_data_sequences(
                df=train_data,
                seq_len=seq_len,
                features=lstm_features,
                target=lstm_target,
                engine_col=engine_col
            )

            lstm_X_test_arr = create_test_data_sequences(
                df=test_data,
                seq_len=seq_len,
                features=lstm_features,
                engine_col=engine_col
            )
            lstm_y_test_arr = test_target_data[lstm_target].values
            
            
            return (lstm_X_train_arr, lstm_y_train_arr, lstm_X_test_arr, lstm_y_test_arr)
        
        except Exception as e:
            raise PredictiveMaintenanceException(e, sys)
    
    def _prepare_clf_data(self, train_data : pd.DataFrame, engine_col : str) -> tuple[np.array, np.array, np.array, np.array]:
        try:
            logging.info("Preparing classifier dataset")

            df = train_data.copy()
            engine_ids = df[engine_col].unique()

            train_clf_engines, val_clf_engines = train_test_split(engine_ids, test_size=self.xgboost_config['validation_size'], random_state=self.model_training_config.random_state)
            
            train_clf_df = df[df[engine_col].isin(train_clf_engines)]
            val_clf_df = df[df[engine_col].isin(val_clf_engines)]

            exclude_clf_features = self.xgboost_config['exclude_columns']
            classifier_target = self.xgboost_config['target']

            clf_X_train_arr = train_clf_df.drop(columns=exclude_clf_features).values
            clf_y_train_arr = train_clf_df[classifier_target].values
            clf_X_val_arr = val_clf_df.drop(columns=exclude_clf_features).values
            clf_y_val_arr = val_clf_df[classifier_target].values

            return (clf_X_train_arr, clf_y_train_arr, clf_X_val_arr, clf_y_val_arr)

        except Exception as e:
            raise PredictiveMaintenanceException(e, sys)
    
    def _prepare_hybrid_test_data(self, test_data : pd.DataFrame, engine_col : str) -> np.ndarray:
        try:
            logging.info("Preparing hybrid test data")

            seq_len = self.model_training_config.sequence_length

            clf_features = [col for col in test_data.columns if col not in self.xgboost_config['exclude_columns']]

            clf_X_test = []
            
            engines = test_data[engine_col].unique()
            for eid in engines:
                engine = test_data[test_data[engine_col] == eid]

                if len(engine) < seq_len:
                    continue

                clf_X_test.append(engine[clf_features].iloc[-1].values)
            
            return np.array(clf_X_test)

        except Exception as e:
            raise PredictiveMaintenanceException(e, sys)

    def initiate_model_training(self) -> ModelTrainerArtifact:
        try:
            logging.info("Initiating Model Training")

            if self.data_validation_artifact.validation_status == "FAIL":
                logging.info("Model Training is stopped because of Data Validation failure")
                return

            train_data = read_csv_file(self.data_validation_artifact.validated_train_data_path)
            test_data = read_csv_file(self.data_validation_artifact.validated_test_data_path)
            test_target_data = read_csv_file(self.data_validation_artifact.validated_test_target_path)
            
            engine_id_column = self.model_training_config.engine_id_column
            
            logging.info("Loading scalers")
            target_scaler = load_obj(self.model_training_config.feature_store.get('target_scaler_path'))
            feature_scaler = load_obj(self.model_training_config.feature_store.get('feature_scaler_path'))

            logging.info("Preparing RUL datasets")
            
            lstm_X_train_arr, lstm_y_train_arr, lstm_X_test_arr, lstm_y_test_arr = self._prepare_rul_data(
                train_data=train_data,
                test_data=test_data,
                test_target_data=test_target_data,
                engine_col=engine_id_column
            )
            clf_X_test_arr = self._prepare_hybrid_test_data(test_data=test_data, engine_col=engine_id_column)

            dataset = AircraftDataset(lstm_X_train_arr, lstm_y_train_arr)
            loader = DataLoader(dataset=dataset, batch_size=self.lstm_config['batch_size'], shuffle=True)

            logging.info("Training LSTM model")
            rul_model = self._train_lstm_model(loader=loader)

            self._evaluate_lstm_model(model=rul_model, X_test=lstm_X_test_arr, y_test=lstm_y_test_arr, target_scaler=target_scaler)

            clf_X_train_arr, clf_y_train_arr, clf_X_val_arr, clf_y_val_arr = self._prepare_clf_data(train_data=train_data, engine_col=engine_id_column)
            
            logging.info("Training XGBoost classifier")
            classifier = self._train_classifier_model(clf_X_train_arr, clf_y_train_arr)
            self._evaluate_classifier_model(model=classifier, X_test=clf_X_val_arr, y_test=clf_y_val_arr)

            self._evaluate_hybrid_model(
                lstm_model=rul_model,
                classifier_model=classifier,
                lstm_X_test=lstm_X_test_arr,
                lstm_y_test=lstm_y_test_arr,
                clf_X_test=clf_X_test_arr,
                target_scaler=target_scaler
            )

            os.makedirs(self.model_training_config.evaluation_report_dir, exist_ok=True)
            with open(self.model_training_config.evaluation_report_path,'w') as file:
                json.dump(self.evaluation_report, file, indent=4)
            
            logging.info("Evaluation Metrics are saved")

            logging.info("Logging models and metrics to MLflow")
            mlflow_manager = MLflowManager(mlflow_config=self.mlflow_config, evaluation_report=self.evaluation_report)
            run_id, exp_id = mlflow_manager._log_hybrid_model(
                model_training_config=self.model_training_config,
                lstm_config=self.lstm_config,
                xgboost_config=self.xgboost_config,
                rul_model=rul_model,
                classifier_model=classifier,
                feature_scaler=feature_scaler,
                target_scaler=target_scaler,
                tracking_uri=MLFLOW_TRACKING_URI,
                failure_threshold=self.model_training_config.failure_threshold
            )
            
            model_trainer_artifact = ModelTrainerArtifact(
                evaluation_report_path=self.model_training_config.evaluation_report_path,
                run_id=run_id,
                experiment_id=exp_id,
                registered_model_name=self.mlflow_config['registered_model_name']
            )

            return model_trainer_artifact
        
        except Exception as e:
            raise PredictiveMaintenanceException(e, sys)