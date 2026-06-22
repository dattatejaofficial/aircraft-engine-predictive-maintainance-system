import pandas as pd
import numpy as np

from utils.feature_extractor import FeatureExtraction
from config import settings

class FeatureProcessor:    
    @staticmethod
    def _extract_features(df : pd.DataFrame, scaler, baseline_map = None):
        extractor = FeatureExtraction(
            df=df,
            exclude_columns=settings.EXCLUDE_COLUMNS,
            feature_columns=settings.FEATURE_COLS,
            degrade_up=settings.DEGRADE_UP,
            degrade_down=settings.DEGRADE_DOWN,
            scaler=scaler,
            lstm_features=settings.LSTM_FEATURES,
            clf_exclude_features=settings.CLASSIFIER_EXCLUDE_FEATURES,
            baseline_map=baseline_map
        )

        return extractor.initiate_feature_extraction()
    
    @staticmethod
    def _create_batch_rul_input(df: pd.DataFrame) -> tuple[np.ndarray, np.array]:
        rul_X = []
        engine_ids = []

        for eid in df['engine_id'].unique():
            engine = df[df['engine_id'] == eid]

            if len(engine) < settings.SEQUENCE_LENGTH:
                continue

            rul_X.append(engine[settings.LSTM_FEATURES].iloc[-settings.SEQUENCE_LENGTH:].values)
        
            engine_ids.append(eid)
        
        return np.asarray(rul_X, dtype=np.float32), np.array(engine_ids)
    
    @staticmethod
    def _create_stream_rul_input(df: pd.DataFrame) -> np.ndarray:
        return np.expand_dims(df[settings.LSTM_FEATURES].values, axis=0).astype(np.float32)
    
    @staticmethod
    def _get_classifier_features(df : pd.DataFrame) -> list[str]:
        return [col for col in df.columns if col not in settings.CLASSIFIER_EXCLUDE_FEATURES]
    
    @staticmethod
    def _create_batch_classifier_input(df : pd.DataFrame) -> np.ndarray:
        clf_features = FeatureProcessor._get_classifier_features(df)
        clf_X = []

        for eid in df['engine_id'].unique():
            engine = df[df['engine_id'] == eid]

            if len(engine) < settings.SEQUENCE_LENGTH:
                continue
            
            clf_X.append(engine[clf_features].iloc[-1].values)
        
        return np.asarray(clf_X, dtype=np.float32)
    
    @staticmethod
    def _create_stream_classifier_input(df: pd.DataFrame) -> np.ndarray:
        clf_features = FeatureProcessor._get_classifier_features(df)
        return np.expand_dims(df[clf_features].iloc[-1].values, axis=0).astype(np.float32)
    
    @staticmethod
    def prepare_batch_data(df: pd.DataFrame, scaler) -> dict:
        lstm_df, clf_df = FeatureProcessor._extract_features(df, scaler)

        rul_X, engine_ids = FeatureProcessor._create_batch_rul_input(lstm_df)
        clf_X = FeatureProcessor._create_batch_classifier_input(clf_df)

        return {
            "engine_ids" : engine_ids,
            "rul_X" : rul_X,
            "clf_X" : clf_X
        }
    
    @staticmethod
    def prepare_stream_data(df: pd.DataFrame, scaler, baseline_map=None) -> dict:
        lstm_df, clf_df = FeatureProcessor._extract_features(df, scaler, baseline_map=baseline_map)

        rul_X = FeatureProcessor._create_stream_rul_input(lstm_df)
        clf_X = FeatureProcessor._create_stream_classifier_input(clf_df)

        return {
            "rul_X" : rul_X,
            "clf_X" : clf_X
        }

    @staticmethod
    def create_baseline_map(record: dict, scaler) -> dict:
        df = pd.DataFrame([record])
        df['cycle_count'] = df['cycle']

        normed = scaler.transform(df[settings.FEATURE_COLS + ['cycle_count']])
        normed_df = pd.DataFrame(normed, columns = [f'{col}_norm' for col in settings.FEATURE_COLS + ['cycle_count']])
        normed_df.drop(columns=['cycle_count_norm'], inplace=True)

        return normed_df.iloc[0].to_dict()