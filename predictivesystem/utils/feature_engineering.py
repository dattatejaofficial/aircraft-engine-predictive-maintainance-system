import sys

from predictivesystem.exception.exception import PredictiveMaintenanceException

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

class FeatureExtraction:
    def __init__(self, df: pd.DataFrame, exclude_columns : list[str], feature_columns : list[str], degrade_up : list[str], degrade_down : list[str], engine_col : str, cycle_col : str, scaler : MinMaxScaler, fit : bool = True):
        self.df = df.copy()

        self.EXCLUDE_COLS = exclude_columns
        self.FEATURE_COLS = feature_columns
        self.DEGRADE_UP = degrade_up
        self.DEGRADE_DOWN = degrade_down
        self.ENGINE_COL = engine_col
        self.CYCLE_COL = cycle_col
        self.SCALER = scaler
        self.FIT = fit
    
    def add_cycle_features(self, df : pd.DataFrame, cycle_col : str) -> pd.DataFrame:
        try:
            df = df.copy()
            df['cycle_count'] = df[cycle_col]

            return df
        
        except Exception as e:
            raise PredictiveMaintenanceException(e, sys)

    def normalize_data(self, df : pd.DataFrame, feature_cols : list[str], scaler : MinMaxScaler, fit=False):
        try:
            df = df.copy()

            if fit:
                normed = scaler.fit_transform(df[feature_cols + ['cycle_count']])
            else:
                normed = scaler.transform(df[feature_cols + ['cycle_count']])

            normed = pd.DataFrame(
                normed,
                columns=[f'{col}_norm' for col in feature_cols + ['cycle_count']],
                index=df.index
            )

            self.SENSOR_NORM_COLS = normed.columns.to_list()
            self.SENSOR_NORM_COLS.remove('cycle_count_norm')

            return pd.concat([df, normed], axis=1)
        
        except Exception as e:
            raise PredictiveMaintenanceException(e, sys)
        
    def add_rolling_features(self, df : pd.DataFrame, feature_cols : list[str], engine_col : str, window = 15) -> pd.DataFrame:
        try:
            df = df.copy()
            new_cols = {}

            for col in feature_cols:
                grp = df.groupby(engine_col)[col]
                roll = grp.rolling(window=window, min_periods=1)

                new_cols[f'{col}_std_{window}'] = roll.std().fillna(0).reset_index(level=0, drop=True)
            
            return pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)
        
        except Exception as e:
            raise PredictiveMaintenanceException(e, sys)
    
    def add_delta_features(self, df : pd.DataFrame, feature_cols : list[str], engine_col : str, lag = 3) -> pd.DataFrame:
        try:
            df = df.copy()
            new_cols = {}

            for col in feature_cols:
                shifted = df.groupby(engine_col)[col].shift(lag)
                shifted = shifted.groupby(df[engine_col]).transform(lambda x: x.bfill())

                new_cols[f'{col}_delta{lag}'] = df[col] - shifted
            
            return pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)
        
        except Exception as e:
            raise PredictiveMaintenanceException(e, sys)
    
    def __rolling_slope(self, series : pd.Series, window=20) -> pd.Series:
        n = len(series)
        slopes = np.full(n, np.nan)
        vals = series.values
        x = np.arange(window, dtype=float)
        x_mean = x.mean()
        x_var = ((x - x_mean) ** 2).sum()

        for i in range(window-1, n):
            y = vals[i - window + 1 : i + 1]
            slopes[i] = np.dot(x - x_mean, y - y.mean()) / x_var
        
        first_valid = next((i for i, v in enumerate(slopes) if not np.isnan(v)), 0)
        slopes[:first_valid] = slopes[first_valid] if first_valid < n else 0.0

        return pd.Series(slopes, index=series.index)
    
    def add_slope_features(self, df : pd.DataFrame, feature_cols : list[str], engine_col : str, window = 20) -> pd.DataFrame:
        try:
            df = df.copy()
            new_cols = {}

            for col in feature_cols:
                key = f'{col}_slope{window}'
                new_cols[key] = df.groupby(engine_col)[col].transform(lambda x: self.__rolling_slope(x, window))
            
            return pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)
        
        except Exception as e:
            raise PredictiveMaintenanceException(e, sys)
    
    def add_health_index(self, df : pd.DataFrame, degrade_up : list[str], degrade_down : list[str]) -> pd.DataFrame:
        try:
            df = df.copy()
            up_contributions = 1.0 - df[degrade_up]
            down_contributions = df[degrade_down]

            all_contributions = pd.concat([up_contributions, down_contributions], axis=1)
            df['health_index'] = 1 - all_contributions.mean(axis=1)

            return df
        
        except Exception as e:
            raise PredictiveMaintenanceException(e, sys)
    
    def add_baseline_deviations(self, df : pd.DataFrame, feature_cols : list[str], engine_col : str) -> pd.DataFrame:
        try:
            df = df.copy()
            new_cols = {}

            eng_first = df.groupby(engine_col)[feature_cols].transform('first')

            for col in feature_cols:
                new_cols[f'{col}_absdev'] = (df[col] - eng_first[col]).abs()
            
            return pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)
        
        except Exception as e:
            raise PredictiveMaintenanceException(e, sys)
    
    def initiate_feature_extraction(self):
        try:
            df = self.df.copy()

            final_df = df.pipe(self.add_cycle_features, self.CYCLE_COL).pipe(self.normalize_data, self.FEATURE_COLS, self.SCALER, self.FIT).pipe(self.add_rolling_features, self.FEATURE_COLS, self.ENGINE_COL).pipe(self.add_delta_features, self.FEATURE_COLS, self.ENGINE_COL).pipe(self.add_slope_features, self.FEATURE_COLS, self.ENGINE_COL).pipe(self.add_health_index, self.DEGRADE_UP, self.DEGRADE_DOWN).pipe(self.add_baseline_deviations, self.SENSOR_NORM_COLS, self.ENGINE_COL)

            final_df.drop(self.EXCLUDE_COLS + self.FEATURE_COLS, inplace=True, axis=1)

            return final_df

        except Exception as e:
            raise PredictiveMaintenanceException(e, sys)