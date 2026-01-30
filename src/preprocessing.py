
## Separation of Concerns
## The data preprocessing logic is separated from the main code and placed in an independent module.
# 分離關注點
# 將資料前處理邏輯從主要程式碼中分離出來，放在獨立的模組中

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def clean_robot_data(df):

    # Cleaning: Drop empty axes (9-14) and convert time format
    # Drop empty columns
    df_cleaned = df.dropna(axis=1, how='all').copy()
    
    # Ensure correct time format
    if 'Time' in df_cleaned.columns:
        df_cleaned['Time'] = pd.to_datetime(df_cleaned['Time'])
        # Calculate elapsed seconds as a feature
        df_cleaned['Elapsed_Seconds'] = (df_cleaned['Time'] - df_cleaned['Time'].min()).dt.total_seconds()
    
    return df_cleaned

def apply_smoothing(df, column_name, window_size=50):

    # Smoothing: Use moving average to filter noise

    df_smoothed = df.copy()
    # Calculate rolling mean
    df_smoothed[f'{column_name}_smooth'] = df_smoothed[column_name].rolling(window=window_size).mean()
    # NaN Fill NaNs from rolling window
    df_smoothed = df_smoothed.bfill()
    return df_smoothed

def prepare_features(df, feature_cols, target_col):

    # Feature preparation: Includes standardization as required by the workshop

    scaler = StandardScaler()
    
    # Extract X and y
    X = df[feature_cols].values
    y = df[target_col].values
    
    # Standardize features (Workshop requirement)
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y, scaler