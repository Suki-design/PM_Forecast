import pandas as pd, numpy as np
from sklearn.preprocessing import StandardScaler
import torch

def split_xy(df: pd.DataFrame, feature_cols, target_col="pm25"):
    X = df[feature_cols].copy()
    y = df[target_col].copy()
    return X, y

def fit_scaler_on_train(train_df: pd.DataFrame, feature_cols):
    scaler = StandardScaler()
    scaler.fit(train_df[feature_cols].values)
    return scaler

def transform(df: pd.DataFrame, scaler: StandardScaler, feature_cols):
    x_scaled = scaler.transform(df[feature_cols].values)
    x_df = pd.DataFrame(x_scaled, index=df.index, columns=feature_cols)
    y_sr = df["pm25"].astype(np.float32)
    return x_df, y_sr

def make_sequences(x_df: pd.DataFrame, y_sr: pd.Series, window_size: int):
    x_np = x_df.to_numpy(dtype=np.float32)
    y_np = y_sr.to_numpy(dtype=np.float32)
    t, d = x_np.shape
    n = t - window_size
    X = np.zeros((n, window_size, d), dtype=np.float32)
    y = np.zeros((n,), dtype=np.float32)
    for i in range(n):
        X[i] = x_np[i : i + window_size]
        y[i] = y_np[i + window_size]
    return torch.from_numpy(X), torch.from_numpy(y)
