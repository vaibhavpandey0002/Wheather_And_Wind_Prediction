"""
WeatherIQ - Feature Engineering Module
Builds chronological supervised features and targets without data leakage,
and performs chronological time-series splitting.
"""

from typing import Tuple, List, Dict, Any, Optional
import pandas as pd
import numpy as np

from src.config import CORE_INPUT_FEATURES, TARGET_FEATURE


def create_supervised_dataset(
    clean_df: pd.DataFrame,
    horizon_steps: int = 1
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Constructs supervised input features (X) and future target (y) chronologically.
    
    Given observations at timestamp t:
        X_t = [temperature_t, relative_humidity_t, wind_speed_t]
        y_t = temperature_{t + horizon_steps}

    Strictly ensures no future information leaks into X.
    
    Returns:
        (X, y, timestamps_X, timestamps_y)
    """
    if len(clean_df) <= horizon_steps:
        raise ValueError(
            f"Dataset length ({len(clean_df)}) is too short for horizon {horizon_steps}."
        )

    df_proc = clean_df.copy().sort_values("timestamp").reset_index(drop=True)

    # Core features at time t
    X = df_proc[CORE_INPUT_FEATURES].iloc[:-horizon_steps].copy()
    ts_X = df_proc["timestamp"].iloc[:-horizon_steps].copy()

    # Target is temperature at t + horizon_steps
    y = df_proc["temperature"].iloc[horizon_steps:].copy().reset_index(drop=True)
    ts_y = df_proc["timestamp"].iloc[horizon_steps:].copy().reset_index(drop=True)

    X.reset_index(drop=True, inplace=True)
    ts_X.reset_index(drop=True, inplace=True)

    return X, y, ts_X, ts_y


def chronological_train_test_split(
    X: pd.DataFrame,
    y: pd.Series,
    timestamps: pd.Series,
    test_ratio: float = 0.20
) -> Dict[str, Any]:
    """
    Splits time-series data chronologically without shuffling.
    The most recent `test_ratio` portion is reserved for the held-out test evaluation.
    """
    n_samples = len(X)
    if n_samples < 20:
        raise ValueError("Insufficient samples for train/test split (minimum 20 required).")

    split_idx = int(n_samples * (1.0 - test_ratio))
    # Ensure at least 5 samples in test set and 10 in train set
    split_idx = max(10, min(split_idx, n_samples - 5))

    X_train = X.iloc[:split_idx].copy().reset_index(drop=True)
    X_test = X.iloc[split_idx:].copy().reset_index(drop=True)
    y_train = y.iloc[:split_idx].copy().reset_index(drop=True)
    y_test = y.iloc[split_idx:].copy().reset_index(drop=True)
    ts_train = timestamps.iloc[:split_idx].copy().reset_index(drop=True)
    ts_test = timestamps.iloc[split_idx:].copy().reset_index(drop=True)

    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "ts_train": ts_train,
        "ts_test": ts_test,
        "train_size": len(X_train),
        "test_size": len(X_test),
        "train_period": (str(ts_train.min()), str(ts_train.max())),
        "test_period": (str(ts_test.min()), str(ts_test.max()))
    }
