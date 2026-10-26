"""
Unit Tests for Preprocessing and Chronological Feature Engineering
"""

import pytest
import pandas as pd
import numpy as np

from src.preprocessing import clean_weather_dataframe
from src.feature_engineering import create_supervised_dataset, chronological_train_test_split


def test_clean_dataframe_preserves_zeros_and_sorts():
    """Verifies that 0.0 values are preserved, nulls interpolated, and order sorted."""
    df = pd.DataFrame({
        "time": ["2024-01-01 02:00:00", "2024-01-01 00:00:00", "2024-01-01 01:00:00", "2024-01-01 03:00:00"],
        "temp": [0.0, 10.0, np.nan, 12.0],  # 0.0 must NOT be treated as missing
        "rh": [50.0, 60.0, 55.0, 52.0],
        "ws": [0.0, 5.0, 4.0, 3.0]          # 0.0 km/h wind is calm, perfectly valid
    })
    mapping = {
        "timestamp": "time",
        "temperature": "temp",
        "relative_humidity": "rh",
        "wind_speed": "ws"
    }
    clean_df, report = clean_weather_dataframe(df, mapping)
    
    # Must be sorted chronologically
    assert list(clean_df["timestamp"]) == sorted(list(clean_df["timestamp"]))
    # 0.0 temperature at 02:00 must be preserved
    row_02 = clean_df[clean_df["timestamp"] == pd.to_datetime("2024-01-01 02:00:00")]
    assert float(row_02["temperature"].iloc[0]) == 0.0
    # 0.0 wind speed at 02:00 must be preserved
    assert float(row_02["wind_speed"].iloc[0]) == 0.0
    # Missing value at 01:00 should be interpolated between 10.0 and 0.0 -> 5.0
    row_01 = clean_df[clean_df["timestamp"] == pd.to_datetime("2024-01-01 01:00:00")]
    assert abs(float(row_01["temperature"].iloc[0]) - 5.0) < 0.1


def test_supervised_dataset_target_alignment_and_no_leakage():
    """Verifies that future target is aligned at horizon without future data leakage."""
    n = 20
    timestamps = [f"2024-01-01 {i:02d}:00:00" for i in range(n)]
    temps = [20.0 + i for i in range(n)]  # 20, 21, 22, ...
    
    df = pd.DataFrame({
        "timestamp": pd.to_datetime(timestamps),
        "temperature": temps,
        "relative_humidity": [50.0] * n,
        "wind_speed": [10.0] * n
    })

    # Horizon = 1 step
    X, y, ts_X, ts_y = create_supervised_dataset(df, horizon_steps=1)

    assert len(X) == n - 1
    assert len(y) == n - 1
    
    # Target y[0] should be temperature at index 1 (21.0)
    assert float(y.iloc[0]) == float(df["temperature"].iloc[1])
    # Feature X[0] temperature should be temperature at index 0 (20.0)
    assert float(X["temperature"].iloc[0]) == float(df["temperature"].iloc[0])

    # Ensure X strictly contains only current features, no future targets
    assert "target_temperature" not in X.columns
    assert set(X.columns) == {"temperature", "relative_humidity", "wind_speed"}


def test_chronological_train_test_split():
    """Verifies that time-series split does not shuffle observations."""
    n = 50
    df = pd.DataFrame({
        "temperature": np.arange(n, dtype=float),
        "relative_humidity": np.full(n, 50.0),
        "wind_speed": np.full(n, 10.0)
    })
    y = pd.Series(np.arange(1, n + 1, dtype=float))
    timestamps = pd.Series(pd.date_range("2024-01-01", periods=n, freq="h"))

    split = chronological_train_test_split(df, y, timestamps, test_ratio=0.2)

    # Train size should be 80% (40), test 20% (10)
    assert split["train_size"] == 40
    assert split["test_size"] == 10
    
    # Train max timestamp must be strictly less than test min timestamp (no future overlap)
    assert pd.to_datetime(split["ts_train"].max()) < pd.to_datetime(split["ts_test"].min())
