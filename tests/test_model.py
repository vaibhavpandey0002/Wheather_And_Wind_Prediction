"""
Unit Tests for Model Training, Persistence Baseline, Serialization, and Prediction
"""

import json
from pathlib import Path
import pytest
import numpy as np
import pandas as pd

from src.train import train_and_select_best_model
from src.predict import predict_temperature, load_model_and_metadata
from src.data_validation import WeatherPredictionInput
from scripts.generate_sample_data import generate_synthetic_weather_dataset


@pytest.fixture(scope="session")
def sample_dataset(tmp_path_factory):
    """Generates a small deterministic dataset for testing."""
    tmp_dir = tmp_path_factory.mktemp("weather_data")
    csv_path = tmp_dir / "test_weather.csv"
    
    np.random.seed(42)
    n = 120
    timestamps = pd.date_range("2024-01-01", periods=n, freq="h")
    temps = 20.0 + 5.0 * np.sin(np.linspace(0, 4 * np.pi, n)) + np.random.normal(0, 0.5, n)
    rh = np.clip(80.0 - 1.5 * (temps - 15.0), 20.0, 95.0)
    ws = np.clip(10.0 + np.random.normal(0, 2.0, n), 1.0, 40.0)

    df = pd.DataFrame({
        "timestamp": timestamps,
        "temperature": temps.round(2),
        "relative_humidity": rh.round(2),
        "wind_speed": ws.round(2)
    })
    df.to_csv(csv_path, index=False)
    return df, csv_path


def test_model_training_and_selection(sample_dataset, tmp_path):
    """Verifies that model training completes, selects best algorithm, and produces valid metrics."""
    df, _ = sample_dataset
    model_path = tmp_path / "test_model.joblib"
    meta_path = tmp_path / "test_meta.json"

    result = train_and_select_best_model(
        clean_df=df,
        horizon_hours=1,
        model_save_path=model_path,
        metadata_save_path=meta_path
    )

    assert model_path.exists()
    assert meta_path.exists()
    
    meta = result["metadata"]
    assert "model_name" in meta
    assert "test_metrics" in meta
    assert meta["test_metrics"]["mae"] > 0
    assert meta["test_metrics"]["rmse"] > 0
    assert "persistence_metrics" in meta
    assert meta["persistence_metrics"]["mae"] > 0


def test_prediction_output_validity(sample_dataset, tmp_path):
    """Verifies that trained model generates finite, realistic predictions."""
    df, _ = sample_dataset
    model_path = tmp_path / "test_model.joblib"
    meta_path = tmp_path / "test_meta.json"

    train_and_select_best_model(
        clean_df=df,
        horizon_hours=1,
        model_save_path=model_path,
        metadata_save_path=meta_path
    )

    pred_input = WeatherPredictionInput(
        temperature=24.0,
        relative_humidity=60.0,
        wind_speed=12.0,
        horizon_hours=1
    )

    pred_result = predict_temperature(
        input_data=pred_input,
        model_path=model_path,
        meta_path=meta_path
    )

    assert isinstance(pred_result.predicted_temperature, float)
    assert np.isfinite(pred_result.predicted_temperature)
    # Predicted temperature should be close to 24°C (reasonable thermal change)
    assert abs(pred_result.predicted_temperature - 24.0) < 15.0
    assert pred_result.horizon_hours == 1
    assert len(pred_result.explanation) > 10


def test_missing_model_raises_file_not_found(tmp_path):
    """Verifies that attempting prediction without a trained model raises FileNotFoundError."""
    non_existent_model = tmp_path / "does_not_exist.joblib"
    non_existent_meta = tmp_path / "does_not_exist.json"

    pred_input = WeatherPredictionInput(
        temperature=20.0,
        relative_humidity=50.0,
        wind_speed=10.0,
        horizon_hours=1
    )

    with pytest.raises(FileNotFoundError):
        predict_temperature(
            input_data=pred_input,
            model_path=non_existent_model,
            meta_path=non_existent_meta
        )
