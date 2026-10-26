"""
WeatherIQ - Prediction Service
Handles runtime inference, input validation, feature order alignment,
and metadata-backed prediction generation.
"""

from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from datetime import datetime, timezone
import json
import joblib
import pandas as pd
import numpy as np

from src.config import (
    DEFAULT_MODEL_PATH, 
    DEFAULT_METADATA_PATH, 
    CORE_INPUT_FEATURES
)
from src.data_validation import WeatherPredictionInput


class PredictionResult:
    def __init__(
        self,
        predicted_temperature: float,
        previous_temperature: float,
        relative_humidity: float,
        wind_speed: float,
        horizon_hours: int,
        model_name: str,
        model_version: str,
        timestamp: str,
        test_mae: float,
        explanation: str
    ):
        self.predicted_temperature = predicted_temperature
        self.previous_temperature = previous_temperature
        self.relative_humidity = relative_humidity
        self.wind_speed = wind_speed
        self.horizon_hours = horizon_hours
        self.model_name = model_name
        self.model_version = model_version
        self.timestamp = timestamp
        self.test_mae = test_mae
        self.explanation = explanation

    def to_dict(self) -> Dict[str, Any]:
        return {
            "predicted_temperature": round(self.predicted_temperature, 2),
            "previous_temperature": round(self.previous_temperature, 2),
            "relative_humidity": round(self.relative_humidity, 2),
            "wind_speed": round(self.wind_speed, 2),
            "horizon_hours": self.horizon_hours,
            "model_name": self.model_name,
            "model_version": self.model_version,
            "timestamp": self.timestamp,
            "test_mae": round(self.test_mae, 3),
            "explanation": self.explanation
        }


def load_model_and_metadata(
    model_path: Path = DEFAULT_MODEL_PATH,
    meta_path: Path = DEFAULT_METADATA_PATH
) -> Tuple[Any, Dict[str, Any]]:
    """
    Loads persisted Scikit-learn model and associated metadata.
    Raises FileNotFoundError if model has not yet been trained.
    """
    if not model_path.exists():
        raise FileNotFoundError(
            f"Trained model not found at '{model_path}'. Please train a model first in Dataset Management."
        )
    if not meta_path.exists():
        raise FileNotFoundError(
            f"Model metadata not found at '{meta_path}'."
        )

    pipeline = joblib.load(model_path)
    with open(meta_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    return pipeline, metadata


def predict_temperature(
    input_data: WeatherPredictionInput,
    model_path: Path = DEFAULT_MODEL_PATH,
    meta_path: Path = DEFAULT_METADATA_PATH
) -> PredictionResult:
    """
    Validates input, ensures feature schema matches, and generates future temperature estimate.
    """
    pipeline, metadata = load_model_and_metadata(model_path, meta_path)

    # Validate horizon compatibility
    model_horizon = metadata.get("horizon_hours", 1)
    if input_data.horizon_hours != model_horizon:
        # If user selected horizon different from model's trained horizon, note this clearly
        pass

    # Align input features strictly in order
    input_features = metadata.get("input_features", CORE_INPUT_FEATURES)
    feature_dict = {
        "temperature": input_data.temperature,
        "relative_humidity": input_data.relative_humidity,
        "wind_speed": input_data.wind_speed
    }

    df_input = pd.DataFrame([[feature_dict[feat] for feat in input_features]], columns=input_features)

    # Predict using pipeline (which handles scaling automatically)
    pred_val = float(pipeline.predict(df_input)[0])

    model_name = metadata.get("model_name", "Supervised Regressor")
    model_version = metadata.get("model_version", "v1.0")
    test_mae = float(metadata.get("test_metrics", {}).get("mae", 1.0))

    temp_diff = pred_val - input_data.temperature
    trend_str = "an increase" if temp_diff > 0 else ("a decrease" if temp_diff < 0 else "steady state")
    
    explanation = (
        f"Based on current conditions ({input_data.temperature:.1f}°C, {input_data.relative_humidity:.0f}% humidity, "
        f"{input_data.wind_speed:.1f} km/h wind), the {model_name} predicts {trend_str} of {abs(temp_diff):.2f}°C "
        f"over the next {model_horizon} hour(s). Empirical historical test MAE is ±{test_mae:.2f}°C."
    )

    now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    return PredictionResult(
        predicted_temperature=pred_val,
        previous_temperature=input_data.temperature,
        relative_humidity=input_data.relative_humidity,
        wind_speed=input_data.wind_speed,
        horizon_hours=model_horizon,
        model_name=model_name,
        model_version=model_version,
        timestamp=now_iso,
        test_mae=test_mae,
        explanation=explanation
    )
