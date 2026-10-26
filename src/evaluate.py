"""
WeatherIQ - Evaluation Module
Computes regression performance metrics (MAE, RMSE, R²), persistence baselines,
and residual distributions.
"""

from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calculates fundamental regression metrics: MAE, RMSE, and R2.
    """
    y_t = np.asarray(y_true, dtype=float)
    y_p = np.asarray(y_pred, dtype=float)

    mae = float(mean_absolute_error(y_t, y_p))
    rmse = float(np.sqrt(mean_squared_error(y_t, y_p)))
    r2 = float(r2_score(y_t, y_p))

    return {
        "mae": round(mae, 3),
        "rmse": round(rmse, 3),
        "r2": round(r2, 4)
    }


def evaluate_model_with_baseline(
    y_test: np.ndarray,
    y_pred: np.ndarray,
    X_test_temp: np.ndarray
) -> Dict[str, Any]:
    """
    Evaluates model predictions and compares them directly against the
    Persistence Baseline (which predicts that future temperature equals current temperature).
    """
    y_test = np.asarray(y_test, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    y_persistence = np.asarray(X_test_temp, dtype=float)

    model_metrics = calculate_metrics(y_test, y_pred)
    persistence_metrics = calculate_metrics(y_test, y_persistence)

    # MAE improvement: positive means model has lower error than persistence baseline
    mae_diff = persistence_metrics["mae"] - model_metrics["mae"]
    percent_improvement = 0.0
    if persistence_metrics["mae"] > 0:
        percent_improvement = (mae_diff / persistence_metrics["mae"]) * 100.0

    residuals = y_test - y_pred

    return {
        "model_metrics": model_metrics,
        "persistence_metrics": persistence_metrics,
        "mae_improvement_c": round(mae_diff, 3),
        "percent_improvement": round(percent_improvement, 2),
        "outperformed_baseline": bool(model_metrics["mae"] < persistence_metrics["mae"]),
        "residuals": residuals.tolist()
    }


def get_metric_explanations() -> Dict[str, str]:
    """
    Returns beginner-friendly educational explanations of ML evaluation metrics.
    """
    return {
        "MAE": (
            "Mean Absolute Error measures the average magnitude of prediction errors "
            "in the exact unit of temperature (°C). For example, an MAE of 1.2°C means "
            "the model's forecasts are on average 1.2°C off from the true observed temperature."
        ),
        "RMSE": (
            "Root Mean Squared Error penalizes larger errors more heavily than small ones. "
            "If RMSE is substantially higher than MAE, it indicates occasional large forecasting mistakes."
        ),
        "R2": (
            "The Coefficient of Determination (R²) measures the proportion of variance in future temperature "
            "explained by the model (1.0 is a perfect fit, 0.0 indicates performance equivalent to the mean). "
            "Note: R² is not an accuracy percentage and should never be reported as one."
        ),
        "Persistence Baseline": (
            "The meteorological benchmark heuristic that predicts future temperature will simply equal "
            "the most recent reading (T_future = T_current). Because weather changes smoothly over hours, "
            "a valid ML model must prove it can beat this naive baseline."
        )
    }
