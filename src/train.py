"""
WeatherIQ - Model Training & Selection Pipeline
Trains candidate regression models (Linear Regression, Random Forest, HistGradientBoosting),
evaluates against Persistence Baseline, selects the best model, and safely serializes artifacts.
"""

import argparse
import sys
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Tuple, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor

from src.config import (
    DEFAULT_MODEL_PATH, 
    DEFAULT_METADATA_PATH, 
    CORE_INPUT_FEATURES, 
    TARGET_FEATURE,
    DEFAULT_HORIZON_HOURS
)
from src.data_loader import load_csv, detect_column_mapping
from src.data_validation import validate_weather_dataframe
from src.preprocessing import clean_weather_dataframe
from src.feature_engineering import create_supervised_dataset, chronological_train_test_split
from src.evaluate import calculate_metrics, evaluate_model_with_baseline


def build_candidate_models() -> Dict[str, Pipeline]:
    """
    Returns candidate model pipelines to be trained and compared.
    """
    return {
        "Linear Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("regressor", LinearRegression())
        ]),
        "Random Forest": Pipeline([
            ("scaler", StandardScaler()),
            ("regressor", RandomForestRegressor(n_estimators=100, max_depth=12, random_state=42))
        ]),
        "HistGradientBoosting": Pipeline([
            ("scaler", StandardScaler()),
            ("regressor", HistGradientBoostingRegressor(max_iter=150, max_depth=8, random_state=42))
        ])
    }


def train_and_select_best_model(
    clean_df: pd.DataFrame,
    horizon_hours: int = DEFAULT_HORIZON_HOURS,
    model_save_path: Path = DEFAULT_MODEL_PATH,
    metadata_save_path: Path = DEFAULT_METADATA_PATH
) -> Dict[str, Any]:
    """
    Executes end-to-end chronological time series modeling:
    1. Constructs features X and horizon target y.
    2. Chronological 3-way split: Train (70%), Validation (15%), Test (15%).
    3. Evaluates all candidates on Validation set to pick the winner.
    4. Refits the winner on Train + Validation.
    5. Evaluates the final selected model on Held-Out Test set.
    6. Compares with Persistence Baseline.
    7. Atomically saves model and metadata (safeguarding prior model).
    """
    # 1. Construct dataset
    X, y, ts_X, ts_y = create_supervised_dataset(clean_df, horizon_steps=horizon_hours)

    n_samples = len(X)
    if n_samples < 30:
        raise ValueError(f"Need at least 30 samples to train and validate, got {n_samples}.")

    # 2. Chronological 3-way partition
    train_end = int(n_samples * 0.70)
    val_end = int(n_samples * 0.85)

    X_train, y_train = X.iloc[:train_end], y.iloc[:train_end]
    X_val, y_val = X.iloc[train_end:val_end], y.iloc[train_end:val_end]
    X_test, y_test = X.iloc[val_end:], y.iloc[val_end:]
    ts_test = ts_y.iloc[val_end:]

    # 3. Fit candidates and evaluate on Validation set
    candidates = build_candidate_models()
    val_results = {}

    for name, pipeline in candidates.items():
        pipeline.fit(X_train, y_train)
        val_pred = pipeline.predict(X_val)
        metrics = calculate_metrics(y_val.to_numpy(), val_pred)
        val_results[name] = {
            "metrics": metrics,
            "pipeline": pipeline
        }

    # Best model by lowest validation MAE
    best_name = min(val_results, key=lambda k: val_results[k]["metrics"]["mae"])
    best_candidate = val_results[best_name]

    # 4. Refit best model on Train + Val combined
    X_train_val = pd.concat([X_train, X_val], axis=0).reset_index(drop=True)
    y_train_val = pd.concat([y_train, y_val], axis=0).reset_index(drop=True)

    final_pipeline = best_candidate["pipeline"]
    final_pipeline.fit(X_train_val, y_train_val)

    # 5. Evaluate on untouched Test Set
    test_pred = final_pipeline.predict(X_test)
    test_evaluation = evaluate_model_with_baseline(
        y_test=y_test.to_numpy(),
        y_pred=test_pred,
        X_test_temp=X_test["temperature"].to_numpy()
    )

    # Evaluate all candidate models on test set for reporting comparison table
    test_comparison = {}
    for name, cand in val_results.items():
        cand["pipeline"].fit(X_train_val, y_train_val)
        pred_i = cand["pipeline"].predict(X_test)
        test_comparison[name] = calculate_metrics(y_test.to_numpy(), pred_i)

    # Persistence baseline metrics
    persistence_metrics = test_evaluation["persistence_metrics"]

    # 6. Generate Model Metadata
    model_version = f"v_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{str(uuid.uuid4())[:6]}"
    
    metadata = {
        "model_version": model_version,
        "model_name": best_name,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "horizon_hours": horizon_hours,
        "input_features": CORE_INPUT_FEATURES,
        "target_feature": TARGET_FEATURE,
        "training_dataset_size": len(X_train_val),
        "test_dataset_size": len(X_test),
        "data_period_start": str(clean_df["timestamp"].min()),
        "data_period_end": str(clean_df["timestamp"].max()),
        "test_period_start": str(ts_test.min()),
        "test_period_end": str(ts_test.max()),
        "validation_results": {k: v["metrics"] for k, v in val_results.items()},
        "test_metrics": test_evaluation["model_metrics"],
        "persistence_metrics": persistence_metrics,
        "mae_improvement_c": test_evaluation["mae_improvement_c"],
        "percent_improvement": test_evaluation["percent_improvement"],
        "outperformed_baseline": test_evaluation["outperformed_baseline"],
        "all_models_comparison": test_comparison,
        # Sample test evaluation records for plotting
        "test_samples": {
            "timestamps": [str(t) for t in ts_test.tolist()[:300]],
            "actual": [round(float(a), 2) for a in y_test.tolist()[:300]],
            "predicted": [round(float(p), 2) for p in test_pred.tolist()[:300]],
            "persistence": [round(float(ps), 2) for ps in X_test["temperature"].tolist()[:300]]
        }
    }

    # 7. Atomic Model Persistence (save to temp file then replace, ensuring existing model is preserved on fail)
    temp_model_path = model_save_path.with_suffix(".tmp")
    temp_meta_path = metadata_save_path.with_suffix(".tmp")

    try:
        joblib.dump(final_pipeline, temp_model_path)
        with open(temp_meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        # Replace existing files
        temp_model_path.replace(model_save_path)
        temp_meta_path.replace(metadata_save_path)
    except Exception as e:
        if temp_model_path.exists():
            temp_model_path.unlink()
        if temp_meta_path.exists():
            temp_meta_path.unlink()
        raise RuntimeError(f"Failed to persist model artifacts: {str(e)}")

    return {
        "pipeline": final_pipeline,
        "metadata": metadata,
        "test_evaluation": test_evaluation
    }


def run_training_cli(args_list: Optional[list] = None) -> int:
    """
    CLI command handler for standalone model training.
    """
    parser = argparse.ArgumentParser(description="WeatherIQ Model Training Pipeline")
    parser.add_argument("--data", type=str, required=True, help="Path to historical weather CSV")
    parser.add_argument("--horizon", type=int, default=1, help="Prediction horizon in hours (default: 1)")
    parser.add_argument("--output-model", type=str, default=str(DEFAULT_MODEL_PATH), help="Model destination path")
    parser.add_argument("--output-meta", type=str, default=str(DEFAULT_METADATA_PATH), help="Metadata destination path")
    parser.add_argument("--timestamp-col", type=str, default=None, help="Explicit timestamp column name")
    parser.add_argument("--temp-col", type=str, default=None, help="Explicit temperature column name")
    parser.add_argument("--humidity-col", type=str, default=None, help="Explicit relative humidity column name")
    parser.add_argument("--wind-col", type=str, default=None, help="Explicit wind speed column name")

    args = parser.parse_args(args_list)

    csv_path = Path(args.data)
    if not csv_path.exists():
        print(f"[ERROR] Data file not found: {csv_path}", file=sys.stderr)
        return 1

    print("==================================================")
    print(" WeatherIQ Model Training Engine")
    print("==================================================")
    print(f"Loading data from: {csv_path}")

    try:
        raw_df = load_csv(csv_path)
        print(f"Loaded {len(raw_df)} rows and {len(raw_df.columns)} columns.")

        # Detect or assign column mappings
        detected = detect_column_mapping(list(raw_df.columns))
        mapping = {
            "timestamp": args.timestamp_col or detected.get("timestamp"),
            "temperature": args.temp_col or detected.get("temperature"),
            "relative_humidity": args.humidity_col or detected.get("relative_humidity"),
            "wind_speed": args.wind_col or detected.get("wind_speed")
        }

        print("Column Mapping:")
        for k, v in mapping.items():
            print(f"  {k} -> {v}")

        # Validate
        val_report = validate_weather_dataframe(raw_df, mapping)
        if not val_report.is_valid:
            print("[ERROR] Dataset validation failed:", file=sys.stderr)
            for err in val_report.error_details:
                print(f"  - {err}", file=sys.stderr)
            return 2

        # Clean
        clean_df, clean_report = clean_weather_dataframe(raw_df, mapping)
        print(f"Data cleaned successfully. Valid records: {len(clean_df)}")
        for msg in clean_report["log_messages"]:
            print(f"  * {msg}")

        # Train and select best model
        print(f"Training models for {args.horizon}-hour ahead forecasting...")
        results = train_and_select_best_model(
            clean_df=clean_df,
            horizon_hours=args.horizon,
            model_save_path=Path(args.output_model),
            metadata_save_path=Path(args.output_meta)
        )

        meta = results["metadata"]
        print("\n==================================================")
        print(f" Selected Best Model: {meta['model_name']} ({meta['model_version']})")
        print("==================================================")
        print(f" Training Samples: {meta['training_dataset_size']}")
        print(f" Test Samples:     {meta['test_dataset_size']}")
        print(f" Test MAE:         {meta['test_metrics']['mae']:.3f} °C")
        print(f" Test RMSE:        {meta['test_metrics']['rmse']:.3f} °C")
        print(f" Test R²:          {meta['test_metrics']['r2']:.4f}")
        print("--------------------------------------------------")
        print(f" Persistence Baseline MAE:  {meta['persistence_metrics']['mae']:.3f} °C")
        print(f" Model MAE Improvement:     {meta['mae_improvement_c']:+.3f} °C ({meta['percent_improvement']:+.1f}%)")
        print("--------------------------------------------------")
        print(f" Model saved to:    {args.output_model}")
        print(f" Metadata saved to: {args.output_meta}")
        print(" Training completed successfully.")
        return 0

    except Exception as e:
        print(f"\n[FATAL ERROR] Training pipeline failed: {str(e)}", file=sys.stderr)
        return 3


if __name__ == "__main__":
    sys.exit(run_training_cli())
