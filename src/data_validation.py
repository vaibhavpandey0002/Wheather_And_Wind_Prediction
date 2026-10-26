"""
WeatherIQ - Data Validation Module
Provides Pydantic models and validation rules for input parameters, 
dataset schemas, and physical boundaries.
"""

from typing import Dict, List, Optional, Tuple, Any
from pydantic import BaseModel, Field, field_validator, model_validator
import pandas as pd
import numpy as np

from src.config import DEFAULT_BOUNDS, CORE_INPUT_FEATURES


class WeatherPredictionInput(BaseModel):
    """
    Validates manual single-point prediction input features.
    """
    temperature: float = Field(..., description="Previous observed temperature in °C")
    relative_humidity: float = Field(..., description="Observed relative humidity in %")
    wind_speed: float = Field(..., description="Observed wind speed in km/h")
    horizon_hours: int = Field(default=1, ge=1, le=72, description="Prediction horizon in hours")

    @field_validator("temperature")
    @classmethod
    def validate_temperature(cls, v: float) -> float:
        if v < DEFAULT_BOUNDS.temp_min or v > DEFAULT_BOUNDS.temp_max:
            raise ValueError(
                f"Temperature {v:.1f}°C is outside plausible meteorological limits "
                f"[{DEFAULT_BOUNDS.temp_min}°C, {DEFAULT_BOUNDS.temp_max}°C]."
            )
        return float(v)

    @field_validator("relative_humidity")
    @classmethod
    def validate_humidity(cls, v: float) -> float:
        if v < DEFAULT_BOUNDS.humidity_min or v > DEFAULT_BOUNDS.humidity_max:
            raise ValueError(
                f"Relative humidity {v:.1f}% must be between "
                f"{DEFAULT_BOUNDS.humidity_min}% and {DEFAULT_BOUNDS.humidity_max}%."
            )
        return float(v)

    @field_validator("wind_speed")
    @classmethod
    def validate_wind(cls, v: float) -> float:
        if v < DEFAULT_BOUNDS.wind_min or v > DEFAULT_BOUNDS.wind_max:
            raise ValueError(
                f"Wind speed {v:.1f} km/h must be non-negative and below "
                f"{DEFAULT_BOUNDS.wind_max} km/h."
            )
        return float(v)


class DatasetValidationReport(BaseModel):
    """
    Validation summary report for uploaded or fetched tabular datasets.
    """
    is_valid: bool
    total_rows: int
    column_mapping: Dict[str, str]
    missing_columns: List[str]
    missing_values: Dict[str, int]
    duplicate_timestamps: int
    sampling_interval_median_minutes: Optional[float] = None
    is_regular_sampling: bool = True
    out_of_bounds_counts: Dict[str, int] = {}
    validation_messages: List[str] = []
    error_details: List[str] = []


def validate_weather_dataframe(
    df: pd.DataFrame, 
    mapping: Dict[str, str],
    min_required_rows: int = 48
) -> DatasetValidationReport:
    """
    Performs comprehensive schema and domain integrity validation on a weather dataframe.
    """
    messages: List[str] = []
    errors: List[str] = []
    missing_cols: List[str] = []
    missing_vals: Dict[str, int] = {}
    out_of_bounds: Dict[str, int] = {}

    expected_logical_cols = ["timestamp", "temperature", "relative_humidity", "wind_speed"]

    # 1. Verify mapped columns exist in df
    for logical_col in expected_logical_cols:
        actual_col = mapping.get(logical_col)
        if not actual_col or actual_col not in df.columns:
            missing_cols.append(logical_col)
            errors.append(f"Missing required logical column '{logical_col}' (mapped to '{actual_col}')")

    if missing_cols:
        return DatasetValidationReport(
            is_valid=False,
            total_rows=len(df),
            column_mapping=mapping,
            missing_columns=missing_cols,
            missing_values={},
            duplicate_timestamps=0,
            validation_messages=messages,
            error_details=errors
        )

    # Work with mapped subset
    ts_col = mapping["timestamp"]
    temp_col = mapping["temperature"]
    hum_col = mapping["relative_humidity"]
    wind_col = mapping["wind_speed"]

    # 2. Minimum row count check
    total_rows = len(df)
    if total_rows < min_required_rows:
        errors.append(
            f"Dataset has only {total_rows} rows. Minimum {min_required_rows} rows required "
            f"for reliable chronological modeling."
        )

    # 3. Missing values count
    for logical, actual in [
        ("timestamp", ts_col), 
        ("temperature", temp_col), 
        ("relative_humidity", hum_col), 
        ("wind_speed", wind_col)
    ]:
        cnt = int(df[actual].isna().sum())
        missing_vals[logical] = cnt
        if cnt > 0:
            messages.append(f"Column '{actual}' ({logical}) has {cnt} missing/null entries.")

    # 4. Check timestamp parsing & duplicates
    duplicate_ts_count = 0
    median_interval = None
    is_regular = True

    try:
        ts_series = pd.to_datetime(df[ts_col], errors="coerce")
        null_ts = int(ts_series.isna().sum())
        if null_ts > 0:
            errors.append(f"Failed to parse {null_ts} timestamps in column '{ts_col}'.")

        # Check for duplicates on valid timestamps
        valid_ts = ts_series.dropna()
        duplicate_ts_count = int(valid_ts.duplicated().sum())
        if duplicate_ts_count > 0:
            messages.append(f"Detected {duplicate_ts_count} duplicate timestamp records.")

        # Sampling interval check
        if len(valid_ts) > 2:
            sorted_ts = valid_ts.sort_values().drop_duplicates()
            diffs = sorted_ts.diff().dropna().dt.total_seconds() / 60.0
            if len(diffs) > 0:
                median_interval = float(diffs.median())
                # If standard deviation of intervals is large, flag irregular sampling
                if diffs.std() > (0.3 * median_interval) and len(diffs) > 10:
                    is_regular = False
                    messages.append(
                        f"Sampling intervals vary noticeably (median: {median_interval:.1f} mins). "
                        "Time series may require interpolation or resampling."
                    )
                else:
                    messages.append(f"Detected uniform sampling interval: ~{median_interval:.0f} minutes.")
    except Exception as ex:
        errors.append(f"Error validating timestamps: {str(ex)}")

    # 5. Numerical bounds checking
    for col_name, log_name, low, high in [
        (temp_col, "temperature", DEFAULT_BOUNDS.temp_min, DEFAULT_BOUNDS.temp_max),
        (hum_col, "relative_humidity", DEFAULT_BOUNDS.humidity_min, DEFAULT_BOUNDS.humidity_max),
        (wind_col, "wind_speed", DEFAULT_BOUNDS.wind_min, DEFAULT_BOUNDS.wind_max),
    ]:
        numeric_series = pd.to_numeric(df[col_name], errors="coerce")
        non_num = int(numeric_series.isna().sum()) - int(df[col_name].isna().sum())
        if non_num > 0:
            errors.append(f"Column '{col_name}' contains {non_num} non-numeric values.")
        
        oob = int(((numeric_series < low) | (numeric_series > high)).sum())
        out_of_bounds[log_name] = oob
        if oob > 0:
            messages.append(
                f"Column '{col_name}' has {oob} values outside normal physical limits [{low}, {high}]."
            )

    is_valid = len(errors) == 0

    return DatasetValidationReport(
        is_valid=is_valid,
        total_rows=total_rows,
        column_mapping=mapping,
        missing_columns=missing_cols,
        missing_values=missing_vals,
        duplicate_timestamps=duplicate_ts_count,
        sampling_interval_median_minutes=median_interval,
        is_regular_sampling=is_regular,
        out_of_bounds_counts=out_of_bounds,
        validation_messages=messages,
        error_details=errors
    )
