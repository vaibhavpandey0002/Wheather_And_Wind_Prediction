"""
WeatherIQ - Preprocessing Module
Cleans raw weather datasets, handles duplicate timestamps, imputes missing values
safely without silent row deletion, and records auditing reports.
"""

from typing import Dict, List, Tuple, Any, Optional
import pandas as pd
import numpy as np

from src.config import DEFAULT_BOUNDS


class CleaningReport:
    def __init__(self):
        self.initial_rows: int = 0
        self.final_rows: int = 0
        self.duplicates_removed: int = 0
        self.invalid_timestamps_dropped: int = 0
        self.interpolated_values: Dict[str, int] = {}
        self.out_of_bounds_clipped: Dict[str, int] = {}
        self.log_messages: List[str] = []

    def to_dict(self) -> Dict[str, Any]:
        return {
            "initial_rows": self.initial_rows,
            "final_rows": self.final_rows,
            "rows_net_change": self.final_rows - self.initial_rows,
            "duplicates_removed": self.duplicates_removed,
            "invalid_timestamps_dropped": self.invalid_timestamps_dropped,
            "interpolated_values": self.interpolated_values,
            "out_of_bounds_clipped": self.out_of_bounds_clipped,
            "log_messages": self.log_messages
        }


def clean_weather_dataframe(
    df: pd.DataFrame,
    mapping: Dict[str, str],
    max_interpolation_gap: int = 3
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Standardizes column names, sorts chronologically, resolves duplicate timestamps,
    safely handles nulls, and validates physical limits.

    Returns:
        (cleaned_df, report_dict)
    """
    report = CleaningReport()
    report.initial_rows = len(df)

    # 1. Select and rename mapped columns to standardized names
    cols_to_extract = {
        mapping["timestamp"]: "timestamp",
        mapping["temperature"]: "temperature",
        mapping["relative_humidity"]: "relative_humidity",
        mapping["wind_speed"]: "wind_speed"
    }

    # Verify presence
    for src_col in cols_to_extract.keys():
        if src_col not in df.columns:
            raise KeyError(f"Mapped column '{src_col}' not found in DataFrame.")

    clean_df = df[list(cols_to_extract.keys())].copy()
    clean_df.rename(columns=cols_to_extract, inplace=True)

    # 2. Parse timestamps
    clean_df["timestamp"] = pd.to_datetime(clean_df["timestamp"], errors="coerce")
    null_ts = clean_df["timestamp"].isna().sum()
    if null_ts > 0:
        report.invalid_timestamps_dropped = int(null_ts)
        report.log_messages.append(
            f"Removed {null_ts} rows with unparseable or null timestamps."
        )
        clean_df = clean_df.dropna(subset=["timestamp"])

    # 3. Coerce numeric columns (ensure zeros are preserved, only truly invalid strings become NaN)
    for num_col in ["temperature", "relative_humidity", "wind_speed"]:
        clean_df[num_col] = pd.to_numeric(clean_df[num_col], errors="coerce")

    # 4. Sort chronologically
    clean_df.sort_values(by="timestamp", inplace=True)

    # 5. Handle duplicate timestamps by taking mean of numeric features
    dups_count = clean_df.duplicated(subset=["timestamp"]).sum()
    if dups_count > 0:
        report.duplicates_removed = int(dups_count)
        clean_df = clean_df.groupby("timestamp", as_index=False).agg({
            "temperature": "mean",
            "relative_humidity": "mean",
            "wind_speed": "mean"
        })
        report.log_messages.append(
            f"Consolidated {dups_count} duplicate timestamp records by taking average readings."
        )

    # 6. Set timestamp as index for time-based interpolation of small gaps
    clean_df.set_index("timestamp", inplace=True)

    for num_col in ["temperature", "relative_humidity", "wind_speed"]:
        nan_before = int(clean_df[num_col].isna().sum())
        if nan_before > 0:
            # Interpolate linearly with a limit on consecutive gaps
            clean_df[num_col] = clean_df[num_col].interpolate(
                method="time", 
                limit=max_interpolation_gap, 
                limit_direction="both"
            )
            # If any remaining at endpoints, backward/forward fill
            clean_df[num_col] = clean_df[num_col].bfill().ffill()
            nan_after = int(clean_df[num_col].isna().sum())
            interpolated_count = nan_before - nan_after
            report.interpolated_values[num_col] = interpolated_count
            report.log_messages.append(
                f"Interpolated {interpolated_count} missing values in '{num_col}'."
            )

    clean_df.reset_index(inplace=True)

    # Drop any remaining unrecoverable rows
    clean_df.dropna(subset=["temperature", "relative_humidity", "wind_speed"], inplace=True)

    # 7. Physical boundary checks and conservative clipping
    # Note: 0 is completely valid for temperature or wind speed; do not touch zero!
    clip_rules = {
        "temperature": (DEFAULT_BOUNDS.temp_min, DEFAULT_BOUNDS.temp_max),
        "relative_humidity": (DEFAULT_BOUNDS.humidity_min, DEFAULT_BOUNDS.humidity_max),
        "wind_speed": (DEFAULT_BOUNDS.wind_min, DEFAULT_BOUNDS.wind_max),
    }

    for col, (low, high) in clip_rules.items():
        oob_mask = (clean_df[col] < low) | (clean_df[col] > high)
        oob_count = int(oob_mask.sum())
        if oob_count > 0:
            clean_df[col] = clean_df[col].clip(lower=low, upper=high)
            report.out_of_bounds_clipped[col] = oob_count
            report.log_messages.append(
                f"Clipped {oob_count} extreme readings in '{col}' to physical limits [{low}, {high}]."
            )

    # Ensure clean rounding
    clean_df["temperature"] = clean_df["temperature"].round(2)
    clean_df["relative_humidity"] = clean_df["relative_humidity"].round(2)
    clean_df["wind_speed"] = clean_df["wind_speed"].round(2)

    report.final_rows = len(clean_df)
    report.log_messages.append(
        f"Preprocessing completed: {report.final_rows} valid chronological records ready."
    )

    return clean_df, report.to_dict()
