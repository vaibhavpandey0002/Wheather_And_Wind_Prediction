"""
WeatherIQ - Data Loader Module
Handles CSV ingestion, heuristic column mapping detection, dataset summaries,
and raw data persistence.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
import io
import pandas as pd
import numpy as np

from src.config import COLUMN_ALIASES, DATA_DIR


def detect_column_mapping(columns: List[str]) -> Dict[str, Optional[str]]:
    """
    Intelligently maps raw CSV column names to logical WeatherIQ features
    using case-insensitive substring and synonym matching.
    """
    clean_cols = {c.strip().lower(): c for c in columns}
    mapping: Dict[str, Optional[str]] = {
        "timestamp": None,
        "temperature": None,
        "relative_humidity": None,
        "wind_speed": None
    }

    # 1. Exact alias match
    for logical, aliases in COLUMN_ALIASES.items():
        for alias in aliases:
            if alias.lower() in clean_cols:
                mapping[logical] = clean_cols[alias.lower()]
                break

    # 2. Substring fallback for unmapped
    for logical, aliases in COLUMN_ALIASES.items():
        if mapping[logical] is None:
            for clean_name, original_name in clean_cols.items():
                if any(alias in clean_name for alias in aliases):
                    mapping[logical] = original_name
                    break

    return mapping


def load_csv(
    source: Union[str, Path, io.BytesIO, io.StringIO], 
    encoding: str = "utf-8"
) -> pd.DataFrame:
    """
    Safely loads a CSV from a file path or file-like buffer into a pandas DataFrame.
    """
    try:
        df = pd.read_csv(source, encoding=encoding)
        return df
    except UnicodeDecodeError:
        # Fallback to latin1 if utf-8 fails
        if hasattr(source, "seek"):
            source.seek(0)
        return pd.read_csv(source, encoding="latin1")


def get_dataset_summary(df: pd.DataFrame, mapping: Dict[str, str]) -> Dict[str, Any]:
    """
    Generates a structured overview of the dataset for dashboard display.
    """
    total_records = len(df)
    cols = list(df.columns)
    
    # Check timestamps if mapped
    ts_min, ts_max, time_span_days = None, None, 0.0
    ts_col = mapping.get("timestamp")
    if ts_col and ts_col in df.columns:
        parsed_ts = pd.to_datetime(df[ts_col], errors="coerce").dropna()
        if not parsed_ts.empty:
            ts_min = parsed_ts.min().strftime("%Y-%m-%d %H:%M")
            ts_max = parsed_ts.max().strftime("%Y-%m-%d %H:%M")
            time_span_days = round((parsed_ts.max() - parsed_ts.min()).total_seconds() / 86400, 1)

    null_counts = {col: int(df[col].isna().sum()) for col in cols}

    return {
        "total_records": total_records,
        "columns": cols,
        "null_counts": null_counts,
        "timestamp_start": ts_min,
        "timestamp_end": ts_max,
        "time_span_days": time_span_days,
        "memory_usage_kb": round(df.memory_usage(deep=True).sum() / 1024, 1)
    }
