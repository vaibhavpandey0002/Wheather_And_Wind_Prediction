"""
Unit Tests for Data Validation & Schema Integrity
"""

import io
import pytest
from pydantic import ValidationError
import pandas as pd

from src.data_validation import WeatherPredictionInput, validate_weather_dataframe
from src.data_loader import load_csv, detect_column_mapping


def test_valid_prediction_input():
    """Verifies that plausible inputs pass validation."""
    inp = WeatherPredictionInput(
        temperature=22.5,
        relative_humidity=55.0,
        wind_speed=14.2,
        horizon_hours=1
    )
    assert inp.temperature == 22.5
    assert inp.relative_humidity == 55.0
    assert inp.wind_speed == 14.2
    assert inp.horizon_hours == 1


def test_invalid_temperature_bounds():
    """Verifies that out-of-bounds temperatures fail validation."""
    with pytest.raises(ValidationError):
        WeatherPredictionInput(temperature=-95.0, relative_humidity=50.0, wind_speed=10.0)

    with pytest.raises(ValidationError):
        WeatherPredictionInput(temperature=85.0, relative_humidity=50.0, wind_speed=10.0)


def test_invalid_humidity_bounds():
    """Verifies that relative humidity outside [0, 100] fails validation."""
    with pytest.raises(ValidationError):
        WeatherPredictionInput(temperature=20.0, relative_humidity=-5.0, wind_speed=10.0)

    with pytest.raises(ValidationError):
        WeatherPredictionInput(temperature=20.0, relative_humidity=105.0, wind_speed=10.0)


def test_invalid_wind_speed():
    """Verifies that negative or excessively high wind speeds fail validation."""
    with pytest.raises(ValidationError):
        WeatherPredictionInput(temperature=20.0, relative_humidity=50.0, wind_speed=-2.0)

    with pytest.raises(ValidationError):
        WeatherPredictionInput(temperature=20.0, relative_humidity=50.0, wind_speed=350.0)


def test_csv_loading_from_buffer():
    """Verifies that load_csv correctly parses valid CSV text."""
    csv_text = "timestamp,temp,humidity,wind\n2024-01-01 00:00:00,20.5,60,10\n"
    buffer = io.StringIO(csv_text)
    df = load_csv(buffer)
    assert len(df) == 1
    assert "temp" in df.columns


def test_missing_column_detection():
    """Verifies that missing essential logical columns are detected and flagged."""
    df = pd.DataFrame({
        "time": ["2024-01-01 00:00:00", "2024-01-01 01:00:00"],
        "temp": [20.0, 21.0]
        # Missing humidity and wind speed
    })
    mapping = {
        "timestamp": "time",
        "temperature": "temp",
        "relative_humidity": "missing_hum",
        "wind_speed": "missing_wind"
    }
    report = validate_weather_dataframe(df, mapping, min_required_rows=2)
    assert not report.is_valid
    assert "relative_humidity" in report.missing_columns
    assert "wind_speed" in report.missing_columns


def test_duplicate_and_missing_timestamps():
    """Verifies that duplicate and null timestamps are detected."""
    df = pd.DataFrame({
        "time": ["2024-01-01 00:00:00", "2024-01-01 00:00:00", None] + [f"2024-01-01 {i:02d}:00:00" for i in range(3, 50)],
        "temp": [20.0] * 50,
        "humidity": [60.0] * 50,
        "wind": [10.0] * 50
    })
    mapping = {
        "timestamp": "time",
        "temperature": "temp",
        "relative_humidity": "humidity",
        "wind_speed": "wind"
    }
    report = validate_weather_dataframe(df, mapping, min_required_rows=40)
    assert report.duplicate_timestamps >= 1
    assert any("Failed to parse" in err for err in report.error_details)
