"""
Unit Tests for Local SQLite Database Operations
"""

import pytest
from pathlib import Path
import pandas as pd

from src.database import (
    init_db,
    save_prediction_record,
    get_predictions_df,
    delete_prediction_record,
    clear_all_predictions
)


@pytest.fixture
def test_db_path(tmp_path):
    """Provides a fresh isolated SQLite database path for testing."""
    return tmp_path / "test_prediction_history.db"


def test_db_init_and_insertion(test_db_path):
    """Verifies table initialization and record insertion."""
    init_db(test_db_path)
    assert test_db_path.exists()

    record = {
        "timestamp": "2024-05-01 12:00:00 UTC",
        "previous_temperature": 28.5,
        "relative_humidity": 45.0,
        "wind_speed": 15.2,
        "horizon_hours": 1,
        "predicted_temperature": 29.8,
        "model_version": "Random Forest (v_test)",
        "notes": "College test observation"
    }

    row_id = save_prediction_record(record, db_path=test_db_path)
    assert row_id == 1

    df = get_predictions_df(db_path=test_db_path)
    assert len(df) == 1
    assert df.iloc[0]["previous_temperature"] == 28.5
    assert df.iloc[0]["predicted_temperature"] == 29.8
    assert df.iloc[0]["notes"] == "College test observation"


def test_db_deletion(test_db_path):
    """Verifies that record deletion and database clearing function correctly."""
    init_db(test_db_path)
    for i in range(3):
        save_prediction_record({
            "timestamp": f"2024-05-01 1{i}:00:00 UTC",
            "previous_temperature": 25.0 + i,
            "relative_humidity": 50.0,
            "wind_speed": 10.0,
            "horizon_hours": 1,
            "predicted_temperature": 26.0 + i,
            "model_version": "test",
            "notes": ""
        }, db_path=test_db_path)

    df_before = get_predictions_df(db_path=test_db_path)
    assert len(df_before) == 3

    # Delete record #2
    deleted = delete_prediction_record(2, db_path=test_db_path)
    assert deleted

    df_after = get_predictions_df(db_path=test_db_path)
    assert len(df_after) == 2
    assert 2 not in df_after["id"].values

    # Clear all
    purged = clear_all_predictions(db_path=test_db_path)
    assert purged == 2
    df_empty = get_predictions_df(db_path=test_db_path)
    assert len(df_empty) == 0
