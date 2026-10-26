"""
WeatherIQ - Local Prediction Database
SQLite persistence layer for storing, filtering, exporting, and managing
historical model predictions.
"""

import sqlite3
from pathlib import Path
from typing import Dict, List, Any, Optional
import pandas as pd

from src.config import DEFAULT_DB_PATH


def get_db_connection(db_path: Path = DEFAULT_DB_PATH) -> sqlite3.Connection:
    """
    Creates and returns an active SQLite database connection with row factory.
    """
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path: Path = DEFAULT_DB_PATH) -> None:
    """
    Initializes the SQLite schema if not already present.
    """
    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS prediction_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                previous_temperature REAL NOT NULL,
                relative_humidity REAL NOT NULL,
                wind_speed REAL NOT NULL,
                horizon_hours INTEGER NOT NULL,
                predicted_temperature REAL NOT NULL,
                model_version TEXT NOT NULL,
                notes TEXT
            )
        """)
        # Index on timestamp for performant chronological filtering
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_pred_timestamp 
            ON prediction_history(timestamp)
        """)
        conn.commit()


def save_prediction_record(
    record: Dict[str, Any], 
    db_path: Path = DEFAULT_DB_PATH
) -> int:
    """
    Inserts a single prediction record into the SQLite database.
    Returns the newly created row ID.
    """
    init_db(db_path)
    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO prediction_history (
                timestamp,
                previous_temperature,
                relative_humidity,
                wind_speed,
                horizon_hours,
                predicted_temperature,
                model_version,
                notes
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            record["timestamp"],
            float(record["previous_temperature"]),
            float(record["relative_humidity"]),
            float(record["wind_speed"]),
            int(record.get("horizon_hours", 1)),
            float(record["predicted_temperature"]),
            str(record.get("model_version", "unknown")),
            record.get("notes", "")
        ))
        conn.commit()
        return int(cursor.lastrowid)


def get_predictions_df(
    limit: Optional[int] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    db_path: Path = DEFAULT_DB_PATH
) -> pd.DataFrame:
    """
    Queries prediction history with optional date range filters, returning a pandas DataFrame.
    """
    init_db(db_path)
    query = "SELECT * FROM prediction_history WHERE 1=1"
    params: List[Any] = []

    if start_date:
        query += " AND timestamp >= ?"
        params.append(start_date)
    if end_date:
        query += " AND timestamp <= ?"
        params.append(end_date)

    query += " ORDER BY id DESC"
    if limit:
        query += " LIMIT ?"
        params.append(limit)

    with get_db_connection(db_path) as conn:
        df = pd.read_sql_query(query, conn, params=params)
        return df


def delete_prediction_record(record_id: int, db_path: Path = DEFAULT_DB_PATH) -> bool:
    """
    Deletes a specific prediction record by its ID.
    """
    init_db(db_path)
    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM prediction_history WHERE id = ?", (record_id,))
        conn.commit()
        return cursor.rowcount > 0


def clear_all_predictions(db_path: Path = DEFAULT_DB_PATH) -> int:
    """
    Clears all records from the prediction history table.
    """
    init_db(db_path)
    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM prediction_history")
        conn.commit()
        return cursor.rowcount
