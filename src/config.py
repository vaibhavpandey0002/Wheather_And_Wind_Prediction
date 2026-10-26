"""
WeatherIQ - Centralized Configuration
Handles application settings, feature constraints, path management, and defaults.
"""

from pathlib import Path
from typing import List, Dict, Any
from pydantic import BaseModel, Field

# Base directories
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
ASSETS_DIR = BASE_DIR / "assets"
SCRIPTS_DIR = BASE_DIR / "scripts"

# Ensure runtime directories exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Database path
DEFAULT_DB_PATH = DATA_DIR / "prediction_history.db"

# Model paths
DEFAULT_MODEL_PATH = MODELS_DIR / "weather_model.joblib"
DEFAULT_METADATA_PATH = MODELS_DIR / "model_metadata.json"

# Core required features
CORE_INPUT_FEATURES = ["temperature", "relative_humidity", "wind_speed"]
TARGET_FEATURE = "target_temperature"

# Default horizons supported
DEFAULT_HORIZON_HOURS = 1
SUPPORTED_HORIZONS = [1, 2, 3, 6, 12, 24]

# Column detection heuristics (case-insensitive aliases)
COLUMN_ALIASES = {
    "timestamp": [
        "time", "timestamp", "date", "datetime", "date_time", "observed_at", 
        "dt", "reading_time", "measurement_time"
    ],
    "temperature": [
        "temperature", "temp", "temp_c", "temperature_c", "temperature_2m", 
        "temp_celsius", "t2m", "air_temperature", "temp_celsius_mean"
    ],
    "relative_humidity": [
        "humidity", "relative_humidity", "relativehumidity_2m", "rh", 
        "rel_humidity", "humidity_pct", "rh_pct", "relative_humidity_pct"
    ],
    "wind_speed": [
        "wind_speed", "windspeed", "wind_speed_10m", "windspeed_10m", 
        "ws", "wind_kmh", "wind_speed_kmh", "wind_kph"
    ]
}

# Physical boundary constraints for validation
class ValidationBounds(BaseModel):
    temp_min: float = -60.0
    temp_max: float = 65.0
    humidity_min: float = 0.0
    humidity_max: float = 100.0
    wind_min: float = 0.0
    wind_max: float = 250.0

DEFAULT_BOUNDS = ValidationBounds()

# Default predefined locations
# Predefined locations across states and countries
PREDEFINED_LOCATIONS: Dict[str, Dict[str, Any]] = {
    "Bhopal (Madhya Pradesh), India": {"lat": 23.2599, "lon": 77.4126, "timezone": "Asia/Kolkata", "base_temp": 26.0},
    "Mumbai (Maharashtra), India": {"lat": 19.0760, "lon": 72.8777, "timezone": "Asia/Kolkata", "base_temp": 30.5},
    "New Delhi (Delhi), India": {"lat": 28.6139, "lon": 77.2090, "timezone": "Asia/Kolkata", "base_temp": 28.0},
    "Bengaluru (Karnataka), India": {"lat": 12.9716, "lon": 77.5946, "timezone": "Asia/Kolkata", "base_temp": 24.0},
    "Kolkata (West Bengal), India": {"lat": 22.5726, "lon": 88.3639, "timezone": "Asia/Kolkata", "base_temp": 29.0},
    "Chennai (Tamil Nadu), India": {"lat": 13.0827, "lon": 80.2707, "timezone": "Asia/Kolkata", "base_temp": 31.5},
    "Jaipur (Rajasthan), India": {"lat": 26.9124, "lon": 75.7873, "timezone": "Asia/Kolkata", "base_temp": 29.5},
    "Srinagar (Jammu & Kashmir), India": {"lat": 34.0837, "lon": 74.7973, "timezone": "Asia/Kolkata", "base_temp": 12.5},
    "Shimla (Himachal Pradesh), India": {"lat": 31.1048, "lon": 77.1734, "timezone": "Asia/Kolkata", "base_temp": 14.0},
    "London, United Kingdom": {"lat": 51.5074, "lon": -0.1278, "timezone": "Europe/London", "base_temp": 13.0},
    "New York, United States": {"lat": 40.7128, "lon": -74.0060, "timezone": "America/New_York", "base_temp": 16.5},
    "Tokyo, Japan": {"lat": 35.6762, "lon": 139.6503, "timezone": "Asia/Tokyo", "base_temp": 18.0},
    "Dubai, United Arab Emirates": {"lat": 25.2048, "lon": 55.2708, "timezone": "Asia/Dubai", "base_temp": 35.0},
    "Paris, France": {"lat": 48.8566, "lon": 2.3522, "timezone": "Europe/Paris", "base_temp": 14.5},
    "Sydney, Australia": {"lat": -33.8688, "lon": 151.2093, "timezone": "Australia/Sydney", "base_temp": 21.0}
}
DEFAULT_LOCATION = "Bhopal (Madhya Pradesh), India"

# Visual Theme Tokens
THEME_TOKENS = {
    "bg": "#0B1120",
    "sidebar_bg": "#111827",
    "card_bg": "#111C30",
    "border": "#263449",
    "primary": "#38BDF8",
    "secondary": "#818CF8",
    "success": "#34D399",
    "warning": "#FBBF24",
    "danger": "#F87171",
    "text_main": "#F8FAFC",
    "text_muted": "#94A3B8"
}
