"""
WeatherIQ - Synthetic Demonstration Data Generator
Generates a deterministic synthetic hourly weather dataset for development, smoke testing,
and college demonstration purposes.

PROMINENT NOTICE:
This script produces SYNTHETIC DEMONSTRATION DATA designed to model diurnal and seasonal
meteorological dynamics. These records are artificially synthesized and MUST NOT be
represented as real historical weather observations.
"""

import sys
from pathlib import Path

# Ensure project root is available in sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from src.config import DATA_DIR


def generate_synthetic_weather_dataset(
    days: int = 90,
    start_date: str = "2024-01-01 00:00:00",
    seed: int = 42,
    output_filename: str = "weather_sample.csv"
) -> Path:
    """
    Synthesizes realistic meteorological time-series:
    - Diurnal (24-hour) sinusoidal temperature curve peaking at ~14:00
    - Multi-week thermal trend (seasonal wave)
    - Inverse humidity correlation with temperature
    - Positively skewed diurnal wind variations (afternoon gusts, calmer nights)
    - Deterministic reproducibility using NumPy fixed seed
    """
    np.random.seed(seed)
    n_hours = days * 24

    base_time = pd.to_datetime(start_date)
    timestamps = [base_time + timedelta(hours=i) for i in range(n_hours)]

    hours = np.array([ts.hour for ts in timestamps])
    day_indices = np.array([i // 24 for i in range(n_hours)])

    # 1. Base seasonal temperature wave (Bhopal-like spring transition: ~18°C base drifting to ~28°C)
    seasonal_trend = 20.0 + 8.0 * np.sin(2 * np.pi * day_indices / 365.0)

    # 2. Diurnal daily cycle: minimum at ~05:00 (hour 5), peak at ~14:00 (hour 14)
    # Peak shift: (hour - 14) * 2pi / 24
    diurnal_cycle = 7.0 * np.cos(2 * np.pi * (hours - 14) / 24.0)

    # 3. Weather front random perturbations (autocorrelated noise)
    noise = np.zeros(n_hours)
    noise[0] = np.random.normal(0, 1)
    for i in range(1, n_hours):
        noise[i] = 0.92 * noise[i - 1] + np.random.normal(0, 0.6)

    temperature = seasonal_trend + diurnal_cycle + noise
    temperature = np.round(temperature, 2)

    # 4. Relative humidity: strongly inversely correlated with temperature
    # (warmer air has higher saturation vapor pressure -> lower RH during midday)
    humidity_base = 90.0 - 1.8 * (temperature - 15.0)
    humidity_noise = np.random.normal(0, 3.5, n_hours)
    relative_humidity = np.clip(humidity_base + humidity_noise, 15.0, 98.0)
    relative_humidity = np.round(relative_humidity, 2)

    # 5. Wind speed: higher in afternoon due to convective boundary layer mixing, lower at night
    diurnal_wind = 6.0 + 4.5 * np.maximum(0, np.sin(np.pi * (hours - 8) / 12.0))
    wind_noise = np.random.exponential(scale=3.0, size=n_hours)
    wind_speed = np.clip(diurnal_wind + wind_noise, 0.5, 65.0)
    wind_speed = np.round(wind_speed, 2)

    df = pd.DataFrame({
        "timestamp": [ts.strftime("%Y-%m-%d %H:%M:%S") for ts in timestamps],
        "temperature": temperature,
        "relative_humidity": relative_humidity,
        "wind_speed": wind_speed,
        "data_source_label": "SYNTHETIC_DEMONSTRATION_DATA"
    })

    out_path = DATA_DIR / output_filename
    df.to_csv(out_path, index=False)
    print(f"Generated {len(df)} synthetic hourly records -> {out_path}")
    print("PROMINENT NOTICE: This dataset is SYNTHETIC DEMONSTRATION DATA.")
    return out_path


if __name__ == "__main__":
    generate_synthetic_weather_dataset()
