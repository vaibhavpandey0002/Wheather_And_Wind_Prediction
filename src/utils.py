"""
WeatherIQ - Utility Functions
Formatting, unit conversion, and file export helpers.
"""

from typing import Union
import pandas as pd


def celsius_to_fahrenheit(celsius: float) -> float:
    """Converts degrees Celsius to Fahrenheit."""
    return (celsius * 9.0 / 5.0) + 32.0


def fahrenheit_to_celsius(fahrenheit: float) -> float:
    """Converts degrees Fahrenheit to Celsius."""
    return (fahrenheit - 32.0) * 5.0 / 9.0


def format_temperature(temp_c: float, unit: str = "°C") -> str:
    """
    Formats temperature value according to selected user display unit.
    Note: Internal calculations are always preserved in °C.
    """
    if unit == "°F":
        return f"{celsius_to_fahrenheit(temp_c):.1f} °F"
    return f"{temp_c:.1f} °C"


def dataframe_to_csv_bytes(df: pd.DataFrame) -> bytes:
    """Encodes a DataFrame into UTF-8 CSV bytes for download."""
    return df.to_csv(index=False).encode("utf-8")
