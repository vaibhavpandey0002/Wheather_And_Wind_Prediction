"""
WeatherIQ - Visual Analytics & Interactive Plotly Charts
Generates high-contrast, modern dark-navy styled Plotly visualizations.
"""

from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from src.config import THEME_TOKENS
from src.utils import celsius_to_fahrenheit


def _apply_dark_theme(fig: go.Figure, title: str, height: int = 420) -> go.Figure:
    """Applies standardized modern SaaS dark aesthetics to Plotly figures."""
    fig.update_layout(
        title=dict(
            text=f"<b>{title}</b>",
            font=dict(family="Inter, -apple-system, sans-serif", size=15, color=THEME_TOKENS["text_main"]),
            x=0.01,
            y=0.96
        ),
        paper_bgcolor=THEME_TOKENS["card_bg"],
        plot_bgcolor="#0A101D",
        font=dict(family="Inter, -apple-system, sans-serif", color=THEME_TOKENS["text_muted"], size=12),
        margin=dict(l=45, r=25, t=55, b=45),
        height=height,
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1.0,
            font=dict(size=11, color=THEME_TOKENS["text_main"]),
            bgcolor="rgba(17, 28, 48, 0.6)"
        )
    )
    fig.update_xaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor="#1E293B",
        zeroline=False,
        linecolor=THEME_TOKENS["border"]
    )
    fig.update_yaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor="#1E293B",
        zeroline=False,
        linecolor=THEME_TOKENS["border"]
    )
    return fig


def create_temperature_trend_chart(
    df: pd.DataFrame, 
    temp_col: str = "temperature",
    ts_col: str = "timestamp",
    unit: str = "°C"
) -> go.Figure:
    """Historical temperature line chart with interactive range selector."""
    fig = go.Figure()

    temps = df[temp_col]
    if unit == "°F":
        temps = temps.apply(celsius_to_fahrenheit)

    fig.add_trace(go.Scatter(
        x=df[ts_col],
        y=temps,
        mode="lines",
        name=f"Observed Temp ({unit})",
        line=dict(color=THEME_TOKENS["primary"], width=2),
        hovertemplate="<b>Date/Time:</b> %{x}<br><b>Temperature:</b> %{y:.1f}" + f" {unit}<extra></extra>"
    ))

    _apply_dark_theme(fig, f"Historical Temperature Trajectory ({unit})")
    fig.update_layout(
        xaxis=dict(
            rangeselector=dict(
                buttons=list([
                    dict(count=24, label="24h", step="hour", stepmode="backward"),
                    dict(count=7, label="7d", step="day", stepmode="backward"),
                    dict(count=30, label="30d", step="day", stepmode="backward"),
                    dict(step="all", label="All")
                ]),
                bgcolor="#162238",
                activecolor="#2563EB",
                font=dict(color="#F8FAFC", size=10)
            ),
            rangeslider=dict(visible=False),
            type="date"
        ),
        yaxis_title=f"Temperature ({unit})"
    )
    return fig


def create_humidity_wind_chart(
    df: pd.DataFrame,
    ts_col: str = "timestamp",
    hum_col: str = "relative_humidity",
    wind_col: str = "wind_speed"
) -> go.Figure:
    """
    Dual-axis time-series visualization for humidity (%) and wind speed (km/h)
    to prevent misleading comparisons across incompatible units.
    """
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Relative humidity (left axis)
    fig.add_trace(
        go.Scatter(
            x=df[ts_col],
            y=df[hum_col],
            name="Relative Humidity (%)",
            line=dict(color=THEME_TOKENS["secondary"], width=1.8),
            hovertemplate="<b>Humidity:</b> %{y:.1f}%<extra></extra>"
        ),
        secondary_y=False
    )

    # Wind speed (right axis)
    fig.add_trace(
        go.Scatter(
            x=df[ts_col],
            y=df[wind_col],
            name="Wind Speed (km/h)",
            line=dict(color=THEME_TOKENS["warning"], width=1.5, dash="dot"),
            hovertemplate="<b>Wind Speed:</b> %{y:.1f} km/h<extra></extra>"
        ),
        secondary_y=True
    )

    _apply_dark_theme(fig, "Atmospheric Moisture & Wind Velocity Dynamics")
    fig.update_yaxes(title_text="Relative Humidity (%)", secondary_y=False, range=[0, 105])
    fig.update_yaxes(title_text="Wind Speed (km/h)", secondary_y=True, range=[0, max(40, df[wind_col].max() * 1.15)])
    return fig


def create_actual_vs_predicted_chart(
    timestamps: List[str],
    actual: List[float],
    predicted: List[float],
    persistence: Optional[List[float]] = None,
    unit: str = "°C"
) -> go.Figure:
    """
    Compares held-out test set observed temperatures with model predictions
    and the naive persistence baseline.
    """
    fig = go.Figure()

    act = [celsius_to_fahrenheit(v) for v in actual] if unit == "°F" else actual
    pred = [celsius_to_fahrenheit(v) for v in predicted] if unit == "°F" else predicted

    fig.add_trace(go.Scatter(
        x=timestamps,
        y=act,
        mode="lines",
        name=f"Actual Observed ({unit})",
        line=dict(color="#94A3B8", width=2.5),
        hovertemplate="<b>Actual:</b> %{y:.2f}" + f" {unit}<extra></extra>"
    ))

    fig.add_trace(go.Scatter(
        x=timestamps,
        y=pred,
        mode="lines",
        name=f"Model Prediction ({unit})",
        line=dict(color=THEME_TOKENS["primary"], width=2.2),
        hovertemplate="<b>Predicted:</b> %{y:.2f}" + f" {unit}<extra></extra>"
    ))

    if persistence is not None:
        pers = [celsius_to_fahrenheit(v) for v in persistence] if unit == "°F" else persistence
        fig.add_trace(go.Scatter(
            x=timestamps,
            y=pers,
            mode="lines",
            name="Persistence Baseline (t-0)",
            line=dict(color="#64748B", width=1.3, dash="dash"),
            hovertemplate="<b>Persistence:</b> %{y:.2f}" + f" {unit}<extra></extra>"
        ))

    _apply_dark_theme(fig, f"Held-Out Test Set: Actual vs Predicted ({unit})")
    fig.update_layout(yaxis_title=f"Temperature ({unit})")
    return fig


def create_residual_plot(
    actual: List[float],
    predicted: List[float],
    unit: str = "°C"
) -> go.Figure:
    """Residual plot: Errors (Actual - Predicted) plotted against Predicted values."""
    residuals = np.array(actual) - np.array(predicted)
    if unit == "°F":
        residuals = residuals * 1.8

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=predicted,
        y=residuals,
        mode="markers",
        marker=dict(
            color=THEME_TOKENS["primary"],
            size=6,
            opacity=0.75,
            line=dict(width=1, color="#1E293B")
        ),
        hovertemplate="<b>Predicted:</b> %{x:.2f}<br><b>Error:</b> %{y:.2f}" + f" {unit}<extra></extra>"
    ))

    # Zero error reference line
    fig.add_hline(y=0.0, line_dash="dash", line_color=THEME_TOKENS["danger"], line_width=1.5)

    _apply_dark_theme(fig, f"Residual Plot (Prediction Errors vs Predicted Value)")
    fig.update_layout(
        xaxis_title=f"Predicted Temperature ({unit})",
        yaxis_title=f"Residual Error [Actual - Predicted] ({unit})"
    )
    return fig


def create_residual_distribution_chart(
    residuals: List[float],
    unit: str = "°C"
) -> go.Figure:
    """Histogram of forecasting residuals illustrating error symmetry and normality."""
    res = np.array(residuals)
    if unit == "°F":
        res = res * 1.8

    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=res,
        nbinsx=35,
        marker=dict(color=THEME_TOKENS["secondary"], line=dict(color="#1E293B", width=1)),
        name="Residuals",
        hovertemplate="<b>Error Bin:</b> %{x:.2f}<br><b>Count:</b> %{y}<extra></extra>"
    ))

    # Reference vertical zero line
    fig.add_vline(x=0.0, line_dash="dash", line_color=THEME_TOKENS["success"], line_width=2)

    _apply_dark_theme(fig, f"Residual Error Distribution ({unit})")
    fig.update_layout(
        xaxis_title=f"Error [Actual - Predicted] ({unit})",
        yaxis_title="Frequency Count"
    )
    return fig


def create_model_comparison_bar_chart(
    comparison_dict: Dict[str, Dict[str, float]],
    persistence_mae: float
) -> go.Figure:
    """Grouped bar chart comparing candidate algorithms against Persistence Baseline."""
    model_names = list(comparison_dict.keys())
    maes = [comparison_dict[m]["mae"] for m in model_names]
    rmses = [comparison_dict[m]["rmse"] for m in model_names]

    # Append Persistence Baseline for direct comparison
    all_names = model_names + ["Persistence Baseline"]
    all_maes = maes + [persistence_mae]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=all_names,
        y=all_maes,
        name="MAE (°C)",
        marker_color=[THEME_TOKENS["primary"] if m != "Persistence Baseline" else "#F59E0B" for m in all_names],
        text=[f"{v:.2f}°C" for v in all_maes],
        textposition="outside"
    ))

    _apply_dark_theme(fig, "Algorithm Performance Comparison (Test MAE)")
    fig.update_layout(
        yaxis_title="Mean Absolute Error (°C)",
        yaxis=dict(range=[0, max(all_maes) * 1.25])
    )
    return fig


def create_correlation_heatmap(df: pd.DataFrame) -> go.Figure:
    """Correlation matrix heatmap of numerical weather features."""
    cols = ["temperature", "relative_humidity", "wind_speed"]
    valid_cols = [c for c in cols if c in df.columns]
    
    corr = df[valid_cols].corr().round(3)

    labels = {
        "temperature": "Temperature",
        "relative_humidity": "Humidity",
        "wind_speed": "Wind Speed"
    }
    axis_labels = [labels.get(c, c) for c in valid_cols]

    fig = go.Figure(data=go.Heatmap(
        z=corr.values,
        x=axis_labels,
        y=axis_labels,
        colorscale=[
            [0.0, "#EF4444"],
            [0.5, "#1E293B"],
            [1.0, "#38BDF8"]
        ],
        zmin=-1.0,
        zmax=1.0,
        text=corr.values,
        texttemplate="%{text:.2f}",
        textfont={"size": 13, "color": "#F8FAFC"},
        colorbar=dict(title="Pearson r")
    ))

    _apply_dark_theme(fig, "Atmospheric Feature Cross-Correlation Matrix", height=380)
    return fig


def create_diurnal_hourly_chart(df: pd.DataFrame, unit: str = "°C") -> go.Figure:
    """Diurnal cycle plot: average temperature by hour of the day (00:00 - 23:00)."""
    df_temp = df.copy()
    df_temp["timestamp"] = pd.to_datetime(df_temp["timestamp"])
    df_temp["hour"] = df_temp["timestamp"].dt.hour

    hourly_stats = df_temp.groupby("hour")["temperature"].agg(["mean", "min", "max"]).reset_index()

    if unit == "°F":
        hourly_stats["mean"] = hourly_stats["mean"].apply(celsius_to_fahrenheit)
        hourly_stats["min"] = hourly_stats["min"].apply(celsius_to_fahrenheit)
        hourly_stats["max"] = hourly_stats["max"].apply(celsius_to_fahrenheit)

    fig = go.Figure()

    # Min-Max shaded envelope
    fig.add_trace(go.Scatter(
        x=hourly_stats["hour"],
        y=hourly_stats["max"],
        mode="lines",
        line=dict(width=0),
        showlegend=False,
        hoverinfo="skip"
    ))
    fig.add_trace(go.Scatter(
        x=hourly_stats["hour"],
        y=hourly_stats["min"],
        mode="lines",
        line=dict(width=0),
        fill="tonexty",
        fillcolor="rgba(56, 189, 248, 0.12)",
        name="Observed Range [Min, Max]",
        hoverinfo="skip"
    ))

    # Mean line
    fig.add_trace(go.Scatter(
        x=hourly_stats["hour"],
        y=hourly_stats["mean"],
        mode="lines+markers",
        name="Diurnal Mean",
        line=dict(color=THEME_TOKENS["primary"], width=2.5),
        marker=dict(size=6),
        hovertemplate="<b>Hour:</b> %{x:02d}:00<br><b>Mean Temp:</b> %{y:.1f}" + f" {unit}<extra></extra>"
    ))

    _apply_dark_theme(fig, f"Diurnal Cycle: 24-Hour Thermal Pattern ({unit})")
    fig.update_layout(
        xaxis=dict(
            tickmode="linear",
            tick0=0,
            dtick=2,
            title="Hour of Day (00:00 to 23:00)"
        ),
        yaxis_title=f"Temperature ({unit})"
    )
    return fig
