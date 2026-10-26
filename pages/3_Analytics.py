"""
WeatherIQ - Weather Analytics Page
Deep exploratory data analysis for meteorological datasets: trends, diurnal cycles,
distributions, and cross-feature correlations.
"""

from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from src.config import DATA_DIR, THEME_TOKENS
from src.ui_helpers import (
    inject_custom_css, 
    init_session_state, 
    render_header, 
    render_metric_card, 
    render_sidebar_branding
)
from src.utils import format_temperature, celsius_to_fahrenheit
from src.charts import (
    create_temperature_trend_chart, 
    create_humidity_wind_chart, 
    create_correlation_heatmap,
    create_diurnal_hourly_chart
)

st.set_page_config(
    page_title="Weather Analytics | WeatherIQ",
    page_icon="📈",
    layout="wide"
)

inject_custom_css()
init_session_state()
render_sidebar_branding()

unit = st.session_state.get("temp_unit", "°C")

render_header(
    title="Meteorological Analytics & Trends",
    subtitle="Interactive exploratory analysis of atmospheric time-series data.",
    badge_text="EDA Suite"
)

# Check active dataset
df = st.session_state.get("active_dataset")
sample_csv = DATA_DIR / "weather_sample.csv"

if df is None and sample_csv.exists():
    try:
        df = pd.read_csv(sample_csv)
        st.session_state["active_dataset"] = df
    except Exception:
        df = None

if df is None or df.empty:
    st.warning("No dataset currently loaded for analytics.")
    st.markdown("""
    <div class="metric-card" style="padding: 2.5rem; text-align: center;">
        <h4 style="color: #38BDF8;">Load Data to Enable Analytics</h4>
        <p style="color: #94A3B8; max-width: 600px; margin: 0 auto 1.5rem auto;">
            Please visit Dataset Management to upload a CSV file or generate synthetic demonstration data.
        </p>
    </div>
    """, unsafe_allow_html=True)
    if st.button("📁 Go to Dataset Management", type="primary"):
        st.switch_page("pages/6_Dataset_Management.py")
    st.stop()

# Ensure timestamp is datetime
df = df.copy()
df["timestamp"] = pd.to_datetime(df["timestamp"])

# ----------------- DATE RANGE FILTERS -----------------
min_date = df["timestamp"].min().date()
max_date = df["timestamp"].max().date()

st.markdown("#### 🔍 Filter Exploration Horizon")
fcol1, fcol2, fcol3 = st.columns([2, 2, 2])

with fcol1:
    selected_start = st.date_input("Start Date", value=min_date, min_value=min_date, max_value=max_date)
with fcol2:
    selected_end = st.date_input("End Date", value=max_date, min_value=min_date, max_value=max_date)
with fcol3:
    st.markdown(f"<div style='margin-top: 1.8rem; color: #94A3B8; font-size: 0.88rem;'>Range span: <b>{(selected_end - selected_start).days + 1} days</b></div>", unsafe_allow_html=True)

# Apply filter
mask = (df["timestamp"].dt.date >= selected_start) & (df["timestamp"].dt.date <= selected_end)
filtered_df = df.loc[mask].copy()

if filtered_df.empty:
    st.error("No records found in selected date interval. Please broaden the range.")
    st.stop()

# ----------------- SUMMARY METRICS -----------------
st.markdown("---")
m1, m2, m3, m4 = st.columns(4)

temps = filtered_df["temperature"]
if unit == "°F":
    temps_disp = temps.apply(celsius_to_fahrenheit)
else:
    temps_disp = temps

with m1:
    render_metric_card("Average Temperature", f"{temps_disp.mean():.1f} {unit}", f"Median: {temps_disp.median():.1f} {unit}", "Mean", "info")
with m2:
    render_metric_card("Min Temperature", f"{temps_disp.min():.1f} {unit}", "Historical minimum", "Low", "info")
with m3:
    render_metric_card("Max Temperature", f"{temps_disp.max():.1f} {unit}", "Historical maximum", "High", "warning")
with m4:
    render_metric_card("Thermal Standard Dev", f"±{temps_disp.std():.2f} {unit}", f"{len(filtered_df):,} samples analyzed", "Dispersion", "info")

# ----------------- VISUALIZATIONS -----------------
st.markdown("### 📊 Continuous Time-Series")
fig_trend = create_temperature_trend_chart(filtered_df, unit=unit)
st.plotly_chart(fig_trend, use_container_width=True)

col_a, col_b = st.columns(2)
with col_a:
    fig_moist = create_humidity_wind_chart(filtered_df)
    st.plotly_chart(fig_moist, use_container_width=True)

with col_b:
    fig_diurnal = create_diurnal_hourly_chart(filtered_df, unit=unit)
    st.plotly_chart(fig_diurnal, use_container_width=True)

# ----------------- DAILY AGGREGATIONS & CORRELATIONS -----------------
st.markdown("### 🗓️ Daily Extremes & Feature Relationships")
col_c, col_d = st.columns([3, 2])

with col_c:
    # Daily aggregation
    daily_df = filtered_df.copy()
    daily_df["date"] = daily_df["timestamp"].dt.date
    daily_stats = daily_df.groupby("date").agg({
        "temperature": ["mean", "min", "max"]
    }).reset_index()
    daily_stats.columns = ["date", "temp_mean", "temp_min", "temp_max"]

    if unit == "°F":
        daily_stats["temp_mean"] = daily_stats["temp_mean"].apply(celsius_to_fahrenheit)
        daily_stats["temp_min"] = daily_stats["temp_min"].apply(celsius_to_fahrenheit)
        daily_stats["temp_max"] = daily_stats["temp_max"].apply(celsius_to_fahrenheit)

    fig_daily = go.Figure()
    fig_daily.add_trace(go.Bar(
        x=daily_stats["date"],
        y=daily_stats["temp_max"] - daily_stats["temp_min"],
        base=daily_stats["temp_min"],
        name="Daily Diurnal Span",
        marker_color="rgba(56, 189, 248, 0.4)",
        hovertemplate="<b>Date:</b> %{x}<br><b>Min:</b> %{base:.1f}" + f" {unit}<extra></extra>"
    ))
    fig_daily.add_trace(go.Scatter(
        x=daily_stats["date"],
        y=daily_stats["temp_mean"],
        mode="lines+markers",
        name="Daily Mean Temp",
        line=dict(color=THEME_TOKENS["primary"], width=2),
        marker=dict(size=4)
    ))
    fig_daily.update_layout(
        title="<b>Daily Thermal Envelope (Min, Max, Mean)</b>",
        paper_bgcolor=THEME_TOKENS["card_bg"],
        plot_bgcolor="#0A101D",
        font=dict(color=THEME_TOKENS["text_muted"], size=12),
        height=380,
        margin=dict(l=45, r=25, t=55, b=45),
        yaxis_title=f"Temperature ({unit})"
    )
    fig_daily.update_xaxes(showgrid=True, gridcolor="#1E293B")
    fig_daily.update_yaxes(showgrid=True, gridcolor="#1E293B")
    st.plotly_chart(fig_daily, use_container_width=True)

with col_d:
    fig_corr = create_correlation_heatmap(filtered_df)
    st.plotly_chart(fig_corr, use_container_width=True)
