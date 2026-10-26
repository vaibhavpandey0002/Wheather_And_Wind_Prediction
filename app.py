"""
WeatherIQ — Intelligent Weather Temperature Prediction System
Main Application Entry Point.
"""

from pathlib import Path
import json
import streamlit as st
import pandas as pd

from src.config import (
    DEFAULT_MODEL_PATH, 
    DEFAULT_METADATA_PATH, 
    DATA_DIR,
    DEFAULT_LOCATION,
    PREDEFINED_LOCATIONS
)
from src.ui_helpers import (
    inject_custom_css, 
    init_session_state, 
    render_header, 
    render_metric_card, 
    render_sidebar_branding
)
from src.utils import format_temperature, celsius_to_fahrenheit
from src.charts import create_temperature_trend_chart, create_humidity_wind_chart
from src.database import get_predictions_df, init_db

# Configure page
st.set_page_config(
    page_title="WeatherIQ — Weather Temperature Prediction System",
    page_icon="🌤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply styling and state
inject_custom_css()
init_session_state()
render_sidebar_branding()
init_db()

# Render top hero banner
render_header(
    title="WeatherIQ — Intelligent Weather Temperature Prediction System",
    subtitle="A complete college machine learning project predicting future atmospheric temperatures using chronological weather observations.",
    badge_text="Production Release v1.0"
)

unit = st.session_state.get("temp_unit", "°C")

# Check if model exists
has_model = DEFAULT_MODEL_PATH.exists() and DEFAULT_METADATA_PATH.exists()
metadata = {}
if has_model:
    try:
        with open(DEFAULT_METADATA_PATH, "r", encoding="utf-8") as f:
            metadata = json.load(f)
    except Exception:
        has_model = False

# Location Station selector
loc_names = list(PREDEFINED_LOCATIONS.keys())
saved_loc = st.session_state.get("current_location", DEFAULT_LOCATION)
loc_idx = loc_names.index(saved_loc) if saved_loc in loc_names else 0

col_sel_loc, col_sel_info = st.columns([2, 3])
with col_sel_loc:
    selected_loc = st.selectbox(
        "Observation Station / Region (State / Country)",
        options=loc_names,
        index=loc_idx,
        help="Select any state or country to immediately sync its regional weather observations."
    )

from src.weather_api import fetch_recent_location_weather

if (
    st.session_state.get("station_data_loaded") != selected_loc
    or st.session_state.get("active_dataset") is None
):
    with st.spinner(f"Loading meteorological observations for {selected_loc}..."):
        loc_df, loc_meta = fetch_recent_location_weather(selected_loc)
        st.session_state["active_dataset"] = loc_df
        st.session_state["active_location_meta"] = loc_meta
        st.session_state["station_data_loaded"] = selected_loc
        st.session_state["current_location"] = selected_loc
        st.session_state["dataset_source_name"] = loc_meta["source"]
        st.session_state["dataset_is_synthetic"] = not loc_meta.get("is_live", False)

active_df = st.session_state.get("active_dataset")
loc_meta = st.session_state.get("active_location_meta", {})
is_live = loc_meta.get("is_live", False)

with col_sel_info:
    status_tag = "✓ LIVE Open-Meteo Feed" if is_live else "Calibrated Regional Climate"
    st.markdown(f"""
    <div class="info-box" style="margin-top: 1.5rem; padding: 0.6rem 1rem;">
        <b>Station:</b> {selected_loc} &nbsp;|&nbsp; 
        <span class="status-badge {'badge-success' if is_live else 'badge-info'}">{status_tag}</span><br>
        <span style="font-size: 0.8rem; color: #94A3B8;">Synced {loc_meta.get('retrieval_timestamp', 'recently')} • {len(active_df):,} hourly observations</span>
    </div>
    """, unsafe_allow_html=True)

# Quick Navigation Grid
st.markdown("### 🧭 Application Modules")
nav_col1, nav_col2, nav_col3, nav_col4 = st.columns(4)

with nav_col1:
    st.markdown("""
    <div class="metric-card" style="height: 140px;">
        <div style="font-size: 1.3rem;">📊 <b>Overview</b></div>
        <div style="color: #94A3B8; font-size: 0.85rem; margin-top: 0.4rem;">
            High-level KPIs, temperature trends, and model verification.
        </div>
    </div>
    """, unsafe_allow_html=True)
    if st.button("Open Dashboard", key="btn_nav_overview", use_container_width=True):
        st.switch_page("pages/1_Overview.py")

with nav_col2:
    st.markdown("""
    <div class="metric-card" style="height: 140px;">
        <div style="font-size: 1.3rem;">🔮 <b>Prediction</b></div>
        <div style="color: #94A3B8; font-size: 0.85rem; margin-top: 0.4rem;">
            Input parameters manually and predict future temperatures.
        </div>
    </div>
    """, unsafe_allow_html=True)
    if st.button("Open Predictor", key="btn_nav_predict", use_container_width=True):
        st.switch_page("pages/2_Prediction.py")

with nav_col3:
    st.markdown("""
    <div class="metric-card" style="height: 140px;">
        <div style="font-size: 1.3rem;">📈 <b>Analytics</b></div>
        <div style="color: #94A3B8; font-size: 0.85rem; margin-top: 0.4rem;">
            Diurnal cycles, moisture patterns, and cross correlations.
        </div>
    </div>
    """, unsafe_allow_html=True)
    if st.button("Open Analytics", key="btn_nav_analytics", use_container_width=True):
        st.switch_page("pages/3_Analytics.py")

with nav_col4:
    st.markdown("""
    <div class="metric-card" style="height: 140px;">
        <div style="font-size: 1.3rem;">📁 <b>Data & Training</b></div>
        <div style="color: #94A3B8; font-size: 0.85rem; margin-top: 0.4rem;">
            Upload CSVs, sync Open-Meteo, and train regression models.
        </div>
    </div>
    """, unsafe_allow_html=True)
    if st.button("Open Data Manager", key="btn_nav_data", use_container_width=True):
        st.switch_page("pages/6_Dataset_Management.py")

st.markdown("<br>", unsafe_allow_html=True)

# ----------------- SYSTEM STATUS SUMMARY -----------------
st.markdown("### ⚡ System Status & Key Performance Indicators")
k1, k2, k3, k4 = st.columns(4)

with k1:
    if active_df is not None and not active_df.empty:
        last_temp = float(active_df.iloc[-1]["temperature"])
        render_metric_card("Latest Recorded Temp", format_temperature(last_temp, unit), "From active dataset", "Observed", "info")
    else:
        render_metric_card("Latest Recorded Temp", "Not available", "No dataset active", "N/A", "warning")

with k2:
    if active_df is not None and not active_df.empty:
        mean_temp = float(active_df["temperature"].mean())
        render_metric_card("Historical Mean Temp", format_temperature(mean_temp, unit), f"Over {len(active_df):,} samples", "Mean", "info")
    else:
        render_metric_card("Historical Mean Temp", "Not available", "No dataset active", "N/A", "warning")

with k3:
    recent_preds = get_predictions_df(limit=1)
    if not recent_preds.empty:
        pred_val = float(recent_preds.iloc[0]["predicted_temperature"])
        render_metric_card("Latest Prediction", format_temperature(pred_val, unit), "Stored in SQLite", "Inference", "success")
    else:
        render_metric_card("Latest Prediction", "No records", "Awaiting first run", "Pending", "warning")

with k4:
    if has_model and "test_metrics" in metadata:
        mae = metadata["test_metrics"]["mae"]
        if unit == "°F":
            mae *= 1.8
        render_metric_card("Model Test MAE", f"±{mae:.2f} {unit}", f"Algorithm: {metadata.get('model_name')}", "Test Score", "success")
    else:
        render_metric_card("Model Status", "Untrained", "Train in Data Manager", "Untrained", "warning")

# First run guidance if neither dataset nor model exists
if active_df is None and not has_model:
    st.markdown("""
    <div class="metric-card" style="margin-top: 1.5rem; padding: 2rem; border: 1px dashed #38BDF8;">
        <h4 style="color: #38BDF8; margin-bottom: 0.5rem;">Quick Start Instructions</h4>
        <p style="color: #94A3B8; font-size: 0.92rem; line-height: 1.6;">
            1. Navigate to <b>Dataset Management</b> (Page 6).<br>
            2. Click <b>Demonstration Data Generator</b> to create sample data, or upload your own weather CSV.<br>
            3. Verify column mappings and click <b>Train Machine Learning Models Now</b>.<br>
            4. Once trained, jump to <b>Temperature Prediction</b> to run real-time forecasts!
        </p>
    </div>
    """, unsafe_allow_html=True)
else:
    # Render preview chart if data exists
    if active_df is not None and not active_df.empty:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("### 📈 Recent Temperature Trajectory Preview")
        fig_preview = create_temperature_trend_chart(active_df.tail(240), unit=unit)
        st.plotly_chart(fig_preview, use_container_width=True)
