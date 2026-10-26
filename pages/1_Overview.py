"""
WeatherIQ - Overview Dashboard
Main intelligence view presenting meteorological KPIs, historical trends,
model verification charts, and database telemetry.
"""

import json
from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np

from src.config import (
    DEFAULT_MODEL_PATH, 
    DEFAULT_METADATA_PATH, 
    DATA_DIR, 
    PREDEFINED_LOCATIONS,
    DEFAULT_LOCATION,
    THEME_TOKENS
)
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
    create_actual_vs_predicted_chart
)
from src.database import get_predictions_df

st.set_page_config(
    page_title="Overview | WeatherIQ",
    page_icon="🌤️",
    layout="wide"
)

inject_custom_css()
init_session_state()
render_sidebar_branding()

render_header(
    title="Weather Intelligence Dashboard",
    subtitle="Analyze historical weather patterns and predict future temperatures using Machine Learning.",
    badge_text="System Telemetry"
)

# Location selector bar
col_loc1, col_loc2 = st.columns([2, 3])
with col_loc1:
    location_names = list(PREDEFINED_LOCATIONS.keys())
    saved_loc = st.session_state.get("current_location", DEFAULT_LOCATION)
    current_idx = location_names.index(saved_loc) if saved_loc in location_names else 0
    selected_loc = st.selectbox(
        "Observation Station / Region (State / Country)",
        options=location_names,
        index=current_idx,
        help="Select any state, country, or global station to load its verified weather readings."
    )

from src.weather_api import fetch_recent_location_weather

# Trigger immediate regional weather sync if location changes or not yet loaded
if (
    st.session_state.get("station_data_loaded") != selected_loc
    or st.session_state.get("active_dataset") is None
):
    with st.spinner(f"Retrieving atmospheric observations for {selected_loc}..."):
        loc_df, loc_meta = fetch_recent_location_weather(selected_loc)
        st.session_state["active_dataset"] = loc_df
        st.session_state["active_location_meta"] = loc_meta
        st.session_state["dataset_source_name"] = loc_meta["source"]
        st.session_state["station_data_loaded"] = selected_loc
        st.session_state["current_location"] = selected_loc
        st.session_state["dataset_is_synthetic"] = not loc_meta.get("is_live", False)

active_df = st.session_state.get("active_dataset")
loc_meta = st.session_state.get("active_location_meta", {})
is_live = loc_meta.get("is_live", False)

with col_loc2:
    status_tag = "✓ LIVE Open-Meteo Feed" if is_live else "Calibrated Regional Climate"
    st.markdown(f"""
    <div class="info-box" style="margin-top: 1.5rem; padding: 0.6rem 1rem;">
        <b>Station:</b> {selected_loc} &nbsp;|&nbsp; 
        <span class="status-badge {'badge-success' if is_live else 'badge-info'}">{status_tag}</span><br>
        <span style="font-size: 0.8rem; color: #94A3B8;">Synced {loc_meta.get('retrieval_timestamp', 'recently')} • {len(active_df):,} hourly observations</span>
    </div>
    """, unsafe_allow_html=True)

# Check for loaded dataset and model metadata
unit = st.session_state.get("temp_unit", "°C")

has_model = DEFAULT_MODEL_PATH.exists() and DEFAULT_METADATA_PATH.exists()
metadata = {}
if has_model:
    try:
        with open(DEFAULT_METADATA_PATH, "r", encoding="utf-8") as f:
            metadata = json.load(f)
    except Exception:
        has_model = False

# If neither dataset nor model exists, display friendly onboarding state
if active_df is None and not has_model:
    st.warning("No historical dataset or trained model found.")
    st.markdown("""
    <div class="hero-card" style="text-align: center; padding: 3rem 1.5rem;">
        <h3 style="color: #38BDF8; margin-bottom: 0.5rem;">Welcome to WeatherIQ!</h3>
        <p style="color: #94A3B8; max-width: 650px; margin: 0 auto 1.5rem auto;">
            To begin exploring weather analytics and predicting temperatures, import a weather CSV file 
            or generate synthetic demonstration data using the Dataset Management module.
        </p>
    </div>
    """, unsafe_allow_html=True)
    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("🚀 Go to Dataset Management", use_container_width=True):
            st.switch_page("pages/6_Dataset_Management.py")
    with col_b:
        if st.button("🔮 Go to Temperature Prediction", use_container_width=True):
            st.switch_page("pages/2_Prediction.py")
    st.stop()

# ----------------- TOP KPI CARDS -----------------
kpi1, kpi2, kpi3, kpi4 = st.columns(4)

# KPI 1: Latest Recorded Temperature
with kpi1:
    if active_df is not None and not active_df.empty and "temperature" in active_df.columns:
        latest_row = active_df.iloc[-1]
        latest_temp = float(latest_row["temperature"])
        val_str = format_temperature(latest_temp, unit)
        ts_str = str(latest_row.get("timestamp", "Latest sample"))
        render_metric_card(
            title="Latest Recorded Temp",
            value=val_str,
            subtext=f"Observed at {ts_str[:16]}",
            badge="Observed",
            badge_type="info"
        )
    else:
        render_metric_card("Latest Recorded Temp", "Not available", "No dataset records", badge="N/A", badge_type="warning")

# KPI 2: Average Historical Temperature
with kpi2:
    if active_df is not None and not active_df.empty and "temperature" in active_df.columns:
        avg_temp = float(active_df["temperature"].mean())
        val_str = format_temperature(avg_temp, unit)
        render_metric_card(
            title="Avg Historical Temp",
            value=val_str,
            subtext=f"Over {len(active_df)} recordings",
            badge="Mean",
            badge_type="info"
        )
    else:
        render_metric_card("Avg Historical Temp", "Not available", "No dataset records", badge="N/A", badge_type="warning")

# KPI 3: Temperature Prediction for the selected region
with kpi3:
    if has_model and active_df is not None and not active_df.empty:
        latest_row = active_df.iloc[-1]
        try:
            from src.data_validation import WeatherPredictionInput
            from src.predict import predict_temperature
            live_pred_inp = WeatherPredictionInput(
                temperature=float(latest_row["temperature"]),
                relative_humidity=float(latest_row["relative_humidity"]),
                wind_speed=float(latest_row["wind_speed"]),
                horizon_hours=metadata.get("horizon_hours", 1)
            )
            res = predict_temperature(live_pred_inp)
            val_str = format_temperature(res.predicted_temperature, unit)
            short_loc = selected_loc.split("(")[0].split(",")[0].strip()
            render_metric_card(
                title=f"{short_loc} Forecast (+{res.horizon_hours}h)",
                value=val_str,
                subtext=f"Model: {res.model_name}",
                badge="Model Forecast",
                badge_type="success"
            )
        except Exception:
            render_metric_card("Temperature Prediction", "Model error", "Check model", badge="Error", badge_type="warning")
    else:
        recent_preds = get_predictions_df(limit=1)
        if not recent_preds.empty:
            last_pred = float(recent_preds.iloc[0]["predicted_temperature"])
            h_hours = int(recent_preds.iloc[0]["horizon_hours"])
            val_str = format_temperature(last_pred, unit)
            render_metric_card(
                title=f"Latest Forecast (+{h_hours}h)",
                value=val_str,
                subtext=f"Horizon: {h_hours} hour(s) ahead",
                badge="Model Forecast",
                badge_type="success"
            )
        else:
            render_metric_card("Temperature Prediction", "No runs yet", "Make prediction in menu", badge="Pending", badge_type="warning")

# KPI 4: Model MAE
with kpi4:
    if has_model and "test_metrics" in metadata:
        mae_val = float(metadata["test_metrics"]["mae"])
        if unit == "°F":
            mae_val = mae_val * 1.8
        model_name = metadata.get("model_name", "Model")
        render_metric_card(
            title="Model Test MAE",
            value=f"±{mae_val:.2f} {unit}",
            subtext=f"Algorithm: {model_name}",
            badge="Test Score",
            badge_type="success"
        )
    else:
        render_metric_card("Model MAE", "Not evaluated", "Train model in Data page", badge="Untrained", badge_type="warning")

# Synthetic data banner if active
if st.session_state.get("dataset_is_synthetic", False):
    st.markdown("""
    <div class="notice-box">
        ⚠️ <b>NOTICE:</b> Currently utilizing <b>SYNTHETIC DEMONSTRATION DATA</b> for station modeling. 
        Synthetic data is intended for architectural smoke testing and viva demonstrations only.
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ----------------- SECTION A & B: CHARTS -----------------
chart_col1, chart_col2 = st.columns([3, 2])

with chart_col1:
    if active_df is not None and not active_df.empty:
        fig_trend = create_temperature_trend_chart(active_df, unit=unit)
        st.plotly_chart(fig_trend, use_container_width=True)
    else:
        st.info("Historical temperature trend will appear once a dataset is loaded.")

with chart_col2:
    if active_df is not None and "relative_humidity" in active_df.columns and "wind_speed" in active_df.columns:
        fig_moist = create_humidity_wind_chart(active_df)
        st.plotly_chart(fig_moist, use_container_width=True)
    else:
        st.info("Moisture and wind analysis will appear once dataset features are mapped.")

# ----------------- SECTION C: ACTUAL VS PREDICTED (HELD OUT TEST SET) -----------------
st.markdown("### 🎯 Model Verification: Actual vs Predicted")
st.caption("Chronological validation on held-out test data comparing model forecasts with ground truth and persistence baseline.")

if has_model and "test_samples" in metadata:
    samples = metadata["test_samples"]
    fig_eval = create_actual_vs_predicted_chart(
        timestamps=samples["timestamps"],
        actual=samples["actual"],
        predicted=samples["predicted"],
        persistence=samples.get("persistence"),
        unit=unit
    )
    st.plotly_chart(fig_eval, use_container_width=True)
else:
    st.markdown("""
    <div class="metric-card" style="text-align: center; padding: 2rem;">
        <span style="font-size: 2rem;">📊</span>
        <div style="font-weight: 600; margin-top: 0.5rem; color: #F8FAFC;">Evaluation Chart Not Available</div>
        <div style="color: #94A3B8; font-size: 0.85rem; margin-top: 0.25rem;">
            Train a model in the Dataset Management page to view actual vs predicted comparisons.
        </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ----------------- SECTION D & E: RECENT PREDICTIONS & TELEMETRY -----------------
bot_col1, bot_col2 = st.columns([3, 2])

with bot_col1:
    st.markdown("### 🕒 Recent Prediction Activity")
    history_df = get_predictions_df(limit=5)
    if not history_df.empty:
        # Display with readable columns
        display_hist = history_df[[
            "id", "timestamp", "previous_temperature", "relative_humidity", 
            "wind_speed", "horizon_hours", "predicted_temperature", "model_version"
        ]].copy()
        
        display_hist.rename(columns={
            "id": "ID",
            "timestamp": "Time",
            "previous_temperature": f"T_prev ({unit})",
            "relative_humidity": "Humidity (%)",
            "wind_speed": "Wind (km/h)",
            "horizon_hours": "Horizon",
            "predicted_temperature": f"Forecast ({unit})",
            "model_version": "Model"
        }, inplace=True)

        if unit == "°F":
            display_hist[f"T_prev ({unit})"] = display_hist[f"T_prev ({unit})"].apply(celsius_to_fahrenheit).round(1)
            display_hist[f"Forecast ({unit})"] = display_hist[f"Forecast ({unit})"].apply(celsius_to_fahrenheit).round(1)

        st.dataframe(display_hist, use_container_width=True, hide_index=True)
    else:
        st.info("No predictions recorded in SQLite yet. Use the Temperature Prediction tab to run inferences.")

with bot_col2:
    st.markdown("### 📡 System & Data Status")
    source_name = st.session_state.get("dataset_source_name", "None")
    record_count = len(active_df) if active_df is not None else 0
    latest_ts = str(active_df.iloc[-1].get("timestamp", "N/A")) if (active_df is not None and not active_df.empty) else "N/A"
    
    status_items = [
        ("Dataset Source", source_name),
        ("Valid Records", f"{record_count:,} observations"),
        ("Latest Observation", latest_ts[:19]),
        ("Model Status", "Trained & Active" if has_model else "Not Trained"),
        ("Model Algorithm", metadata.get("model_name", "N/A") if has_model else "N/A"),
        ("Model Version", metadata.get("model_version", "N/A") if has_model else "N/A"),
        ("Forecast Horizon", f"{metadata.get('horizon_hours', 1)} hour(s)" if has_model else "1 hour")
    ]

    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    for label, val in status_items:
        st.markdown(f"""
        <div style="display: flex; justify-content: space-between; padding: 0.35rem 0; border-bottom: 1px solid #1E293B; font-size: 0.88rem;">
            <span style="color: #94A3B8;">{label}:</span>
            <span style="font-weight: 600; color: #F8FAFC;">{val}</span>
        </div>
        """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
