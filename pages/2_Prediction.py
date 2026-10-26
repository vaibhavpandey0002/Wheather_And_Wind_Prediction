"""
WeatherIQ - Temperature Prediction Page
Allows users to enter observed weather parameters manually, validate them against
physical bounds, run ML inference, persist the result, and download reports.
"""

from pathlib import Path
import json
import streamlit as st
import pandas as pd
from pydantic import ValidationError

from src.config import (
    DEFAULT_MODEL_PATH, 
    DEFAULT_METADATA_PATH, 
    DEFAULT_BOUNDS, 
    SUPPORTED_HORIZONS,
    PREDEFINED_LOCATIONS,
    DEFAULT_LOCATION
)
from src.ui_helpers import (
    inject_custom_css, 
    init_session_state, 
    render_header, 
    render_sidebar_branding
)
from src.utils import format_temperature, celsius_to_fahrenheit, dataframe_to_csv_bytes
from src.data_validation import WeatherPredictionInput
from src.predict import predict_temperature, load_model_and_metadata
from src.database import save_prediction_record

st.set_page_config(
    page_title="Predict Temperature | WeatherIQ",
    page_icon="🔮",
    layout="wide"
)

inject_custom_css()
init_session_state()
render_sidebar_branding()

unit = st.session_state.get("temp_unit", "°C")

render_header(
    title="Temperature Prediction Engine",
    subtitle="Estimate future atmospheric temperatures using machine learning models trained on meteorological sequences.",
    badge_text="Inference Engine"
)

# Check if model exists
model_available = DEFAULT_MODEL_PATH.exists() and DEFAULT_METADATA_PATH.exists()
metadata = {}
if model_available:
    try:
        _, metadata = load_model_and_metadata()
    except Exception as e:
        model_available = False

if not model_available:
    st.error("No trained machine learning model found in storage.")
    st.markdown("""
    <div class="metric-card" style="padding: 2.5rem; text-align: center;">
        <h4 style="color: #F87171; margin-bottom: 0.5rem;">Model Artifacts Missing</h4>
        <p style="color: #94A3B8; max-width: 600px; margin: 0 auto 1.5rem auto;">
            In order to generate real, non-fabricated temperature forecasts, a model must first 
            be trained on historical weather data.
        </p>
    </div>
    """, unsafe_allow_html=True)
    if st.button("👉 Proceed to Dataset Management & Model Training", type="primary"):
        st.switch_page("pages/6_Dataset_Management.py")
    st.stop()

# Display active model telemetry
model_name = metadata.get("model_name", "Trained Regressor")
model_version = metadata.get("model_version", "v1.0")
trained_horizon = metadata.get("horizon_hours", 1)
test_mae = metadata.get("test_metrics", {}).get("mae", 0.0)

st.markdown(f"""
<div class="info-box" style="margin-bottom: 1.5rem;">
    <b>Active Inference Model:</b> <code>{model_name}</code> ({model_version}) | 
    <b>Trained Horizon:</b> {trained_horizon} Hour(s) ahead | 
    <b>Empirical Test Error:</b> ±{test_mae:.2f}°C
</div>
""", unsafe_allow_html=True)

from src.weather_api import fetch_recent_location_weather

# Form for manual input
col_form, col_result = st.columns([1, 1], gap="large")

with col_form:
    st.markdown("### 📝 Meteorological Input Parameters")
    st.caption("Select a state/country station to auto-fill its latest verified observations, or enter custom values.")

    # Station / Region selector
    loc_names = list(PREDEFINED_LOCATIONS.keys())
    saved_loc = st.session_state.get("current_location", DEFAULT_LOCATION)
    loc_idx = loc_names.index(saved_loc) if saved_loc in loc_names else 0

    chosen_region = st.selectbox(
        "🌍 Station Region (State / Country)",
        options=loc_names,
        index=loc_idx,
        help="Select any region to automatically load its temperature, humidity, and wind speed."
    )

    # Sync regional data if region changed or not yet loaded
    if (
        st.session_state.get("station_data_loaded") != chosen_region 
        or st.session_state.get("active_dataset") is None
    ):
        with st.spinner(f"Retrieving atmospheric observations for {chosen_region}..."):
            loc_df, loc_meta = fetch_recent_location_weather(chosen_region)
            st.session_state["active_dataset"] = loc_df
            st.session_state["active_location_meta"] = loc_meta
            st.session_state["station_data_loaded"] = chosen_region
            st.session_state["current_location"] = chosen_region

    # Read latest observations for the selected region
    reg_df = st.session_state.get("active_dataset")
    if reg_df is not None and not reg_df.empty:
        last_obs = reg_df.iloc[-1]
        default_temp = float(last_obs["temperature"])
        default_hum = float(last_obs["relative_humidity"])
        default_wind = float(last_obs["wind_speed"])
        obs_time = str(last_obs.get("timestamp", "Recent"))[:16]
    else:
        default_temp, default_hum, default_wind = 25.0, 60.0, 12.0
        obs_time = "Manual default"

    st.markdown(f"""
    <div style="font-size: 0.8rem; color: #38BDF8; margin-bottom: 0.75rem;">
        📍 <b>Current readings for {chosen_region}:</b> {default_temp:.1f}°C • {default_hum:.0f}% RH • {default_wind:.1f} km/h (at {obs_time})
    </div>
    """, unsafe_allow_html=True)

    with st.form("prediction_input_form"):
        # Temperature input
        temp_input = st.number_input(
            label="Previous Observed Temperature (°C)",
            min_value=-70.0,
            max_value=70.0,
            value=default_temp,
            step=0.1,
            format="%.1f",
            help=f"Meteorological valid limits: {DEFAULT_BOUNDS.temp_min}°C to {DEFAULT_BOUNDS.temp_max}°C"
        )

        # Humidity input
        humidity_input = st.number_input(
            label="Relative Humidity (%)",
            min_value=-10.0,
            max_value=120.0,
            value=default_hum,
            step=1.0,
            format="%.1f",
            help="Physically bounded between 0% and 100%"
        )

        # Wind speed input
        wind_input = st.number_input(
            label="Wind Speed (km/h)",
            min_value=-10.0,
            max_value=300.0,
            value=default_wind,
            step=0.5,
            format="%.1f",
            help=f"Valid velocity range: 0 to {DEFAULT_BOUNDS.wind_max} km/h"
        )

        # Horizon selector (trained horizon highlighted)
        horizon_options = [trained_horizon]
        horizon_choice = st.selectbox(
            "Forecast Horizon",
            options=horizon_options,
            index=0,
            help=f"The active model was specifically fitted for a {trained_horizon}-hour lookahead."
        )

        notes_input = st.text_input(
            "Observation Notes (Optional)",
            value=f"Station observation at {chosen_region.split('(')[0].strip()}",
            placeholder="e.g., Afternoon thunderstorm precursor, college lab observation"
        )

        submit_btn = st.form_submit_button("⚡ Predict Temperature", type="primary", use_container_width=True)

with col_result:
    st.markdown("### 📊 Forecast Results & Telemetry")
    
    if submit_btn:
        # 1. Pydantic input validation
        validation_error = None
        try:
            validated_input = WeatherPredictionInput(
                temperature=temp_input,
                relative_humidity=humidity_input,
                wind_speed=wind_input,
                horizon_hours=horizon_choice
            )
        except ValidationError as ve:
            validation_error = ve

        if validation_error:
            st.error("Input Validation Failed")
            for err in validation_error.errors():
                st.markdown(f"- **{err['loc'][0]}**: {err['msg']}")
        else:
            # 2. Run inference
            with st.spinner("Executing model pipeline and feature scaling..."):
                try:
                    result = predict_temperature(validated_input)

                    # 3. Save to SQLite database
                    db_record = {
                        "timestamp": result.timestamp,
                        "previous_temperature": result.previous_temperature,
                        "relative_humidity": result.relative_humidity,
                        "wind_speed": result.wind_speed,
                        "horizon_hours": result.horizon_hours,
                        "predicted_temperature": result.predicted_temperature,
                        "model_version": f"{result.model_name} ({result.model_version})",
                        "notes": notes_input.strip()
                    }
                    save_prediction_record(db_record)

                    # Format temperature according to display unit
                    disp_pred = format_temperature(result.predicted_temperature, unit)
                    disp_prev = format_temperature(result.previous_temperature, unit)

                    delta = result.predicted_temperature - result.previous_temperature
                    delta_str = f"{delta:+.2f}°C" if unit == "°C" else f"{(delta * 1.8):+.2f}°F"

                    st.markdown(f"""
                    <div class="metric-card" style="border: 2px solid #38BDF8; background: #0E1A33; padding: 2rem;">
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <span class="status-badge badge-success">Forecast Generated</span>
                            <span style="font-size: 0.8rem; color: #94A3B8;">{result.timestamp}</span>
                        </div>
                        <div style="margin-top: 1rem; color: #94A3B8; font-size: 0.9rem; text-transform: uppercase;">
                            Predicted Temperature (+{result.horizon_hours}h)
                        </div>
                        <div style="font-size: 3rem; font-weight: 800; color: #38BDF8; margin: 0.25rem 0;">
                            {disp_pred}
                        </div>
                        <div style="font-size: 0.95rem; color: {'#34D399' if delta >= 0 else '#F87171'};">
                            Expected Thermal Shift: <b>{delta_str}</b> compared to previous {disp_prev}
                        </div>
                        <hr style="border: none; border-top: 1px solid #1E293B; margin: 1.25rem 0;">
                        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 0.75rem; font-size: 0.85rem;">
                            <div><span style="color: #64748B;">Observed Temp:</span> <b>{disp_prev}</b></div>
                            <div><span style="color: #64748B;">Humidity:</span> <b>{result.relative_humidity:.1f}%</b></div>
                            <div><span style="color: #64748B;">Wind Velocity:</span> <b>{result.wind_speed:.1f} km/h</b></div>
                            <div><span style="color: #64748B;">Algorithm:</span> <b>{result.model_name}</b></div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    st.markdown("#### 💡 Forecasting Rationale")
                    st.info(result.explanation)

                    # Export single prediction result as CSV
                    res_df = pd.DataFrame([result.to_dict()])
                    csv_bytes = dataframe_to_csv_bytes(res_df)
                    st.download_button(
                        label="📥 Download This Forecast Result (CSV)",
                        data=csv_bytes,
                        file_name=f"forecast_{result.timestamp.replace(':', '-').replace(' ', '_')}.csv",
                        mime="text/csv",
                        use_container_width=True
                    )

                except Exception as ex:
                    st.error(f"Prediction failed: {str(ex)}")

    else:
        st.markdown("""
        <div class="metric-card" style="padding: 3rem; text-align: center;">
            <div style="font-size: 2.5rem; margin-bottom: 0.5rem;">⏱️</div>
            <div style="color: #F8FAFC; font-weight: 600;">Awaiting Input</div>
            <div style="color: #94A3B8; font-size: 0.85rem; margin-top: 0.25rem;">
                Adjust meteorological variables on the left and click <b>Predict Temperature</b>.
            </div>
        </div>
        """, unsafe_allow_html=True)
