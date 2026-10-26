"""
WeatherIQ - Dataset Management & Ingestion Center
Upload historical weather CSVs, auto-detect or configure column mappings,
audit data quality, trigger Open-Meteo historical fetching, and retrain models.
"""

from pathlib import Path
import io
import streamlit as st
import pandas as pd
import numpy as np

from src.config import (
    DATA_DIR, 
    DEFAULT_MODEL_PATH, 
    DEFAULT_METADATA_PATH, 
    PREDEFINED_LOCATIONS,
    DEFAULT_LOCATION,
    SUPPORTED_HORIZONS
)
from src.ui_helpers import (
    inject_custom_css, 
    init_session_state, 
    render_header, 
    render_metric_card, 
    render_sidebar_branding
)
from src.utils import dataframe_to_csv_bytes
from src.data_loader import load_csv, detect_column_mapping, get_dataset_summary
from src.data_validation import validate_weather_dataframe
from src.preprocessing import clean_weather_dataframe
from src.weather_api import fetch_open_meteo_historical_data, WeatherAPIError
from src.train import train_and_select_best_model
from scripts.generate_sample_data import generate_synthetic_weather_dataset

st.set_page_config(
    page_title="Dataset Management | WeatherIQ",
    page_icon="📁",
    layout="wide"
)

inject_custom_css()
init_session_state()
render_sidebar_branding()

render_header(
    title="Dataset Management & Model Pipeline",
    subtitle="Ingest historical observations, inspect data distributions, audit schemas, and train models.",
    badge_text="Data Pipeline"
)

# ----------------- TABS FOR DATA SOURCES -----------------
tab_upload, tab_sample, tab_api = st.tabs([
    "📤 Upload Weather CSV", 
    "🧪 Demonstration Data Generator", 
    "🌐 Open-Meteo Meteorological Archive"
])

# Container to hold ingested raw dataframe
raw_df: Optional[pd.DataFrame] = None
data_source_label = ""
is_synthetic = False

# TAB 1: CSV UPLOAD
with tab_upload:
    st.markdown("#### Upload Historical Observations (CSV)")
    st.caption("Upload any CSV containing timestamps, temperatures, humidity, and wind speed.")
    uploaded_file = st.file_uploader(
        "Select CSV file", 
        type=["csv"], 
        help="Supports standard comma-separated tabular files."
    )
    if uploaded_file is not None:
        try:
            raw_df = load_csv(uploaded_file)
            data_source_label = f"Uploaded CSV: {uploaded_file.name}"
            is_synthetic = False
            st.success(f"Successfully loaded '{uploaded_file.name}' ({len(raw_df):,} rows).")
        except Exception as e:
            st.error(f"Failed to read CSV: {str(e)}")

# TAB 2: DEMO GENERATOR
with tab_sample:
    st.markdown("#### Generate Calibrated Demonstration Weather Data")
    st.markdown("""
    <div class="notice-box">
        <b>NOTICE:</b> This generator synthesizes deterministic hourly weather observations 
        (diurnal oscillations, temperature wave, relative humidity and wind variations). 
        It is explicitly labeled as <b>SYNTHETIC DEMONSTRATION DATA</b>.
    </div>
    """, unsafe_allow_html=True)
    
    col_g1, col_g2 = st.columns([2, 1])
    with col_g1:
        demo_days = st.slider("Duration of synthetic observations", min_value=14, max_value=180, value=90, step=7)
    with col_g2:
        if st.button("⚡ Generate Demo Dataset", type="primary"):
            with st.spinner("Synthesizing hourly weather dynamics..."):
                sample_file = generate_synthetic_weather_dataset(days=demo_days)
                raw_df = pd.read_csv(sample_file)
                data_source_label = f"Synthetic Demo Dataset ({demo_days} days, Bhopal)"
                is_synthetic = True
                st.session_state["active_dataset"] = raw_df
                st.session_state["dataset_source_name"] = data_source_label
                st.session_state["dataset_is_synthetic"] = True
                st.success(f"Generated {len(raw_df):,} hourly synthetic demonstration records!")

# TAB 3: OPEN-METEO ARCHIVE
with tab_api:
    st.markdown("#### Sync from Open-Meteo Historical Archive")
    st.caption("Download real historical ground station observations via Open-Meteo API.")
    
    col_a1, col_a2 = st.columns(2)
    with col_a1:
        station_choice = st.selectbox("Location Station", options=list(PREDEFINED_LOCATIONS.keys()), index=0)
        station_info = PREDEFINED_LOCATIONS[station_choice]
    
    with col_a2:
        col_d1, col_d2 = st.columns(2)
        with col_d1:
            api_start = st.date_input("Start Date", value=pd.to_datetime("2024-01-01").date())
        with col_d2:
            api_end = st.date_input("End Date", value=pd.to_datetime("2024-03-31").date())

    if st.button("🌐 Fetch Historical Observations"):
        with st.spinner(f"Connecting to Open-Meteo for {station_choice}..."):
            try:
                df_api, api_meta = fetch_open_meteo_historical_data(
                    latitude=station_info["lat"],
                    longitude=station_info["lon"],
                    start_date=str(api_start),
                    end_date=str(api_end),
                    timezone_str=station_info["timezone"]
                )
                raw_df = df_api
                data_source_label = f"Open-Meteo Archive ({station_choice})"
                is_synthetic = False
                st.session_state["active_dataset"] = raw_df
                st.session_state["dataset_source_name"] = data_source_label
                st.session_state["dataset_is_synthetic"] = False
                st.success(f"Retrieved {len(raw_df):,} verified records from Open-Meteo!")
            except WeatherAPIError as we:
                st.error(f"API Error: {str(we)}")
            except Exception as ex:
                st.error(f"Unexpected error: {str(ex)}")

# If no new dataset selected in this session tab, check if one is already cached in session state
if raw_df is None and st.session_state.get("active_dataset") is not None:
    raw_df = st.session_state.get("active_dataset")
    data_source_label = st.session_state.get("dataset_source_name", "Active Workspace Dataset")
    is_synthetic = st.session_state.get("dataset_is_synthetic", False)

if raw_df is None:
    st.info("Please select one of the tabs above to provide or generate a dataset.")
    st.stop()

st.markdown("---")

# ----------------- COLUMN MAPPING INTERFACE -----------------
st.markdown("### 🔀 Step 1: Column Schema Mapping")
st.caption("Map CSV columns to the standardized WeatherIQ features. Automatic detection has been applied.")

detected = detect_column_mapping(list(raw_df.columns))
available_cols = list(raw_df.columns)

m_col1, m_col2, m_col3, m_col4 = st.columns(4)

with m_col1:
    default_ts_idx = available_cols.index(detected["timestamp"]) if detected.get("timestamp") in available_cols else 0
    mapped_ts = st.selectbox("Timestamp Column", options=available_cols, index=default_ts_idx)

with m_col2:
    default_temp_idx = available_cols.index(detected["temperature"]) if detected.get("temperature") in available_cols else 0
    mapped_temp = st.selectbox("Temperature Column (°C)", options=available_cols, index=default_temp_idx)

with m_col3:
    default_hum_idx = available_cols.index(detected["relative_humidity"]) if detected.get("relative_humidity") in available_cols else 0
    mapped_hum = st.selectbox("Relative Humidity Column (%)", options=available_cols, index=default_hum_idx)

with m_col4:
    default_wind_idx = available_cols.index(detected["wind_speed"]) if detected.get("wind_speed") in available_cols else 0
    mapped_wind = st.selectbox("Wind Speed Column (km/h)", options=available_cols, index=default_wind_idx)

mapping = {
    "timestamp": mapped_ts,
    "temperature": mapped_temp,
    "relative_humidity": mapped_hum,
    "wind_speed": mapped_wind
}

# ----------------- DATASET VALIDATION & PREVIEW -----------------
st.markdown("### 🔍 Step 2: Quality Audit & Validation")

val_report = validate_weather_dataframe(raw_df, mapping)

if not val_report.is_valid:
    st.error("Dataset Failed Integrity Checks:")
    for err in val_report.error_details:
        st.markdown(f"- ❌ {err}")
else:
    st.success("Dataset Schema & Data Types are Valid!")
    if val_report.validation_messages:
        with st.expander("ℹ️ Data Integrity Notes & Findings", expanded=True):
            for msg in val_report.validation_messages:
                st.markdown(f"- 🔎 {msg}")

# Summary Metrics
summary = get_dataset_summary(raw_df, mapping)
v1, v2, v3, v4 = st.columns(4)
with v1:
    render_metric_card("Total Records", f"{summary['total_records']:,}", "Rows in file", "Volume", "info")
with v2:
    render_metric_card("Date Span", f"{summary['time_span_days']} days", f"{summary['timestamp_start'][:10] if summary['timestamp_start'] else 'N/A'}", "Duration", "info")
with v3:
    render_metric_card("Missing Values", f"{sum(val_report.missing_values.values()):,}", "Across mapped columns", "Quality", "info")
with v4:
    render_metric_card("Duplicate Timestamps", f"{val_report.duplicate_timestamps}", "To be resolved", "Integrity", "warning" if val_report.duplicate_timestamps > 0 else "info")

# Raw Preview Table
st.markdown("#### Raw Dataset Sample (First 5 Rows)")
st.dataframe(raw_df.head(5), use_container_width=True)

# ----------------- DATA CLEANING & REPROCESSING -----------------
st.markdown("### 🧹 Step 3: Clean & Standardize Dataset")

clean_df, clean_report = clean_weather_dataframe(raw_df, mapping)

with st.expander("📋 Preprocessing & Cleaning Audit Report", expanded=False):
    st.markdown(f"- **Initial Row Count:** {clean_report['initial_rows']:,}")
    st.markdown(f"- **Final Cleaned Rows:** {clean_report['final_rows']:,}")
    st.markdown(f"- **Duplicate Timestamps Consolidated:** {clean_report['duplicates_removed']}")
    st.markdown(f"- **Missing Timestamps Dropped:** {clean_report['invalid_timestamps_dropped']}")
    for num_col, cnt in clean_report["interpolated_values"].items():
        st.markdown(f"- **Time-Interpolated Values in '{num_col}':** {cnt}")
    for col, cnt in clean_report["out_of_bounds_clipped"].items():
        st.markdown(f"- **Clipped Out-of-Bounds Values in '{col}':** {cnt}")

# Option to export cleaned CSV
csv_clean_bytes = dataframe_to_csv_bytes(clean_df)
st.download_button(
    label="📥 Download Cleaned Dataset (CSV)",
    data=csv_clean_bytes,
    file_name="weatheriq_cleaned_dataset.csv",
    mime="text/csv"
)

# Cache cleaned dataset into session state
st.session_state["active_dataset"] = clean_df
st.session_state["active_mapping"] = {
    "timestamp": "timestamp",
    "temperature": "temperature",
    "relative_humidity": "relative_humidity",
    "wind_speed": "wind_speed"
}
st.session_state["dataset_source_name"] = data_source_label
st.session_state["dataset_is_synthetic"] = is_synthetic

# ----------------- MODEL TRAINING SECTION -----------------
st.markdown("---")
st.markdown("### 🤖 Step 4: Machine Learning Model Training")
st.caption("Train and compare candidate algorithms: Linear Regression, Random Forest, HistGradientBoosting.")

tcol1, tcol2 = st.columns([1, 1])

with tcol1:
    horizon_hours_choice = st.selectbox(
        "Select Prediction Lookahead Horizon (Hours)",
        options=SUPPORTED_HORIZONS,
        index=0,
        help="Default is 1 hour ahead for hourly observations."
    )

with tcol2:
    st.markdown("<br>", unsafe_allow_html=True)
    train_btn = st.button("🚀 Train Machine Learning Models Now", type="primary", use_container_width=True)

if train_btn:
    with st.spinner(f"Splitting data chronologically and training candidate models for +{horizon_hours_choice}h horizon..."):
        try:
            train_results = train_and_select_best_model(
                clean_df=clean_df,
                horizon_hours=horizon_hours_choice
            )
            meta = train_results["metadata"]
            st.success(f"Model Training Complete! Selected Winner: **{meta['model_name']}** ({meta['model_version']})")
            
            # Comparison summary
            st.markdown(f"""
            <div class="metric-card" style="border: 2px solid #34D399; padding: 1.5rem; margin-top: 1rem;">
                <div style="font-size: 1.1rem; font-weight: 700; color: #34D399;">
                    ✓ Best Model: {meta['model_name']}
                </div>
                <div style="margin-top: 0.5rem; color: #F8FAFC; font-size: 0.95rem;">
                    Held-Out Test MAE: <b>{meta['test_metrics']['mae']:.3f}°C</b> | 
                    RMSE: <b>{meta['test_metrics']['rmse']:.3f}°C</b> | 
                    R²: <b>{meta['test_metrics']['r2']:.4f}</b>
                </div>
                <div style="color: #94A3B8; font-size: 0.85rem; margin-top: 0.25rem;">
                    Persistence Baseline MAE: <b>{meta['persistence_metrics']['mae']:.3f}°C</b> 
                    (Model improvement: <b>{meta['mae_improvement_c']:+.3f}°C / {meta['percent_improvement']:+.1f}%</b>)
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.balloons()
        except Exception as e:
            st.error(f"Training failed: {str(e)}")
