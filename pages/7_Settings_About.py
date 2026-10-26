"""
WeatherIQ - Settings, Project Methodology & Documentation
Global settings, meteorological methodology, model version audits,
and college viva explanation guide.
"""

from pathlib import Path
import json
import streamlit as st

from src.config import (
    DEFAULT_MODEL_PATH, 
    DEFAULT_METADATA_PATH, 
    DEFAULT_LOCATION,
    PREDEFINED_LOCATIONS,
    DEFAULT_HORIZON_HOURS,
    SUPPORTED_HORIZONS
)
from src.ui_helpers import (
    inject_custom_css, 
    init_session_state, 
    render_header, 
    render_metric_card, 
    render_sidebar_branding
)
from src import __version__

st.set_page_config(
    page_title="Settings & About | WeatherIQ",
    page_icon="⚙️",
    layout="wide"
)

inject_custom_css()
init_session_state()
render_sidebar_branding()

render_header(
    title="Settings, Methodology & System Architecture",
    subtitle="Configuration parameters, mathematical formulations, and engineering architecture.",
    badge_text="System Specs"
)

# Load model metadata if present
has_model = DEFAULT_MODEL_PATH.exists() and DEFAULT_METADATA_PATH.exists()
metadata = {}
if has_model:
    try:
        with open(DEFAULT_METADATA_PATH, "r", encoding="utf-8") as f:
            metadata = json.load(f)
    except Exception:
        has_model = False

# ----------------- SECTION 1: GLOBAL PREFERENCES -----------------
st.markdown("### ⚙️ User Preferences & Runtime Configuration")

scol1, scol2, scol3 = st.columns(3)

with scol1:
    unit_choice = st.selectbox(
        "Temperature Display Unit",
        options=["°C", "°F"],
        index=0 if st.session_state.get("temp_unit", "°C") == "°C" else 1,
        help="Internal calculations are always computed in degrees Celsius. Only UI display is converted."
    )
    st.session_state["temp_unit"] = unit_choice

with scol2:
    loc_names = list(PREDEFINED_LOCATIONS.keys())
    current_loc = st.session_state.get("current_location", DEFAULT_LOCATION)
    loc_choice = st.selectbox(
        "Default Reference Station",
        options=loc_names,
        index=loc_names.index(current_loc) if current_loc in loc_names else 0
    )
    st.session_state["current_location"] = loc_choice

with scol3:
    st.markdown(f"""
    <div class="metric-card" style="padding: 1rem 1.25rem;">
        <div class="metric-title">Software Version</div>
        <div style="font-size: 1.2rem; font-weight: 700; color: #38BDF8;">WeatherIQ v{__version__}</div>
        <div style="font-size: 0.8rem; color: #94A3B8;">Python 3.11 • Streamlit • Scikit-learn</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# ----------------- SECTION 2: MODEL TELEMETRY -----------------
st.markdown("### 🤖 Deployed Model Registry")
if has_model:
    mcol1, mcol2 = st.columns(2)
    with mcol1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Active Model Metadata</div>
            <div style="font-size: 0.9rem; margin-top: 0.5rem; line-height: 1.8;">
                <b>Algorithm:</b> {metadata.get('model_name')}<br>
                <b>Version Hash:</b> <code>{metadata.get('model_version')}</code><br>
                <b>Trained Horizon:</b> {metadata.get('horizon_hours')} Hour(s) ahead<br>
                <b>Input Features:</b> <code>{", ".join(metadata.get('input_features', []))}</code><br>
                <b>Target Variable:</b> <code>{metadata.get('target_feature')}</code>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with mcol2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Performance Telemetry (Held-out Test Set)</div>
            <div style="font-size: 0.9rem; margin-top: 0.5rem; line-height: 1.8;">
                <b>Test MAE:</b> ±{metadata.get('test_metrics', {}).get('mae', 0):.3f} °C<br>
                <b>Test RMSE:</b> ±{metadata.get('test_metrics', {}).get('rmse', 0):.3f} °C<br>
                <b>Test R²:</b> {metadata.get('test_metrics', {}).get('r2', 0):.4f}<br>
                <b>Persistence Baseline MAE:</b> ±{metadata.get('persistence_metrics', {}).get('mae', 0):.3f} °C<br>
                <b>MAE Improvement:</b> {metadata.get('mae_improvement_c', 0):+.3f} °C ({metadata.get('percent_improvement', 0):+.1f}%)
            </div>
        </div>
        """, unsafe_allow_html=True)
else:
    st.info("No active model deployed. Train a model in Dataset Management.")

st.markdown("---")

# ----------------- SECTION 3: PROJECT METHODOLOGY & VIVA NOTES -----------------
st.markdown("### 📚 Project Methodology & College Viva Preparation")

with st.expander("🎓 1. Problem Statement & Mathematical Formulation", expanded=True):
    st.markdown("""
    **Problem Statement:**
    Short-term temperature forecasting is vital for agriculture, renewable energy management, and smart urban infrastructure. 
    This system predicts the future temperature $T_{t+h}$ at lookahead horizon $h$ (default 1 hour) 
    given current atmospheric parameters observed at time $t$:
    
    $$\hat{T}_{t+h} = f(T_t, RH_t, WS_t)$$
    
    Where:
    - $T_t$: Temperature at observation time $t$ (°C)
    - $RH_t$: Relative Humidity at observation time $t$ (%)
    - $WS_t$: Wind Speed at observation time $t$ (km/h)
    """)

with st.expander("🧪 2. Machine Learning Pipeline & Preventing Data Leakage", expanded=True):
    st.markdown("""
    - **No Future Information Leakage:** Features $X_t$ are constructed exclusively from measurements at time $t$ or earlier. 
      The target $y_t = T_{t+h}$ is formed by chronological shifting.
    - **Chronological Time-Series Splitting:** Standard cross-validation randomly shuffles observations, which would cause severe 
      temporal data leakage due to strong autocorrelation in weather. WeatherIQ uses strict chronological partitioning:
      - Training (70%)
      - Validation (15%) for model selection
      - Untouched Held-Out Test (15%) for final evaluation
    - **Pipeline Scaling:** Feature normalization (StandardScaler) is fitted strictly on training data and stored inside the 
      Scikit-learn pipeline, avoiding leakage into validation or testing data.
    """)

with st.expander("⚖️ 3. Why the Persistence Baseline is Crucial", expanded=True):
    st.markdown("""
    In weather forecasting, temperature changes gradually from hour to hour. A trivial naive predictor, 
    the **Persistence Baseline**, predicts that future temperature equals current temperature:
    
    $$\hat{T}_{t+h}^{persistence} = T_t$$
    
    In a college demonstration, reporting high $R^2$ or low MAE without comparison to the persistence baseline is 
    scientifically flawed. A valid model must prove it genuinely captures thermodynamic transitions by outperforming 
    this baseline.
    """)

with st.expander("⚠️ 4. Real-World Data vs Synthetic Demonstration Data", expanded=True):
    st.markdown("""
    - **Synthetic Demonstration Data:** Generated using deterministic mathematical formulations (sinusoidal diurnal waves, 
      seasonal drift, physical humidity correlations). It is invaluable for UI smoke testing, offline presentations, 
      and automated unit tests.
    - **Scientific Caution:** Synthetic data CANNOT be used to substantiate real-world predictive validity. Real-world validation 
      must use actual ground-truth meteorological stations (e.g. from the Open-Meteo archive or uploaded CSVs).
    """)

with st.expander("🛠️ 5. Technology Stack & Architecture", expanded=False):
    st.markdown("""
    - **Backend & ML:** Python 3.11, Scikit-learn (Linear Regression, Random Forest, HistGradientBoosting), Pandas, NumPy, Joblib, Pydantic v2
    - **Database:** SQLite (local zero-configuration persistence with connection pooling and indexing)
    - **Frontend:** Streamlit Multipage Engine, Plotly Interactive Visuals, Custom Modern SaaS CSS
    - **Testing:** Pytest automated suite covering schema validation, chronological splits, leakage prevention, persistence baseline, and inference sanity
    """)
