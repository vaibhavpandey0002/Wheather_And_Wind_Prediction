"""
WeatherIQ - Model Performance & Evaluation Dashboard
Rigorous evaluation on held-out test data, residual diagnostics,
model comparison benchmarks, and persistence baselines.
"""

from pathlib import Path
import json
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from src.config import DEFAULT_MODEL_PATH, DEFAULT_METADATA_PATH, THEME_TOKENS
from src.ui_helpers import (
    inject_custom_css, 
    init_session_state, 
    render_header, 
    render_metric_card, 
    render_sidebar_branding
)
from src.utils import format_temperature, celsius_to_fahrenheit
from src.evaluate import get_metric_explanations
from src.charts import (
    create_actual_vs_predicted_chart,
    create_residual_plot,
    create_residual_distribution_chart,
    create_model_comparison_bar_chart
)

st.set_page_config(
    page_title="Model Performance | WeatherIQ",
    page_icon="🎯",
    layout="wide"
)

inject_custom_css()
init_session_state()
render_sidebar_branding()

unit = st.session_state.get("temp_unit", "°C")

render_header(
    title="Model Evaluation & Verification Suite",
    subtitle="Objective performance benchmarks on held-out chronological test data.",
    badge_text="Model Audit"
)

# Load metadata
has_model = DEFAULT_MODEL_PATH.exists() and DEFAULT_METADATA_PATH.exists()
metadata = {}

if has_model:
    try:
        with open(DEFAULT_METADATA_PATH, "r", encoding="utf-8") as f:
            metadata = json.load(f)
    except Exception:
        has_model = False

if not has_model or "test_metrics" not in metadata:
    st.warning("No evaluated machine learning model is currently active.")
    st.markdown("""
    <div class="metric-card" style="padding: 2.5rem; text-align: center;">
        <h4 style="color: #FBBF24;">Evaluation Status: Not Evaluated</h4>
        <p style="color: #94A3B8; max-width: 600px; margin: 0 auto 1.5rem auto;">
            Model evaluation metrics and residual diagnostics will be computed automatically 
            upon completing a training run in the Dataset Management page.
        </p>
    </div>
    """, unsafe_allow_html=True)
    if st.button("🚀 Train Model Now", type="primary"):
        st.switch_page("pages/6_Dataset_Management.py")
    st.stop()

# ----------------- 1. TELEMETRY & TOP METRICS -----------------
test_metrics = metadata["test_metrics"]
persistence = metadata.get("persistence_metrics", {"mae": 0.0, "rmse": 0.0, "r2": 0.0})
mae_val = test_metrics["mae"]
rmse_val = test_metrics["rmse"]
r2_val = test_metrics["r2"]
pers_mae = persistence.get("mae", 0.0)

if unit == "°F":
    mae_val *= 1.8
    rmse_val *= 1.8
    pers_mae *= 1.8

kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)

with kpi_col1:
    render_metric_card(
        title="Test MAE",
        value=f"±{mae_val:.2f} {unit}",
        subtext="Mean Absolute Error",
        badge="Primary Metric",
        badge_type="info"
    )

with kpi_col2:
    render_metric_card(
        title="Test RMSE",
        value=f"±{rmse_val:.2f} {unit}",
        subtext="Root Mean Squared Error",
        badge="Loss Penalty",
        badge_type="info"
    )

with kpi_col3:
    render_metric_card(
        title="R² Score",
        value=f"{r2_val:.4f}",
        subtext="Explained Variance (max 1.0)",
        badge="Fit Quality",
        badge_type="success" if r2_val > 0.5 else "warning"
    )

with kpi_col4:
    diff_mae = pers_mae - mae_val
    pct = metadata.get("percent_improvement", 0.0)
    render_metric_card(
        title="vs Persistence Baseline",
        value=f"{diff_mae:+.2f} {unit}",
        subtext=f"{pct:+.1f}% improvement over T(t)",
        badge="Beats Baseline" if diff_mae > 0 else "Matches Baseline",
        badge_type="success" if diff_mae > 0 else "warning"
    )

# ----------------- 2. METRIC EXPLANATIONS EXPANDER -----------------
with st.expander("ℹ️ Scientific Explanations of Regression Metrics (Viva / Demo Notes)", expanded=False):
    explanations = get_metric_explanations()
    for name, expl in explanations.items():
        st.markdown(f"**{name}:** {expl}")

# ----------------- 3. DATASET PARTITION AUDIT -----------------
st.markdown("### 📋 Training & Testing Configuration Audit")
audit_col1, audit_col2, audit_col3 = st.columns(3)

with audit_col1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Model Specifications</div>
        <div style="font-size: 0.95rem; margin-top: 0.5rem;">
            <b>Algorithm:</b> {metadata.get('model_name')}<br>
            <b>Version:</b> <code>{metadata.get('model_version')}</code><br>
            <b>Forecast Horizon:</b> {metadata.get('horizon_hours')} hour(s) ahead
        </div>
    </div>
    """, unsafe_allow_html=True)

with audit_col2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Chronological Partition</div>
        <div style="font-size: 0.95rem; margin-top: 0.5rem;">
            <b>Training Samples:</b> {metadata.get('training_dataset_size'):,}<br>
            <b>Test Samples:</b> {metadata.get('test_dataset_size'):,}<br>
            <b>Split Strategy:</b> Strict Chronological
        </div>
    </div>
    """, unsafe_allow_html=True)

with audit_col3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Held-Out Test Window</div>
        <div style="font-size: 0.95rem; margin-top: 0.5rem;">
            <b>Test Start:</b> {metadata.get('test_period_start', '')[:16]}<br>
            <b>Test End:</b> {metadata.get('test_period_end', '')[:16]}<br>
            <b>Shuffle Leakage:</b> Prevented (0%)
        </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ----------------- 4. EVALUATION CHARTS -----------------
samples = metadata.get("test_samples", {})
if samples and "timestamps" in samples:
    # A. Actual vs Predicted Chart
    st.markdown("### 📉 Chronological Prediction Tracking")
    fig_track = create_actual_vs_predicted_chart(
        timestamps=samples["timestamps"],
        actual=samples["actual"],
        predicted=samples["predicted"],
        persistence=samples.get("persistence"),
        unit=unit
    )
    st.plotly_chart(fig_track, use_container_width=True)

    # B & C: Residual Diagnostics
    st.markdown("### 🔬 Residual Diagnostics (Error Analysis)")
    rcol1, rcol2 = st.columns(2)

    with rcol1:
        fig_res_plot = create_residual_plot(
            actual=samples["actual"],
            predicted=samples["predicted"],
            unit=unit
        )
        st.plotly_chart(fig_res_plot, use_container_width=True)

    with rcol2:
        residuals = np.array(samples["actual"]) - np.array(samples["predicted"])
        fig_dist = create_residual_distribution_chart(residuals.tolist(), unit=unit)
        st.plotly_chart(fig_dist, use_container_width=True)

    # D & E: Model Comparison Bar Chart & Error Over Time
    st.markdown("### 🏆 Candidate Algorithm Comparison")
    comp_col1, comp_col2 = st.columns([1, 1])

    with comp_col1:
        comp_dict = metadata.get("all_models_comparison", {})
        if comp_dict:
            fig_comp = create_model_comparison_bar_chart(
                comparison_dict=comp_dict,
                persistence_mae=metadata.get("persistence_metrics", {}).get("mae", 1.0)
            )
            st.plotly_chart(fig_comp, use_container_width=True)
        else:
            st.info("Algorithm benchmark data not available.")

    with comp_col2:
        # Error over time
        residuals = np.array(samples["actual"]) - np.array(samples["predicted"])
        if unit == "°F":
            residuals *= 1.8

        fig_eot = go.Figure()
        fig_eot.add_trace(go.Scatter(
            x=samples["timestamps"],
            y=np.abs(residuals),
            mode="lines",
            name="Absolute Error",
            line=dict(color=THEME_TOKENS["warning"], width=1.8),
            hovertemplate="<b>Time:</b> %{x}<br><b>Abs Error:</b> %{y:.2f}" + f" {unit}<extra></extra>"
        ))
        fig_eot.add_hline(y=mae_val, line_dash="dash", line_color=THEME_TOKENS["primary"], annotation_text=f"Mean Error ({mae_val:.2f} {unit})")
        fig_eot.update_layout(
            title="<b>Forecasting Error Magnitude Over Test Time</b>",
            paper_bgcolor=THEME_TOKENS["card_bg"],
            plot_bgcolor="#0A101D",
            font=dict(color=THEME_TOKENS["text_muted"], size=12),
            height=420,
            margin=dict(l=45, r=25, t=55, b=45),
            yaxis_title=f"|Actual - Predicted| ({unit})"
        )
        fig_eot.update_xaxes(showgrid=True, gridcolor="#1E293B")
        fig_eot.update_yaxes(showgrid=True, gridcolor="#1E293B")
        st.plotly_chart(fig_eot, use_container_width=True)
