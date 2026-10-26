"""
WeatherIQ - Prediction History Audit Log
Local SQLite persistence interface for reviewing, filtering, exporting,
and managing all past model inferences.
"""

from pathlib import Path
import streamlit as st
import pandas as pd

from src.ui_helpers import (
    inject_custom_css, 
    init_session_state, 
    render_header, 
    render_metric_card, 
    render_sidebar_branding
)
from src.utils import format_temperature, celsius_to_fahrenheit, dataframe_to_csv_bytes
from src.database import (
    get_predictions_df, 
    delete_prediction_record, 
    clear_all_predictions,
    init_db
)

st.set_page_config(
    page_title="Prediction History | WeatherIQ",
    page_icon="📜",
    layout="wide"
)

inject_custom_css()
init_session_state()
render_sidebar_branding()

unit = st.session_state.get("temp_unit", "°C")

render_header(
    title="Prediction History & Audit Trail",
    subtitle="Query, filter, export, and manage inferences stored in the local SQLite database.",
    badge_text="SQLite Audit Log"
)

# Fetch data
df_history = get_predictions_df()

if df_history.empty:
    st.info("The prediction database is currently empty.")
    st.markdown("""
    <div class="metric-card" style="padding: 2.5rem; text-align: center;">
        <h4 style="color: #38BDF8;">No Forecasts Recorded Yet</h4>
        <p style="color: #94A3B8; max-width: 600px; margin: 0 auto 1.5rem auto;">
            Run predictions using the Temperature Prediction page. Every forecast, its input parameters, 
            model version, and timestamp will be automatically logged here.
        </p>
    </div>
    """, unsafe_allow_html=True)
    if st.button("🔮 Run a Prediction", type="primary"):
        st.switch_page("pages/2_Prediction.py")
    st.stop()

# ----------------- FILTERS AND CONTROLS -----------------
st.markdown("#### 🔍 Filter & Sort Records")
fcol1, fcol2, fcol3 = st.columns([2, 2, 2])

with fcol1:
    search_query = st.text_input("Search Notes or Model Version", placeholder="Type keywords...")

with fcol2:
    sort_order = st.selectbox("Sort Order", options=["Newest First (Default)", "Oldest First", "Highest Predicted Temp", "Lowest Predicted Temp"])

with fcol3:
    records_limit = st.slider("Record Limit", min_value=10, max_value=500, value=100, step=10)

# Filter application
filtered = df_history.copy()

if search_query:
    q = search_query.lower()
    filtered = filtered[
        filtered["notes"].astype(str).str.lower().str.contains(q) | 
        filtered["model_version"].astype(str).str.lower().str.contains(q)
    ]

if sort_order == "Newest First (Default)":
    filtered.sort_values(by="id", ascending=False, inplace=True)
elif sort_order == "Oldest First":
    filtered.sort_values(by="id", ascending=True, inplace=True)
elif sort_order == "Highest Predicted Temp":
    filtered.sort_values(by="predicted_temperature", ascending=False, inplace=True)
elif sort_order == "Lowest Predicted Temp":
    filtered.sort_values(by="predicted_temperature", ascending=True, inplace=True)

filtered = filtered.head(records_limit)

# Metric Summary
m1, m2, m3, m4 = st.columns(4)
with m1:
    render_metric_card("Total Inferences", f"{len(df_history)}", "Logged in SQLite", "Total", "info")
with m2:
    mean_pred = filtered["predicted_temperature"].mean()
    val_disp = format_temperature(mean_pred, unit)
    render_metric_card("Mean Forecast", val_disp, f"Across {len(filtered)} records", "Mean", "info")
with m3:
    max_pred = filtered["predicted_temperature"].max()
    val_disp = format_temperature(max_pred, unit)
    render_metric_card("Peak Forecast", val_disp, "Highest predicted", "Max", "warning")
with m4:
    min_pred = filtered["predicted_temperature"].min()
    val_disp = format_temperature(min_pred, unit)
    render_metric_card("Lowest Forecast", val_disp, "Minimum predicted", "Min", "info")

st.markdown("---")

# Format for display table
display_df = filtered.copy()
if unit == "°F":
    display_df["previous_temperature"] = display_df["previous_temperature"].apply(celsius_to_fahrenheit).round(1)
    display_df["predicted_temperature"] = display_df["predicted_temperature"].apply(celsius_to_fahrenheit).round(1)

display_df.rename(columns={
    "id": "ID",
    "timestamp": "Timestamp",
    "previous_temperature": f"T_prev ({unit})",
    "relative_humidity": "Humidity (%)",
    "wind_speed": "Wind (km/h)",
    "horizon_hours": "Horizon (hrs)",
    "predicted_temperature": f"Forecast ({unit})",
    "model_version": "Model Version",
    "notes": "Observation Notes"
}, inplace=True)

st.dataframe(display_df, use_container_width=True, hide_index=True)

# ----------------- CSV EXPORT & RECORD DELETION -----------------
st.markdown("### 💾 Export & Database Maintenance")
dcol1, dcol2 = st.columns([1, 1])

with dcol1:
    st.markdown("#### Export Records")
    csv_bytes = dataframe_to_csv_bytes(display_df)
    st.download_button(
        label="📥 Download Filtered History as CSV",
        data=csv_bytes,
        file_name="weatheriq_prediction_history.csv",
        mime="text/csv",
        use_container_width=True
    )

with dcol2:
    st.markdown("#### Delete Records")
    with st.expander("⚠️ Database Record Deletion"):
        rec_ids = filtered["id"].tolist()
        if rec_ids:
            selected_id_to_delete = st.selectbox("Select Record ID to Delete", options=rec_ids)
            confirm_single = st.checkbox(f"Confirm deletion of record #{selected_id_to_delete}")
            if st.button("🗑️ Delete Selected Record", type="primary", disabled=not confirm_single):
                if delete_prediction_record(selected_id_to_delete):
                    st.success(f"Record #{selected_id_to_delete} deleted successfully.")
                    st.rerun()

        st.markdown("<hr style='border: none; border-top: 1px solid #1E293B;'>", unsafe_allow_html=True)
        confirm_all = st.checkbox("Confirm wipe: Delete ALL history records permanently")
        if st.button("🔥 Wipe All Prediction History", disabled=not confirm_all):
            deleted_count = clear_all_predictions()
            st.success(f"Purged {deleted_count} prediction records from SQLite database.")
            st.rerun()
