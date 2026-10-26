"""
WeatherIQ - UI Components & Layout Helpers
Provides shared styling injection, branding headers, metric cards,
and session state helpers for Streamlit pages.
"""

from pathlib import Path
from typing import Optional, Dict, Any
import streamlit as st

from src.config import ASSETS_DIR, DEFAULT_LOCATION, THEME_TOKENS
from src.utils import format_temperature


def inject_custom_css() -> None:
    """Injects custom CSS from assets/styles.css into the active Streamlit page."""
    css_file = ASSETS_DIR / "styles.css"
    if css_file.exists():
        with open(css_file, "r", encoding="utf-8") as f:
            css_content = f.read()
        st.markdown(f"<style>{css_content}</style>", unsafe_allow_html=True)
    else:
        # Graceful fallback styling
        st.markdown("""
        <style>
        .stApp { background-color: #0B1120; color: #F8FAFC; }
        </style>
        """, unsafe_allow_html=True)


def init_session_state() -> None:
    """Initializes standard global session state variables."""
    if "temp_unit" not in st.session_state:
        st.session_state["temp_unit"] = "°C"
    if "current_location" not in st.session_state:
        st.session_state["current_location"] = DEFAULT_LOCATION
    if "active_dataset" not in st.session_state:
        st.session_state["active_dataset"] = None
    if "active_mapping" not in st.session_state:
        st.session_state["active_mapping"] = None
    if "dataset_source_name" not in st.session_state:
        st.session_state["dataset_source_name"] = "No dataset loaded"
    if "dataset_is_synthetic" not in st.session_state:
        st.session_state["dataset_is_synthetic"] = False


def render_header(title: str, subtitle: str, badge_text: str = "WeatherIQ ML") -> None:
    """Renders a standardized premium header banner."""
    st.markdown(f"""
    <div class="hero-card">
        <div style="display: flex; justify-content: space-between; align-items: flex-start; flex-wrap: wrap; gap: 0.5rem;">
            <div>
                <div class="hero-title">
                    <span>⚡</span> {title}
                </div>
                <div class="hero-subtitle">{subtitle}</div>
            </div>
            <div>
                <span class="status-badge badge-info">{badge_text}</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_metric_card(
    title: str, 
    value: str, 
    subtext: str = "", 
    badge: Optional[str] = None, 
    badge_type: str = "info"
) -> None:
    """Renders a standardized styled metric card."""
    badge_html = f'<span class="status-badge badge-{badge_type}" style="margin-left:auto;">{badge}</span>' if badge else ""
    st.markdown(f"""
    <div class="metric-card">
        <div style="display: flex; align-items: center; justify-content: space-between;">
            <div class="metric-title">{title}</div>
            {badge_html}
        </div>
        <div class="metric-value">{value}</div>
        <div class="metric-sub">{subtext}</div>
    </div>
    """, unsafe_allow_html=True)


def render_sidebar_branding() -> None:
    """Renders product branding and quick preferences in the sidebar."""
    with st.sidebar:
        st.markdown(f"""
        <div style="padding: 0.75rem 0.5rem 1.25rem 0.5rem; border-bottom: 1px solid #1E293B; margin-bottom: 1rem;">
            <div style="display: flex; align-items: center; gap: 0.75rem;">
                <div style="width: 38px; height: 38px; border-radius: 8px; background: linear-gradient(135deg, #0284C7, #38BDF8); display: flex; align-items: center; justify-content: center; font-size: 20px;">
                    🌤️
                </div>
                <div>
                    <div style="font-weight: 700; font-size: 1.15rem; color: #FFFFFF; letter-spacing: -0.02em;">WeatherIQ</div>
                    <div style="font-size: 0.75rem; color: #94A3B8;">Intelligent Forecasting</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.caption("⚙️ GLOBAL PREFERENCES")
        unit_choice = st.radio(
            "Temperature Display Unit",
            options=["°C", "°F"],
            index=0 if st.session_state.get("temp_unit", "°C") == "°C" else 1,
            horizontal=True,
            key="global_temp_unit_radio"
        )
        st.session_state["temp_unit"] = unit_choice

        st.markdown("<hr style='border: none; border-top: 1px solid #1E293B; margin: 1rem 0;'>", unsafe_allow_html=True)
        st.caption("WeatherIQ v1.0 • College ML Project")
