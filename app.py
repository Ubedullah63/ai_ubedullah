"""
app.py
Main entrypoint for the AutoPredict AI Dealership Forecasting Suite.
Run locally via: streamlit run app.py
"""

import os
import streamlit as st
import pandas as pd
from dotenv import load_dotenv

from utils.prediction_engine import INITIAL_RECORDS

# Load .env file if available
load_dotenv()

st.set_page_config(
    page_title="AutoPredict AI | Dealership Sales Forecasting",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)


def init_session_state():
    """Initializes shared session state variables across all pages."""
    if "historical_df" not in st.session_state:
        st.session_state.historical_df = pd.DataFrame(INITIAL_RECORDS)
    if "timeframe" not in st.session_state:
        st.session_state.timeframe = "7d"
    if "api_key" not in st.session_state:
        st.session_state.api_key = os.environ.get("GEMINI_API_KEY", "")
    if "sim_controls" not in st.session_state:
        st.session_state.sim_controls = {
            "marketing_spend_delta": 15,
            "dealer_discount": 5.0,
            "avg_interest_rate": 5.9,
            "lead_quality_index": 1.15,
            "inventory_adequacy": 95,
        }
    if "ai_report_text" not in st.session_state:
        st.session_state.ai_report_text = ""


init_session_state()

# Global Sidebar Controls
st.sidebar.title("🚗 AutoPredict AI")
st.sidebar.caption("Dealership Predictive Intelligence Suite V4.5")

st.sidebar.markdown("---")
st.sidebar.subheader("⚙️ Horizon Selector")
st.session_state.timeframe = st.sidebar.selectbox(
    "Forecast Window",
    ["today", "tomorrow", "7d", "14d"],
    index=2,
    format_func=lambda x: {
        "today": "Today (24h)",
        "tomorrow": "Tomorrow (48h)",
        "7d": "Next 7 Days",
        "14d": "Next 14 Days"
    }[x]
)

st.sidebar.markdown("---")
st.sidebar.subheader("🔑 Gemini API Key")
key_input = st.sidebar.text_input(
    "Google AI Studio Key",
    value=st.session_state.api_key,
    type="password",
    help="Optional: Provide a key to run live Gemini 3 Flash generations. If left blank, offline heuristics are used."
)
if key_input != st.session_state.api_key:
    st.session_state.api_key = key_input

st.sidebar.markdown("---")
st.sidebar.info("💡 **Navigation Guide**: Select a page from the sidebar above to access the Dashboard, Simulation Lab, Telemetry Editor, or AI Briefings.")

# Welcome Hero on main app root
st.markdown("# 🚗 AutoPredict AI Dealership Forecasting Suite")
st.markdown(
    "Welcome to the **AutoPredict AI** multi-page intelligence platform. "
    "Use the left sidebar to navigate across the specialized modules:"
)

c1, c2 = st.columns(2)
with c1:
    st.markdown("""
    - **📊 1_Dashboard**: Visual executive KPIs, actuals vs predicted sales curve with 95% confidence bands, powertrain mix, and conversion funnel.
    - **🧪 2_Simulation_Lab**: What-If scenario sliders (Marketing, Discount %, Customer APR, Inventory Health) that dynamically recalculate forecasts.
    - **💬 5_Dealership_AI_Chatbot**: Interactive conversational advisor grounded in your live telemetry to answer operational, pricing, and GSM questions.
    """)
with c2:
    st.markdown("""
    - **📝 3_Telemetry_Records**: Ingest new showroom data, edit existing numbers inline, or reset baseline data.
    - **🤖 4_AI_Intelligence_Brief**: Generate structured executive briefings with Gemini 3 Flash, customize tones/directives, and export Markdown.
    """)

st.markdown("---")
st.caption("Active Target Forecast Horizon: **" + st.session_state.timeframe.upper() + "** | Ingested Telemetry Entries: **" + str(len(st.session_state.historical_df)) + "**")