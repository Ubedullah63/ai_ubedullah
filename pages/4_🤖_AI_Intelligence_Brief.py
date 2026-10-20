"""
pages/4_🤖_AI_Intelligence_Brief.py
Automated Gemini AI Executive Intelligence Briefings.
"""

from datetime import datetime
import streamlit as st

from utils.prediction_engine import (
    compute_sales_forecast,
    compute_executive_kpis,
    get_segment_mix,
)
from utils.gemini_service import request_gemini_briefing

if "historical_df" not in st.session_state:
    st.switch_page("app.py")

st.markdown("## 🤖 Gemini AI Executive Intelligence Brief")
st.caption("Automated automotive briefings synthesized using Gemini 3 Flash and live dealership telemetry.")

# Compute state parameters
df_forecast = compute_sales_forecast(
    st.session_state.historical_df,
    st.session_state.sim_controls,
    st.session_state.timeframe
)
kpis = compute_executive_kpis(df_forecast, st.session_state.sim_controls)
segment_mix = get_segment_mix(st.session_state.sim_controls)

rc1, rc2, rc3 = st.columns([1, 1, 1])
tone = rc1.selectbox("Report Tone", [
    "Balanced Executive",
    "Aggressive Sales Growth",
    "Risk-Averse / Cost-Control",
    "Investor & Board Ready"
])
focus = rc2.selectbox("Strategic Directive", [
    "Max Margin & Inventory Turnover",
    "Weekend Demand Preparation",
    "EV Transition & Tax Credit Allocation",
    "F&I Desk & Interest Rate Defense"
])
generate_btn = rc3.button("✨ Generate Executive Briefing", type="primary", use_container_width=True)

if generate_btn or not st.session_state.ai_report_text:
    with st.spinner("Calling Gemini 3 Flash to evaluate inventory risk, discount elasticity, and synthesize battle plans..."):
        report = request_gemini_briefing(
            api_key=st.session_state.api_key,
            timeframe=st.session_state.timeframe,
            kpis=kpis,
            sim_controls=st.session_state.sim_controls,
            segment_mix=segment_mix,
            tone=tone,
            focus=focus
        )
        st.session_state.ai_report_text = report

st.markdown("---")
st.markdown(st.session_state.ai_report_text)

st.download_button(
    label="📥 Download Briefing as Markdown",
    data=st.session_state.ai_report_text,
    file_name=f"Automotive_Briefing_{st.session_state.timeframe}_{datetime.now().strftime('%Y%m%d')}.md",
    mime="text/markdown"
)