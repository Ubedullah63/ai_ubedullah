"""
pages/2_🧪_Simulation_Lab.py
Interactive What-If Simulation Engine for adjusting macro and dealership levers.
"""

import streamlit as st
import plotly.express as px

from utils.prediction_engine import compute_sales_forecast, compute_executive_kpis

if "historical_df" not in st.session_state:
    st.switch_page("app.py")

st.markdown("## 🧪 What-If Sales Forecast Simulation Engine")
st.caption("Adjust macroeconomic, inventory, and marketing levers to see live impact on unit sales and gross revenue.")

# Quick Presets
st.markdown("#### Scenario Presets")
b1, b2, b3, b4 = st.columns(4)

if b1.button("🚀 Weekend Sales Blitz", use_container_width=True):
    st.session_state.sim_controls = {
        "marketing_spend_delta": 35,
        "dealer_discount": 6.5,
        "avg_interest_rate": 5.4,
        "lead_quality_index": 1.25,
        "inventory_adequacy": 98,
    }
    st.rerun()

if b2.button("🛡️ High APR Defense", use_container_width=True):
    st.session_state.sim_controls = {
        "marketing_spend_delta": -10,
        "dealer_discount": 3.5,
        "avg_interest_rate": 7.6,
        "lead_quality_index": 0.95,
        "inventory_adequacy": 88,
    }
    st.rerun()

if b3.button("⚡ EV Fleet Push", use_container_width=True):
    st.session_state.sim_controls = {
        "marketing_spend_delta": 25,
        "dealer_discount": 7.0,
        "avg_interest_rate": 5.2,
        "lead_quality_index": 1.20,
        "inventory_adequacy": 100,
    }
    st.rerun()

if b4.button("🔄 Reset to Baseline", use_container_width=True):
    st.session_state.sim_controls = {
        "marketing_spend_delta": 15,
        "dealer_discount": 5.0,
        "avg_interest_rate": 5.9,
        "lead_quality_index": 1.15,
        "inventory_adequacy": 95,
    }
    st.rerun()

st.markdown("---")

# Sliders
c1, c2 = st.columns(2)
with c1:
    st.session_state.sim_controls["marketing_spend_delta"] = st.slider(
        "Marketing / Ad Spend Delta (%)",
        min_value=-40, max_value=60,
        value=int(st.session_state.sim_controls["marketing_spend_delta"]),
        step=5
    )
    st.session_state.sim_controls["dealer_discount"] = st.slider(
        "Average Dealer Discount (% MSRP)",
        min_value=1.0, max_value=10.0,
        value=float(st.session_state.sim_controls["dealer_discount"]),
        step=0.5
    )
    st.session_state.sim_controls["avg_interest_rate"] = st.slider(
        "Customer Financing APR (%)",
        min_value=2.9, max_value=9.9,
        value=float(st.session_state.sim_controls["avg_interest_rate"]),
        step=0.1
    )

with c2:
    st.session_state.sim_controls["lead_quality_index"] = st.slider(
        "Lead Intent Quality Index (1.0 = Baseline)",
        min_value=0.8, max_value=1.4,
        value=float(st.session_state.sim_controls["lead_quality_index"]),
        step=0.05
    )
    st.session_state.sim_controls["inventory_adequacy"] = st.slider(
        "Lot Inventory Adequacy (%)",
        min_value=65, max_value=100,
        value=int(st.session_state.sim_controls["inventory_adequacy"]),
        step=1
    )

# Compute simulated outcomes
df_forecast = compute_sales_forecast(
    st.session_state.historical_df,
    st.session_state.sim_controls,
    st.session_state.timeframe
)
kpis = compute_executive_kpis(df_forecast, st.session_state.sim_controls)

st.info(
    f"💡 **Simulated Outcome:** Deliveries = **{kpis['total_units']} Vehicles** | "
    f"Gross Revenue = **${kpis['total_revenue'] / 1e6:.2f}M** | "
    f"Closing Rate = **{kpis['lead_to_deal_rate']}%**"
)

# Revenue Breakdown Bar Chart
fig_bar = px.bar(
    df_forecast,
    x="day",
    y="projected_revenue",
    color="predicted_units",
    labels={"projected_revenue": "Projected Revenue ($)", "day": "Day of Week", "predicted_units": "Units"},
    title="Projected Daily Gross Revenue ($)",
    template="plotly_dark",
    color_continuous_scale="Blues"
)
fig_bar.update_layout(height=360, margin=dict(l=20, r=20, t=40, b=20))
st.plotly_chart(fig_bar, use_container_width=True)