"""
pages/1_📊_Dashboard.py
Executive Dealership Sales Dashboard with interactive Plotly visual analytics.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

from utils.prediction_engine import (
    compute_sales_forecast,
    compute_executive_kpis,
    get_segment_mix,
)

# Ensure state is present
if "historical_df" not in st.session_state:
    st.switch_page("app.py")

st.markdown("## 📊 Executive Dealership Sales Dashboard")
st.caption("Real-time telemetry, econometric forecasts, segment distribution, and throughput funnel.")

# Compute dynamic metrics
df_forecast = compute_sales_forecast(
    st.session_state.historical_df,
    st.session_state.sim_controls,
    st.session_state.timeframe
)
kpis = compute_executive_kpis(df_forecast, st.session_state.sim_controls)
segment_mix = get_segment_mix(st.session_state.sim_controls)

# Metric Row
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Projected Deliveries", f"{kpis['total_units']} units", "+14.2% pacing")
c2.metric("Gross Revenue", f"${kpis['total_revenue'] / 1e6:.2f}M", "+18.5% volume")
c3.metric("Average ASP", f"${kpis['avg_asp'] / 1000:.1f}k", "EV & SUV Lead")
c4.metric("Close Rate", f"{kpis['lead_to_deal_rate']}%", f"{kpis['total_test_drives']} test drives")
c5.metric("Model Confidence", f"{kpis['accuracy_score']}%", "High Precision")

st.markdown("---")

# Main Charts: Daily Curve & Segment Mix
col_chart, col_mix = st.columns([2, 1])

with col_chart:
    st.subheader("Daily Car Sales: Actual Deliveries vs Predictive Model")
    hist_df = st.session_state.historical_df.sort_values("date")
    fig_timeline = go.Figure()

    # Actuals
    fig_timeline.add_trace(go.Scatter(
        x=hist_df["date"],
        y=hist_df["actual_sales"],
        mode="lines+markers",
        name="Actual Deliveries",
        line=dict(color="#3b82f6", width=3),
        fill="tozeroy",
        fillcolor="rgba(59, 130, 246, 0.15)"
    ))

    # Forecast
    fig_timeline.add_trace(go.Scatter(
        x=df_forecast["date"],
        y=df_forecast["predicted_units"],
        mode="lines+markers",
        name="Forecasted Sales",
        line=dict(color="#a855f7", width=3, dash="dash"),
        marker=dict(size=7, color="#a855f7")
    ))

    # 95% Confidence Band
    fig_timeline.add_trace(go.Scatter(
        x=list(df_forecast["date"]) + list(df_forecast["date"])[::-1],
        y=list(df_forecast["confidence_high"]) + list(df_forecast["confidence_low"])[::-1],
        fill="toself",
        fillcolor="rgba(168, 85, 247, 0.12)",
        line=dict(color="rgba(255,255,255,0)"),
        hoverinfo="skip",
        showlegend=True,
        name="95% Confidence Band"
    ))

    fig_timeline.update_layout(
        template="plotly_dark",
        margin=dict(l=20, r=20, t=30, b=20),
        height=360,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig_timeline, use_container_width=True)

with col_mix:
    st.subheader("Vehicle Segment Allocation")
    df_seg = pd.DataFrame(segment_mix)
    fig_pie = px.pie(
        df_seg,
        names="name",
        values="value",
        color="name",
        color_discrete_map={row["name"]: row["color"] for row in segment_mix},
        hole=0.55
    )
    fig_pie.update_layout(
        template="plotly_dark",
        margin=dict(l=10, r=10, t=10, b=10),
        height=360,
        showlegend=True,
        legend=dict(orientation="h", yanchor="top", y=-0.1)
    )
    st.plotly_chart(fig_pie, use_container_width=True)

# Funnel & Day-by-Day Forecast Schedule
col_funnel, col_schedule = st.columns([1, 2])

with col_funnel:
    st.subheader("Dealership Conversion Funnel")
    funnel_steps = ["Total Leads", "Test Drives", "Credit Desk Submissions", "Final Deliveries"]
    funnel_vals = [
        kpis["total_leads"],
        kpis["total_test_drives"],
        int(round(kpis["total_units"] * 1.35)),
        kpis["total_units"]
    ]
    fig_funnel = go.Figure(go.Funnel(
        y=funnel_steps,
        x=funnel_vals,
        textinfo="value+percent previous",
        marker=dict(color=["#3b82f6", "#6366f1", "#8b5cf6", "#10b981"])
    ))
    fig_funnel.update_layout(
        template="plotly_dark",
        margin=dict(l=10, r=10, t=20, b=10),
        height=320
    )
    st.plotly_chart(fig_funnel, use_container_width=True)

with col_schedule:
    st.subheader("Upcoming Forecast Schedule")
    display_df = df_forecast[["date", "day", "projected_leads", "projected_test_drives", "predicted_units", "projected_revenue"]].copy()
    display_df["projected_revenue"] = display_df["projected_revenue"].apply(lambda v: f"${v:,}")
    display_df.columns = ["Date", "Day", "Exp. Leads", "Test Drives", "Predicted Units", "Est. Revenue"]
    st.dataframe(display_df, use_container_width=True, hide_index=True)