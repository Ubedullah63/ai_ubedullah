"""
pages/5_💬_Dealership_AI_Chatbot.py
Interactive AI Automotive Sales & Inventory Advisor Chatbot.
Supports LangChain (LCEL) with streaming tokens and grounded showroom telemetry.
"""

import json
import os
import streamlit as st

from utils.prediction_engine import (
    compute_sales_forecast,
    compute_executive_kpis,
    get_segment_mix,
)
from utils.langchain_service import stream_langchain_chat

# Ensure state is initialized
if "historical_df" not in st.session_state:
    st.switch_page("app.py")

st.markdown("## 💬 AutoPredict Dealership AI Chatbot")
st.caption("Conversational intelligence powered by **LangChain** and grounded in your live showroom telemetry.")

# Compute current state context
df_forecast = compute_sales_forecast(
    st.session_state.historical_df,
    st.session_state.sim_controls,
    st.session_state.timeframe,
)
kpis = compute_executive_kpis(df_forecast, st.session_state.sim_controls)
segment_mix = get_segment_mix(st.session_state.sim_controls)

# Construct telemetry grounding context
dealership_context = {
    "forecastHorizon": st.session_state.timeframe,
    "totalUnitsPredicted": kpis["total_units"],
    "projectedRevenue": f"${(kpis['total_revenue'] / 1e6):.2f}M",
    "avgSellingPrice": f"${kpis['avg_asp']:,}",
    "leadToDealRate": f"{kpis['lead_to_deal_rate']}%",
    "modelConfidence": f"{kpis['accuracy_score']}%",
    "totalLeads": kpis["total_leads"],
    "totalTestDrives": kpis["total_test_drives"],
    "operatingLevers": {
        "marketingDeltaPct": st.session_state.sim_controls.get("marketing_spend_delta", 0),
        "dealerDiscountPct": st.session_state.sim_controls.get("dealer_discount", 5.0),
        "customerAPR": st.session_state.sim_controls.get("avg_interest_rate", 5.9),
        "inventoryAdequacy": st.session_state.sim_controls.get("inventory_adequacy", 95),
        "leadQualityIndex": st.session_state.sim_controls.get("lead_quality_index", 1.15),
    },
    "segments": [f"{s['name']}: {s['value']}%" for s in segment_mix],
    "historicalRecordsCount": len(st.session_state.historical_df),
}

# Chat session state initialization
if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = [
        {
            "role": "assistant",
            "content": (
                "👋 Hello! I am your **AutoPredict Dealership Strategist** (running via **LangChain**). "
                f"I'm tracking your **{st.session_state.timeframe.upper()} forecast** "
                f"({kpis['total_units']} projected deliveries, ${kpis['total_revenue'] / 1e6:.2f}M revenue). "
                "Ask me about showroom pacing, financing sensitivity, discounting tactics, or weekend desk staffing!"
            ),
        }
    ]

# Sidebar actions for chatbot
with st.sidebar:
    st.markdown("---")
    st.markdown("### 💬 LangChain Chat Settings")
    use_langchain = st.toggle("Enable LangChain Engine", value=True, help="Routes chat queries through LangChain LCEL & ChatGoogleGenerativeAI.")
    if st.button("🧹 Clear Chat History", use_container_width=True):
        st.session_state.chat_messages = [
            {
                "role": "assistant",
                "content": "Conversation cleared. How can I help optimize your dealership operations today?",
            }
        ]
        st.rerun()

# Quick suggestion prompt chips
st.markdown("##### 💡 Suggested Questions")
q_col1, q_col2, q_col3 = st.columns(3)

preset_query = None
if q_col1.button("📉 How does APR affect this week's sales?", use_container_width=True):
    preset_query = "How is our customer financing APR affecting this week's predicted sales and test-drive conversion?"
if q_col2.button("🚗 Which segments should we stock more of?", use_container_width=True):
    preset_query = "Based on our current segment mix and projected demand, which vehicles should we prioritize with OEM allocations?"
if q_col3.button("🎯 Saturday closing strategy for GSM", use_container_width=True):
    preset_query = "Give me a concrete 3-step action plan for the General Sales Manager (GSM) and F&I desk to maximize Saturday contract closings."

# Display existing chat history
for msg in st.session_state.chat_messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


def offline_heuristic_reply(last_user_msg: str, telemetry_summary: dict) -> str:
    """Local rule-based generator when no key is set or offline."""
    user_txt = last_user_msg.lower()
    units = telemetry_summary["totalUnitsPredicted"]
    apr = telemetry_summary["operatingLevers"]["customerAPR"]
    discount = telemetry_summary["operatingLevers"]["dealerDiscountPct"]
    rev = telemetry_summary["projectedRevenue"]

    if "apr" in user_txt or "interest" in user_txt or "rate" in user_txt:
        return (
            f"### APR Sensitivity Analysis (Offline Mode)\n\n"
            f"Customer APR is set at **{apr}%**. Across our **{units} projected unit deliveries**, "
            f"every 50 bps increase in customer financing rates typically reduces showroom appointment holds by **6-8%**.\n\n"
            f"**Recommended Tactical Moves:**\n"
            f"- **Captive Finance Subvention:** Work with lenders on promotional 1.9%–3.9% APR for older inventory.\n"
            f"- **Payment Framing:** Train desk managers to negotiate bi-weekly payment options rather than total APR.\n"
            f"- **Lease Penetration:** Increase lease quotes where factory subvention cushions customer monthly payments."
        )
    elif "stock" in user_txt or "segment" in user_txt or "inventory" in user_txt:
        top_seg = telemetry_summary["segments"][0] if telemetry_summary["segments"] else "Electric SUV"
        return (
            f"### Inventory Allocation Directives (Offline Mode)\n\n"
            f"Projections highlight **{top_seg}** as your highest velocity category.\n\n"
            f"**Stocking Strategy:**\n"
            f"- **Maintain 45-day supply:** Prevent lot stock-outs on leading trims.\n"
            f"- **Limit Slow-Moving Units:** Keep units with over 60 lot days to a minimum to curtail floorplan interest expenses.\n"
            f"- **Service Drive Appraisals:** Use the service lane to acquire clean used trades and build inventory margin."
        )
    else:
        return (
            f"### Dealership Executive Advisory (Offline Mode)\n\n"
            f"For the **{telemetry_summary['forecastHorizon'].upper()} horizon**, you are pacing toward **{units} deliveries** "
            f"and **{rev} gross revenue** with an average discount of **{discount}%**.\n\n"
            f"**Key Priorities:**\n"
            f"1. **Enforce 100% TO Policy:** No customer leaves without a manager greeting.\n"
            f"2. **Backend Protection:** Maintain a strict **$1,500+ PVR** target in the F&I box to protect margins against front-end discounting.\n"
            f"3. **BDC Speed-to-Lead:** Respond to internet leads within 8 minutes for maximum show-rate."
        )


# Capture user prompt
user_query = st.chat_input("Ask about sales forecasts, inventory levers, pricing...") or preset_query

if user_query:
    # Append & display user prompt
    st.session_state.chat_messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    # Generate assistant response
    with st.chat_message("assistant"):
        api_key = st.session_state.api_key or os.environ.get("GEMINI_API_KEY", "")

        if api_key and use_langchain:
            try:
                # Stream directly using LangChain
                response_stream = stream_langchain_chat(
                    messages=st.session_state.chat_messages,
                    telemetry_summary=dealership_context,
                    api_key=api_key,
                )
                full_reply = st.write_stream(response_stream)
            except Exception as e:
                st.warning(f"LangChain notice: {str(e)}. Falling back to offline advisor.")
                full_reply = offline_heuristic_reply(user_query, dealership_context)
                st.markdown(full_reply)
        else:
            full_reply = offline_heuristic_reply(user_query, dealership_context)
            st.markdown(full_reply)

    # Save reply to history
    st.session_state.chat_messages.append({"role": "assistant", "content": full_reply})