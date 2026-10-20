"""
utils/gemini_service.py
Google Gemini 3 Flash client and heuristic offline fallback generator.
"""

import os
import json
import requests
import streamlit as st


def request_gemini_briefing(
    api_key: str,
    timeframe: str,
    kpis: dict,
    sim_controls: dict,
    segment_mix: list,
    tone: str,
    focus: str
) -> str:
    """
    Issues a structured prompt to Gemini 3 Flash API to generate executive retail guidance.
    Falls back gracefully if the API is unreachable or no key is provided.
    """
    if not api_key:
        api_key = os.environ.get("GEMINI_API_KEY", "")

    summary = {
        "forecastHorizon": timeframe,
        "totalUnitsExpected": kpis["total_units"],
        "projectedGrossRevenue": f"${(kpis['total_revenue'] / 1e6):.2f}M",
        "averageSellingPrice": f"${kpis['avg_asp']:,}",
        "leadToCloseRatio": f"{kpis['lead_to_deal_rate']}%",
        "modelConfidenceIndex": f"{kpis['accuracy_score']}%",
        "dealershipOperatingLevers": {
            "marketingSpendAdjustment": f"{sim_controls.get('marketing_spend_delta', 0)}%",
            "dealerDiscountPct": f"{sim_controls.get('dealer_discount', 5.0)}%",
            "customerAPR": f"{sim_controls.get('avg_interest_rate', 5.9)}%",
            "lotStockAdequacy": f"{sim_controls.get('inventory_adequacy', 95)}%"
        },
        "projectedVehicleSegments": ", ".join([f"{s['name']} ({s['value']}%)" for s in segment_mix])
    }

    system_prompt = (
        "You are the Chief Analytics Officer and Automotive Dealership Strategist. "
        "Provide razor-sharp C-suite retail intelligence, margin analyses, and operational battle plans."
    )

    user_prompt = f"""
Generate an incisive, highly structured C-Suite Automotive Sales Intelligence Brief:

DEALERSHIP TELEMETRY & MODEL STATE:
{json.dumps(summary, indent=2)}

REPORT TONE: {tone}
STRATEGIC DIRECTIVE: {focus}

STRUCTURE USING CLEAN MARKDOWN HEADERS:
# 1. EXECUTIVE DIGEST & PACE ASSESSMENT
# 2. FUNNEL EFFICIENCY & CONVERSION DYNAMICS
# 3. MACRO RISKS, APR SENSITIVITY & MARGIN HEALTH
# 4. VEHICLE SEGMENT MIX RECOMMENDATIONS
# 5. 7-DAY GSM & F&I OPERATIONAL BATTLE PLAN
"""

    if api_key:
        try:
            api_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-3-flash-preview:generateContent?key={api_key}"
            payload = {
                "contents": [{"parts": [{"text": user_prompt}]}],
                "systemInstruction": {"parts": [{"text": system_prompt}]}
            }
            res = requests.post(
                api_url,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=25
            )
            if res.status_code == 200:
                data = res.json()
                text = data["candidates"][0]["content"]["parts"][0]["text"]
                return text
        except Exception as e:
            st.warning(f"Gemini API request note: {str(e)}. Generating local executive fallback.")

    return build_offline_fallback_brief(summary, focus)


def build_offline_fallback_brief(summary: dict, focus: str) -> str:
    """
    Deterministic rule-based executive brief generator used when offline or without an API key.
    """
    return f"""# EXECUTIVE SALES INTELLIGENCE BRIEF
**Strategic Directive:** {focus}
**Target Period:** {summary['forecastHorizon'].upper()}
**Projected Deliveries:** {summary['totalUnitsExpected']} Units | **Gross Revenue:** {summary['projectedGrossRevenue']}
**Funnel Closing Rate:** {summary['leadToCloseRatio']} | **Model Confidence:** {summary['modelConfidenceIndex']}

---

### 1. EXECUTIVE DIGEST & PACE ASSESSMENT
Showroom pacing indicates delivery of **{summary['totalUnitsExpected']} vehicles** across the active horizon at an Average Selling Price (ASP) of **{summary['averageSellingPrice']}**. Weekend foot traffic represents approximately **58% of scheduled contract closings**, necessitating optimized staffing in business and finance offices.

### 2. FUNNEL EFFICIENCY & CONVERSION DYNAMICS
- **Demand Driver:** {summary['dealershipOperatingLevers']['marketingSpendAdjustment']} advertising delta sustains high-intent digital inquiries.
- **Showroom Test-Drive Conversion:** Holding strong above regional averages; customer willingness to commit is heavily influenced by discount transparency.

### 3. MACRO RISKS, APR SENSITIVITY & MARGIN HEALTH
- **Interest Rate Sensitivity:** Customer APR at **{summary['dealershipOperatingLevers']['customerAPR']}** creates resistance in subprime brackets. Recommend subvented dealer APR buy-downs.
- **Discount Protection:** Dealer discount level at **{summary['dealershipOperatingLevers']['dealerDiscountPct']}** requires strict backend F&I penetration ($1,400+ PVR).

### 4. VEHICLE SEGMENT MIX RECOMMENDATIONS
- **Top In-Demand:** Prioritize inventory replenishment on {summary['projectedVehicleSegments'].split(',')[0]}.
- **Turnover Speed:** Transition aging inventory into promotional lease terms to mitigate floorplan interest penalties.

### 5. 7-DAY OPERATIONAL BATTLE PLAN
- **General Sales Manager (GSM):** Stagger desk shifts for Saturday volume surges; enforce 100% turnover (TO) protocol on all walkouts.
- **F&I Director:** Pre-screen online applications via soft credit pulls before customer arrival.
- **Digital Marketing Lead:** Reallocate 20% of display campaign budgets toward in-market search queries.
"""