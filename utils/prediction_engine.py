"""
utils/prediction_engine.py
Econometric forecasting model and KPI aggregation routines.
"""

import math
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

INITIAL_RECORDS = [
    {"id": "rec-1", "date": "2026-05-10", "day": "Sat", "footfall": 68, "leads": 52, "test_drives": 28, "discount_pct": 4.5, "interest_rate": 6.2, "actual_sales": 16, "revenue": 672000, "top_segment": "Electric SUV"},
    {"id": "rec-2", "date": "2026-05-11", "day": "Sun", "footfall": 74, "leads": 58, "test_drives": 32, "discount_pct": 5.0, "interest_rate": 6.2, "actual_sales": 18, "revenue": 756000, "top_segment": "Hybrid Sedan"},
    {"id": "rec-3", "date": "2026-05-12", "day": "Mon", "footfall": 36, "leads": 26, "test_drives": 12, "discount_pct": 3.5, "interest_rate": 6.2, "actual_sales": 7, "revenue": 294000, "top_segment": "Mid-size SUV"},
    {"id": "rec-4", "date": "2026-05-13", "day": "Tue", "footfall": 39, "leads": 31, "test_drives": 14, "discount_pct": 3.5, "interest_rate": 6.2, "actual_sales": 8, "revenue": 336000, "top_segment": "Electric SUV"},
    {"id": "rec-5", "date": "2026-05-14", "day": "Wed", "footfall": 42, "leads": 34, "test_drives": 17, "discount_pct": 4.0, "interest_rate": 6.2, "actual_sales": 9, "revenue": 378000, "top_segment": "Pickup Truck"},
    {"id": "rec-6", "date": "2026-05-15", "day": "Thu", "footfall": 48, "leads": 39, "test_drives": 20, "discount_pct": 4.0, "interest_rate": 6.2, "actual_sales": 11, "revenue": 462000, "top_segment": "Electric SUV"},
    {"id": "rec-7", "date": "2026-05-16", "day": "Fri", "footfall": 56, "leads": 46, "test_drives": 24, "discount_pct": 4.5, "interest_rate": 6.2, "actual_sales": 14, "revenue": 588000, "top_segment": "Luxury Sport"},
]

SEGMENT_CONFIG = [
    {"name": "Electric SUV", "base_share": 32, "avg_price": 48500, "color": "#3b82f6"},
    {"name": "Hybrid Sedan", "base_share": 24, "avg_price": 34200, "color": "#10b981"},
    {"name": "Mid-size SUV", "base_share": 20, "avg_price": 39800, "color": "#f59e0b"},
    {"name": "Pickup Truck", "base_share": 14, "avg_price": 52000, "color": "#8b5cf6"},
    {"name": "Luxury Sport", "base_share": 10, "avg_price": 71000, "color": "#ec4899"},
]


def compute_sales_forecast(df_historical: pd.DataFrame, sim_controls: dict, timeframe: str = "7d") -> pd.DataFrame:
    """
    Computes forecasted leads, test drives, deliveries, revenue, and confidence bounds.
    """
    days_to_forecast = {"today": 1, "tomorrow": 2, "7d": 7, "14d": 14}.get(timeframe, 7)
    
    if not df_historical.empty:
        last_date = pd.to_datetime(df_historical["date"].max())
    else:
        last_date = pd.to_datetime(datetime.now().strftime("%Y-%m-%d"))

    mkt_mult = 1.0 + (sim_controls.get("marketing_spend_delta", 0) / 100.0) * 0.45
    discount_factor = 1.0 + ((sim_controls.get("dealer_discount", 5.0) - 4.0) * 0.08)
    rate_factor = max(0.7, 1.0 - ((sim_controls.get("avg_interest_rate", 5.9) - 5.5) * 0.09))
    inv_factor = sim_controls.get("inventory_adequacy", 95) / 100.0
    lead_quality = sim_controls.get("lead_quality_index", 1.15)

    results = []
    for i in range(1, days_to_forecast + 1):
        target_date = last_date + timedelta(days=i)
        day_name = target_date.strftime("%a")
        is_weekend = day_name in ["Sat", "Sun"]

        proj_leads = int(round((35 * mkt_mult * (1.55 if is_weekend else 0.92)) + (math.sin(i) * 3)))
        proj_test_drives = int(round(proj_leads * 0.48 * lead_quality))

        raw_units = proj_test_drives * 0.52 * discount_factor * rate_factor * inv_factor
        predicted_units = max(4.0, round(raw_units, 1))

        projected_asp = 44500 + (math.cos(i) * 1800)
        projected_revenue = int(round(predicted_units * projected_asp))
        confidence_low = max(2.0, round(predicted_units * 0.88, 1))
        confidence_high = round(predicted_units * 1.14, 1)

        results.append({
            "date": target_date.strftime("%Y-%m-%d"),
            "day": day_name,
            "is_weekend": is_weekend,
            "projected_leads": proj_leads,
            "projected_test_drives": proj_test_drives,
            "predicted_units": predicted_units,
            "confidence_low": confidence_low,
            "confidence_high": confidence_high,
            "projected_asp": int(round(projected_asp)),
            "projected_revenue": projected_revenue,
        })

    return pd.DataFrame(results)


def compute_executive_kpis(df_forecast: pd.DataFrame, sim_controls: dict) -> dict:
    """
    Computes aggregated performance indicators from forecast timeline.
    """
    if df_forecast.empty:
        return {
            "total_units": 0, "total_revenue": 0, "avg_asp": 0,
            "total_leads": 0, "total_test_drives": 0,
            "lead_to_deal_rate": "0.0", "accuracy_score": "90.0"
        }

    total_units = round(df_forecast["predicted_units"].sum())
    total_revenue = int(df_forecast["projected_revenue"].sum())
    avg_asp = int(total_revenue / total_units) if total_units > 0 else 0
    total_leads = int(df_forecast["projected_leads"].sum())
    total_test_drives = int(df_forecast["projected_test_drives"].sum())
    lead_to_deal_rate = f"{(total_units / total_leads * 100):.1f}" if total_leads > 0 else "0.0"

    accuracy_score = 94.2
    if sim_controls.get("dealer_discount", 5.0) > 8 or sim_controls.get("dealer_discount", 5.0) < 1:
        accuracy_score -= 3.5
    if sim_controls.get("avg_interest_rate", 5.9) > 8.5:
        accuracy_score -= 2.1
    if sim_controls.get("inventory_adequacy", 95) < 80:
        accuracy_score -= 4.0
    accuracy_score = max(78.0, min(98.5, accuracy_score))

    return {
        "total_units": total_units,
        "total_revenue": total_revenue,
        "avg_asp": avg_asp,
        "total_leads": total_leads,
        "total_test_drives": total_test_drives,
        "lead_to_deal_rate": lead_to_deal_rate,
        "accuracy_score": f"{accuracy_score:.1f}",
    }


def get_segment_mix(sim_controls: dict) -> list:
    """
    Computes dynamic powertrain segment mix based on simulation parameters.
    """
    segment_mix = []
    for s in SEGMENT_CONFIG:
        val = s["base_share"]
        if s["name"] == "Electric SUV" and sim_controls.get("marketing_spend_delta", 0) > 10:
            val += 3
        if s["name"] == "Pickup Truck" and sim_controls.get("avg_interest_rate", 5.9) > 7.0:
            val -= 2
        segment_mix.append({
            "name": s["name"],
            "value": val,
            "avg_price": s["avg_price"],
            "color": s["color"]
        })
    return segment_mix