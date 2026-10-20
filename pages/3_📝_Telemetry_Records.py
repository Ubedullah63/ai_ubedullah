"""
pages/3_📝_Telemetry_Records.py
Data ingestion form and interactive table for incoming daily dealership metrics.
"""

from datetime import datetime
import streamlit as st
import pandas as pd

from utils.prediction_engine import INITIAL_RECORDS, SEGMENT_CONFIG

if "historical_df" not in st.session_state:
    st.switch_page("app.py")

st.markdown("## 📝 Dealership Sales Telemetry Records")
st.caption("View, insert, and modify daily showroom traffic, test drives, and completed unit deliveries.")

# Ingestion Form
with st.expander("➕ Ingest Daily Telemetry Entry", expanded=False):
    with st.form("new_telemetry_form"):
        fc1, fc2, fc3, fc4 = st.columns(4)
        in_date = fc1.date_input("Date", datetime.now())
        in_footfall = fc2.number_input("Foot Traffic", min_value=1, value=50)
        in_leads = fc3.number_input("Digital Leads", min_value=1, value=35)
        in_test_drives = fc4.number_input("Test Drives", min_value=0, value=20)

        fc5, fc6, fc7, fc8 = st.columns(4)
        in_discount = fc5.number_input("Discount %", min_value=0.0, max_value=25.0, value=4.0, step=0.1)
        in_rate = fc6.number_input("Interest Rate %", min_value=1.0, max_value=15.0, value=6.0, step=0.1)
        in_sales = fc7.number_input("Completed Deliveries", min_value=0, value=12)
        in_segment = fc8.selectbox("Top Segment", [s["name"] for s in SEGMENT_CONFIG])

        submitted = st.form_submit_button("Log Telemetry Entry", type="primary")
        if submitted:
            new_row = {
                "id": f"rec-{int(datetime.now().timestamp())}",
                "date": in_date.strftime("%Y-%m-%d"),
                "day": in_date.strftime("%a"),
                "footfall": int(in_footfall),
                "leads": int(in_leads),
                "test_drives": int(in_test_drives),
                "discount_pct": float(in_discount),
                "interest_rate": float(in_rate),
                "actual_sales": int(in_sales),
                "revenue": int(in_sales * 42000),
                "top_segment": in_segment
            }
            st.session_state.historical_df = pd.concat([
                st.session_state.historical_df,
                pd.DataFrame([new_row])
            ], ignore_index=True)
            st.success("Telemetry record logged successfully!")
            st.rerun()

# Editable Table
st.markdown("#### Historical Records")
st.caption("You can edit cells directly inside the table below to recalibrate the model.")

edited_df = st.data_editor(
    st.session_state.historical_df,
    use_container_width=True,
    num_rows="dynamic",
    key="data_editor_historical"
)

if not edited_df.equals(st.session_state.historical_df):
    st.session_state.historical_df = edited_df
    st.toast("Model recalibrated with updated data!", icon="✅")

if st.button("Reset Telemetry to Standard Sample"):
    st.session_state.historical_df = pd.DataFrame(INITIAL_RECORDS)
    st.rerun()