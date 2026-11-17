"""Streamlit demo for the demand forecast API.

    1. start the API:   uvicorn src.api.main:app --port 8010      (or: docker compose up)
    2. start the demo:  streamlit run app/streamlit_app.py
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # so "src" can be imported

import pandas as pd  # noqa: E402
import requests  # noqa: E402
import streamlit as st  # noqa: E402

from src.demo.schedule import default_schedule, forecasts_to_frame, schedule_to_payload  # noqa: E402

st.set_page_config(page_title="Store demand forecast", layout="wide")
st.title("Store demand forecast")
st.caption("Daily sales forecast for a Rossmann store, served by the FastAPI model. "
           "Historical Kaggle data (2013-2015), so this is a portfolio demo and not a live system.")

api_url = st.sidebar.text_input("API address", os.getenv("API_URL", "http://127.0.0.1:8010"))


@st.cache_data(ttl=30)
def get_info(url: str) -> dict:
    r = requests.get(f"{url}/model-info", timeout=10)
    r.raise_for_status()
    return r.json()


try:
    info = get_info(api_url)
except Exception as e:
    st.error(f"Could not reach the API at {api_url}.")
    st.info("Start it with `uvicorn src.api.main:app --port 8010` or `docker compose up`, "
            "then reload this page.")
    st.caption(str(e))
    st.stop()

last_date = pd.Timestamp(info["last_history_date"]).date()
horizon = info["horizon_days"]
st.sidebar.markdown(
    f"**Model:** {info['model_version']}  \n"
    f"**History ends:** {last_date}  \n"
    f"**Can forecast:** {info['forecast_window_start']} to {info['forecast_window_end']}  \n"
    f"**Stores:** {info['n_stores']}")
if info.get("backtest_wape") is not None:
    st.sidebar.markdown(f"**Backtest WAPE:** {info['backtest_wape']}")

col1, col2 = st.columns(2)
store = col1.number_input("Store id", min_value=1, max_value=99999, value=1, step=1)
n_days = col2.slider("Days to forecast", 1, horizon, 14)

st.subheader("Schedule")
st.caption("Edit the schedule: tick Promo on days a promotion will run, "
           "untick Is open for days the store is closed.")
sched = st.data_editor(default_schedule(last_date, n_days), hide_index=True,
                       disabled=["date"], key=f"schedule_{last_date}_{n_days}",
                       width="stretch")
compare = st.checkbox("Also show the forecast with no promotions at all")

if st.button("Forecast", type="primary"):
    def call(force_no_promo: bool):
        payload = schedule_to_payload(store, sched, force_no_promo)
        r = requests.post(f"{api_url}/forecast", json=payload, timeout=60)
        if r.status_code != 200:
            st.error(f"API error {r.status_code}: {r.json().get('detail', r.text)}")
            st.stop()
        return forecasts_to_frame(r.json())

    main = call(False).rename(columns={"predicted_sales": "Your schedule"})
    chart = main.set_index("date")
    total, open_days = main["Your schedule"].sum(), int((main["Your schedule"] > 0).sum())

    c1, c2, c3 = st.columns(3)
    c1.metric("Total predicted sales", f"{total:,.0f}")
    c2.metric("Open days", open_days)
    c3.metric("Average per open day", f"{total / max(open_days, 1):,.0f}")

    if compare:
        base = call(True).rename(columns={"predicted_sales": "No promotions"}).set_index("date")
        chart = chart.join(base)
        base_total = chart["No promotions"].sum()
        if base_total > 0:
            st.metric("Model's estimate of the promo effect on total sales",
                      f"{(total / base_total - 1) * 100:+.1f}%")
        st.caption("This is what the model predicts, not a proven causal effect of promotions.")

    if compare:
        st.line_chart(chart)        # two lines: easy to compare, nothing stacked
    else:
        st.bar_chart(chart["Your schedule"])
    st.dataframe(chart.reset_index().assign(date=lambda d: d["date"].dt.date),
                 hide_index=True, width="stretch")