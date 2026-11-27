"""Streamlit demo for the demand forecast API.

Hosted:  Streamlit Community Cloud, calling the live API on AWS Lambda.
Locally: streamlit run app/streamlit_app.py
         (set API_URL=http://127.0.0.1:8010 to use a local API instead of AWS)
"""
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # so "src" can be imported

import altair as alt
import pandas as pd
import requests
import streamlit as st

from src.demo.schedule import default_schedule, forecasts_to_frame, schedule_to_payload

DATE_AXIS = alt.Axis(format="%a %d %b", labelAngle=-45)   # e.g. "Sat 01 Aug" for every tick
LIVE_API = "https://hgdnakicif.execute-api.ap-southeast-2.amazonaws.com"
REPO = "https://github.com/ameerykb/Retail-demand-forecasting"


def default_api_url() -> str:
    try:
        if "API_URL" in st.secrets:
            return st.secrets["API_URL"]
    except Exception:          # no secrets file locally
        pass
    return os.getenv("API_URL", LIVE_API)


def call_api(method: str, url: str, retries: int = 3, **kwargs) -> requests.Response:
    """Lambda sleeps when idle; the first call can take ~20 s or time out once. Retry."""
    for attempt in range(retries):
        try:
            r = requests.request(method, url, timeout=45, **kwargs)
            if r.status_code not in (502, 503, 504):
                return r
        except requests.RequestException:
            if attempt == retries - 1:
                raise
        time.sleep(2)
    return r


@st.cache_data(ttl=300, show_spinner=False)
def get_info(url: str) -> dict:
    r = call_api("GET", f"{url}/model-info")
    r.raise_for_status()
    return r.json()


st.set_page_config(page_title="Retail demand forecast", page_icon="📈", layout="wide")
st.title("📈 Retail demand forecast")
st.markdown(
    "Pick a Rossmann store, plan its promotions and closed days, and get a **daily sales "
    "forecast with an 80% range**. The model (LightGBM) runs as an API on **AWS Lambda**. "
    f"[Source code]({REPO}) · [API docs]({LIVE_API}/docs)")
st.caption("Trained on Kaggle's historical Rossmann data (2013 to July 2015), so forecasts "
           "start in August 2015. A portfolio demo, not a production system.")

with st.sidebar:
    st.header("Model")
    with st.expander("Advanced: API address"):
        api_url = st.text_input("API address", default_api_url()).rstrip("/")

try:
    with st.spinner("Waking up the model on AWS (the first request can take ~20 seconds)..."):
        info = get_info(api_url)
except Exception as e:
    st.error(f"Could not reach the API at {api_url}. Please refresh in a few seconds.")
    st.caption(str(e))
    st.stop()

last_date = pd.Timestamp(info["last_history_date"]).date()
horizon = info["horizon_days"]
stores = info.get("store_ids") or list(range(1, info["n_stores"] + 1))
with st.sidebar:
    st.markdown(
        f"**Version:** `{info['model_version']}`  \n"
        f"**Trained on:** {info['trained_on']}  \n"
        f"**History ends:** {last_date}  \n"
        f"**Forecast window:** {info['forecast_window_start']} to {info['forecast_window_end']}  \n"
        f"**Stores:** {len(stores):,}")
    if info.get("backtest_wape") is not None:
        st.metric("Backtest error (WAPE)", f"{info['backtest_wape']:.1%}",
                  help="Weighted absolute percentage error in a 4-fold rolling backtest, "
                       "42 days ahead. Lower is better.")

col1, col2 = st.columns(2)
store = col1.selectbox("Store", stores, index=0)
n_days = col2.slider("Days to forecast", 1, horizon, 14)

st.subheader("Schedule")
st.caption("Tick **promo** on days a promotion runs; untick **is_open** for days the store is closed.")
sched = st.data_editor(default_schedule(last_date, n_days), hide_index=True,
                       disabled=["date"], key=f"schedule_{last_date}_{n_days}", width="stretch")
compare = st.checkbox("Compare with the same days without any promotions")

if st.button("Forecast", type="primary"):
    def forecast(no_promo: bool) -> pd.DataFrame:
        r = call_api("POST", f"{api_url}/forecast",
                     json=schedule_to_payload(store, sched, no_promo))
        if r.status_code != 200:
            st.error(f"API error {r.status_code}: {r.json().get('detail', r.text)}")
            st.stop()
        return forecasts_to_frame(r.json())

    with st.spinner("Forecasting..."):
        main = forecast(False)
        base = forecast(True) if compare else None

    total, open_days = main["predicted_sales"].sum(), int((main["predicted_sales"] > 0).sum())
    c1, c2, c3 = st.columns(3)
    c1.metric("Total predicted sales", f"{total:,.0f}")
    c2.metric("Open days", open_days)
    c3.metric("Average per open day", f"{total / max(open_days, 1):,.0f}")

    has_range = "lower" in main and main["lower"].notna().all()
    layers = []
    if has_range:
        layers.append(alt.Chart(main).mark_area(opacity=0.25, color="#2563eb").encode(
            x=alt.X("date:T", title=None, axis=DATE_AXIS), y=alt.Y("lower:Q", title="Sales"),
            y2="upper:Q",
            tooltip=[alt.Tooltip("date:T"), alt.Tooltip("lower:Q", format=",.0f"),
                     alt.Tooltip("upper:Q", format=",.0f")]))
    lines = main.assign(series="Your schedule")[["date", "predicted_sales", "series"]]
    if base is not None:
        lines = pd.concat([lines, base.assign(series="No promotions")[
            ["date", "predicted_sales", "series"]]])
    layers.append(alt.Chart(lines).mark_line(point=True).encode(
        x=alt.X("date:T", title=None, axis=DATE_AXIS), y=alt.Y("predicted_sales:Q", title="Sales"),
        # explicit domain: "Your schedule" is always blue like its range band
        color=alt.Color("series:N", title=None,
                        scale=alt.Scale(domain=["Your schedule", "No promotions"],
                                        range=["#2563eb", "#9ca3af"])),
        tooltip=[alt.Tooltip("date:T"), "series:N",
                 alt.Tooltip("predicted_sales:Q", format=",.0f")]))
    st.altair_chart(alt.layer(*layers).properties(height=360), width="stretch")
    if has_range:
        st.caption("Shaded band: 80% prediction range (calibrated so that about 8 in 10 real "
                   "values fall inside it in the backtest).")

    if base is not None and base["predicted_sales"].sum() > 0:
        lift = total / base["predicted_sales"].sum() - 1
        st.metric("Model's estimate of the promotions' effect on total sales", f"{lift:+.1%}")
        st.caption("A model prediction, not a proven causal effect of promotions.")

    table = main.assign(date=main["date"].dt.date)
    if base is not None:
        table["no_promotions"] = base["predicted_sales"].to_numpy()
    st.dataframe(table, hide_index=True, width="stretch")
