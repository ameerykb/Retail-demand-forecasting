"""Small helpers for the Streamlit demo (no Streamlit code here, so they are easy to test)."""
import datetime as dt
import pandas as pd


def default_schedule(last_date: dt.date, n_days: int) -> pd.DataFrame:
    """Next n_days after last_date: open Monday-Saturday, closed Sunday, no promo, no holidays."""
    rows = []
    for i in range(n_days):
        d = last_date + dt.timedelta(days=i + 1)
        rows.append({"date": d, "is_open": d.weekday() != 6, "promo": False,
                     "state_holiday": False, "school_holiday": False})
    return pd.DataFrame(rows)


def schedule_to_payload(store: int, sched: pd.DataFrame, force_no_promo: bool = False) -> dict:
    """Turn the (possibly edited) schedule table into the JSON the API expects."""
    days = [{"date": pd.Timestamp(r.date).date().isoformat(),
             "is_open": int(r.is_open),
             "promo": 0 if force_no_promo else int(r.promo),
             "state_holiday": int(r.state_holiday),
             "school_holiday": int(r.school_holiday)}
            for r in sched.itertuples()]
    return {"store": int(store), "days": days}


def forecasts_to_frame(response: dict) -> pd.DataFrame:
    df = pd.DataFrame(response["forecasts"])
    df["date"] = pd.to_datetime(df["date"])
    return df
