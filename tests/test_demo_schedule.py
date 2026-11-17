import datetime as dt
import pandas as pd
from src.api.schemas import ForecastRequest
from src.demo.schedule import default_schedule, forecasts_to_frame, schedule_to_payload

LAST = dt.date(2015, 7, 31)  # a Friday


def test_default_schedule_shape_and_sundays():
    s = default_schedule(LAST, 7)                  # 1 Aug (Sat) to 7 Aug (Fri): exactly one Sunday
    assert len(s) == 7 and s["date"].iloc[0] == dt.date(2015, 8, 1)
    sundays = s[[d.weekday() == 6 for d in s["date"]]]
    assert len(sundays) == 1 and not sundays["is_open"].any()
    assert s[~s["is_open"]].shape[0] == 1          # only Sundays are closed
    assert not s["promo"].any()


def test_payload_is_valid_for_the_api_schema():
    s = default_schedule(LAST, 14)
    s.loc[2, "promo"] = True
    payload = schedule_to_payload(5, s)
    req = ForecastRequest(**payload)               # raises if the shape is wrong
    assert req.store == 5 and len(req.days) == 14 and req.days[2].promo == 1


def test_force_no_promo_removes_all_promos():
    s = default_schedule(LAST, 7)
    s["promo"] = True
    payload = schedule_to_payload(1, s, force_no_promo=True)
    assert all(d["promo"] == 0 for d in payload["days"])


def test_forecasts_to_frame():
    df = forecasts_to_frame({"forecasts": [{"date": "2015-08-01", "predicted_sales": 10.5}]})
    assert list(df.columns) == ["date", "predicted_sales"]
    assert df["date"].iloc[0] == pd.Timestamp("2015-08-01")
