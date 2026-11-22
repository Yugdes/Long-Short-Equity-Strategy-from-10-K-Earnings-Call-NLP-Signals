import numpy as np
import pandas as pd
from src.data.prices import monthly_from_daily, split_adjust_shares


def _daily():
    dates = pd.bdate_range("2020-01-01", "2020-03-31")
    raw = np.full(len(dates), 100.0)
    raw[dates >= "2020-02-14"] = 50.0             # 2:1 split ex 2020-02-14, no economic change
    split = np.where(dates == pd.Timestamp("2020-02-14"), 2.0, 0.0)
    close = raw / 2.0 * np.where(dates >= "2020-02-14", 2.0, 1.0)   # Yahoo: adjusted to today's basis
    return pd.DataFrame({"date": dates, "ticker": "X", "adj": close, "close": close,
                         "volume": 1000.0, "div": 0.0, "split": split}), dates


def test_mcap_invariant_to_split():
    daily, _ = _daily()
    m = monthly_from_daily(daily)
    splits = daily[daily["split"] > 0][["ticker", "date", "split"]].rename(columns={"split": "ratio"})
    # 1,000 raw shares reported as of Jan 31 (pre-split), 2,000 as of Mar 31 (post-split)
    sh = split_adjust_shares(pd.Series([1000.0, 2000.0]), pd.Series(pd.to_datetime(["2020-01-31", "2020-03-31"])),
                             pd.Series(["X", "X"]), splits)
    assert sh.tolist() == [2000.0, 2000.0]
    mcap = m["prc"].values * 2000.0
    assert np.allclose(mcap, 100_000.0)          # raw 100 * 1000 = 50 * 2000 throughout


def test_monthly_return_compounds_daily():
    dates = pd.bdate_range("2021-01-01", "2021-02-26")
    rng = np.random.default_rng(0)
    r = rng.normal(0, 0.01, len(dates))
    adj = 10 * np.cumprod(1 + r)
    d = pd.DataFrame({"date": dates, "ticker": "Y", "adj": adj, "close": adj, "volume": 1.0, "div": 0.0, "split": 0.0})
    m = monthly_from_daily(d)
    feb = dates[dates.month == 2]
    expected = adj[dates == feb[-1]][0] / adj[dates == dates[dates.month == 1][-1]][0] - 1
    assert np.isclose(m.loc[m["month"] == pd.Period("2021-02", "M"), "ret"].iloc[0], expected)


def test_missing_month_breaks_return():
    dates = pd.to_datetime(["2021-01-29", "2021-03-31"])
    d = pd.DataFrame({"date": dates, "ticker": "Z", "adj": [10.0, 11.0], "close": [10.0, 11.0],
                      "volume": 1.0, "div": 0.0, "split": 0.0})
    assert monthly_from_daily(d)["ret"].isna().all()


def test_clean_returns_flags_errors_not_crashes():
    from src.data.prices import clean_returns
    m = pd.DataFrame({"ticker": "A", "month": pd.period_range("2020-01", periods=6, freq="M"),
                      "ret": [-1.5, -0.9, 4.0, -0.8, 12.0, 0.5]})
    out, st = clean_returns(m)
    assert out["ret"].isna().tolist() == [True, False, True, True, True, False]   # -90% crash kept
    assert st["total_blanked"] == 4
