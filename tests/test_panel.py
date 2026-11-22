"""Merge and look-ahead tests. Synthetic unit tests always run; invariants on the real panel run
when data/processed/panel.parquet exists."""
import numpy as np
import pandas as pd
import pytest

from src.features.characteristics import asof_join, latest_facts
from src.features.timing import risk_first_month
from src.utils.config import path, load_config

P = lambda s: pd.Period(s, "M")


def test_asof_join_never_uses_future_release():
    left = pd.DataFrame({"cik": [1, 1, 1, 2], "month": pd.PeriodIndex(["2019-02", "2019-03", "2019-05", "2019-03"], freq="M")})
    right = pd.DataFrame({"cik": [1, 1, 2], "first": pd.PeriodIndex(["2019-03", "2019-05", "2019-04"], freq="M"), "v": [10, 20, 30]})
    out = asof_join(left, right, "first", ["v"])
    assert np.isnan(out["v"].iloc[0])          # Feb: nothing released yet
    assert out["v"].iloc[1] == 10              # Mar: first release
    assert out["v"].iloc[2] == 20              # May: second release
    assert np.isnan(out["v"].iloc[3])          # other firm, released only in April


def test_latest_facts_amendment_does_not_override_newer_period():
    f = pd.DataFrame({"cik": [1, 1, 1],
                      "end": pd.to_datetime(["2019-12-31", "2020-12-31", "2019-12-31"]),
                      "filed": pd.to_datetime(["2020-02-20", "2021-02-20", "2021-06-01"]),
                      "x": [1.0, 2.0, 1.5]})
    out = latest_facts(f, ["x"], lag_months=1)
    assert out["x"].tolist() == [1.0, 2.0]       # the June-2021 restatement of 2019 is ignored
    assert [str(u) for u in out["usable"]] == ["2020-03", "2021-03"]


# ---------------- invariants on the real panel ----------------
@pytest.fixture(scope="module")
def panel():
    f = path("processed") / "panel.parquet"
    if not f.exists():
        pytest.skip("panel not built")
    return pd.read_parquet(f)


def test_no_signal_before_release(panel):
    lag = load_config()["timing"]["risk_lag_bdays"]
    p = panel[panel["filing_date"].notna()]
    first = risk_first_month(p["filing_date"], lag)
    assert (pd.PeriodIndex(first).asi8 <= pd.PeriodIndex(p["month"]).asi8).all()
    avail = p["filing_date"] + pd.offsets.BDay(lag)
    assert (avail < p["month"].dt.to_timestamp()).all()


def test_staleness_cap(panel):
    p = panel[panel["filing_month"].notna()]
    age = pd.PeriodIndex(p["month"]).asi8 - pd.PeriodIndex(p["filing_month"]).asi8
    assert age.max() <= load_config()["timing"]["risk_max_age_months"]
    assert age.min() >= 1


def test_marketing_not_before_first_month(panel):
    p = panel[panel["mkt_first_month"].notna()]
    age = pd.PeriodIndex(p["month"]).asi8 - pd.PeriodIndex(p["mkt_first_month"]).asi8
    assert age.min() >= 0 and age.max() < load_config()["timing"]["mkt_max_age_months"]


def test_controls_lagged(panel):
    # reversal control must equal the previous month's return of the same firm, never ret_t
    p = panel.sort_values(["cik", "month"])
    nxt = p.groupby("cik")["month"].shift(-1)
    consec = (pd.PeriodIndex(nxt).asi8 - pd.PeriodIndex(p["month"]).asi8) == 1
    lead_rev = p.groupby("cik")["rev_1"].shift(-1)
    ok = consec & lead_rev.notna() & p["ret"].notna()
    assert np.allclose(lead_rev[ok], p["ret"][ok], atol=1e-5)


def test_unique_keys_and_universe(panel):
    assert not panel.duplicated(["cik", "month"]).any()
    e = panel[panel["elig"]]
    assert (e["prc_l"] >= 5).all() and (e["mcap_l"] >= 100).all()
    assert not e["fin_util"].astype(bool).any()
    assert e["month"].min() >= P(load_config()["sample"]["primary_start"])
