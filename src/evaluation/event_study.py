"""Filing-date event study (D-033): how does the market react around a 10-K, by its risk language?

Events: non-empty 10-K risk sections of firms in the primary universe (eligible in the filing month), 2010-2024.
Day 0 is the first trading day on or after the filing date. Abnormal return = stock return - market return
(FF daily Mkt-RF + RF). Outcomes:
  car01      CAR over days [0, +1] (the announcement reaction)
  abs_car01  |CAR[0, +1]| (how much information the filing moves)
  car2_60    CAR over days [+2, +60] (post-filing drift)
  dlogvol    log(std of AR over [+2, +61] / std of AR over [-250, -21]) (change in idiosyncratic-ish volatility)
Signals are the panel's centred cross-sectional ranks from the first month the filing is usable (point in time);
controls are the panel's control ranks measured before the filing month. Estimates are Fama-MacBeth across
filing years. This is a descriptive test of what the text tells investors, not a trading rule.
"""
import numpy as np
import pandas as pd

from src.data.factors import load_ff_daily
from src.data.risk import FACTORS
from src.features.signals import FEATURE_GROUPS
from src.features.timing import risk_first_month
from src.utils.config import load_config, path

SIGNALS = ["r_risk_all", "r_operations_prob", "r_nonop_prob", "r_d_risk_all", "r_d_operations_prob"] + \
          [f"r_{f}_prob" for f in FACTORS if f != "operations"]
OUTCOMES = ["car01", "abs_car01", "car2_60", "dlogvol"]


def daily_abnormal(tickers) -> pd.DataFrame:
    """[ticker, date, ar] from the frozen Yahoo daily bars; returns across gaps > 7 calendar days are dropped."""
    parts = []
    for f in sorted((path("raw") / "yahoo").glob("batch_*.parquet")):
        d = pd.read_parquet(f, columns=["date", "ticker", "adj"])
        parts.append(d[d["ticker"].isin(tickers)])
    d = pd.concat(parts).dropna().sort_values(["ticker", "date"])
    d["ret"] = d.groupby("ticker")["adj"].pct_change()
    gap = d.groupby("ticker")["date"].diff().dt.days
    d.loc[gap > 7, "ret"] = np.nan
    ff = load_ff_daily()
    d = d.merge(ff[["date", "mktrf", "rf"]], on="date", how="inner")
    d["ar"] = d["ret"] - (d["mktrf"] + d["rf"])
    return d[["ticker", "date", "ar"]]


def event_windows(ar: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    """events [eid, ticker, filing_date] -> outcomes per eid. Windows must be >= 90% complete."""
    out = []
    ev = dict(tuple(events.groupby("ticker")))
    for tk, a in ar.groupby("ticker"):
        if tk not in ev:
            continue
        g = ev[tk]
        dates, r = a["date"].values, a["ar"].values
        for eid, fd in zip(g["eid"], g["filing_date"].values):
            i0 = np.searchsorted(dates, fd)            # first trading day on/after the filing date
            if i0 < 250 or i0 + 62 > len(dates):
                continue

            def win(lo, hi, need):
                w = r[i0 + lo: i0 + hi + 1]
                return w if np.isfinite(w).sum() >= need * len(w) else None
            w01, w260, post, pre = win(0, 1, 1.0), win(2, 60, 0.9), win(2, 61, 0.9), win(-250, -21, 0.9)
            if w01 is None or w260 is None or post is None or pre is None:
                continue
            c01 = np.nansum(w01)
            out.append({"eid": eid, "car01": c01, "abs_car01": abs(c01), "car2_60": np.nansum(w260),
                        "dlogvol": np.log(np.nanstd(post) / np.nanstd(pre))})
    return pd.DataFrame(out)


def build_events() -> pd.DataFrame:
    feat = pd.read_parquet(path("processed") / "features.parquet")
    ctrl = FEATURE_GROUPS["controls"]
    lag = load_config()["timing"]["risk_lag_bdays"]
    risk = pd.read_parquet(path("interim") / "risk.parquet")
    risk = risk[(risk["words"] > 0) & risk["filing_date"].between("2010-01-01", "2024-12-31")].copy()
    risk["fm"] = risk["filing_date"].dt.to_period("M")
    risk["first"] = risk_first_month(risk["filing_date"], lag)
    pre = feat[feat["elig"] == 1][["cik", "month"] + ctrl].rename(columns={"month": "fm"}) if "elig" in feat else \
        feat[["cik", "month"] + ctrl].rename(columns={"month": "fm"})
    sig = feat[["cik", "month", "operations_prob"] + SIGNALS].rename(columns={"month": "first", "operations_prob": "ops_live"})
    e = risk[["cik", "filing_date", "fm", "first", "operations_prob"]].merge(pre, on=["cik", "fm"]).merge(sig, on=["cik", "first"])
    e = e[np.isclose(e["ops_live"], e["operations_prob"], atol=1e-4)]   # the live signal is this filing's
    sm = pd.read_parquet(path("interim") / "security_master.parquet")[["cik", "ticker"]]
    e = e.merge(sm, on="cik").reset_index(drop=True)
    e["eid"] = np.arange(len(e))
    return e


def fama_macbeth_years(e: pd.DataFrame, y: str, x: str, ctrl: list) -> dict:
    import statsmodels.api as sm
    coefs = []
    for yr, g in e.dropna(subset=[y, x] + ctrl).groupby(e["filing_date"].dt.year):
        if len(g) < 100:
            continue
        fit = sm.OLS(g[y], sm.add_constant(g[[x] + ctrl])).fit()
        coefs.append(fit.params[x])
    c = pd.Series(coefs)
    return {"coef": c.mean(), "t": c.mean() / (c.std(ddof=1) / np.sqrt(len(c))), "n_years": len(c),
            "pos_years": float((c > 0).mean())}


def run() -> dict:
    e = build_events()
    ar = daily_abnormal(set(e["ticker"]))
    e = e.merge(event_windows(ar, e[["eid", "ticker", "filing_date"]]), on="eid")
    # winsorise outcomes at 1/99% within year (a few penny-stock blowups dominate raw CARs)
    yr = e["filing_date"].dt.year
    for y in OUTCOMES:
        lo, hi = e.groupby(yr)[y].transform(lambda s: s.quantile(0.01)), e.groupby(yr)[y].transform(lambda s: s.quantile(0.99))
        e[y] = e[y].clip(lo, hi)
    ctrl = FEATURE_GROUPS["controls"]
    rows = []
    for x in SIGNALS:
        for y in OUTCOMES:
            for spec, c in (("alone", []), ("with_controls", ctrl)):
                rows.append({"signal": x, "outcome": y, "spec": spec, **fama_macbeth_years(e, y, x, c)})
    fm = pd.DataFrame(rows)
    q = []
    for x in ["r_risk_all", "r_operations_prob", "r_finance_prob", "r_d_risk_all"]:
        b = e.groupby(yr)[x].transform(lambda s: np.ceil(s.rank(pct=True, method="first") * 5)).astype(int)
        m = e.groupby(b)[OUTCOMES].mean()
        q += [{"signal": x, "quintile": k, "n": int((b == k).sum()), **m.loc[k].to_dict()} for k in m.index]
    quint = pd.DataFrame(q)
    summ = pd.DataFrame([{"events": len(e), "firms": e["cik"].nunique(), "first": str(e["filing_date"].min().date()),
                          "last": str(e["filing_date"].max().date()), **{f"mean_{y}": e[y].mean() for y in OUTCOMES}}])
    T = path("results") / "tables"
    fm.to_csv(T / "event_study_fm.csv", index=False)
    quint.to_csv(T / "event_study_quintiles.csv", index=False)
    summ.to_csv(T / "event_study_summary.csv", index=False)
    return {"fm": fm, "quintiles": quint, "summary": summ}
