"""Survivorship audit (D-032): who is missing from the free price data, and does it bias the risk-signal results?

Free prices cover only firms that still trade, so firms that later went bankrupt, were acquired or deregistered are
absent. Their SEC filing histories are not: bulk `submissions.zip` keeps every filer. For every 10-K in the risk
dataset this module records
  - whether the firm ever had CRSP-listed equity (it has a CRSP/Compustat Merged link in the WRDS link table),
  - whether we observe its stock price in the 12 months after the filing (i.e. it is in our sample),
  - its fate: still filing in 2025+, or exited (last SEC filing) after a bankruptcy (8-K item 1.03), after a merger
    (merger proxy / tender-offer forms), or otherwise,
  - whether it filed for bankruptcy within 36 months of the 10-K.
These are outcome flags for an audit, never inputs to a trading signal.
"""
import json
import zipfile

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from src.data.risk import FACTORS
from src.data.sec import download
from src.utils.config import path

MERGER_FORMS = {"DEFM14A", "PREM14A", "DEFM14C", "PREM14C", "SC 14D9", "SC TO-T", "SC 13E3"}
ACTIVE_FROM = pd.Timestamp("2025-01-01")      # still filing after this date = survivor
HORIZON_M = 36                                # bankruptcy / exit horizon after a 10-K


def filing_history(ciks) -> pd.DataFrame:
    """Per CIK: last filing date of any form, first-ever bankruptcy 8-K dates, last merger-form date, SIC.
    Reads `recent` plus the overflow files of each CIK in the bulk submissions zip. Cached."""
    out_f = path("interim") / "sec_fate.parquet"
    if out_f.exists():
        return pd.read_parquet(out_f)
    zpath = download("submissions")
    with zipfile.ZipFile(zpath) as z:
        names = set(z.namelist())
        rows = []
        for cik in sorted(set(int(c) for c in ciks)):
            n = f"CIK{cik:010d}.json"
            if n not in names:
                continue
            js = json.loads(z.read(n))
            blocks = [js["filings"]["recent"]] + [json.loads(z.read(f["name"])) for f in js["filings"].get("files", [])
                                                  if f["name"] in names]
            last, bk, mg = None, [], None
            for b in blocks:
                for form, date, items in zip(b["form"], b["filingDate"], b.get("items", [""] * len(b["form"]))):
                    d = pd.Timestamp(date)
                    last = d if last is None or d > last else last
                    if form in ("8-K", "8-K/A") and "1.03" in (items or "").split(","):
                        bk.append(d)
                    if form in MERGER_FORMS and (mg is None or d > mg):
                        mg = d
            sic = js.get("sic")
            rows.append({"cik": cik, "last_filing": last, "bk_dates": sorted(set(bk)), "last_merger_form": mg,
                         "sic": float(sic) if sic else np.nan})
    df = pd.DataFrame(rows)
    df.to_parquet(out_f, index=False)
    return df


def classify_fate(h: pd.DataFrame) -> pd.Series:
    """survivor | bankrupt | merged | exit_other (priority in that order for exited firms)."""
    exited = h["last_filing"] < ACTIVE_FROM
    bk_near = [any((h_last - d).days <= 3 * 365 for d in ds) for h_last, ds in zip(h["last_filing"], h["bk_dates"])]
    mg_near = (h["last_filing"] - h["last_merger_form"]).dt.days <= 2 * 365
    return pd.Series(np.select([~exited, pd.Series(bk_near, index=h.index), mg_near.fillna(False)],
                               ["survivor", "bankrupt", "merged"], "exit_other"), index=h.index)


def build_audit() -> pd.DataFrame:
    """One row per non-empty 10-K risk section of a (once) CRSP-listed, non-financial, non-utility firm."""
    risk = pd.read_parquet(path("interim") / "risk.parquet")
    risk = risk[risk["words"] > 0].copy()
    lk = pd.read_csv(path("raw") / "cik_gvkey.csv", usecols=["cik", "source"])
    listed = set(lk.loc[lk["source"] == "CRSP/Compustat Merged", "cik"].astype(int))
    risk = risk[risk["cik"].isin(listed)]
    h = filing_history(risk["cik"].unique())
    h["fate"] = classify_fate(h)
    a = risk.merge(h, on="cik", how="left")
    a = a[~(a["sic"].between(6000, 6999) | a["sic"].between(4900, 4949))]
    # in our sample = a monthly price observed within 12 months after the filing month
    sm = pd.read_parquet(path("interim") / "security_master.parquet")[["cik", "ticker"]]
    px = pd.read_parquet(path("interim") / "prices_monthly.parquet", columns=["ticker", "month", "ret"]).dropna()
    px = px.merge(sm, on="ticker")[["cik", "month"]].drop_duplicates()
    a["fm"] = a["filing_date"].dt.to_period("M")
    m = a[["cik", "fm"]].drop_duplicates().merge(px, on="cik")
    m = m[(m["month"] > m["fm"]) & (m["month"] <= m["fm"] + 12)]
    a["covered"] = a.set_index(["cik", "fm"]).index.isin(m.set_index(["cik", "fm"]).index)
    # outcomes within 36 months of the filing
    end = a["filing_date"] + pd.DateOffset(months=HORIZON_M)
    a["bk_36m"] = [any(f < d <= e for d in ds) for f, e, ds in zip(a["filing_date"], end, a["bk_dates"])]
    a["exit_36m"] = (a["fate"] != "survivor") & (a["last_filing"] <= end)
    a["merged_36m"] = a["exit_36m"] & (a["fate"] == "merged")
    a["year"] = a["filing_date"].dt.year
    # within-year percentile ranks, centred (same scale as the panel's r_ signals)
    for f in FACTORS:
        a[f"r_{f}"] = a.groupby("year")[f"{f}_prob"].rank(pct=True) - 0.5
    a["r_risk_all"] = a[[f"r_{f}" for f in FACTORS]].mean(axis=1)
    a["nonop"] = a[[f"{f}_prob" for f in FACTORS if f != "operations"]].mean(axis=1)
    a["r_nonop"] = a.groupby("year")["nonop"].rank(pct=True) - 0.5
    return a.drop(columns=["bk_dates"])


def coverage_by_year(a: pd.DataFrame) -> pd.DataFrame:
    g = a.groupby("year")
    out = pd.DataFrame({"filings": g.size(), "covered": g["covered"].mean()})
    for f in ["survivor", "bankrupt", "merged", "exit_other"]:
        out[f"missing_{f}"] = a[~a["covered"]].groupby("year")["fate"].apply(lambda s: (s == f).sum()) / out["filings"]
    return out.fillna(0).reset_index()


def fate_risk_regressions(a: pd.DataFrame, years=(2006, 2022)) -> pd.DataFrame:
    """Linear probability models: outcome within 36 months on one within-year risk rank, filing-year FE,
    SEs clustered by firm. Filings up to 2022 so the 36-month window is observed."""
    d = a[a["year"].between(*years)].copy()
    rows = []
    for y in ["bk_36m", "merged_36m", "exit_36m", "covered"]:
        d[y] = d[y].astype(float)
        for s in ["r_risk_all", "r_nonop"] + [f"r_{f}" for f in FACTORS]:
            fit = smf.ols(f"{y} ~ {s} + C(year)", d).fit(cov_type="cluster", cov_kwds={"groups": d["cik"]})
            rows.append({"outcome": y, "signal": s, "coef": fit.params[s], "t": fit.tvalues[s],
                         "base_rate": d[y].mean(), "n": int(fit.nobs)})
    return pd.DataFrame(rows)


def missing_by_decile(a: pd.DataFrame, years=(2010, 2021)) -> pd.DataFrame:
    """Share of listed filings absent from our price data, by within-year decile of a risk signal (dev years)."""
    d = a[a["year"].between(*years)]
    rows = []
    for s in ["r_risk_all", "r_nonop", "r_operations"]:
        dec = d.groupby("year")[s].transform(lambda x: np.ceil(x.rank(pct=True, method="first") * 10)).astype(int)
        g = d.groupby(dec)
        for k, gg in g:
            rows.append({"signal": s, "decile": k, "filings": len(gg), "missing": 1 - gg["covered"].mean(),
                         "missing_bankrupt": ((~gg["covered"]) & (gg["fate"] == "bankrupt")).mean(),
                         "missing_merged": ((~gg["covered"]) & (gg["fate"] == "merged")).mean()})
    return pd.DataFrame(rows)


def breakeven(miss: pd.DataFrame, sorts: pd.DataFrame) -> pd.DataFrame:
    """Monthly return gap g = (missing firms - observed firms in the same decile), common to both deciles, at which the
    observed top-minus-bottom spread would vanish once the missing firms were added back:
    true decile return = observed + missing_share * g, so spread_true = spread_obs + (m_top - m_bottom) * g = 0
    gives g* = -spread_obs / (m_top - m_bottom). g* > 0 means the missing firms would have to OUTperform."""
    m = {"r_risk_all": "r_risk_all", "r_nonop": "r_nonop_prob", "r_operations": "r_operations_prob"}
    rows = []
    for s, sig in m.items():
        q = miss[miss["signal"] == s].set_index("decile")["missing"]
        for w in ("ew", "vw"):
            ls = sorts[(sorts["signal"] == sig) & (sorts["weight"] == w)]["ls_mean_m"].iloc[0]
            gap = q.loc[10] - q.loc[1]
            rows.append({"signal": sig, "weight": w, "ls_mean_m": ls, "missing_top": q.loc[10], "missing_bottom": q.loc[1],
                         "breakeven_gap_m": -ls / gap if abs(gap) > 1e-9 else np.nan})
    return pd.DataFrame(rows)


def run() -> dict:
    a = build_audit()
    T = path("results") / "tables"
    out = {"coverage": coverage_by_year(a), "fate_lpm": fate_risk_regressions(a), "missing_decile": missing_by_decile(a)}
    out["breakeven"] = breakeven(out["missing_decile"], pd.read_csv(T / "single_signal_sorts_dev.csv"))
    fate = a.drop_duplicates("cik")["fate"].value_counts()
    out["summary"] = pd.DataFrame({"fate": fate.index, "firms": fate.values})
    for k, v in out.items():
        v.to_csv(T / f"survivorship_{k}.csv", index=False)
    return out
