"""Point-in-time stock-month panel.

Row (cik, month=t): returns realised in month t (targets) and features known before t begins:
  * controls: end of month t-1 (characteristics shifted by one month)
  * risk text: latest 10-K with risk_first_month <= t, dropped once t > filing month + 15 (D-005, D-007)
  * marketing: latest call quarter with first_month <= t < first_month + 6 (D-006, D-007)
Firm-history transforms (changes, surprises) are computed at the filing/quarter level from past
releases only; cross-sectional transforms happen per month in signals.py.
"""
import numpy as np
import pandas as pd

from src.data.factors import load_ff_factors
from src.data.marketing import load_marketing, CONSTRUCTS, ZCA
from src.data.prices import build_monthly, split_adjust_shares
from src.data.risk import load_risk_scores, FACTORS
from src.data.sec import parse_companyfacts
from src.data.security_master import build_security_master
from src.features.characteristics import (annual_fundamentals, asof_join, full_monthly_grid,
                                          latest_facts, return_characteristics, shares_table)
from src.features.timing import risk_first_month
from src.utils.config import load_config, path

RISK_BASE = [f"{f}_prob" for f in FACTORS]


def risk_filing_features(risk: pd.DataFrame) -> pd.DataFrame:
    """Per-filing features from the firm's own past filings only."""
    r = risk.sort_values(["cik", "filing_date"]).copy()
    r["dens_ops"] = r["operations_cnt"] / r["words"]          # dictionary density (count per word)
    r["log_words"] = np.log(r["words"].where(r["words"] > 0))
    r["nonop_prob"] = r[[c for c in RISK_BASE if c != "operations_prob"]].mean(axis=1)
    base = RISK_BASE + ["nonop_prob", "dens_ops", "log_words"]
    g = r.groupby("cik")
    gap = (r["filing_date"] - g["filing_date"].shift(1)).dt.days
    for c in base:
        prev = g[c].shift(1)
        r[f"d_{c}"] = (r[c] - prev).where(gap <= 548)
        past_mean = g[c].transform(lambda x: x.shift(1).expanding(min_periods=2).mean())
        r[f"s_{c}"] = r[c] - past_mean
    r["risk_first_month"] = risk_first_month(r["filing_date"], load_config()["timing"]["risk_lag_bdays"])
    r["filing_month"] = r["filing_date"].dt.to_period("M")
    return r


def marketing_quarter_features(mk: pd.DataFrame) -> pd.DataFrame:
    m = mk.sort_values(["cik", "year", "quarter"]).copy()
    m["qidx"] = m["year"] * 4 + m["quarter"]
    g = m.groupby("cik")
    cols = list(CONSTRUCTS) + list(ZCA.values()) + ["an_cust"]
    prev_q = g["qidx"].shift(1)
    for c in CONSTRUCTS:
        m[f"dq_{c}"] = (m[c] - g[c].shift(1)).where(m["qidx"] - prev_q == 1)
    # YoY: same calendar quarter one year earlier
    yo = m[["cik", "qidx"] + list(CONSTRUCTS)].copy()
    yo["qidx"] += 4
    m = m.merge(yo, on=["cik", "qidx"], how="left", suffixes=("", "_ly"))
    for c in CONSTRUCTS:
        m[f"dy_{c}"] = m[c] - m.pop(f"{c}_ly")
    keep = ["cik", "first_month", "document_length"] + cols + [f"dq_{c}" for c in CONSTRUCTS] + [f"dy_{c}" for c in CONSTRUCTS]
    return m[keep]


def build_panel(force: bool = False) -> pd.DataFrame:
    out = path("processed") / "panel.parquet"
    if out.exists() and not force:
        return pd.read_parquet(out)
    cfg = load_config()
    sm = build_security_master()
    pm, splits = build_monthly()
    ff = load_ff_factors()
    rf = ff.set_index("month")["rf"]
    mkt = ff.set_index("month")["mktrf"] + rf

    # --- prices on a gap-free grid, characteristics at end of month m ---
    pm = pm[pm["ticker"].isin(sm["ticker"])]
    g = full_monthly_grid(pm)
    g = return_characteristics(g, mkt)
    g = g.merge(sm[["ticker", "cik"]], on="ticker", how="left")

    # market cap at end of m: latest share count usable by m, put on Yahoo's split basis
    sh_raw, fu_raw = parse_companyfacts(set(sm["cik"]))
    sh = latest_facts(shares_table(sh_raw), ["shares"], cfg["timing"]["shares_lag_months"])
    sh = sh.merge(sm[["cik", "ticker"]], on="cik")
    sh["shares_adj"] = split_adjust_shares(sh["shares"], sh["end"], sh["ticker"], splits)
    # D-021: drop unit-error share counts: >10x jump vs the median of the firm's previous 3 counts (past only)
    trail = sh.groupby("cik")["shares_adj"].transform(lambda x: x.shift(1).rolling(3, min_periods=1).median())
    jump = (sh["shares_adj"] / trail)
    sh = sh[~((jump > 10) | (jump < 0.1))]
    sh_j = asof_join(g, sh, "usable", ["shares_adj", "end"]).rename(columns={"end": "sh_end"})
    g = g.join(sh_j)
    month_end = g["month"].dt.to_timestamp(how="end").dt.normalize()
    g["prc_raw"] = split_adjust_shares(g["prc"], month_end, g["ticker"], splits)   # undo later splits
    stale = (month_end - g["sh_end"]).dt.days > 400
    g["mcap"] = (g["prc"] * g["shares_adj"] / 1e6).where(~stale)        # $ millions
    g.loc[(g["mcap"] > 5e6) | (g["prc_raw"] > 1e5), "mcap"] = np.nan    # > $5T or > $100k/share: data error
    # D-021: split-invariant cross-check. Dollar volume / mcap outside [0.1%, 500%] per month means
    # mis-scaled shares or a broken split history, so mcap is treated as missing.
    turnover = np.exp(g["ldvol_3"]) / (g["mcap"] * 1e6)
    g.loc[(turnover < 1e-3) | (turnover > 5), "mcap"] = np.nan
    g["size"] = np.log(g["mcap"])

    fa = latest_facts(annual_fundamentals(fu_raw), ["be", "at", "rev", "cogs", "at_prev"],
                      cfg["timing"]["fund_lag_months"])
    fa_j = asof_join(g, fa, "usable", ["be", "at", "rev", "cogs", "at_prev", "end"]).rename(columns={"end": "fa_end"})
    g = g.join(fa_j)
    fstale = (g["month"].dt.to_timestamp(how="end") - g["fa_end"]).dt.days > 18 * 31
    g.loc[fstale, ["be", "at", "rev", "cogs", "at_prev"]] = np.nan
    g.loc[g["mcap"] * 1e6 > 500 * g["at"], "mcap"] = np.nan          # D-021: > 500x book assets is a data error
    g["size"] = np.log(g["mcap"])
    g["bm"] = np.log((g["be"] / (g["mcap"] * 1e6)).where(g["be"] > 0))
    g["ag"] = (g["at"] / g["at_prev"] - 1).where(g["at_prev"] > 0)
    g["gp"] = (g["rev"] - g["cogs"]) / g["at"]
    ctrl = ["size", "bm", "mom_12_2", "rev_1", "vol_12", "beta_36", "illiq_12", "ldvol_3", "ag", "gp"]
    lag_cols = ctrl + ["prc_raw", "mcap", "hist"]

    # --- holding month t: realised returns at t, controls from m = t-1 ---
    p = g[["cik", "ticker", "month", "ret"]].copy()
    lagged = g[["cik", "month"] + lag_cols].copy()
    lagged["month"] = lagged["month"] + 1
    p = p.merge(lagged.rename(columns={"prc_raw": "prc_l", "mcap": "mcap_l", "hist": "hist_l"}), on=["cik", "month"], how="left")
    p["rf"] = p["month"].map(rf)
    p["exret"] = p["ret"] - p["rf"]
    p = p.sort_values(["cik", "month"]).reset_index(drop=True)
    gp_ = p.groupby("cik")
    r1, r2 = gp_["ret"].shift(-1), gp_["ret"].shift(-2)
    p["ret_3m"] = (1 + p["ret"]) * (1 + r1) * (1 + r2) - 1
    p["exret_3m"] = p["ret_3m"] - (p["rf"] + p["month"].map(rf.shift(-1)) + p["month"].map(rf.shift(-2)))
    p["fvol_12"] = gp_["ret"].transform(lambda x: x[::-1].rolling(12, min_periods=10).std()[::-1])

    # --- risk text (as of t, with staleness cap) ---
    rk = risk_filing_features(load_risk_scores())
    rcols = [c for c in rk.columns if c.endswith(("_prob", "_cnt")) or c.startswith(("d_", "s_"))] + \
            ["dens_ops", "log_words", "words", "optional", "filing_date", "filing_month"]
    rcols = list(dict.fromkeys(rcols))
    rj = asof_join(p, rk, "risk_first_month", rcols)
    age = pd.PeriodIndex(p["month"]).asi8 - pd.PeriodIndex(rj["filing_month"]).asi8
    rj.loc[~(age <= cfg["timing"]["risk_max_age_months"]), rcols] = np.nan
    p = p.join(rj)

    # --- marketing (as of t, live for mkt_max_age_months) ---
    mq = marketing_quarter_features(load_marketing(set(sm["cik"])))
    mcols = [c for c in mq.columns if c not in ("cik", "first_month")] + ["mkt_first_month"]
    mq["mkt_first_month"] = mq["first_month"]
    mj = asof_join(p, mq, "first_month", mcols)
    mage = pd.PeriodIndex(p["month"]).asi8 - pd.PeriodIndex(mj["mkt_first_month"]).asi8
    mj.loc[~(mage < cfg["timing"]["mkt_max_age_months"]), mcols] = np.nan
    p = p.join(mj)

    # --- universe flags (D-008, D-009, D-018) ---
    p = p.merge(sm[["cik", "ff48", "ff12", "fin_util", "exchange"]], on="cik", how="left")
    u, s = cfg["universe"], cfg["sample"]
    base = (p["operations_prob"].notna() & p["ret"].notna() & (p["prc_l"] >= u["min_price"])
            & (p["hist_l"] >= u["min_history_months"]) & ~p["fin_util"].astype(bool) & (p["month"] <= pd.Period(s["end"], "M")))
    p["elig_ext"] = base & (p["month"] >= pd.Period(s["extended_start"], "M"))
    p["elig"] = base & (p["mcap_l"] >= u["min_mcap_musd"]) & (p["month"] >= pd.Period(s["primary_start"], "M"))
    p = p[p["elig_ext"] | p["elig"]].reset_index(drop=True)
    f64 = p.select_dtypes("float64").columns
    p[f64] = p[f64].astype("float32")
    p.to_parquet(out, index=False)
    return p
