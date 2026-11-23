"""Replication of Astvansh & Simpson (2026), Table 7 (D-034): operational risk in year y -> ln(COGS) in year y+1.

Their sample merges the risk scores with Compustat (N = 67,727 firm-years, 2005-2024). We have no Compustat, so the
accounting variables come from SEC XBRL companyfacts, which exist from 2009-2011 onward but cover every filer,
including firms that later disappeared (so this sample is not survivor-only).
  y        = the fiscal year reported in the 10-K (latest annual period end in facts filed with that 10-K)
  DV       = ln(COGS) for the next fiscal year (period ending 330-400 days later), COGS winsorised at 1%/99%
  controls = ln(total assets_y) [their Table 5 definition], Item 1A optional, ln(Item 1A words), R&D_y / assets_y
             (missing R&D = 0)
  risk     = operations probability; non-operational risk = sum of the other 7 probabilities; both mean-centred
  FE       = firm (CIK) and fiscal year; SEs clustered by firm. Columns (1)-(4) as in their Table 7.
"""
import json
import zipfile

import numpy as np
import pandas as pd

from src.data.risk import FACTORS
from src.data.sec import _facts, _zip_members, download
from src.utils.config import path

CONCEPTS = {"cogs": ["CostOfRevenue", "CostOfGoodsAndServicesSold", "CostOfGoodsSold"], "at": ["Assets"],
            "rd": ["ResearchAndDevelopmentExpense", "ResearchAndDevelopmentExpenseExcludingAcquiredInProcessCost"]}
PAPER = {"(2) op": (0.222, 0.038), "(4) op": (0.247, 0.041), "(4) op x nonop": (0.298, 0.125), "N": 67727}
# Astvansh & Simpson (2026) Table 7, coefficient (SE), columns (1)-(4), transcribed from the published paper
PAPER_T7 = {
    "ln_at": [(0.487, 0.014), (0.486, 0.014), (0.486, 0.014), (0.486, 0.014)],
    "optional": [(-0.104, 0.025), (-0.093, 0.025), (-0.089, 0.025), (-0.083, 0.025)],
    "ln_words": [(-0.006, 0.003), (-0.007, 0.003), (-0.004, 0.003), (0.001, 0.004)],
    "rd_at": [(0.004, 0.014)] * 4,
    "op": [None, (0.222, 0.038), (0.225, 0.038), (0.247, 0.041)],
    "nonop": [None, None, (-0.034, 0.022), (-0.046, 0.024)],
    "op_x_nonop": [None, None, None, (0.298, 0.125)],
}


def xbrl_annual(ciks) -> pd.DataFrame:
    """10-K FY facts for COGS, assets, R&D of every CIK (all filers, not just priced ones). Cached."""
    out_f = path("interim") / "xbrl_replication.parquet"
    if out_f.exists():
        return pd.read_parquet(out_f)
    zpath = download("companyfacts")
    rows = []
    with zipfile.ZipFile(zpath) as z:
        for cik, n in _zip_members(zpath, ciks).items():
            js = json.loads(z.read(n))
            for item, cons in CONCEPTS.items():
                for k, con in enumerate(cons):
                    for f in _facts(js, "us-gaap", con, "USD"):
                        if f.get("form") in ("10-K", "10-K/A") and f.get("fp") == "FY":
                            rows.append((cik, item, k, f.get("start"), f.get("end"), f.get("filed"), f.get("val")))
    df = pd.DataFrame(rows, columns=["cik", "item", "pri", "start", "end", "filed", "val"])
    for c in ("start", "end", "filed"):
        df[c] = pd.to_datetime(df[c], errors="coerce")
    df = df.dropna(subset=["end", "filed", "val"])
    df.to_parquet(out_f, index=False)
    return df


def build_sample() -> pd.DataFrame:
    risk = pd.read_parquet(path("interim") / "risk.parquet")
    risk = risk[risk["words"] > 0].copy()
    x = xbrl_annual(risk["cik"].unique())
    dur = (x["end"] - x["start"]).dt.days
    x = x[(x["item"] == "at") | dur.between(330, 400)]
    # one value per (cik, item, period end): highest-priority concept, earliest filing (the original report)
    x = x.sort_values(["cik", "item", "end", "pri", "filed"]).drop_duplicates(["cik", "item", "end"])
    vals = x.pivot_table(index=["cik", "end"], columns="item", values="val", aggfunc="first").reset_index()
    # fiscal year y of each 10-K = latest annual period end among facts filed within 3 days of the filing date
    fx = pd.read_parquet(path("interim") / "xbrl_replication.parquet", columns=["cik", "end", "filed"])
    m = risk[["cik", "filing_date"]].merge(fx, on="cik")
    m = m[(m["filed"] - m["filing_date"]).abs().dt.days <= 3]
    yend = m.groupby(["cik", "filing_date"])["end"].max().rename("y_end").reset_index()
    s = risk.merge(yend, on=["cik", "filing_date"])
    s = s.merge(vals[["cik", "end", "at", "rd"]].rename(columns={"end": "y_end"}), on=["cik", "y_end"], how="left")
    nxt = vals[["cik", "end", "cogs"]].dropna().rename(columns={"end": "n_end", "cogs": "cogs_next"})
    s = s.merge(nxt, on="cik")
    s = s[((s["n_end"] - s["y_end"]).dt.days).between(330, 400)].drop_duplicates(["cik", "filing_date"])
    s = s[(s["cogs_next"] > 0) & (s["at"] > 0)].copy()
    s["fyear"] = np.where(s["y_end"].dt.month >= 6, s["y_end"].dt.year, s["y_end"].dt.year - 1)
    s = s.sort_values("filing_date").drop_duplicates(["cik", "fyear"], keep="last")
    lo, hi = s["cogs_next"].quantile([0.01, 0.99])
    s["ln_cogs_next"] = np.log(s["cogs_next"].clip(lo, hi))
    s["ln_at"] = np.log(s["at"])
    s["rd_at"] = (s["rd"].fillna(0) / s["at"]).clip(*((s["rd"].fillna(0) / s["at"]).quantile([0.01, 0.99])))
    s["ln_words"] = np.log(s["words"])
    s["optional"] = s["optional"].astype(float)
    s["op"] = s["operations_prob"] - s["operations_prob"].mean()
    nonop = s[[f"{f}_prob" for f in FACTORS if f != "operations"]].sum(axis=1)
    s["nonop"] = nonop - nonop.mean()
    s["op_x_nonop"] = s["op"] * s["nonop"]
    return s


def run() -> dict:
    from linearmodels.panel import PanelOLS
    s = build_sample()
    d = s.set_index(["cik", "fyear"])
    base = ["ln_at", "optional", "ln_words", "rd_at"]
    specs = {"(1)": base, "(2)": base + ["op"], "(3)": base + ["op", "nonop"], "(4)": base + ["op", "nonop", "op_x_nonop"]}
    rows = []
    for name, xs in specs.items():
        fit = PanelOLS(d["ln_cogs_next"], d[xs], entity_effects=True, time_effects=True).fit(
            cov_type="clustered", cluster_entity=True)
        for v in xs:
            pc = PAPER_T7[v][int(name[1]) - 1]
            rows.append({"model": name, "var": v, "coef": fit.params[v], "se": fit.std_errors[v], "t": fit.tstats[v],
                         "paper_coef": pc[0] if pc else None, "paper_se": pc[1] if pc else None,
                         "n": int(fit.nobs), "firms": int(d.index.get_level_values(0).nunique()),
                         "r2_within": fit.rsquared_within})
    out = pd.DataFrame(rows)
    summ = pd.DataFrame([{"n": len(s), "firms": s["cik"].nunique(), "fyear_min": int(s["fyear"].min()),
                          "fyear_max": int(s["fyear"].max()), "mean_op": s["operations_prob"].mean(),
                          "paper_op_coef": PAPER["(4) op"][0], "paper_op_se": PAPER["(4) op"][1],
                          "paper_inter_coef": PAPER["(4) op x nonop"][0], "paper_inter_se": PAPER["(4) op x nonop"][1],
                          "paper_n": PAPER["N"]}])
    T = path("results") / "tables"
    out.to_csv(T / "replication_cogs.csv", index=False)
    summ.to_csv(T / "replication_summary.csv", index=False)
    return {"table": out, "summary": summ}
