"""Data-pipeline sanity checks: factor replication and the coverage funnel (dev window only)."""
import pandas as pd

from src.backtest.portfolios import sort_portfolios
from src.data.factors import load_ff_factors
from src.data.prices import build_monthly
from src.data.risk import load_risk_scores
from src.data.security_master import build_security_master
from src.utils.config import path
from src.utils.lock import dev_only


def factor_replication(panel: pd.DataFrame) -> pd.DataFrame:
    """Momentum and size long-shorts built from our own returns vs French UMD / SMB."""
    ff = load_ff_factors().set_index("month")
    rows = []
    for name, mask, sig, fac, sign in [("mom_decile_LS_ew", "elig", "mom_12_2", "mom", 1),
                                       ("size_decile_SmB_ew", "elig", "size", "smb", -1),
                                       ("mom_decile_LS_ext_ew", "elig_ext", "mom_12_2", "mom", 1)]:
        d = dev_only(panel[panel[mask]])
        rets, _ = sort_portfolios(d, sig, n=10, weight="ew")
        ls = sign * rets["LS"]
        j = pd.concat([ls.rename("ours"), ff[fac]], axis=1, join="inner").dropna()
        rows.append({"series": name, "french": fac, "months": len(j), "corr": j.corr().iloc[0, 1],
                     "mean_ours": j["ours"].mean(), "mean_french": j[fac].mean()})
    out = pd.DataFrame(rows)
    out.to_csv(path("results") / "tables" / "sanity_factor_replication.csv", index=False)
    return out


def coverage_funnel(panel: pd.DataFrame) -> pd.DataFrame:
    """Risk firm-years by filing year surviving each data stage (counts of distinct CIKs)."""
    r = load_risk_scores()
    sm = build_security_master()
    pm, _ = build_monthly()
    have_px = set(sm.loc[sm["ticker"].isin(pm["ticker"].unique()), "cik"])
    p = panel.copy()
    p["fy"] = p["filing_date"].dt.year
    ext = p[p["elig_ext"]].groupby("fy")["cik"].nunique()
    pri = p[p["elig"]].groupby("fy")["cik"].nunique()
    f = r.groupby("file_year").agg(filings=("cik", "size"))
    f["nonempty_item1a"] = r[r["words"] > 0].groupby("file_year").size()
    f["current_sec_ticker"] = r[r["cik"].isin(sm["cik"])].groupby("file_year").size()
    f["yahoo_prices"] = r[r["cik"].isin(have_px)].groupby("file_year").size()
    f["in_extended_universe"] = ext
    f["in_primary_universe"] = pri
    f = f.fillna(0).astype(int)
    f["pct_primary_of_filings"] = (f["in_primary_universe"] / f["filings"]).round(3)
    f.to_csv(path("results") / "tables" / "coverage_funnel.csv")
    return f
