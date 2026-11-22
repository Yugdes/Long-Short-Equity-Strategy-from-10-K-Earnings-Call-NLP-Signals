"""Rung 1-2 of the model ladder: single-signal decile sorts and Fama-MacBeth, development window only."""
import pandas as pd

from src.backtest.metrics import perf_stats
from src.backtest.portfolios import sort_portfolios
from src.backtest.regressions import factor_alpha, fama_macbeth
from src.data.factors import load_ff_factors
from src.evaluation.speclog import log_spec
from src.features.signals import FEATURE_GROUPS
from src.utils.config import path
from src.utils.lock import dev_only

# Pre-specified in RESEARCH_DESIGN.md §2 (hypothesis -> signal). LS = top decile - bottom decile.
SIGNALS = {
    "H1": ["r_operations_prob", "r_ops_ind", "r_risk_all", "r_nonop_prob", "r_dens_ops"],
    "H2": ["r_d_operations_prob", "r_d_risk_all", "r_d_nonop_prob", "r_s_operations_prob"],
    "H3": ["r_mkt_capab", "r_mkt_excel", "r_mkt_orient", "r_dy_mkt_capab", "r_dy_mkt_excel"],
    "explore_factors": [f"r_{f}_prob" for f in ["accounting", "finance", "international", "legal",
                                                 "management", "marketing", "technology"]],
}
MKT_SIGNALS = set(SIGNALS["H3"])


def run_single_signals(feat: pd.DataFrame) -> pd.DataFrame:
    ff = load_ff_factors()
    dev = dev_only(feat)
    rows = []
    for hyp, sigs in SIGNALS.items():
        for s in sigs:
            d = dev[dev["has_mkt"] == 1] if s in MKT_SIGNALS else dev
            # imputed zeros are "no information": sort only rows with the raw input present
            raw = s[2:] if s.startswith("r_") else s
            if raw in d.columns:
                d = d[d[raw].notna()]
            for w in ("ew", "vw"):
                rets, _ = sort_portfolios(d, s, n=10, weight=w)
                ls = rets["LS"]
                a = factor_alpha(ls, ff, "FF5+Mom")
                st = perf_stats(ls)
                row = {"hyp": hyp, "signal": s, "weight": w, "ls_mean_m": ls.mean(), "t_mean": st["t_mean_nw"],
                       "sharpe": st["sharpe"], "alpha_ann": a["alpha_ann"], "t_alpha": a["t_alpha"],
                       "n_months": st["n_months"], "avg_names": d.groupby("month").size().mean() / 10}
                rows.append(row)
                log_spec("single_sort", f"{s}|{w}|d10", ls, hyp=hyp, t_alpha=a["t_alpha"])
    out = pd.DataFrame(rows)
    out.to_csv(path("results") / "tables" / "single_signal_sorts_dev.csv", index=False)
    return out


def run_fama_macbeth(feat: pd.DataFrame) -> pd.DataFrame:
    """exret_t on each signal alone and with the 10 control ranks (all centred ranks)."""
    dev = dev_only(feat)
    ctrl = FEATURE_GROUPS["controls"]
    rows = []
    for hyp, sigs in SIGNALS.items():
        for s in sigs:
            d = dev[dev["has_mkt"] == 1] if s in MKT_SIGNALS else dev
            for label, xs in (("alone", [s]), ("with_controls", [s] + ctrl)):
                fm = fama_macbeth(d, "exret", xs)
                rows.append({"hyp": hyp, "signal": s, "spec": label, "coef": fm.loc[s, "coef"],
                             "t_nw": fm.loc[s, "t_nw"], "n_months": fm.loc[s, "n_months"]})
    out = pd.DataFrame(rows)
    out.to_csv(path("results") / "tables" / "fama_macbeth_dev.csv", index=False)
    return out


def run_vol_prediction(feat: pd.DataFrame) -> pd.DataFrame:
    """H1(a): Fama-MacBeth of next-12-month realised volatility on risk levels, with controls.
    Formation months <= 2020-12 so the forward window never reaches the locked test period (D-024)."""
    d = feat[feat["month"] <= pd.Period("2020-12", "M")].copy()
    d["fvol_rank"] = d.groupby("month")["fvol_12"].rank(pct=True) - 0.5
    ctrl = FEATURE_GROUPS["controls"]
    rows = []
    for s in SIGNALS["H1"] + SIGNALS["explore_factors"]:
        for label, xs in (("alone", [s]), ("with_controls", [s] + ctrl)):
            fm = fama_macbeth(d, "fvol_rank", xs)
            rows.append({"signal": s, "spec": label, "coef": fm.loc[s, "coef"], "t_nw": fm.loc[s, "t_nw"],
                         "n_months": fm.loc[s, "n_months"]})
    out = pd.DataFrame(rows)
    out.to_csv(path("results") / "tables" / "vol_prediction_dev.csv", index=False)
    return out
