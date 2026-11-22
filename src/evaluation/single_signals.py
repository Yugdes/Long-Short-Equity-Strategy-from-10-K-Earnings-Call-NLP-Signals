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


def run_interaction_fm(feat: pd.DataFrame) -> pd.DataFrame:
    """H4: Fama-MacBeth of exret on OpRisk, a marketing construct, their product and the 10 controls
    (marketing-covered rows). Predicted sign of the interaction: positive (marketing attenuates the risk penalty)."""
    dev = dev_only(feat)
    d = dev[dev["has_mkt"] == 1]
    ctrl = FEATURE_GROUPS["controls"]
    rows = []
    for c in ["mkt_orient", "mkt_capab", "mkt_excel"]:
        for risk, x in (("r_operations_prob", f"x_ops_{c}"), ("r_d_operations_prob", f"x_dops_{c}")):
            fm = fama_macbeth(d, "exret", [risk, f"r_{c}", x] + ctrl)
            rows.append({"interaction": x, "coef": fm.loc[x, "coef"], "t_nw": fm.loc[x, "t_nw"],
                         "risk_t": fm.loc[risk, "t_nw"], "mkt_t": fm.loc[f"r_{c}", "t_nw"], "n_months": fm.loc[x, "n_months"]})
    out = pd.DataFrame(rows)
    out.to_csv(path("results") / "tables" / "interaction_fm_dev.csv", index=False)
    return out


def run_double_sort(feat: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """H4, pre-registered 3x3 conditional double sort (D-031): terciles of operational risk (level, or change vs the
    previous 10-K) each month, then terciles of a marketing construct within each risk tercile. Marketing-covered
    rows, dev window, month-t excess returns. 'cushion' = (high-minus-low marketing among high-risk firms) minus
    (the same among low-risk firms); H4 predicts > 0."""
    from src.backtest.metrics import nw_tstat
    from src.backtest.portfolios import double_sort
    ff = load_ff_factors()
    dev = dev_only(feat)
    dev = dev[dev["has_mkt"] == 1]
    rows, cells = [], []
    for risk in ("r_operations_prob", "r_d_operations_prob"):
        for c in ("mkt_orient", "mkt_capab", "mkt_excel"):
            d = dev[dev[risk[2:]].notna() & dev[c].notna()]          # raw inputs present (no imputed zeros)
            for w in ("ew", "vw"):
                C = double_sort(d, risk, f"r_{c}", n=3, weight=w, ret_col="exret")
                for (i, j), s in C.items():
                    cells.append({"risk": risk, "mkt": c, "weight": w, "risk_t": i, "mkt_t": j,
                                  "mean_m": s.mean(), "t_nw": nw_tstat(s)})
                series = {"mkt_spread_high_risk": C[(3, 3)] - C[(3, 1)],
                          "mkt_spread_low_risk": C[(1, 3)] - C[(1, 1)],
                          "risk_spread_low_mkt": C[(3, 1)] - C[(1, 1)],
                          "risk_spread_high_mkt": C[(3, 3)] - C[(1, 3)]}
                series["cushion"] = series["mkt_spread_high_risk"] - series["mkt_spread_low_risk"]
                for name, s in series.items():
                    s = s.dropna()
                    a = factor_alpha(s, ff, "FF5+Mom")
                    rows.append({"risk": risk, "mkt": c, "weight": w, "series": name, "mean_m": s.mean(),
                                 "t_nw": nw_tstat(s), "alpha_ann": a["alpha_ann"], "t_alpha": a["t_alpha"],
                                 "n_months": len(s)})
                    if name in ("mkt_spread_high_risk", "cushion"):          # the two tradeable H4 portfolios
                        log_spec("single_sort", f"{risk}x{c}|{w}|3x3|{name}", s, hyp="H4", t_alpha=a["t_alpha"])
    out, cell_df = pd.DataFrame(rows), pd.DataFrame(cells)
    out.to_csv(path("results") / "tables" / "double_sort_dev.csv", index=False)
    cell_df.to_csv(path("results") / "tables" / "double_sort_cells_dev.csv", index=False)
    return out, cell_df
