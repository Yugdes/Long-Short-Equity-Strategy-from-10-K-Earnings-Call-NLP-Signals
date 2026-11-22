"""Robustness, placebo and overfitting diagnostics (brief Phase 6)."""
import itertools

import numpy as np
import pandas as pd

from src.backtest.metrics import perf_stats, deflated_sharpe
from src.backtest.portfolios import sort_portfolios
from src.backtest.regressions import factor_alpha
from src.evaluation.predictive import ic_series
from src.evaluation.speclog import load_specs

SUBPERIODS = {"2010-2013": ("2010-01", "2013-12"), "2014-2017": ("2014-01", "2017-12"),
              "2018-2021": ("2018-01", "2021-12"), "2020 (COVID)": ("2020-01", "2020-12"),
              "2008-2009 (GFC, ext. EW)": ("2008-01", "2009-12"), "test 2022-01 to 2025-05": ("2022-01", "2025-05")}


def subperiod_table(ls: pd.Series, ff: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name, (a, b) in SUBPERIODS.items():
        r = ls[(ls.index >= pd.Period(a, "M")) & (ls.index <= pd.Period(b, "M"))].dropna()
        if len(r) < 6:
            continue
        st = perf_stats(r)
        al = factor_alpha(r, ff, "FF5+Mom") if len(r) >= 24 else {"alpha_ann": np.nan, "t_alpha": np.nan}
        rows.append({"period": name, "months": len(r), "ann_mean": st["ann_mean"], "sharpe": st["sharpe"],
                     "t_mean": st["t_mean_nw"], "alpha_ann": al["alpha_ann"], "t_alpha": al["t_alpha"]})
    return pd.DataFrame(rows)


def size_buckets(d: pd.DataFrame, score: str = "pred") -> pd.DataFrame:
    d = d.copy()
    d["size_b"] = d.groupby("month")["mcap_l"].transform(lambda x: pd.qcut(x.rank(method="first"), 3, labels=["small", "mid", "large"]))
    rows = []
    for b, g in d.groupby("size_b", observed=True):
        rets, _ = sort_portfolios(g, score, n=5, weight="ew")
        st = perf_stats(rets["LS"])
        ic = ic_series(g, score, "y_rank")
        rows.append({"bucket": b, "ls_q5_ew_ann": st["ann_mean"], "sharpe": st["sharpe"], "t_mean": st["t_mean_nw"],
                     "ic_mean": ic.mean(), "names_per_month": g.groupby("month").size().mean()})
    return pd.DataFrame(rows)


def industry_ic(d: pd.DataFrame, score: str = "pred") -> pd.DataFrame:
    rows = []
    for ind, g in d.groupby("ff12"):
        ic = ic_series(g, score, "y_rank").dropna()
        if len(ic) > 24:
            rows.append({"ff12": int(ind), "ic_mean": ic.mean(), "ic_t": ic.mean() / ic.std() * np.sqrt(len(ic)),
                         "names_per_month": g.groupby("month").size().mean()})
    return pd.DataFrame(rows)


def placebo_shuffle(d: pd.DataFrame, score: str = "pred", n_draws: int = 500, seed: int = 0) -> dict:
    """Shuffle the score within each month: the null distribution of decile LS Sharpe (EW) and mean IC."""
    rng = np.random.default_rng(seed)
    real_rets, _ = sort_portfolios(d, score, n=10, weight="ew")
    real_sr = perf_stats(real_rets["LS"])["sharpe"]
    real_ic = ic_series(d, score, "y_rank").mean()
    sr, ic = [], []
    base = d[["month", "cik", "ret", "y_rank", score]].dropna().copy()
    for _ in range(n_draws):
        base["shuf"] = base.groupby("month")[score].transform(lambda x: rng.permutation(x.values))
        r, _ = sort_portfolios(base, "shuf", n=10, weight="ew")
        sr.append(perf_stats(r["LS"])["sharpe"])
        ic.append(ic_series(base, "shuf", "y_rank").mean())
    sr, ic = np.array(sr), np.array(ic)
    return {"real_sharpe": real_sr, "null_sharpe_mean": sr.mean(), "null_sharpe_p95": np.quantile(sr, 0.95),
            "p_value_sharpe": float((sr >= real_sr).mean()), "real_ic": real_ic, "null_ic_p95": np.quantile(ic, 0.95),
            "p_value_ic": float((ic >= real_ic).mean()), "n_draws": n_draws, "null_sharpes": sr, "null_ics": ic}


def pbo_cscv(R: pd.DataFrame, n_blocks: int = 16) -> dict:
    """Probability of backtest overfitting (Bailey, Borwein, López de Prado & Zhu 2017).
    R: months x strategies matrix of returns. Choose the best in-sample Sharpe on half the blocks;
    PBO = share of splits where that strategy ranks below the median out of sample."""
    R = R.dropna()
    blocks = np.array_split(np.arange(len(R)), n_blocks)
    logits = []
    for is_idx in itertools.combinations(range(n_blocks), n_blocks // 2):
        ii = np.concatenate([blocks[i] for i in is_idx])
        oo = np.concatenate([blocks[i] for i in range(n_blocks) if i not in is_idx])
        sr_is = R.iloc[ii].mean() / R.iloc[ii].std()
        sr_oo = R.iloc[oo].mean() / R.iloc[oo].std()
        best = sr_is.idxmax()
        w = (sr_oo.rank()[best]) / (len(sr_oo) + 1)
        logits.append(np.log(w / (1 - w)))
    logits = np.array(logits)
    return {"pbo": float((logits <= 0).mean()), "n_splits": len(logits), "n_strategies": R.shape[1],
            "logit_median": float(np.median(logits))}


def dsr_for(ls: pd.Series, stage_filter=None) -> dict:
    specs = load_specs()
    if stage_filter:
        specs = specs[specs["stage"].isin(stage_filter)]
    r = ls.dropna()
    st = perf_stats(r)
    return deflated_sharpe(r.mean() / r.std(), len(r), st["skew"], st["kurt"], specs["sr_m"].dropna().values)
