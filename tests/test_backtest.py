import numpy as np
import pandas as pd
import statsmodels.api as sm

from src.backtest.metrics import perf_stats, max_drawdown, deflated_sharpe
from src.backtest.portfolios import sort_portfolios, ls_turnover, net_of_costs
from src.backtest.regressions import factor_alpha, fama_macbeth


def _toy():
    # 4 stocks, 2 months, quantiles n=2 (low = P1, high = P2)
    return pd.DataFrame({
        "month": pd.PeriodIndex(["2020-01"] * 4 + ["2020-02"] * 4, freq="M"),
        "cik": [1, 2, 3, 4, 1, 2, 3, 4],
        "score": [1, 2, 3, 4, 4, 2, 3, 1],
        "ret": [0.10, 0.00, 0.05, -0.10, 0.02, 0.04, 0.00, 0.06],
        "mcap_l": [100, 300, 100, 100, 100, 100, 300, 100.0],
    })


def test_ew_toy_returns():
    rets, _ = sort_portfolios(_toy(), "score", n=2, weight="ew", min_names=1)
    jan = rets.loc[pd.Period("2020-01", "M")]
    assert np.isclose(jan["P1"], 0.05) and np.isclose(jan["P2"], -0.025) and np.isclose(jan["LS"], -0.075)


def test_vw_toy_returns():
    rets, _ = sort_portfolios(_toy(), "score", n=2, weight="vw", min_names=1, size_col="mcap_l")
    jan = rets.loc[pd.Period("2020-01", "M")]
    assert np.isclose(jan["P1"], (100 * 0.10 + 300 * 0.0) / 400)
    assert np.isclose(jan["P2"], (100 * 0.05 - 100 * 0.10) / 200)


def test_double_sort_is_conditional():
    # 8 stocks, n=2: first sort on a, then b *within* each a-half
    from src.backtest.portfolios import double_sort
    d = pd.DataFrame({"month": pd.PeriodIndex(["2020-01"] * 8, freq="M"), "cik": range(8),
                      "a": [1, 2, 3, 4, 5, 6, 7, 8], "b": [8, 7, 6, 5, 1, 2, 3, 4],
                      "ret": [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08]})
    c = double_sort(d, "a", "b", n=2, min_names=1).iloc[0]
    # low a = {0..3}: low b = {2,3}, high b = {0,1}; high a = {4..7}: low b = {4,5}, high b = {6,7}
    assert np.isclose(c[(1, 1)], 0.035) and np.isclose(c[(1, 2)], 0.015)
    assert np.isclose(c[(2, 1)], 0.055) and np.isclose(c[(2, 2)], 0.075)
    assert double_sort(d, "a", "b", n=2, min_names=3).iloc[0].isna().all()


def test_turnover_and_costs_toy():
    rets, h = sort_portfolios(_toy(), "score", n=2, weight="ew", min_names=1)
    to = ls_turnover(h, n=2)
    # Jan: build both legs -> 1 + 1. Feb long leg {1,3}: drifted Jan long {3:1.05,4:0.9}/1.95 vs new {1:.5,3:.5}
    d3, d4 = 1.05 / 1.95, 0.90 / 1.95
    exp_long_feb = abs(0.5 - 0) + abs(0.5 - d3) + abs(0 - d4)
    assert np.isclose(to.loc[pd.Period("2020-01", "M"), "total"], 2.0)
    assert np.isclose(to.loc[pd.Period("2020-02", "M"), "long"], exp_long_feb)
    net = net_of_costs(rets["LS"], to["total"], bps=25)
    assert np.isclose(net.iloc[0], rets["LS"].iloc[0] - 2.0 * 0.0025)


def test_min_names_drops_thin_bins():
    rets, _ = sort_portfolios(_toy(), "score", n=2, weight="ew", min_names=3)
    assert rets.empty or rets["LS"].isna().all()


def test_perf_stats_known_series():
    r = pd.Series([0.1, -0.2, 0.05, 0.05])
    s = perf_stats(r)
    assert np.isclose(s["sharpe"], r.mean() / r.std(ddof=1) * np.sqrt(12))
    assert np.isclose(s["max_dd"], -0.2)          # wealth 1.10 -> 0.88


def test_factor_alpha_matches_statsmodels():
    rng = np.random.default_rng(0)
    months = pd.period_range("2010-01", periods=120, freq="M")
    f = pd.DataFrame({"month": months, "mktrf": rng.normal(0.01, 0.04, 120), "smb": rng.normal(0, 0.02, 120),
                      "hml": rng.normal(0, 0.02, 120), "rmw": rng.normal(0, 0.02, 120), "cma": rng.normal(0, 0.02, 120),
                      "mom": rng.normal(0, 0.03, 120), "rf": 0.001})
    r = pd.Series(0.002 + 0.5 * f["mktrf"].values + rng.normal(0, 0.01, 120), index=months)
    out = factor_alpha(r, f, "CAPM", nw_lags=6)
    ref = sm.OLS(r.values, sm.add_constant(f["mktrf"].values)).fit(cov_type="HAC", cov_kwds={"maxlags": 6})
    assert np.isclose(out["alpha_m"], ref.params[0]) and np.isclose(out["t_alpha"], ref.tvalues[0])


def test_fama_macbeth_recovers_slope():
    rng = np.random.default_rng(1)
    rows = []
    for m in pd.period_range("2015-01", periods=60, freq="M"):
        x = rng.normal(size=200)
        rows.append(pd.DataFrame({"month": m, "x": x, "y": 0.01 * x + rng.normal(0, 0.05, 200)}))
    fm = fama_macbeth(pd.concat(rows), "y", ["x"])
    assert abs(fm.loc["x", "coef"] - 0.01) < 0.003 and fm.loc["x", "t_nw"] > 3


def test_deflated_sharpe_penalises_many_trials():
    few = deflated_sharpe(0.2, 120, 0.0, 3.0, np.random.default_rng(0).normal(0, 0.05, 2))
    many = deflated_sharpe(0.2, 120, 0.0, 3.0, np.random.default_rng(0).normal(0, 0.05, 500))
    assert many["dsr"] < few["dsr"] and many["sr0_monthly"] > few["sr0_monthly"]


def test_pbo_low_for_one_dominant_strategy_high_for_noise():
    from src.evaluation.robustness import pbo_cscv
    pbos = [pbo_cscv(pd.DataFrame(np.random.default_rng(s).normal(0, 0.02, (160, 8))), n_blocks=8)["pbo"]
            for s in range(20)]
    assert 0.35 < np.mean(pbos) < 0.65                              # pure noise: ~0.5 on average
    noise = pd.DataFrame(np.random.default_rng(0).normal(0, 0.02, (160, 8)))
    good = noise.copy()
    good[0] += 0.01                                                 # one genuinely better strategy
    assert pbo_cscv(good, n_blocks=8)["pbo"] < 0.1
