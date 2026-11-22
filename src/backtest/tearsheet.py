"""Strategy tear sheet for one score: sorts x weights x neutrality, costs, metrics, alphas, legs, benchmarks.

Portfolios are always formed over the full span of `d`, so turnover and costs are continuous. `window`
restricts only the statistics (e.g. dev vs locked test), never the portfolio history (D-029).
"""
import pandas as pd

from src.backtest.metrics import perf_stats, drawdown_series
from src.backtest.portfolios import sort_portfolios, ls_turnover, net_of_costs
from src.backtest.regressions import all_alphas, factor_alpha
from src.data.factors import load_ff_factors
from src.data.prices import build_monthly
from src.utils.config import load_config


def benchmarks() -> pd.DataFrame:
    ff = load_ff_factors().set_index("month")
    pm, _ = build_monthly()
    spy = pm[pm["ticker"] == "SPY"].set_index("month")["ret"]
    return pd.DataFrame({"mkt_excess": ff["mktrf"], "rf": ff["rf"], "spy_excess": spy - ff["rf"]})


def lowvol_factor(d: pd.DataFrame) -> pd.Series:
    """Low-minus-high 12-month-volatility decile spread (capped VW) inside this same universe: a factor that
    shares the sample's survivorship, so alpha against it isolates anything beyond the low-vol anomaly."""
    x = d.dropna(subset=["vol_12"]).assign(neg_vol=lambda z: -z["vol_12"])
    return sort_portfolios(x, "neg_vol", n=10, weight="vw")[0]["LS"].rename("lowvol")


def _win(s: pd.Series, window):
    if window is None:
        return s
    a, b = (pd.Period(w, "M") for w in window)
    return s[(s.index >= a) & (s.index <= b)]


def tearsheet(d: pd.DataFrame, score: str = "pred", window: tuple | None = None) -> dict:
    """d: feature rows (month, cik, ret, mcap_cap, vol_12, ff12) merged with `score`."""
    cfg = load_config()["portfolio"]
    ff = load_ff_factors()
    ffx = ff.merge(lowvol_factor(d).reset_index(), on="month", how="left")
    bm = benchmarks()
    variants, series, decile_bars = [], {}, {}
    for n in (10, 5):
        for w in ("vw", "ew"):
            for neutral in (None, "ff12"):
                tag = f"{'d' if n == 10 else 'q'}{n}_{w}{'_ff12' if neutral else ''}"
                rets, h = sort_portfolios(d, score, n=n, weight=w, neutral=neutral, min_names=cfg["min_names"])
                to = ls_turnover(h, n)
                ls_full = rets["LS"].dropna()
                ls = _win(ls_full, window)
                tov = to["total"].reindex(ls.index)
                if window is None or ls_full.index.min() >= ls.index.min():
                    tov = tov.iloc[1:]                       # skip the initial build of the very first month
                row = {"variant": tag, **perf_stats(ls), "turnover_m": tov.mean(),
                       "names_per_leg": h[(h["bin"] == n) & h["month"].isin(ls.index)].groupby("month").size().mean()}
                for bps in cfg["costs_bps"]:
                    net = _win(net_of_costs(ls_full, to["total"], bps), window)
                    st = perf_stats(net)
                    a = factor_alpha(net, ff, "FF5+Mom")
                    row.update({f"net{bps}_ann_mean": st["ann_mean"], f"net{bps}_sharpe": st["sharpe"],
                                f"net{bps}_alpha_ann": a["alpha_ann"], f"net{bps}_t_alpha": a["t_alpha"]})
                ga = factor_alpha(ls, ff, "FF5+Mom")
                gl = factor_alpha(ls, ffx, "FF5+Mom+LowVol")
                row.update({"alpha_ann": ga["alpha_ann"], "t_alpha": ga["t_alpha"],
                            "alpha_lv_ann": gl["alpha_ann"], "t_alpha_lv": gl["t_alpha"]})
                row["breakeven_bps"] = ls.mean() / row["turnover_m"] * 1e4 if row["turnover_m"] > 0 else float("nan")
                variants.append(row)
                series[tag] = {"gross": ls_full, "net25": net_of_costs(ls_full, to["total"], 25),
                               "long_ex": rets[f"P{n}"] - bm["rf"].reindex(rets.index),
                               "short_ex": rets["P1"] - bm["rf"].reindex(rets.index)}
                decile_bars[tag] = _win(rets[[f"P{i}" for i in range(1, n + 1)]], window).mean()
    prim = {k: _win(v, window) for k, v in series["d10_vw"].items()}
    alphas = pd.concat([all_alphas(prim["net25"], ffx).assign(series="LS net 25bps"),
                        all_alphas(prim["gross"], ffx).assign(series="LS gross"),
                        all_alphas(prim["long_ex"], ffx, excess=False).assign(series="long leg (excess)"),
                        all_alphas(prim["short_ex"], ffx, excess=False).assign(series="short leg (excess)")])
    return {"variants": pd.DataFrame(variants), "series": series, "decile_bars": decile_bars, "alphas": alphas,
            "benchmarks": bm, "drawdown": drawdown_series(series["d10_vw"]["net25"]), "lowvol": ffx.set_index("month")["lowvol"]}
