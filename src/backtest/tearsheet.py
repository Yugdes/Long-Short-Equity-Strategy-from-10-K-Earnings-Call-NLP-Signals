"""Strategy tear sheet for one score: sorts x weights x neutrality, costs, metrics, alphas, legs, benchmarks."""
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


def tearsheet(d: pd.DataFrame, score: str = "pred") -> dict:
    """d: feature rows (month, cik, ret, mcap_cap, mcap_l, ff12) merged with `score`."""
    cfg = load_config()["portfolio"]
    ff = load_ff_factors()
    bm = benchmarks()
    variants, series, decile_bars = [], {}, {}
    for n in (10, 5):
        for w in ("vw", "ew"):
            for neutral in (None, "ff12"):
                tag = f"{'d' if n == 10 else 'q'}{n}_{w}{'_ff12' if neutral else ''}"
                rets, h = sort_portfolios(d, score, n=n, weight=w, neutral=neutral, min_names=cfg["min_names"])
                to = ls_turnover(h, n)
                ls = rets["LS"].dropna()
                row = {"variant": tag, **{k: v for k, v in perf_stats(ls).items()},
                       "turnover_m": to["total"].reindex(ls.index).iloc[1:].mean(),
                       "names_per_leg": h[h["bin"] == n].groupby("month").size().mean()}
                for bps in cfg["costs_bps"]:
                    net = net_of_costs(ls, to["total"], bps)
                    st = perf_stats(net)
                    a = factor_alpha(net, ff, "FF5+Mom")
                    row.update({f"net{bps}_ann_mean": st["ann_mean"], f"net{bps}_sharpe": st["sharpe"],
                                f"net{bps}_alpha_ann": a["alpha_ann"], f"net{bps}_t_alpha": a["t_alpha"]})
                gross_a = factor_alpha(ls, ff, "FF5+Mom")
                row.update({"alpha_ann": gross_a["alpha_ann"], "t_alpha": gross_a["t_alpha"]})
                # break-even one-way cost: gross mean / average traded fraction
                row["breakeven_bps"] = ls.mean() / row["turnover_m"] * 1e4 if row["turnover_m"] > 0 else float("nan")
                variants.append(row)
                series[tag] = {"gross": ls, "net25": net_of_costs(ls, to["total"], 25),
                               "long_ex": rets[f"P{n}"] - bm["rf"].reindex(rets.index),
                               "short_ex": rets["P1"] - bm["rf"].reindex(rets.index)}
                decile_bars[tag] = rets[[f"P{i}" for i in range(1, n + 1)]].mean()
    prim = series["d10_vw"]
    alphas = all_alphas(prim["net25"], ff).assign(series="LS net 25bps")
    legs = pd.concat([all_alphas(prim["long_ex"], ff).assign(series="long leg (excess)"),
                      all_alphas(prim["short_ex"], ff).assign(series="short leg (excess)")])
    return {"variants": pd.DataFrame(variants), "series": series, "decile_bars": decile_bars,
            "alphas": pd.concat([alphas, all_alphas(prim["gross"], ff).assign(series="LS gross"), legs]),
            "benchmarks": bm, "drawdown": drawdown_series(prim["net25"])}
