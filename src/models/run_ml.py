"""Walk-forward ML ladder on the development window (test years 2014-2021). D-023 sets the grid.

Usage: python -m src.models.run_ml [--horizon 1|3] [--final]   (--final needs the test window unlocked)
Out:   data/processed/preds/{set}__{model}__h{h}.parquet, results/tables/ml_summary_*.csv
"""
import argparse

import numpy as np
import pandas as pd

from src.backtest.portfolios import sort_portfolios
from src.backtest.regressions import factor_alpha
from src.backtest.metrics import perf_stats
from src.data.factors import load_ff_factors
from src.evaluation.predictive import summarize
from src.evaluation.speclog import log_spec
from src.features.signals import FEATURE_GROUPS as G
from src.models.learners import LEARNERS
from src.models.walkforward import run_walkforward
from src.utils.config import load_config, path

SETS = {
    "C": G["controls"],
    "C+R": G["controls"] + G["risk"],
    "C+R+M": G["controls"] + G["risk"] + G["marketing"],
    "ALL": G["controls"] + G["risk"] + G["marketing"] + G["interactions"],
    "TEXT": G["risk"] + G["marketing"] + G["interactions"],
}
PLAN = {"enet": list(SETS), "lgbm": list(SETS), "xgb": ["C", "ALL"], "mlp": ["C", "ALL"]}
PLAN_H3 = {"enet": ["ALL"], "lgbm": ["C", "ALL"]}          # 3-month horizon is a robustness check only (D-025)
PLAN_FINAL = {"enet": ["C", "ALL"], "lgbm": ["C", "ALL"], "xgb": ["C", "ALL"], "mlp": ["C", "ALL"]}


def evaluate(feat, preds, name, ff, stage="ml_dev", target="y_rank"):
    d = feat.merge(preds, on=["cik", "month"])
    s = summarize(d, "pred", target)
    for w in ("vw", "ew"):
        rets, _ = sort_portfolios(d, "pred", n=10, weight=w)
        a = factor_alpha(rets["LS"], ff, "FF5+Mom")
        st = perf_stats(rets["LS"])
        s.update({f"{w}_ls_mean_m": rets["LS"].mean(), f"{w}_sharpe": st["sharpe"],
                  f"{w}_alpha_ann": a["alpha_ann"], f"{w}_t_alpha": a["t_alpha"]})
        log_spec(stage, f"{name}|{w}|d10", rets["LS"], ic=s["ic_mean"], t_alpha=a["t_alpha"])
    return s


def main(horizon=1, final=False):
    cfg = load_config()
    feat = pd.read_parquet(path("processed") / "features.parquet")
    target = "y_rank" if horizon == 1 else "y_rank_3m"
    first, last = (cfg["sample"]["first_test_year"], cfg["sample"]["last_dev_year"]) if not final else (2022, 2025)
    out_dir = path("processed") / "preds"
    out_dir.mkdir(exist_ok=True)
    ff = load_ff_factors()
    rows = []
    plan = PLAN_FINAL if final else PLAN_H3 if horizon == 3 else PLAN
    for model, sets in plan.items():
        for s in sets:
            f = out_dir / f"{s}__{model}__h{horizon}{'_final' if final else ''}.parquet"
            if f.exists():
                pr = pd.read_parquet(f)
            else:
                print(f"[{model} | {s} | h={horizon}] {len(SETS[s])} features")
                pr = run_walkforward(feat, SETS[s], target, LEARNERS[model], first, last, horizon=horizon)
                if horizon > 1 and not final:   # D-029: a dev label must not read the locked window
                    pr = pr[pr["month"] <= pd.Period(f"{last}-12", "M") - (horizon - 1)]
                pr.to_parquet(f, index=False)
            rows.append({"set": s, "model": model, **evaluate(feat, pr, f"{s}|{model}|h{horizon}", ff,
                                                                "ml_final" if final else "ml_dev", target)})
    # ensembles: mean prediction of the 4 learners (all predict the same centred-rank target)
    for s in (("C", "ALL") if horizon == 1 else ()):
        ps = [pd.read_parquet(out_dir / f"{s}__{m}__h{horizon}{'_final' if final else ''}.parquet") for m in LEARNERS]
        # pre-registered equal-weight RANK ensemble (RESEARCH_DESIGN §3; corrected in D-029): each learner's
        # predictions are ranked within the month first, so no learner dominates through its scale
        ps = [p.assign(pred=p.groupby("month")["pred"].rank(pct=True) - 0.5) for p in ps]
        ens = pd.concat(ps).groupby(["cik", "month"], as_index=False)["pred"].mean()
        ens.to_parquet(out_dir / f"{s}__ens__h{horizon}{'_final' if final else ''}.parquet", index=False)
        row = {"set": s, "model": "ens", **evaluate(feat, ens, f"{s}|ens|h{horizon}", ff, "ml_final" if final else "ml_dev")}
        row["oos_r2_rank"] = float("nan")   # undefined for a rank ensemble (its scale is the target's, not a forecast's)
        rows.append(row)
    res = pd.DataFrame(rows)
    res.to_csv(path("results") / "tables" / f"ml_summary_{'final' if final else 'dev'}_h{horizon}.csv", index=False)
    print(res.round(4).to_string())
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizon", type=int, default=1)
    ap.add_argument("--final", action="store_true")
    a = ap.parse_args()
    main(a.horizon, a.final)
