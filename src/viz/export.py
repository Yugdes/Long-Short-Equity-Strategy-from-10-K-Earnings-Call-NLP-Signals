"""Phase 5-6 analysis and website data export. Every number on the site comes from here.

run_analysis(): tear sheet of the pre-specified primary strategy, robustness, placebo, overfitting checks,
                SHAP -> results/analysis/*.pkl
export_json():  results/json/*.json (website reads only these)
"""
import json
import pickle

import numpy as np
import pandas as pd

from src.backtest.tearsheet import tearsheet
from src.data.factors import load_ff_factors
from src.data.marketing import CONSTRUCTS
from src.data.risk import FACTORS, load_risk_scores
from src.evaluation import robustness as rb
from src.evaluation.explain import shap_analysis
from src.evaluation.predictive import ic_series, summarize
from src.models.learners import LEARNERS
from src.models.run_ml import SETS
from src.models.walkforward import run_walkforward
from src.utils.config import load_config, path
from src.utils.lock import dev_only, is_unlocked

PRIMARY = ("ALL", "ens")          # RESEARCH_DESIGN §3: ensemble, all features, VW decile, net 25 bps


def _adir():
    d = path("results") / "analysis"
    d.mkdir(exist_ok=True)
    return d


def _preds(s, m, h=1, final=False):
    f = path("processed") / "preds" / f"{s}__{m}__h{h}{'_final' if final else ''}.parquet"
    return pd.read_parquet(f)


def _feat(name="features.parquet"):
    return pd.read_parquet(path("processed") / name)


def _oos(feat, s, m, h=1):
    """Dev OOS predictions; once unlocked, the final-test predictions are appended."""
    p = _preds(s, m, h)
    if is_unlocked():
        p = pd.concat([p, _preds(s, m, h, final=True)])
    return feat.merge(p, on=["cik", "month"])


def alt_lag_runs(variants: dict) -> pd.DataFrame:
    """Rebuild the panel under alternative timing, then rerun LGBM-ALL walk-forward (dev window)."""
    from src.features.panel import build_panel
    from src.features.signals import make_features
    cfg = load_config()
    rows = []
    for name, timing in variants.items():
        f = path("processed") / f"features_{name}.parquet"
        if not f.exists():
            p = build_panel(force=True, timing=timing, out_name=f"panel_{name}.parquet")
            make_features(p, "elig").to_parquet(f, index=False)
        ft = pd.read_parquet(f)
        pf = path("processed") / "preds" / f"ALL__lgbm__h1_{name}.parquet"
        if not pf.exists():
            run_walkforward(ft, SETS["ALL"], "y_rank", LEARNERS["lgbm"], cfg["sample"]["first_test_year"],
                            cfg["sample"]["last_dev_year"]).to_parquet(pf, index=False)
        d = ft.merge(pd.read_parquet(pf), on=["cik", "month"])
        rows.append({"variant": name, **summarize(d)})
    return pd.DataFrame(rows)


def shuffled_label_test(feat: pd.DataFrame, seed: int = 0) -> dict:
    """Retrain LGBM-ALL on targets shuffled within each month: the OOS IC must collapse to ~0."""
    cfg = load_config()
    pf = path("processed") / "preds" / "ALL__lgbm__h1_shuffled.parquet"
    rng = np.random.default_rng(seed)
    f = feat.copy()
    f["y_shuf"] = f.groupby("month")["y_rank"].transform(lambda x: rng.permutation(x.values))
    if not pf.exists():
        run_walkforward(f, SETS["ALL"], "y_shuf", LEARNERS["lgbm"], cfg["sample"]["first_test_year"],
                        cfg["sample"]["last_dev_year"]).to_parquet(pf, index=False)
    d = feat.merge(pd.read_parquet(pf), on=["cik", "month"])
    return summarize(d)


def run_analysis():
    ff = load_ff_factors()
    feat = _feat()
    res = {}
    # ---- primary strategy + model ladder comparison (dev; + test after unlock) ----
    d = _oos(feat, *PRIMARY)
    res["tear"] = tearsheet(d)
    res["ic_primary"] = ic_series(d, "pred", "y_rank")
    res["ic_controls"] = ic_series(_oos(feat, "C", "ens"), "pred", "y_rank")
    # H5: paired test of the IC added by text features, same rows and months (dev window)
    from src.backtest.metrics import nw_tstat
    inc = {}
    for m in ("ens", "enet", "lgbm"):
        a = ic_series(_oos(feat, "ALL", m), "pred", "y_rank")
        b = ic_series(_oos(feat, "C", m), "pred", "y_rank")
        diff = dev_only((a - b).dropna().to_frame("d"), None)["d"]
        inc[m] = {"ic_all": float(dev_only(a.to_frame("a"), None)["a"].mean()), "ic_controls": float(dev_only(b.to_frame("b"), None)["b"].mean()),
                  "diff_mean": float(diff.mean()), "diff_t_nw": nw_tstat(diff), "months_text_better": float((diff > 0).mean())}
    res["ic_increment"] = inc
    # does the forecast predict returns beyond the 10 characteristics? (FM, dev window)
    from src.backtest.regressions import fama_macbeth
    from src.features.signals import FEATURE_GROUPS
    rows = []
    for s_, m in (("ALL", "ens"), ("C", "ens"), ("ALL", "lgbm"), ("TEXT", "lgbm")):
        dd = dev_only(_oos(feat, s_, m))
        dd["r_pred"] = dd.groupby("month")["pred"].rank(pct=True) - 0.5
        for label, xs in (("alone", ["r_pred"]), ("with_controls", ["r_pred"] + FEATURE_GROUPS["controls"])):
            fm = fama_macbeth(dd, "exret", xs)
            rows.append({"forecast": f"{s_}|{m}", "spec": label, "coef": fm.loc["r_pred", "coef"], "t_nw": fm.loc["r_pred", "t_nw"]})
    res["fm_forecast"] = pd.DataFrame(rows)
    res["subperiods"] = rb.subperiod_table(res["tear"]["series"]["d10_vw"]["net25"], ff)
    res["size_buckets"] = rb.size_buckets(dev_only(d))
    res["industry_ic"] = rb.industry_ic(dev_only(d))
    placebo = rb.placebo_shuffle(dev_only(d), n_draws=500)
    res["placebo"] = placebo
    res["shuffled_label"] = shuffled_label_test(feat)
    # PBO across the ML grid (dev, VW decile LS gross)
    from src.backtest.portfolios import sort_portfolios
    R = {}
    for f_ in sorted((path("processed") / "preds").glob("*__h1.parquet")):
        s, m, _ = f_.stem.split("__")
        dd = feat.merge(pd.read_parquet(f_), on=["cik", "month"])
        R[f"{s}|{m}"] = sort_portfolios(dev_only(dd), "pred", n=10, weight="vw")[0]["LS"]
    res["pbo"] = rb.pbo_cscv(pd.DataFrame(R), n_blocks=16)
    res["dsr"] = rb.dsr_for(dev_only(res["tear"]["series"]["d10_vw"]["net25"].to_frame("r"), None)["r"],
                            stage_filter=["single_sort", "ml_dev"])
    res["alt_lags"] = alt_lag_runs({"mkt_lag0": {"mkt_buffer_months": 0}, "mkt_lag2": {"mkt_buffer_months": 2},
                                    "risk_lag10bd": {"risk_lag_bdays": 10}})
    res["horizon3"] = pd.read_csv(path("results") / "tables" / "ml_summary_dev_h3.csv")
    # extended EW sample incl. 2008-09: single-signal sorts only (no size data before XBRL)
    from src.backtest.portfolios import sort_portfolios as sp
    fe = _feat("features_ext.parquet")
    ext = {}
    for s in ["r_nonop_prob", "r_risk_all", "r_operations_prob", "r_d_operations_prob"]:
        raw = s[2:] if s[2:] in fe.columns else "operations_prob"
        ext[s] = sp(dev_only(fe[fe[raw].notna()]), s, n=10, weight="ew")[0]["LS"]
    res["extended_ew"] = ext
    res["shap"] = shap_analysis(feat)
    with open(_adir() / "analysis.pkl", "wb") as fh:
        pickle.dump(res, fh)
    return res


# ------------------------------------------------------------------ JSON export
def _ser(s: pd.Series, digits=6):
    s = s.dropna()
    return {"x": [str(i) for i in s.index], "y": [round(float(v), digits) for v in s.values]}


def _tab(df: pd.DataFrame, digits=5):
    out = df.replace([np.inf, -np.inf], np.nan).copy()
    for c in out.select_dtypes("number").columns:
        out[c] = out[c].round(digits)
    return json.loads(out.to_json(orient="records"))


def _write(name, obj):
    d = path("results") / "json"
    d.mkdir(exist_ok=True)
    (d / f"{name}.json").write_text(json.dumps(obj, indent=1, default=str), encoding="utf-8")


def export_json():
    with open(_adir() / "analysis.pkl", "rb") as fh:
        res = pickle.load(fh)
    cfg = load_config()
    T = path("results") / "tables"
    feat = _feat()
    tear = res["tear"]
    prim = tear["series"]["d10_vw"]
    bm = tear["benchmarks"]
    var = tear["variants"].set_index("variant")
    dev_net = dev_only(prim["net25"].to_frame("r"), None)["r"]
    pv = var.loc["d10_vw"]
    ml_dev = pd.read_csv(T / "ml_summary_dev_h1.csv")
    prim_dev = ml_dev[(ml_dev["set"] == PRIMARY[0]) & (ml_dev["model"] == PRIMARY[1])].iloc[0]
    from src.backtest.metrics import perf_stats
    from src.backtest.regressions import factor_alpha
    dev_stats = perf_stats(dev_net)
    dev_alpha = factor_alpha(dev_net, load_ff_factors(), "FF5+Mom")

    # --- summary / hero ---
    risk = load_risk_scores()
    summary = {
        "unlocked": is_unlocked(), "test_window": [cfg["test"]["start"], cfg["sample"]["end"]],
        "dev_window": ["2014-01", "2021-12"],
        "n_filings": int(len(risk)), "n_firms_scores": int(risk["cik"].nunique()),
        "n_stock_months": int(len(feat)), "n_firms_panel": int(feat["cik"].nunique()),
        "median_names_per_month": float(feat.groupby("month").size().median()),
        "primary": {"label": "Ensemble, all features, value-weighted top-minus-bottom decile, net of 25 bps",
                    "dev_ic_mean": prim_dev["ic_mean"], "dev_ic_t": prim_dev["ic_t_nw"],
                    "dev_net_sharpe": dev_stats["sharpe"], "dev_net_ann_mean": dev_stats["ann_mean"],
                    "dev_net_alpha_ann": dev_alpha["alpha_ann"], "dev_net_t_alpha": dev_alpha["t_alpha"],
                    "turnover_m": pv["turnover_m"], "names_per_leg": pv["names_per_leg"]},
        "dsr": res["dsr"], "pbo": res["pbo"],
        "placebo": {k: v for k, v in res["placebo"].items() if not isinstance(v, np.ndarray)},
        "shuffled_label": res["shuffled_label"],
        "ic_increment": res["ic_increment"],
    }
    if is_unlocked():
        test_net = prim["net25"][prim["net25"].index >= pd.Period(cfg["test"]["start"], "M")]
        st = perf_stats(test_net)
        summary["primary"].update({"test_net_sharpe": st["sharpe"], "test_net_ann_mean": st["ann_mean"],
                                   "test_months": st["n_months"], "test_t_mean": st["t_mean_nw"]})
        ic = res["ic_primary"]
        summary["primary"]["test_ic_mean"] = float(ic[ic.index >= pd.Period(cfg["test"]["start"], "M")].mean())
    _write("summary", summary)

    # --- dataset explorer ---
    r = risk[risk["words"] > 0].copy()
    r = r[r["file_year"].between(2006, 2024)]
    from src.data.security_master import build_security_master
    from src.data.factors import load_sic_map
    sm = build_security_master()
    r = r.merge(sm[["cik", "ff12"]], on="cik", how="left")
    names12 = load_sic_map("sic12").drop_duplicates("ind").set_index("ind")["ind_name"].to_dict()
    names12[12] = "Other"
    by_year = r.groupby("file_year")[[f"{f}_prob" for f in FACTORS]].median()
    by_ind = r[r["ff12"] > 0].groupby("ff12")[[f"{f}_prob" for f in FACTORS]].median()
    mk = pd.read_parquet(path("interim") / "marketing.parquet")
    mk_year = mk[mk["year"].between(2004, 2025)].groupby("year")[list(CONSTRUCTS)].mean()
    words = r.groupby("file_year")["words"].median()
    _write("explorer", {
        "factors": FACTORS, "years": [int(y) for y in by_year.index],
        "risk_by_year": {f: [round(v, 4) for v in by_year[f"{f}_prob"]] for f in FACTORS},
        "industries": [names12.get(int(i), str(i)) for i in by_ind.index],
        "risk_by_industry": {f: [round(v, 4) for v in by_ind[f"{f}_prob"]] for f in FACTORS},
        "words_by_year": [int(v) for v in words.values],
        "marketing_years": [int(y) for y in mk_year.index],
        "marketing_by_year": {c: [round(v, 4) for v in mk_year[c]] for c in CONSTRUCTS},
        "filings_by_year": [int(v) for v in r.groupby("file_year").size().values],
        "coverage": _tab(pd.read_csv(T / "coverage_funnel.csv")),
        "corr_constructs": _tab(mk[list(CONSTRUCTS)].corr().round(3).reset_index()),
    })

    # --- results ---
    cum = lambda s: (1 + s.fillna(0)).cumprod()
    idx = prim["gross"].index
    _write("strategy", {
        "cum_gross": _ser(cum(prim["gross"])), "cum_net25": _ser(cum(prim["net25"])),
        "cum_long_ex": _ser(cum(prim["long_ex"].reindex(idx))), "cum_short_ex": _ser(cum(prim["short_ex"].reindex(idx))),
        "cum_market_ex": _ser(cum(bm["mkt_excess"].reindex(idx))),
        "drawdown": _ser(tear["drawdown"]),
        "decile_bars": {k: [round(float(v) * 12, 5) for v in s.values] for k, s in tear["decile_bars"].items()},
        "variants": _tab(tear["variants"]),
        "alphas": _tab(tear["alphas"]),
        "ic_primary": _ser(res["ic_primary"]), "ic_controls": _ser(res["ic_controls"]),
        "test_start": cfg["test"]["start"], "unlocked": is_unlocked(),
    })
    _write("models", {
        "ladder_dev": _tab(ml_dev),
        "single_sorts_dev": _tab(pd.read_csv(T / "single_signal_sorts_dev.csv")),
        "fama_macbeth_dev": _tab(pd.read_csv(T / "fama_macbeth_dev.csv")),
        "vol_prediction_dev": _tab(pd.read_csv(T / "vol_prediction_dev.csv")),
        "interaction_fm_dev": _tab(pd.read_csv(T / "interaction_fm_dev.csv")),
        "fm_forecast": _tab(res["fm_forecast"]),
        "diagnostics_dev": _tab(pd.read_csv(T / "ml_diagnostics_dev.csv")),
        **({"ladder_final": _tab(pd.read_csv(T / "ml_summary_final_h1.csv"))}
           if (T / "ml_summary_final_h1.csv").exists() else {}),
    })
    _write("robustness", {
        "subperiods": _tab(res["subperiods"]), "size_buckets": _tab(res["size_buckets"]),
        "industry_ic": _tab(res["industry_ic"].assign(name=lambda x: x["ff12"].map(names12))),
        "alt_lags": _tab(res["alt_lags"]), "horizon3": _tab(res["horizon3"]),
        "placebo_null_sharpes": [round(float(v), 4) for v in res["placebo"]["null_sharpes"]],
        "placebo_real_sharpe": res["placebo"]["real_sharpe"],
        "extended_ew": {k: {"cum": _ser(cum(v)), "stats": perf_stats(v)} for k, v in res["extended_ew"].items()},
        "spec_count": int(len(pd.read_csv(path("results") / "spec_log.csv"))),
    })
    sh = res["shap"]
    _write("shap", {
        "importance": _tab(sh["importance"].head(25)),
        "groups": {k: round(float(v), 6) for k, v in sh["groups"].items()},
        "dependence": {k: {"x": [round(float(v), 4) for v in g["x"]], "shap": [round(float(v), 6) for v in g["shap"]]}
                       for k, g in sh["dependence"].items()},
        "model": sh["chosen"],
    })
