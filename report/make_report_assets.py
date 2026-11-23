"""Generate every number, table and figure used by report/main.tex from the pipeline outputs.

Reads only results/json/*.json and results/tables/*.csv (written by run_all.py), so the report can never disagree
with the pipeline. Writes
    report/generated/numbers.tex   \\newcommand macros for in-text numbers
    report/generated/tab_*.tex     table bodies (booktabs)
    report/figures/*.pdf           figures
Run:  python report/make_report_assets.py
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
J, T = ROOT / "results" / "json", ROOT / "results" / "tables"
OUT, FIG = ROOT / "report" / "generated", ROOT / "report" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)

FACTORS = ["accounting", "finance", "international", "legal", "management", "marketing", "operations", "technology"]
COL = {"accounting": "#0072b2", "finance": "#009e73", "international": "#56b4e9", "legal": "#cc79a7", "management": "#8a6a2c",
       "marketing": "#e69f00", "operations": "#d55e00", "technology": "#6a5acd"}
INK, TEAL, AMBER, RED, GREY = "#1b2a41", "#0e6b6b", "#a86a12", "#a3352b", "#8792a6"
plt.rcParams.update({"font.family": "serif", "font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": "#e3e6ea", "grid.linewidth": 0.6, "legend.frameon": False,
                     "savefig.bbox": "tight", "pdf.fonttype": 42})


def js(n):
    return json.loads((J / f"{n}.json").read_text(encoding="utf-8"))


def num(v, d=2, pct=False, sign=False):
    """LaTeX-safe number: math minus, optional percent (v is a fraction), n/a for missing."""
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "n/a"
    x = v * 100 if pct else v
    s = f"{x:+.{d}f}" if sign else f"{x:.{d}f}"
    s = s.replace("-", r"\ensuremath{-}")
    return s + (r"\%" if pct else "")


def intc(v):
    return f"{int(round(v)):,}".replace(",", "{,}")


MACROS = {}


def M(name, val):
    assert name.isalpha(), name
    MACROS[name] = val


def write_table(name, header, rows, colspec, panels=None):
    """rows: list of lists of strings; panels: {row_index: 'Panel title'} inserted before that row."""
    lines = [rf"\begin{{tabular}}{{{colspec}}}", r"\toprule", " & ".join(header) + r" \\", r"\midrule"]
    n = len(header)
    for i, r in enumerate(rows):
        if panels and i in panels:
            if i:
                lines.append(r"\midrule")
            lines.append(rf"\multicolumn{{{n}}}{{l}}{{\textit{{{panels[i]}}}}} \\")
        lines.append(" & ".join(r) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (OUT / f"tab_{name}.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


SIGNAME = {
    "r_operations_prob": "Operations risk", "r_ops_ind": "Operations risk, industry-demeaned", "r_risk_all": "Aggregate risk (8 factors)",
    "r_nonop_prob": "Non-operational risk (7 factors)", "r_dens_ops": "Operations count density",
    "r_d_operations_prob": r"$\Delta$ operations risk", "r_d_risk_all": r"$\Delta$ aggregate risk", "r_d_nonop_prob": r"$\Delta$ non-operational risk",
    "r_s_operations_prob": "Operations risk surprise", "r_mkt_capab": "Marketing capabilities", "r_mkt_excel": "Marketing excellence",
    "r_mkt_orient": "Market orientation", "r_dy_mkt_capab": r"$\Delta_{\text{YoY}}$ marketing capabilities",
    "r_dy_mkt_excel": r"$\Delta_{\text{YoY}}$ marketing excellence",
}
for f in FACTORS:
    SIGNAME.setdefault(f"r_{f}_prob", f"{f.capitalize()} risk")
SETNAME = {"C": "Controls only", "C+R": "+ risk text", "C+R+M": "+ marketing", "ALL": "+ interactions (all)", "TEXT": "Text only"}
MODNAME = {"enet": "Elastic net", "lgbm": "LightGBM", "xgb": "XGBoost", "mlp": "Neural net", "ens": "Rank ensemble"}
VARNAME = {"d10_vw": "Deciles, VW", "d10_vw_ff12": "Deciles, VW, sector-neutral", "d10_ew": "Deciles, EW", "d10_ew_ff12": "Deciles, EW, sector-neutral",
           "q5_vw": "Quintiles, VW", "q5_vw_ff12": "Quintiles, VW, sector-neutral", "q5_ew": "Quintiles, EW", "q5_ew_ff12": "Quintiles, EW, sector-neutral"}


def tstr(t, d=2):
    """t-statistic, bold if |t| >= 3 (the pre-registered hurdle)."""
    s = num(t, d)
    return rf"\textbf{{{s}}}" if t is not None and np.isfinite(t) and abs(t) >= 3 else s


# ------------------------------------------------------------------ summary numbers
sm, ex, st, mo, rb, sh, sd = (js(n) for n in ["summary", "explorer", "strategy", "models", "robustness", "shap", "studies"])
p = sm["primary"]
M("NFilings", intc(sm["n_filings"]))
M("NFilingsScored", intc(sm["n_filings_scored"]))
M("NFirmsScores", intc(sm["n_firms_scores"]))
M("NFeatures", str(sm["n_features_all"]))
M("NStockMonths", intc(sm["n_stock_months"]))
M("NFirmsPanel", intc(sm["n_firms_panel"]))
M("MedianNames", intc(sm["median_names_per_month"]))
M("TestMonths", str(sm["test_months"]))
M("DevMonths", str(sm["dev_months"]))
M("TestStart", sm["test_window"][0])
M("TestEnd", sm["test_window"][1])
M("DevIC", num(p["dev_ic_mean"], 3))
M("DevICt", num(p["dev_ic_t"], 2))
M("TestIC", num(p["test_ic_mean"], 3))
M("TestICt", num(p["test_ic_t"], 2))
M("TestICControls", num(p["test_ic_controls"], 3))
M("DevNetSharpe", num(p["dev_net_sharpe"], 2))
M("DevNetMean", num(p["dev_net_ann_mean"], 1, pct=True))
M("DevNetAlpha", num(p["dev_net_alpha_ann"], 1, pct=True))
M("DevNetAlphat", num(p["dev_net_t_alpha"], 2))
M("TestNetSharpe", num(p["test_net_sharpe"], 2))
M("TestGrossSharpe", num(p["test_gross_sharpe"], 2))
M("TestNetMean", num(p["test_net_ann_mean"], 1, pct=True))
M("TestNetMeant", num(p["test_t_mean"], 2))
M("TestNetAlpha", num(p["test_net_alpha_ann"], 1, pct=True))
M("TestNetAlphat", num(p["test_net_t_alpha"], 2))
M("Turnover", num(p["turnover_m"], 0, pct=True))
M("TurnoverOneWay", num(p["turnover_m"] / 4, 0, pct=True))
M("NamesPerLeg", intc(p["names_per_leg"]))
d = sm["dsr"]
M("DSR", num(d["dsr"], 2)); M("DSRTrials", str(d["n_trials"])); M("DSRsrzero", num(d["sr0_monthly"] * np.sqrt(12), 2))
b = sm["pbo"]
M("PBO", num(b["pbo"], 2)); M("PBOsplits", intc(b["n_splits"])); M("PBOstrategies", str(b["n_strategies"]))
pl = sm["placebo"]
M("PlaceboDraws", str(pl["n_draws"])); M("PlaceboPsharpe", num(pl["p_value_sharpe"], 3)); M("PlaceboPic", num(pl["p_value_ic"], 3))
M("PlaceboRealSharpe", num(pl["real_sharpe"], 2)); M("PlaceboNullNinetyFive", num(pl["null_sharpe_p95"], 2))
M("PlaceboNullICNinetyFive", num(pl["null_ic_p95"], 4))
sl = sm["shuffled_label"]
M("ShuffledIC", num(sl["ic_mean"], 4)); M("ShuffledICt", num(sl["ic_t_nw"], 2))
inc = sm["ic_increment"]
for k, nm in (("ens", "Ens"), ("enet", "Enet"), ("lgbm", "Lgbm")):
    M(f"Inc{nm}", num(inc[k]["diff_mean"], 4)); M(f"Inc{nm}t", num(inc[k]["diff_t_nw"], 2))
    M(f"Inc{nm}All", num(inc[k]["ic_all"], 4)); M(f"Inc{nm}Ctrl", num(inc[k]["ic_controls"], 4))
    M(f"Inc{nm}Share", num(inc[k]["months_text_better"], 0, pct=True))
td = ex["text_diag"]
M("OpsCountRho", num(td["ops_prob_count_spearman"], 2)); M("OpsWordsRho", num(td["ops_prob_words_spearman"], 2))
cc = pd.DataFrame(ex["corr_constructs"]).set_index("index")
offd = [cc.iloc[i, j] for i in range(3) for j in range(3) if i < j]
M("ConstructCorrLo", num(min(offd), 2)); M("ConstructCorrHi", num(max(offd), 2))
M("SpecCount", str(rb["spec_count"]))
fmf = {(r["forecast"], r["spec"]): r for r in mo["fm_forecast"]}
M("FMForecastt", num(fmf[("ALL|ens", "with_controls")]["t_nw"], 2))

# volatility
vol = pd.DataFrame(mo["vol_prediction_dev"])
vc = vol[vol["spec"] == "with_controls"].set_index("signal")
M("VolOpst", num(vc.loc["r_operations_prob", "t_nw"], 2)); M("VolAllt", num(vc.loc["r_risk_all", "t_nw"], 2))
f8 = vc.loc[[f"r_{f}_prob" for f in FACTORS], "t_nw"]
M("VolSigCount", str(int((f8.abs() >= 3).sum()))); M("VolUp", str(int((f8 >= 3).sum()))); M("VolDown", str(int((f8 <= -3).sum())))
for f in FACTORS:
    M(f"Vol{f.capitalize()}t", num(vc.loc[f"r_{f}_prob", "t_nw"], 1))

# single sorts / FM headline
ss = pd.DataFrame(mo["single_sorts_dev"])
fm = pd.DataFrame(mo["fama_macbeth_dev"])
fmc = fm[fm["spec"] == "with_controls"].set_index("signal")
vw = ss[ss["weight"] == "vw"].set_index("signal")
ew = ss[ss["weight"] == "ew"].set_index("signal")
M("RiskAllFMt", num(fmc.loc["r_risk_all", "t_nw"], 2)); M("RiskAllFMcoef", num(fmc.loc["r_risk_all", "coef"], 2, pct=True))
M("RiskAllVWmean", num(vw.loc["r_risk_all", "ls_mean_m"], 2, pct=True)); M("RiskAllVWt", num(vw.loc["r_risk_all", "t_mean"], 2))
M("RiskAllEWmean", num(ew.loc["r_risk_all", "ls_mean_m"], 2, pct=True)); M("RiskAllEWt", num(ew.loc["r_risk_all", "t_mean"], 2))
M("RiskAllVWalphat", num(vw.loc["r_risk_all", "t_alpha"], 2)); M("RiskAllEWalphat", num(ew.loc["r_risk_all", "t_alpha"], 2))
M("OpsVWmean", num(vw.loc["r_operations_prob", "ls_mean_m"], 2, pct=True)); M("OpsVWt", num(vw.loc["r_operations_prob", "t_mean"], 2))
M("OpsFMt", num(fmc.loc["r_operations_prob", "t_nw"], 2))
h1abs = vw.loc[vw["hyp"] == "H1", "t_alpha"].abs().max()
M("HoneMaxAbsAlphat", num(h1abs, 2))
h2 = pd.concat([ss[(ss["hyp"] == "H2") & (ss["weight"] == "vw")]["t_alpha"], fm[(fm["hyp"] == "H2") & (fm["spec"] == "with_controls")]["t_nw"]])
M("HtwoMinT", num(h2.min(), 2)); M("HtwoMaxAbsT", num(h2.abs().max(), 2)); M("HtwoN", str(len(h2)))
h3 = pd.concat([ss[(ss["hyp"] == "H3") & (ss["weight"] == "vw")].set_index("signal")["t_alpha"],
                fm[(fm["hyp"] == "H3") & (fm["spec"] == "with_controls")].set_index("signal")["t_nw"]])
M("HthreeMaxT", num(h3.max(), 2)); M("HthreeN", str(len(h3)))
M("HthreeBest", SIGNAME[h3.idxmax()].replace(r"$\Delta_{\text{YoY}}$", r"$\Delta_{\text{YoY}}$"))
allsorts_best = ss["t_alpha"].abs().max()
M("SingleSortMaxAbsAlphat", num(allsorts_best, 2)); M("SingleSortCount", str(len(ss)))

# H4
ix = pd.DataFrame(mo["interaction_fm_dev"])
M("HfourFMmax", num(ix["t_nw"].max(), 2))
ds = pd.DataFrame(mo["double_sort_dev"])
cu = ds[(ds["series"] == "cushion") & (ds["weight"] == "vw")]
lv, ch = cu[cu["risk"] == "r_operations_prob"]["t_alpha"], cu[cu["risk"] == "r_d_operations_prob"]["t_alpha"]
M("CushionLevelLo", num(lv.min(), 2)); M("CushionLevelHi", num(lv.max(), 2))
M("CushionChangeLo", num(ch.min(), 2)); M("CushionChangeHi", num(ch.max(), 2))
hs = ds[(ds["series"] == "mkt_spread_high_risk") & (ds["weight"] == "vw") & (ds["risk"] == "r_d_operations_prob")]
M("HighRiskSpreadLo", num(hs["mean_m"].min(), 2, pct=True)); M("HighRiskSpreadHi", num(hs["mean_m"].max(), 2, pct=True))
M("HighRiskSpreadTlo", num(hs["t_alpha"].min(), 1)); M("HighRiskSpreadThi", num(hs["t_alpha"].max(), 1))
lad = pd.DataFrame(mo["ladder_dev"])
dl = lambda s, m: lad[(lad["set"] == s) & (lad["model"] == m)]["ic_mean"].iloc[0]
M("HfourICEnet", num(dl("ALL", "enet") - dl("C+R+M", "enet"), 4)); M("HfourICLgbm", num(dl("ALL", "lgbm") - dl("C+R+M", "lgbm"), 4))
dg = {r["set"]: r for r in mo["diagnostics_dev"]}
M("DiagC", num(dl("C", "lgbm"), 4)); M("DiagFlag", num(dg["C+flag"]["ic_mean"], 4)); M("DiagCM", num(dg["C+M"]["ic_mean"], 4))
M("DiagFlagShare", num((dg["C+flag"]["ic_mean"] - dl("C", "lgbm")) / (dg["C+M"]["ic_mean"] - dl("C", "lgbm")), 0, pct=True))
M("LadderMinT", num(lad["ic_t_nw"].min(), 1)); M("LadderMaxT", num(lad["ic_t_nw"].max(), 1))

# SHAP
g = sh["groups"]; tot = sum(g.values())
for k, nm in (("marketing", "Mkt"), ("controls", "Ctrl"), ("risk", "Risk"), ("interactions", "Inter")):
    M(f"Shap{nm}", num(g[k] / tot, 0, pct=True))
M("ShapLeaves", str(sh["model"]["num_leaves"])); M("ShapTrees", str(sh["model"]["n_estimators"]))
M("ShapTop", sh["importance"][0]["feature"].replace("_", r"\_"))

# subperiods / size / alt lags
sp = {r["period"]: r for r in rb["subperiods"]}
M("SubAt", num(sp["2014-2017"]["t_alpha"], 2)); M("SubBt", num(sp["2018-2021"]["t_alpha"], 2))
M("CovidMean", num(sp["2020 (COVID)"]["ann_mean"], 1, pct=True))
sb = {r["bucket"]: r for r in rb["size_buckets"]}
M("SmallIC", num(sb["small"]["ic_mean"], 3)); M("LargeIC", num(sb["large"]["ic_mean"], 3)); M("MidIC", num(sb["mid"]["ic_mean"], 3))
M("SmallSharpe", num(sb["small"]["sharpe"], 2)); M("SmallLSt", num(sb["small"]["t_mean"], 2))
ee = rb["extended_ew"]["r_risk_all"]["stats"]
M("ExtRiskAllSharpe", num(ee["sharpe"], 2)); M("ExtRiskAllt", num(ee["t_mean_nw"], 2)); M("ExtRiskAllMean", num(ee["ann_mean"], 1, pct=True))
M("ExtRiskAllDD", num(ee["max_dd"], 0, pct=True))

# ------------------------------------------------------------------ supplementary studies
sv = sd["survivorship"]
cov = {r["year"]: r for r in sv["coverage"]}
M("CovTwoThousandSix", num(cov[2006]["covered"], 0, pct=True)); M("CovTwentyTwentyFour", num(cov[2024]["covered"], 0, pct=True))
fate = {r["fate"]: r["firms"] for r in sv["summary"]}
M("FateTotal", intc(sum(fate.values())))
for k, nm in (("survivor", "Surv"), ("merged", "Merged"), ("bankrupt", "Bankrupt"), ("exit_other", "Other")):
    M(f"Fate{nm}", intc(fate.get(k, 0)))
mdec = {(r["signal"], r["decile"]): r for r in sv["missing_decile"]}
M("MissLowRisk", num(mdec[("r_risk_all", 1)]["missing"], 0, pct=True)); M("MissHighRisk", num(mdec[("r_risk_all", 10)]["missing"], 0, pct=True))
be = {(r["signal"], r["weight"]): r for r in sv["breakeven"]}
M("BreakevenVW", num(be[("r_risk_all", "vw")]["breakeven_gap_m"], 1, pct=True))
M("BreakevenEW", num(be[("r_risk_all", "ew")]["breakeven_gap_m"], 1, pct=True))
M("BreakevenOpsVW", num(be[("r_operations_prob", "vw")]["breakeven_gap_m"], 1, pct=True))
lp = {(r["outcome"], r["signal"]): r for r in sv["fate_lpm"]}
M("BkFinancePP", num(lp[("bk_36m", "r_finance")]["coef"] * 100, 1)); M("BkFinancet", num(lp[("bk_36m", "r_finance")]["t"], 1))
M("BkOpsPP", num(lp[("bk_36m", "r_operations")]["coef"] * 100, 1)); M("BkOpst", num(lp[("bk_36m", "r_operations")]["t"], 1))
M("BkBase", num(lp[("bk_36m", "r_risk_all")]["base_rate"], 1, pct=True)); M("MgBase", num(lp[("merged_36m", "r_risk_all")]["base_rate"], 1, pct=True))
M("BkN", intc(lp[("bk_36m", "r_risk_all")]["n"]))
M("MgTecht", num(lp[("merged_36m", "r_technology")]["t"], 1))
ev = sd["event"]
es = ev["summary"][0]
M("EvN", intc(es["events"])); M("EvFirms", intc(es["firms"]))
efm = {(r["signal"], r["outcome"], r["spec"]): r for r in ev["fm"]}
M("EvLegalAbst", num(efm[("r_legal_prob", "abs_car01", "with_controls")]["t"], 1))
drift = [abs(r["t"]) for r in ev["fm"] if r["outcome"] == "car2_60" and r["spec"] == "with_controls"]
M("EvDriftMaxAbst", num(max(drift), 1))
M("EvMeanAbsCAR", num(es["mean_abs_car01"], 1, pct=True))
M("EvRiskAllCARt", num(efm[("r_risk_all", "car01", "with_controls")]["t"], 1))
rp = sd["replication"]
rs = rp["summary"][0]
rt = {(r["model"], r["var"]): r for r in rp["table"]}
M("RepN", intc(rs["n"])); M("RepFirms", intc(rs["firms"])); M("RepYearLo", str(rs["fyear_min"])); M("RepYearHi", str(rs["fyear_max"]))
M("RepOpFour", num(rt[("(4)", "op")]["coef"], 3)); M("RepOpFourSE", num(rt[("(4)", "op")]["se"], 3)); M("RepOpFourt", num(rt[("(4)", "op")]["t"], 2))
M("RepOpTwo", num(rt[("(2)", "op")]["coef"], 3)); M("RepOpTwoSE", num(rt[("(2)", "op")]["se"], 3))
M("RepInter", num(rt[("(4)", "op_x_nonop")]["coef"], 3)); M("RepIntert", num(rt[("(4)", "op_x_nonop")]["t"], 2))
M("RepSize", num(rt[("(4)", "ln_at")]["coef"], 3))
M("PaperN", intc(rs["paper_n"]))

# ------------------------------------------------------------------ tables
# coverage funnel
fun = pd.read_csv(T / "coverage_funnel.csv")
fun = fun[fun["file_year"].between(2006, 2024) & (fun["file_year"] % 2 == 0)]
write_table("funnel", ["Filing year", "10-K filings", "Non-empty Item 1A", "Current SEC ticker", "Yahoo prices", "Extended universe", "Primary universe", r"Primary / filings"],
            [[str(r.file_year), intc(r.filings), intc(r.nonempty_item1a), intc(r.current_sec_ticker), intc(r.yahoo_prices),
              intc(r.in_extended_universe), intc(r.in_primary_universe), num(r.pct_primary_of_filings, 0, pct=True)] for r in fun.itertuples()],
            "lrrrrrrr")

# single signals
rows, panels = [], {}
for hyp, title in (("H1", "H1: risk levels"), ("H2", "H2: risk changes"), ("H3", "H3: marketing emphasis"), ("explore_factors", "Exploratory: individual risk factors")):
    panels[len(rows)] = title
    for s in vw[vw["hyp"] == hyp].index:
        f_ = fmc.loc[s]
        rows.append([SIGNAME[s], num(vw.loc[s, "ls_mean_m"], 2, pct=True), tstr(vw.loc[s, "t_mean"]), num(vw.loc[s, "alpha_ann"], 1, pct=True),
                     tstr(vw.loc[s, "t_alpha"]), tstr(ew.loc[s, "t_alpha"]), num(f_["coef"] * 100, 2), tstr(f_["t_nw"])])
write_table("single", ["Signal", "VW L/S / mo", "$t$", r"VW $\alpha$ / yr", r"$t_\alpha$", r"EW $t_\alpha$", "FM slope (\%)", "FM $t$"], rows, "lrrrrrrr", panels)

# volatility
vols = ["r_operations_prob", "r_risk_all", "r_nonop_prob", "r_ops_ind", "r_dens_ops"] + [f"r_{f}_prob" for f in FACTORS if f != "operations"]
va = vol[vol["spec"] == "alone"].set_index("signal")
write_table("vol", ["Signal", "Slope alone", "$t$", "Slope with controls", "$t$"],
            [[SIGNAME[s], num(va.loc[s, "coef"], 3), tstr(va.loc[s, "t_nw"]), num(vc.loc[s, "coef"], 3), tstr(vc.loc[s, "t_nw"])] for s in vols],
            "lrrrr", {0: "Pre-registered (H1a)", 5: "Exploratory: individual factors"})

# H4
CON = {"mkt_orient": "market orientation", "mkt_capab": "marketing capabilities", "mkt_excel": "marketing excellence"}
def rlab(is_change, c):
    risk = r"$\Delta$ operations" if is_change else "Operations level"
    return risk + r" $\times$ " + CON[c]


rows = [[rlab(r["interaction"].startswith("x_dops"), r["interaction"].split("_", 2)[2]), num(r["coef"] * 100, 2), tstr(r["t_nw"]),
         tstr(r["risk_t"]), tstr(r["mkt_t"])] for r in mo["interaction_fm_dev"]]
write_table("hfourfm", ["Risk $\\times$ marketing", "Interaction slope (\\%)", "$t$", "Risk $t$", "Marketing $t$"], rows, "lrrrr")
rows = []
for r in cu.sort_values(["risk", "mkt"], ascending=[False, True]).itertuples():
    hr = ds[(ds["risk"] == r.risk) & (ds["mkt"] == r.mkt) & (ds["weight"] == "vw") & (ds["series"] == "mkt_spread_high_risk")].iloc[0]
    lr = ds[(ds["risk"] == r.risk) & (ds["mkt"] == r.mkt) & (ds["weight"] == "vw") & (ds["series"] == "mkt_spread_low_risk")].iloc[0]
    ew_ = ds[(ds["risk"] == r.risk) & (ds["mkt"] == r.mkt) & (ds["weight"] == "ew") & (ds["series"] == "cushion")].iloc[0]
    rows.append([rlab(r.risk == "r_d_operations_prob", r.mkt), num(hr["mean_m"], 2, pct=True), tstr(hr["t_alpha"]), num(lr["mean_m"], 2, pct=True),
                 num(r.mean_m, 2, pct=True), tstr(r.t_nw), tstr(r.t_alpha), tstr(ew_["t_alpha"])])
write_table("hfourds", ["Risk $\\times$ marketing", "High-risk spread", r"$t_\alpha$", "Low-risk spread", "Cushion", "$t$", r"$t_\alpha$", r"EW $t_\alpha$"],
            rows, "lrrrrrrr")

# ML ladder
def ladder(rows_):
    out = []
    for r in rows_:
        out.append([SETNAME.get(r["set"], r["set"]), MODNAME.get(r["model"], r["model"]), num(r["ic_mean"], 4), tstr(r["ic_t_nw"]),
                    num(r["ic_pos_months"], 0, pct=True), num(r["hit_rate"], 1, pct=True), num(r["vw_sharpe"], 2), tstr(r["vw_t_alpha"]), tstr(r["ew_t_alpha"])])
    return out
order = {"C": 0, "C+R": 1, "C+R+M": 2, "ALL": 3, "TEXT": 4}
mord = {"enet": 0, "lgbm": 1, "xgb": 2, "mlp": 3, "ens": 4}
ld = sorted(mo["ladder_dev"], key=lambda r: (order[r["set"]], mord[r["model"]]))
lf = sorted(mo["ladder_final"], key=lambda r: (order[r["set"]], mord[r["model"]]))
hdr = ["Features", "Model", "Rank IC", "$t$", "IC $>0$", "Hit rate", "VW Sharpe", r"VW $t_\alpha$", r"EW $t_\alpha$"]
write_table("ladder", hdr, ladder(ld) + ladder(lf), "llrrrrrrr",
            {0: f"Panel A: development, 2014--2021 ({sm['dev_months']} months)", len(ld): f"Panel B: locked test, {sm['test_window'][0]} to {sm['test_window'][1]} ({sm['test_months']} months)"})

# strategy variants
def variants(vs):
    out = []
    for r in vs:
        out.append([VARNAME[r["variant"]], num(r["ann_mean"], 1, pct=True), num(r["ann_vol"], 1, pct=True), num(r["sharpe"], 2), tstr(r["t_mean_nw"]),
                    num(r["max_dd"], 0, pct=True), num(r["turnover_m"], 0, pct=True), num(r["net25_sharpe"], 2), tstr(r["net25_t_alpha"]),
                    tstr(r["t_alpha_lv"]), num(r["breakeven_bps"], 0)])
    return out
write_table("variants", ["Construction", "Mean / yr", "Vol / yr", "Sharpe", "$t$", "Max DD", "Turnover", "Net Sharpe", r"Net $t_\alpha$", r"$t_\alpha$ +LowVol", "Break-even (bp)"],
            variants(st["variants"]) + variants(st["variants_test"]), "lrrrrrrrrrr",
            {0: "Panel A: development, 2014--2021", len(st["variants"]): f"Panel B: locked test, {sm['test_window'][0]} to {sm['test_window'][1]}"})

# alphas of the primary strategy
def alpha_rows(al):
    a = pd.DataFrame(al)
    out = []
    for m in ["CAPM", "FF3", "FF5", "FF5+Mom", "FF5+Mom+LowVol"]:
        row = [m.replace("+", " + ")]
        for s in ["LS gross", "LS net 25bps", "long leg (excess)", "short leg (excess)"]:
            x = a[(a["model"] == m) & (a["series"] == s)]
            row += [num(x["alpha_ann"].iloc[0], 1, pct=True), tstr(x["t_alpha"].iloc[0])] if len(x) else ["", ""]
        out.append(row)
    return out
write_table("alphas", ["Model", r"Gross $\alpha$", "$t$", r"Net $\alpha$", "$t$", r"Long $\alpha$", "$t$", r"Short $\alpha$", "$t$"],
            alpha_rows(st["alphas"]) + alpha_rows(st["alphas_test"]), "lrrrrrrrr",
            {0: "Panel A: development, 2014--2021", 5: f"Panel B: locked test, {sm['test_window'][0]} to {sm['test_window'][1]}"})

# robustness: subperiods, size, alt lags, h3, diagnostics
rows = [[r["period"].replace("test ", "Test, ").replace(" to ", "--"), str(r["months"]), num(r["ann_mean"], 1, pct=True), num(r["sharpe"], 2), tstr(r["t_mean"]),
         num(r["alpha_ann"], 1, pct=True) if r["alpha_ann"] is not None else "--", tstr(r["t_alpha"]) if r["t_alpha"] is not None else "--"] for r in rb["subperiods"]]
write_table("sub", ["Period", "Months", "Net mean / yr", "Sharpe", "$t$", r"$\alpha$ / yr", r"$t_\alpha$"], rows, "lrrrrrr")
rows = [[r["bucket"].capitalize(), intc(r["names_per_month"]), num(r["ic_mean"], 3), num(r["ls_q5_ew_ann"], 1, pct=True), num(r["sharpe"], 2), tstr(r["t_mean"])]
        for r in rb["size_buckets"]]
write_table("size", ["Size tercile", "Names / mo", "Mean IC", "EW Q5$-$Q1 / yr", "Sharpe", "$t$"], rows, "lrrrrr")
LAG = {"mkt_lag0": "Call scores one month earlier (no buffer)", "mkt_lag2": "Call scores one month later", "risk_lag10bd": "10-K scores 10 business days after filing"}
DIAG = {"C+flag": "Controls + call-coverage flag", "C+M_noflag": "Controls + marketing, no flag", "C+M": "Controls + marketing",
        "C+R+Mraw": "Controls + risk + raw marketing components"}
base = lad[(lad["set"] == "ALL") & (lad["model"] == "lgbm")].iloc[0]
rows = [["Baseline (LightGBM, all features)", num(base["ic_mean"], 4), tstr(base["ic_t_nw"]), num(base["hit_rate"], 1, pct=True)]]
rows += [[LAG[r["variant"]], num(r["ic_mean"], 4), tstr(r["ic_t_nw"]), num(r["hit_rate"], 1, pct=True)] for r in rb["alt_lags"]]
rows += [[DIAG[r["set"]], num(r["ic_mean"], 4), tstr(r["ic_t_nw"]), num(r["hit_rate"], 1, pct=True)] for r in mo["diagnostics_dev"]]
rows += [[f"3-month target: {SETNAME[r['set']]}, {MODNAME[r['model']]}", num(r["ic_mean"], 4), tstr(r["ic_t_nw"]), num(r["hit_rate"], 1, pct=True)] for r in rb["horizon3"]]
write_table("altspec", ["Specification (development OOS, 2014--2021)", "Rank IC", "$t$", "Hit rate"], rows, "lrrr",
            {0: "Panel A: timing variants", 4: "Panel B: feature diagnostics", 8: "Panel C: horizon"})
ext = rb["extended_ew"]
rows = [[SIGNAME[k], str(v["stats"]["n_months"]), num(v["stats"]["ann_mean"], 1, pct=True), num(v["stats"]["sharpe"], 2), tstr(v["stats"]["t_mean_nw"]),
         num(v["stats"]["max_dd"], 0, pct=True)] for k, v in ext.items()]
write_table("ext", ["Signal (EW decile L/S, 2006--2021)", "Months", "Mean / yr", "Sharpe", "$t$", "Max DD"], rows, "lrrrrr")

# survivorship
cv = pd.DataFrame(sv["coverage"])
cv = cv[cv["year"].between(2006, 2024) & (cv["year"] % 3 == 0)]
write_table("survcov", ["Filing year", "Filings", "Covered", "Still filing", "Acquired", "Bankrupt", "Other exit"],
            [[str(r.year), intc(r.filings), num(r.covered, 0, pct=True), num(r.missing_survivor, 0, pct=True), num(r.missing_merged, 0, pct=True),
              num(r.missing_bankrupt, 0, pct=True), num(r.missing_exit_other, 0, pct=True)] for r in cv.itertuples()], "lrrrrrr")
SV = {"r_risk_all": "Aggregate risk (8 factors)", "r_nonop": "Non-operational risk"} | {f"r_{f}": f"{f.capitalize()} risk" for f in FACTORS}
rows = []
for s in ["r_risk_all", "r_nonop"] + [f"r_{f}" for f in FACTORS]:
    row = [SV[s]]
    for o in ["bk_36m", "merged_36m", "exit_36m"]:
        row += [num(lp[(o, s)]["coef"] * 100, 1), tstr(lp[(o, s)]["t"])]
    rows.append(row)
write_table("survlpm", ["Risk measure", "Bankruptcy (pp)", "$t$", "Acquisition (pp)", "$t$", "Any exit (pp)", "$t$"], rows, "lrrrrrr")
rows = [[SIGNAME[r["signal"]], r["weight"].upper(), num(r["ls_mean_m"], 2, pct=True), num(r["missing_top"], 0, pct=True), num(r["missing_bottom"], 0, pct=True),
         num(r["breakeven_gap_m"], 1, pct=True)] for r in sv["breakeven"]]
write_table("survbe", ["Signal", "Weights", "Observed L/S / mo", "Missing, top decile", "Missing, bottom decile", "Break-even gap / mo"], rows, "llrrrr")

# event study
EVS = ["r_risk_all", "r_operations_prob", "r_nonop_prob", "r_d_risk_all", "r_d_operations_prob"] + [f"r_{f}_prob" for f in FACTORS if f != "operations"]
rows = []
for s in EVS:
    row = [SIGNAME[s]]
    for o in ["car01", "abs_car01", "car2_60", "dlogvol"]:
        r = efm[(s, o, "with_controls")]
        row += [num(r["coef"], 3) if o == "dlogvol" else num(r["coef"], 2, pct=True), tstr(r["t"])]
    rows.append(row)
write_table("event", ["Risk measure", "CAR$[0,1]$", "$t$", "$|$CAR$[0,1]|$", "$t$", "CAR$[2,60]$", "$t$", r"$\Delta\ln\sigma$", "$t$"], rows, "lrrrrrrrr")

# replication
VAR = {"ln_at": r"$\ln$(total assets)$_{y}$", "optional": "Item 1A optional", "ln_words": r"$\ln$(Item 1A words)", "rd_at": "R\\&D / assets",
       "op": "Operational risk", "nonop": "Non-operational risk", "op_x_nonop": r"Operational $\times$ non-operational"}
rows = []
for v in ["op", "nonop", "op_x_nonop", "ln_at", "optional", "ln_words", "rd_at"]:
    row = [VAR[v]]
    for m in ["(2)", "(4)"]:
        r = rt.get((m, v))
        if r is None:
            row += ["", "", ""]
            continue
        row += [num(r["coef"], 3) + f" ({num(r['se'], 3)})",
                (num(r["paper_coef"], 3) + f" ({num(r['paper_se'], 3)})") if r["paper_coef"] is not None else "", ""]
    rows.append([x for i, x in enumerate(row) if i not in (3, 6)])
rows.append(["Firm-years", intc(rs["n"]), intc(rs["paper_n"]), intc(rs["n"]), intc(rs["paper_n"])])
rows.append(["Firm and fiscal-year FE", "Yes", "Yes", "Yes", "Yes"])
write_table("rep", ["", "This study (2)", "Paper (2)", "This study (4)", "Paper (4)"], rows, "lrrrr")

# ------------------------------------------------------------------ figures
def shade(ax, x):
    if st["unlocked"]:
        xs = [pd.Period(v, "M").to_timestamp() for v in x]
        a, b_ = pd.Period(st["test_start"], "M").to_timestamp(), pd.Period(st["test_end"], "M").to_timestamp()
        ax.axvspan(a, b_, color="#eef0f3", zorder=0, lw=0)
        ax.text(a, ax.get_ylim()[1], " locked test", va="top", fontsize=7.5, color=GREY)


def ts(s):
    return [pd.Period(v, "M").to_timestamp() for v in s["x"]], s["y"]


fig, ax = plt.subplots(figsize=(6.4, 3.0))
for k, lab, c, lw, ls in (("cum_net25", "Long-short, net of 25 bp", INK, 1.8, "-"), ("cum_gross", "Long-short, gross", INK, 0.9, ":"),
                          ("cum_long_ex", "Long leg, excess", TEAL, 1.1, "-"), ("cum_short_ex", "Short leg (held long), excess", AMBER, 1.1, "-"),
                          ("cum_market_ex", "Market, excess", GREY, 1.0, "--")):
    x, y = ts(st[k]); ax.plot(x, y, color=c, lw=lw, ls=ls, label=lab)
ax.set_yscale("log"); ax.set_ylabel("Growth of \\$1 (log scale)")
ax.set_yticks([0.75, 1, 1.5, 2, 3]); ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:g}")); ax.yaxis.set_minor_formatter(plt.NullFormatter())
shade(ax, st["cum_net25"]["x"]); ax.legend(fontsize=7, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.1))
fig.savefig(FIG / "cum.pdf"); plt.close(fig)

fig, ax = plt.subplots(figsize=(6.4, 2.6))
for k, lab, c in (("ic_primary", "Ensemble, all features", INK), ("ic_controls", "Ensemble, controls only", TEAL)):
    x, y = ts(st[k]); r = pd.Series(y).rolling(12).mean()
    ax.plot(x, r, color=c, lw=1.4, label=lab)
ax.axhline(0, color=GREY, lw=0.8); ax.set_ylabel("12-month mean rank IC")
shade(ax, st["ic_primary"]["x"]); ax.legend(fontsize=7.5, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.12))
fig.savefig(FIG / "ic.pdf"); plt.close(fig)

fig, axs = plt.subplots(1, 2, figsize=(6.4, 2.4), sharey=False)
for ax, k, t_ in ((axs[0], "decile_bars", "Development, 2014--2021"), (axs[1], "decile_bars_test", "Locked test")):
    v = st[k]["d10_vw"]; ax.bar([f"{i+1}" for i in range(10)], v, color=[TEAL if x >= 0 else RED for x in v])
    ax.set_title(t_.replace("--", "–"), fontsize=8.5); ax.set_xlabel("Forecast decile (1 = lowest)"); ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
axs[0].set_ylabel("Mean return / yr (VW)")
fig.tight_layout(); fig.savefig(FIG / "deciles.pdf"); plt.close(fig)

fig, ax = plt.subplots(figsize=(4.6, 2.4))
ax.hist(rb["placebo_null_sharpes"], bins=30, color="#c9d0d4", edgecolor="white")
ax.axvline(rb["placebo_real_sharpe"], color=RED, lw=1.6, label="Actual forecast")
ax.set_xlabel("Gross annualised Sharpe ratio, development window"); ax.set_ylabel("Placebo draws"); ax.legend(fontsize=7.5)
fig.savefig(FIG / "placebo.pdf"); plt.close(fig)

imp = pd.DataFrame(sh["importance"]).head(20)[::-1]
gcol = {"controls": GREY, "risk": RED, "marketing": AMBER, "interactions": TEAL}
fig, ax = plt.subplots(figsize=(5.2, 3.8))
from matplotlib.patches import Patch
READ = {"rev_1": "1-month reversal", "vol_12": "12-month volatility", "illiq_12": "Amihud illiquidity", "gp": "Gross profitability",
        "beta_36": "CAPM beta", "has_mkt": "Call-coverage flag", "size": "Size", "ldvol_3": "Dollar volume", "bm": "Book-to-market",
        "mom_12_2": "Momentum 12-2", "ag": "Asset growth", "mkt_capab": "Marketing capabilities", "mkt_orient": "Market orientation",
        "mkt_excel": "Marketing excellence", "an_cust": "Analysts' customer orientation"}


def readable(f):
    f = f.replace("r_", "", 1)
    if f in READ:
        return READ[f]
    if f.startswith("zca_"):
        return "Mkt. component: " + f[4:].replace("_", " ").replace("customeorientation", "customer orientation")
    if f.endswith("_prob"):
        return f[:-5].replace("d_", "change in ").capitalize() + " risk"
    return f.replace("_", " ")


ax.barh([readable(f) for f in imp["feature"]], imp["mean_abs_shap"], color=[gcol[g_] for g_ in imp["group"]])
ax.set_xlabel("Mean |SHAP| (prediction units)"); ax.tick_params(axis="y", labelsize=7)
ax.xaxis.set_major_locator(plt.MaxNLocator(4))
ax.legend(handles=[Patch(color=c, label=g_.capitalize()) for g_, c in gcol.items()], fontsize=7.5, loc="lower right")
fig.savefig(FIG / "shap.pdf"); plt.close(fig)

fig, ax = plt.subplots(figsize=(5.2, 2.4))
tv = [vc.loc[f"r_{f}_prob", "t_nw"] for f in FACTORS]
ax.bar(FACTORS, tv, color=[COL[f] for f in FACTORS]); ax.axhline(3, color=GREY, ls="--", lw=0.8); ax.axhline(-3, color=GREY, ls="--", lw=0.8)
ax.axhline(0, color=INK, lw=0.6); ax.set_ylabel("Newey-West $t$ (with controls)"); ax.tick_params(axis="x", labelrotation=30, labelsize=7.5)
fig.savefig(FIG / "vol.pdf"); plt.close(fig)

cvf = pd.DataFrame(sv["coverage"]); cvf = cvf[cvf["year"].between(2006, 2024)]
fig, ax = plt.subplots(figsize=(6.4, 2.6))
bottom = np.zeros(len(cvf))
for k, lab, c in (("covered", "In our price data", TEAL), ("missing_survivor", "Missing, still filing", GREY), ("missing_merged", "Missing, acquired", AMBER),
                  ("missing_bankrupt", "Missing, bankrupt", RED), ("missing_exit_other", "Missing, other exit", "#d5dade")):
    ax.bar(cvf["year"], cvf[k], bottom=bottom, color=c, label=lab, width=0.8); bottom += cvf[k].values
ax.set_ylim(0, 1); ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}")); ax.set_xlabel("Filing year")
ax.set_xticks(range(2006, 2025, 2))
ax.legend(fontsize=7, ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.22))
fig.savefig(FIG / "survivorship.pdf"); plt.close(fig)

fig, axs = plt.subplots(1, 2, figsize=(6.4, 2.5))
yrs = ex["years"]
for f in FACTORS:
    axs[0].plot(yrs, ex["risk_by_year"][f], color=COL[f], lw=1.2, label=f)
axs[0].set_ylabel("Mean probability"); axs[0].set_title("Risk scores by filing year", fontsize=8.5)
axs[0].legend(fontsize=6.5, ncol=4, loc="upper center", bbox_to_anchor=(1.1, -0.12))
axs[1].plot(yrs, np.array(ex["words_by_year"]) / 1000, color=INK, lw=1.4); axs[1].set_ylabel("Median words (thousands)")
axs[1].set_title("Item 1A length", fontsize=8.5)
fig.tight_layout(); fig.savefig(FIG / "scores.pdf"); plt.close(fig)

# ------------------------------------------------------------------ write macros
(OUT / "numbers.tex").write_text("% generated by report/make_report_assets.py -- do not edit\n" +
                                 "\n".join(rf"\newcommand{{\{k}}}{{{v}}}" for k, v in sorted(MACROS.items())) + "\n", encoding="utf-8")
print(f"{len(MACROS)} macros, {len(list(OUT.glob('tab_*.tex')))} tables, {len(list(FIG.glob('*.pdf')))} figures")
