"""Generate and execute the six narrated notebooks (they only read pipeline outputs and call src/).

    python scripts/make_notebooks.py            # writes notebooks/*.ipynb and executes them in place
"""
from pathlib import Path

import nbformat as nbf
from nbconvert.preprocessors import ExecutePreprocessor

ROOT = Path(__file__).resolve().parents[1]
NB = ROOT / "notebooks"
SETUP = """import sys, os
sys.path.insert(0, os.path.abspath('..'))
import pandas as pd, numpy as np, matplotlib.pyplot as plt
pd.set_option('display.width', 160); pd.set_option('display.max_columns', 30)
from src.utils.config import path
T = path('results') / 'tables'"""

BOOKS = {
"01_eda": [
("md", "# 01 · Exploratory data analysis\nWhat the two score datasets contain, how they link, and how much survives into the tradeable universe. Numbers are produced by `scripts/profile_raw_data.py` and the loaders in `src/data/`."),
("code", SETUP + "\nfrom src.data.risk import load_risk_scores, FACTORS\nfrom src.data.marketing import CONSTRUCTS\nrisk = load_risk_scores(); mk = pd.read_parquet(path('interim')/'marketing.parquet')\nprint(risk.shape, risk.cik.nunique(), mk.shape, mk.cik.nunique())"),
("md", "## Risk scores by filing year\n`FYEAR` in the source file equals the filing year for every row, so all timing keys on the filing date. 2005 filings mostly lack Item 1A (mandatory only from December 2005)."),
("code", "pd.read_csv(T/'data_audit'/'risk_by_fyear.csv', index_col=0).round(3)"),
("code", "fig, ax = plt.subplots(figsize=(9,4))\nr = risk[risk.words>0]\nfor f in FACTORS: ax.plot(r.groupby('file_year')[f+'_prob'].median(), label=f)\nax.set_title('Median transformer probability by filing year'); ax.legend(ncol=4, fontsize=8); plt.show()"),
("md", "## Marketing scores\nThe quarter is the call's calendar quarter (verified on IPO firms in DATA_AUDIT §3). The three constructs are nearly collinear."),
("code", "print(mk[list(CONSTRUCTS)].corr().round(3)); pd.read_csv(T/'marketing_link_stats.csv', index_col=0)"),
("md", "## Coverage funnel\nHow many filings per year reach the primary universe (current ticker, Yahoo prices, price ≥ $5, cap ≥ $100M, not a financial or utility)."),
("code", "pd.read_csv(T/'coverage_funnel.csv', index_col=0)"),
],
"02_text_diagnostics": [
("md", "# 02 · Text-signal diagnostics\nAre the transformer probabilities measuring what they claim? Correlation with the dictionary counts they were trained to imitate, dependence on text length, persistence across a firm's filings, and cross-factor correlation."),
("code", SETUP + "\nfrom src.data.risk import load_risk_scores, FACTORS\nr = load_risk_scores(); r = r[r.words >= 1000]"),
("code", "rows = []\nfor f in FACTORS:\n    rows.append({'factor': f, 'pearson_prob_count': r[f+'_prob'].corr(r[f+'_cnt']), 'spearman_prob_count': r[f+'_prob'].corr(r[f+'_cnt'], method='spearman'),\n                 'spearman_prob_words': r[f+'_prob'].corr(r.words, method='spearman'), 'spearman_density_prob': (r[f+'_cnt']/r.words).corr(r[f+'_prob'], method='spearman')})\nd = pd.DataFrame(rows).set_index('factor').round(3); d.to_csv(T/'text_diagnostics.csv'); d"),
("md", "Low probability–count correlations and a *negative* relation with length mean the probability behaves like a share of risk language, not an amount. Ranking within each month's cross-section (the design used throughout) removes the secular length trend."),
("code", "s = r.sort_values(['cik','filing_date']); g = s.groupby('cik')\npersist = {f: s[f+'_prob'].corr(g[f+'_prob'].shift(1)) for f in FACTORS}\npd.Series(persist, name='autocorr_consecutive_filings').round(3)"),
("code", "r[[f+'_prob' for f in FACTORS]].corr().round(2)"),
],
"03_panel_and_timing": [
("md", "# 03 · Point-in-time panel\nThe stock-month panel and the checks that it uses no future information. The same invariants are asserted in `tests/test_panel.py`."),
("code", SETUP + "\nfrom src.features.timing import risk_first_month\np = pd.read_parquet(path('processed')/'panel.parquet'); print(p.shape)"),
("code", "x = p[p.filing_date.notna()]\nlag = (pd.PeriodIndex(x.month).asi8 - pd.PeriodIndex(x.filing_month).asi8)\nprint('months since filing: min', lag.min(), 'max', lag.max())\nprint('all signals strictly before month start:', ((x.filing_date + pd.offsets.BDay(2)) < x.month.dt.to_timestamp()).all())"),
("code", "pd.read_csv(T/'sanity_factor_replication.csv')"),
("md", "Our own momentum and size long-shorts track Ken French's UMD and SMB closely, which validates the free price pipeline."),
("code", "pd.read_csv(T/'return_cleaning.csv', index_col=0)"),
("code", "e = p[p.elig]; e.groupby(e.month.dt.year).size().plot(kind='bar', figsize=(9,3), title='Primary-universe stock-months per year'); plt.show()"),
],
"04_models": [
("md", "# 04 · Models\nThe ladder from single-signal sorts to the ensemble, development window only."),
("code", SETUP),
("code", "pd.read_csv(T/'single_signal_sorts_dev.csv').round(4)"),
("code", "pd.read_csv(T/'fama_macbeth_dev.csv').round(4)"),
("code", "pd.read_csv(T/'vol_prediction_dev.csv').round(4)"),
("code", "pd.read_csv(T/'ml_summary_dev_h1.csv').round(4)"),
("code", "pd.read_csv(T/'ml_summary_dev_h3.csv').round(4)"),
("code", "import os\nf = T/'ml_summary_final_h1.csv'\npd.read_csv(f).round(4) if os.path.exists(f) else 'test window still locked'"),
],
"05_backtest": [
("md", "# 05 · Strategy tear sheet\nThe pre-specified primary strategy: ensemble forecast, value-weighted (capped) decile long-short, net of 25 bps."),
("code", SETUP + "\nimport pickle\nres = pickle.load(open(path('results')/'analysis'/'analysis.pkl','rb')); tear = res['tear']"),
("code", "tear['variants'].set_index('variant').round(4).T"),
("code", "tear['alphas'].round(4)"),
("code", "s = tear['series']['d10_vw']\nfig, ax = plt.subplots(figsize=(9,4))\nfor k in ['gross','net25','long_ex','short_ex']: ax.plot(s[k].index.to_timestamp(), (1+s[k].fillna(0)).cumprod(), label=k)\nax.set_yscale('log'); ax.legend(); ax.set_title('Growth of $1'); plt.show()"),
("code", "res['subperiods'].round(3)"),
],
"06_explainability": [
("md", "# 06 · Robustness and explainability"),
("code", SETUP + "\nimport pickle\nres = pickle.load(open(path('results')/'analysis'/'analysis.pkl','rb'))"),
("code", "print('PBO', res['pbo']); print('DSR', res['dsr']); print({k: v for k, v in res['placebo'].items() if not hasattr(v, '__len__')}); print('shuffled labels', res['shuffled_label'])"),
("code", "res['size_buckets'].round(4)"),
("code", "res['industry_ic'].round(4)"),
("code", "res['alt_lags'].round(4)"),
("code", "sh = res['shap']; print(sh['groups'].round(5)); sh['importance'].head(20).round(5)"),
("code", "fig, axs = plt.subplots(2, 3, figsize=(11,6))\nfor ax, (f, d) in zip(axs.flat, sh['dependence'].items()): ax.plot(d['x'], d['shap'], marker='o'); ax.set_title(f, fontsize=9)\nplt.tight_layout(); plt.show()"),
],
}

if __name__ == "__main__":
    NB.mkdir(exist_ok=True)
    for name, cells in BOOKS.items():
        nb = nbf.v4.new_notebook()
        nb.cells = [nbf.v4.new_markdown_cell(c) if k == "md" else nbf.v4.new_code_cell(c) for k, c in cells]
        ExecutePreprocessor(timeout=1800, kernel_name="python3").preprocess(nb, {"metadata": {"path": str(NB)}})
        nbf.write(nb, NB / f"{name}.ipynb")
        print("executed", name)
