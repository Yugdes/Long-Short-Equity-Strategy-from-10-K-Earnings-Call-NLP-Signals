# Reading the Risk Factors

**Do 10-K risk-factor disclosures and earnings-call marketing language predict U.S. stock returns?**

A point-in-time, walk-forward study combining two published text datasets with free market data:
- Astvansh & Simpson (2026, *MSOM*): transformer scores for 8 risk factors in Item 1A of 131,920 10-K filings.
- Damavandi, Mai & Astvansh (2025, *Marketing Letters*): SBERT scores for 19 marketing-emphasis components of earnings calls.

The pipeline builds a monthly stock panel in which every number is public before the month begins. It tests pre-registered hypotheses with portfolio sorts and Fama-MacBeth regressions, climbs a model ladder (elastic net, LightGBM, XGBoost, neural net, ensemble), and backtests a long-short strategy with costs, factor alphas, robustness, placebo and overfitting checks, and SHAP. The final 41 months (2022-01 → 2025-05) were locked until the design was frozen.

<!-- HEADLINE-RESULTS -->

## How to run

```bash
# 1. environment (Python 3.10)
py -3.10 -m venv venv
venv\Scripts\python -m pip install -r requirements-lock.txt

# 2. everything, from raw data to website data (stages reuse their caches)
venv\Scripts\python run_all.py              # download → panel → features → tests → single → ml → analysis → export → notebooks
venv\Scripts\python run_all.py --from panel # rebuild every derived file from the frozen raw downloads
venv\Scripts\python -m pytest -q            # test suite (timing / look-ahead, prices, portfolios, metrics, walk-forward)

# 3. website (Vite + React + TypeScript + Plotly), reads results/json
cd website
npm install
npm run dev      # local server
npm run build    # static files in website/dist
```

Raw inputs:
- **In the repo:** the risk scores (`data/raw/risk/`) and marketing scores (`data/raw/marketing/`).
- **Not in the repo:** the WRDS CIK–GVKEY link table (`data/raw/cik_gvkey.csv`), which is licensed and gitignored.
- **Downloaded on first run, then frozen:** Yahoo prices (`data/raw/yahoo/`), SEC bulk XBRL and submissions (`data/raw/sec/`), and Ken French factors (`data/raw/french/`). Set the `SEC_USER_AGENT` environment variable to your own contact string before downloading from the SEC.

A full rebuild takes about 2–3 hours on an 8-core laptop, most of it the walk-forward models.

## Repository map

```
config/config.yaml        all parameters: timing rules, sample windows, universe, costs, test lock
run_all.py                one-command pipeline
src/data/                 risk.py, marketing.py (GVKEY→CIK links), sec.py (XBRL, SIC), prices.py (Yahoo),
                          factors.py (French), security_master.py
src/features/             timing.py (release rules), characteristics.py, panel.py (point-in-time panel), signals.py
src/models/               walkforward.py, learners.py, run_ml.py, diagnostics.py
src/backtest/             portfolios.py, metrics.py, regressions.py, tearsheet.py
src/evaluation/           single_signals.py, predictive.py, robustness.py, explain.py (SHAP), sanity.py, speclog.py
src/viz/export.py         analysis + JSON export for the website
src/utils/                config.py, lock.py (locked-test-window guard)
tests/                    pytest suite
notebooks/                01_eda … 06_explainability (executed; read pipeline outputs)
results/                  tables/, json/ (website data), spec_log.csv (multiple-testing log)
website/                  the presentation site
docs/                     PLAN, PROGRESS, DECISIONS, DATA_AUDIT, ENVIRONMENT, RESEARCH_DESIGN, LITERATURE
```

## Key documents
- `docs/RESEARCH_DESIGN.md`: hypotheses H1–H5 and the primary specification, written before any return was examined.
- `docs/DECISIONS.md`: every methodological choice with alternatives and rationale, including the audit of earlier work, the test unlock (D-027), a look-ahead bug caught through the test window (D-028), and the independent reviewer's fixes (D-029).
- `docs/DATA_AUDIT.md`: inventory, coverage, identifiers and timing evidence for every input.

## Limitations (summary)
- **Survivorship.** Free price data only covers firms that still trade, and the bias is stronger in early years than late ones.
- **Current, not historical, identifiers** (industry, ticker, exchange).
- **Size and book data start in 2009–2011** (XBRL).
- **Untested score validity:** the risk scores come from transformers trained on synthetic sentences with no human validation on real filings.

Full discussion: the Limitations section of the website.

## Authors
Developed by Yug Mitulkumar Desai and Arin Mehta as part of MS 499 (Independent Research).

## Citation
Astvansh, V., & Simpson, J. J. (2026). A firm's operational risk: Data set and empirical evidence. *Manufacturing & Service Operations Management*, 28(1), 326–341.
Damavandi, H., Mai, F., & Astvansh, V. (2025). A new technique for measuring a firm's marketing emphasis. *Marketing Letters*, 36(4), 823–837.

Research and teaching use only; not investment advice.
