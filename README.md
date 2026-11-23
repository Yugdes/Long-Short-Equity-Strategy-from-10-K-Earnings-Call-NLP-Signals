# Reading the Risk Factors

**Do 10-K risk-factor disclosures and earnings-call marketing language predict U.S. stock returns?**

A point-in-time, walk-forward study combining two published text datasets with free market data:
- Astvansh & Simpson (2026, *MSOM*): transformer scores for 8 risk factors in Item 1A of 131,920 10-K filings.
- Damavandi, Mai & Astvansh (2025, *Marketing Letters*): SBERT scores for 19 marketing-emphasis components of earnings calls.

The pipeline builds a monthly stock panel in which every number is public before the month begins. It tests pre-registered hypotheses with portfolio sorts and Fama-MacBeth regressions, climbs a model ladder (elastic net, LightGBM, XGBoost, neural net, ensemble), and backtests a long-short strategy with costs, factor alphas, robustness, placebo and overfitting checks, and SHAP. The final 41 months (2022-01 → 2025-05) were locked until the design was frozen.

<!-- HEADLINE-RESULTS -->
**Headline results** (generated from `results/json` by `src/viz/export.py`):

| Measure | Result |
|---|---|
| Sample | 2,247 firms, 205,574 stock-months, median 1118 stocks per month (2010-01 to 2025-05) |
| Forecast quality (ensemble, all features): mean monthly rank IC | 0.041 (t = 3.6) in 2014–2021; 0.060 (t = 4.5) in the locked 2022-01 to 2025-05 test |
| Text over controls only (paired IC difference, 2014–2021) | 0.0066 (t = 1.6); test-window controls-only IC 0.064 |
| Primary strategy (capped-VW decile L/S, net of 25 bps): Sharpe | −0.12 (2014–2021); 0.51 (test) |
| Primary strategy: FF5 + momentum alpha, net | −4.9% a year (t = −0.9); test −4.6% a year (t = −0.8) |
| Turnover and breadth | buys plus sells in both legs equal 214% of one leg's capital per month (about 54% one-way per leg); 116 names per leg |
| Multiple testing | deflated Sharpe probability 0.08 over 123 development specs; probability of backtest overfitting 0.88 |
| Placebo and shuffled labels | IC p = 0.000, Sharpe p = 0.042; shuffled-label IC 0.0073 |
| Disclosed risk and future volatility | pre-registered test fails: operations risk t = 0.1, aggregate risk t = −1.0 (predicted positive); exploratory: 5 of 8 individual factors have t-statistics of 3 or more in size after controls (2 higher, 3 lower) |
| Survivorship audit (SEC filing histories) | our prices cover 26% of once-listed 10-K filers in 2006 and 80% in 2024; the low-risk decile loses the most firms, so missing firms would have to outperform by 1.7% a month to erase the aggregate-risk spread; finance-risk language predicts bankruptcy within 3 years (t = 8.6) |
| Replication of Astvansh & Simpson (2026), operations risk → next-year ln(COGS) | 0.289 (SE 0.107) on 34,960 XBRL firm-years vs 0.247 (0.041) in the paper; their risk interaction does not replicate (t = 0.4) |

**Bottom line:** the forecasts rank stocks better than chance, but the long-short strategy earns no significant alpha after standard factors and costs, and the text adds little beyond a controls-only model. Individual disclosed risk factors do carry information about future volatility, though not in the pre-registered form.
<!-- /HEADLINE-RESULTS -->

## How to run

```bash
# 1. environment (Python 3.10)
py -3.10 -m venv venv
venv\Scripts\python -m pip install -r requirements-lock.txt

# 2. everything, from raw data to website data (stages reuse their caches)
venv\Scripts\python run_all.py              # download → panel → features → tests → single → ml → analysis → studies → export → notebooks
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
src/evaluation/           single_signals.py, predictive.py, robustness.py, explain.py (SHAP), sanity.py, speclog.py,
                          survivorship.py (SEC-history audit), event_study.py (filing-date reactions), replication.py (source paper)
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
- `docs/DECISIONS.md`: every methodological choice with alternatives and rationale, including the audit of earlier work, the test unlock (D-027), a look-ahead bug caught through the test window (D-028), the independent reviewer's fixes (D-029), and the supplementary studies (D-031 to D-034).
- `docs/DATA_AUDIT.md`: inventory, coverage, identifiers and timing evidence for every input.

## Limitations (summary)
- **Survivorship.** Free price data only covers firms that still trade, and the bias is stronger in early years than late ones. An audit using SEC filing histories (which include the missing firms) shows that the bias works against, not for, the risk premium; it cannot score the machine-learning strategy's short leg.
- **Current, not historical, identifiers** (industry, ticker, exchange).
- **Size and book data start in 2009–2011** (XBRL).
- **Untested score validity:** the risk scores come from transformers trained on synthetic sentences with no human validation on real filings.

Full discussion: the Limitations section of the website.

## Authors
Developed by Yug Mitulkumar Desai and Arin Mehta as part of MS 499 (Independent Research). Supervisor: Professor Marcos Severo.

## Citation
Astvansh, V., & Simpson, J. J. (2026). A firm's operational risk: Data set and empirical evidence. *Manufacturing & Service Operations Management*, 28(1), 326–341.
Damavandi, H., Mai, F., & Astvansh, V. (2025). A new technique for measuring a firm's marketing emphasis. *Marketing Letters*, 36(4), 823–837.

Research and teaching use only; not investment advice.
