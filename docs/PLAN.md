# Implementation Plan: NLP-signal long-short equity research pipeline

> **For agentic workers:** executed natively with superpowers:executing-plans (see D-017). Steps use checkbox (`- [ ]`) syntax for tracking. Progress is mirrored in `docs/PROGRESS.md`.

**Goal:** Measure, with point-in-time discipline, whether 10-K risk-factor scores and earnings-call marketing-emphasis scores predict next-month cross-sectional returns beyond known factors, and turn the predictions into an honestly evaluated long-short strategy presented as a static website.

**Architecture:** A staged, cached pipeline (`run_all.py`): raw → interim → processed parquet → results (CSV/PNG/JSON). The website reads only `results/json/`. Every transform that touches the cross-section runs per month on that month's eligible universe. A single guard module enforces the locked test window.

**Tech stack:** Python 3.10 (pandas, numpy, statsmodels, scikit-learn, LightGBM, XGBoost, SHAP, yfinance, requests), pytest. The website uses Vite + React + TypeScript + Plotly.

**Spec:** `../PROJECT_BRIEF.md` (one level up), plus `docs/DECISIONS.md`, `docs/DATA_AUDIT.md` and `docs/RESEARCH_DESIGN.md` (Phase 1).

## Global constraints

- Never `git push`, `git remote`, or `gh`. Local commits only.
- No signal is used before `release date + lag` (risk: filing + 2 business days, then the next full month; marketing: quarter-end month + 2) (D-005, D-006).
- Locked test window **2022-01 → 2025-06**. Before the unlock is logged in DECISIONS.md, no function may compute or print a metric on these months (enforced by `src/utils/lock.py`).
- Scaling, imputation, feature selection and tuning use training-window data only (inside each walk-forward fold).
- Every number on the site or in docs is produced by code in this repo (`results/json/*.json`).
- No final report until the user asks.
- The SEC User-Agent comes from `config.sec_user_agent` or the env `SEC_USER_AGENT`, never the user's email.
- Memory: build panels as float32. Never hold daily prices for all tickers in one frame; aggregate per batch.

## Review focus (failure modes that matter most)

1. **A signal leaks into its own filing month**, e.g. a filing on the last day of a month used for that month. Pinned by `tests/test_timing.py::test_month_end_filing_skips_next_partial`.
2. **Cross-sectional ranks include firms not yet public.** Pinned by `tests/test_panel.py::test_rank_uses_only_same_month_eligible`.
3. **A split misprices market cap** (shares and price on different split bases). Pinned by `tests/test_prices.py::test_mcap_invariant_to_split`.
4. **A walk-forward fold whose training labels overlap the test window** (embargo). Pinned by `tests/test_walkforward.py::test_train_label_month_before_test_start`.
5. **Locked months leaking into dev metrics or tuning.** Pinned by `tests/test_lock.py::test_guard_raises_on_locked_months`.

---

## File structure

```
config/config.yaml                 all parameters (windows, lags, filters, costs, model grids)
run_all.py                         one command: python run_all.py [--stage X] [--force]
src/utils/config.py                load_config(), path(key), sec_headers()
src/utils/lock.py                  TEST_START, assert_unlocked(), dev_only(df)
src/data/risk.py                   load_risk_scores()
src/data/marketing.py              load_marketing(), link_marketing_to_cik()
src/data/sec.py                    bulk downloads; parse_sic(), parse_shares(), parse_fundamentals(), tickers_exchange()
src/data/prices.py                 download_daily(), monthly_from_daily()
src/data/factors.py                load_ff_factors(), load_ff_industries(scheme)
src/data/security_master.py        build_security_master()
src/features/timing.py             risk_first_month(), marketing_first_month()
src/features/characteristics.py    controls from returns/fundamentals (size, bm, mom, rev, vol, beta, illiq, ag, prof)
src/features/panel.py              build_panel(): stock-month frame, as-of joins with caps, universe flags, targets
src/features/signals.py            add_signal_features(): ranks, z, changes, surprises, interactions, constructs
src/backtest/portfolios.py         sort_portfolios(), weights, turnover, costs
src/backtest/metrics.py            perf metrics (kept from earlier work after review)
src/backtest/regressions.py        factor_alpha(), fama_macbeth()
src/models/walkforward.py          splits(), run_walkforward()
src/models/learners.py             fit/predict for enet, lgbm, xgb, mlp; ensemble
src/evaluation/predictive.py       ic_series(), oos_r2(), hit_rate()
src/evaluation/robustness.py       subperiods, size buckets, industries, lags, horizons, placebo, DSR, PBO
src/evaluation/explain.py          SHAP global and dependence
src/viz/figures.py, src/viz/export.py   PNGs for docs/notebooks; JSON for the website
notebooks/01_eda … 06_explainability.ipynb   narrated; call src/ only
website/                           Vite + React + TS
tests/                             pytest suite (below)
```

---

## Phase 0: environment, audit, plan ✅ (2026-11-22)
- [x] ENVIRONMENT.md, pinned requirements, DATA_AUDIT.md (profiler `scripts/profile_raw_data.py`), DECISIONS.md (audit + D-001…D-016), this plan.
- [x] Remove old artefacts per the audit table; new skeleton; commit.

## Phase 1: research grounding
- [ ] Literature-review subagent (Firecrawl): 10–15 verified works → `docs/LITERATURE.md`, with verified bibliographic details only.
- [ ] `docs/RESEARCH_DESIGN.md`: score construction and validity for both papers, 3–5 hypotheses with the predicted sign and the test that decides each, primary specification, and the multiple-testing budget (spec log).

## Phase 2: point-in-time panel

### Task 2.1: config, lock guard, timing rules
**Files:** `config/config.yaml`, `src/utils/config.py`, `src/utils/lock.py`, `src/features/timing.py`; tests `tests/test_timing.py`, `tests/test_lock.py`.
**Produces:** `risk_first_month(filing_date: Series[datetime]) -> Series[Period[M]]`; `marketing_first_month(year, quarter, buffer_months=1) -> Series[Period[M]]`; `dev_only(df, month_col="month") -> DataFrame`; `assert_unlocked()`.
- [ ] Tests: filing 2019-02-27 → 2019-04; 2019-02-12 → 2019-03; 2019-02-28 (Thu) → avail 2019-03-04 → 2019-04 (`test_month_end_filing_skips_next_partial`); Q1-2020 → 2020-05; Q4-2020 → 2021-02; `dev_only` drops ≥ 2022-01; `assert_unlocked` raises while `config.test.unlocked` is false.
- [ ] Implement, run `pytest tests/test_timing.py tests/test_lock.py`, commit.

### Task 2.2: raw loaders (risk, marketing, link)
**Files:** `src/data/risk.py`, `src/data/marketing.py`; test `tests/test_loaders.py`.
**Produces:** `load_risk_scores() -> DataFrame[cik:int, filing_date, file_year, words, optional, {f}_prob, {f}_cnt for 8 factors]`, where `words==0` ⇒ probabilities and counts are NaN. `load_marketing() -> DataFrame[cik, year, quarter, first_month, 19 raw, 19 zca, mo, mc, me, doc_len, an_cust]`. CIK is resolved via `cik_gvkey.csv` rows whose link window covers the quarter end; multi-GVKEY cells are exploded; ambiguous many-to-one resolutions are dropped and counted.
- [ ] Tests on real files: risk 131,920 rows and unique (cik, filing_date); operations prob mean 0.177 ± 0.001 over all rows; marketing has no duplicate (cik, year, quarter); construct means use 8/8/3 components.
- [ ] Implement, cache to `data/interim/*.parquet`, commit.

### Task 2.3: SEC, Yahoo, French downloads + security master
**Files:** `src/data/sec.py`, `src/data/prices.py`, `src/data/factors.py`, `src/data/security_master.py`; test `tests/test_prices.py`.
**Produces:** `data/interim/` parquet tables. `prices_monthly[ticker, month, ret, prc_splitadj, dvol, split_factor_after]`; `shares[cik, filed, end, shares]`; `fund[cik, filed, end, be, at, oi, rev, cogs]`; `secmaster[cik, ticker, exchange, sic, ff48, ff12]`; `factors[month, mktrf, smb, hml, rmw, cma, rf, mom]`.
- [ ] `test_mcap_invariant_to_split`: a synthetic 2:1 split leaves the computed market cap unchanged across the split month. `test_monthly_return_compounds_daily`: compounding daily returns gives the monthly return.
- [ ] Downloads run resumably in the background (Yahoo in batches of 50 cached as `data/raw/yahoo/batch_*.parquet`; SEC bulk zips in `data/raw/sec/`).
- [ ] Sanity: a momentum decile L/S built from our returns correlates > 0.6 with French UMD over 2010–2021, and size L/S correlates with SMB > 0.6. Results go to `results/tables/sanity_factor_replication.csv`.

### Task 2.4: panel
**Files:** `src/features/characteristics.py`, `src/features/panel.py`; test `tests/test_panel.py`.
**Produces:** `data/processed/panel.parquet` with one row per (cik, month), `month` = holding month t. Contents: return and excess return in t (`ret`, `exret`) and the t+1..t+3 horizon targets; controls measured at end of t−1; risk fields as-of `risk_first_month ≤ t` with `t ≤ filing_month + 15`; marketing as-of `first_month ≤ t < first_month + 6`; `elig_primary` and `elig_ext` flags (D-009).
- [ ] Tests: `test_no_signal_before_release` (for every row, `risk_first_month ≤ month` and `filing_date + 2BD < month start`); `test_staleness_cap`; `test_rank_uses_only_same_month_eligible`; `test_controls_lagged` (mom/vol at t never include ret_t); `test_missing_month_breaks_forward_return`.
- [ ] Coverage funnel by year → `results/tables/coverage_funnel.csv`. Commit.

## Phase 3: features
### Task 3.1: signal features
**Files:** `src/features/signals.py`; test `tests/test_signals.py`.
**Produces:** `add_signal_features(panel) -> panel` with, per month over eligible rows: percentile ranks and z-scores of the 8 probabilities, the dictionary density (count/words), log words, NonOp (mean rank of the 7), ΔX vs the previous filing (≤ 18 months apart), surprise (X minus the firm's expanding mean of past filings), marketing constructs (8/8/3 means of ZCA components) with QoQ and YoY changes, interactions (OpRisk × NonOp, OpRisk × each marketing construct), and industry-demeaned (FF48) versions of the key ranks. Text features also go into `FEATURE_GROUPS: dict[str, list[str]]` (`controls`, `risk`, `marketing`, `interactions`).
- [ ] Tests: ranks within [0, 1] per month; Δ is NaN when the gap > 18 months; surprise uses only prior filings (the first filing gives NaN). Commit.

## Phase 4: models
### Task 4.1: predictive evaluation + walk-forward
**Files:** `src/evaluation/predictive.py`, `src/models/walkforward.py`, `src/models/learners.py`; tests `tests/test_walkforward.py`, `tests/test_predictive.py`.
**Produces:** `splits(months, first_test_year=2014, last_test_year=2021, embargo=1)` returns folds (train months ≤ Dec(Y−1) − embargo − horizon, test = year Y). `run_walkforward(panel, features, target, learner, grid) -> preds[cik, month, pred]`; tuning uses the last 2 training years as validation, inside the fold. Learners: `enet`, `lgbm`, `xgb`, `mlp`, `ens` (rank-average). `ic_series`, `oos_r2` (zero benchmark, GKX 2020), `hit_rate`.
- [ ] Tests: `test_train_label_month_before_test_start`; scalers and imputers are fitted on training rows only (`test_preprocess_fit_on_train_only`); IC equals the scipy Spearman on a toy example.
- [ ] Ladder (dev window only): single-signal sorts → Fama-MacBeth → enet → lgbm/xgb → mlp → ens; feature sets controls / +risk / +marketing / +interactions on matched samples. Every spec is appended to `results/spec_log.csv`.

## Phase 5: strategy & backtest
### Task 5.1: portfolios, costs, regressions
**Files:** `src/backtest/portfolios.py`, `src/backtest/regressions.py`, `src/backtest/metrics.py` (reviewed); tests `tests/test_portfolios.py`, `tests/test_regressions.py`, `tests/test_metrics.py`.
**Produces:** `sort_portfolios(panel, score, n=10, weight="ew"|"vw", neutral=None|"ff12", min_names=20) -> (rets[month, P1..Pn, LS, long, short], weights)`; turnover from actual weight changes, including drift; `net_returns(gross, turnover, bps)`; `factor_alpha(r, factors, model, nw_lags=6)`; `fama_macbeth(panel, y, xs, nw_lags=6)`.
- [ ] Tests: a hand-computed 5-stock, 3-month toy example (weights, LS, turnover, 25 bps cost); alpha and t match a direct statsmodels HAC fit; Sharpe and max drawdown on a known series.
- [ ] Tear sheet: deciles and quintiles, EW/VW/FF12-neutral, costs of 10/25/50 bps, Sharpe, Sortino, MDD, Calmar, CAPM/FF5+Mom alphas with NW t-stats, long vs short leg, vs market and vs single signals.

## Phase 6: robustness & explainability
- [ ] Subperiods (2010–13, 2014–17, 2018–21; extended EW sample covering 2008–09; 2020), size buckets and ex-microcaps, FF12 industries, alternative lags (marketing +1/+3; risk June-rebalance), horizons (1/3 months).
- [ ] Placebo: 500 within-month shuffles of the prediction → null distribution of LS Sharpe and IC. Deflated Sharpe with N = number of specs in the spec log. PBO via CSCV (16 blocks) across the model × feature-set grid.
- [ ] SHAP on the frozen LightGBM (≤ 50k sampled dev rows): global importance, dependence for the top 6 features, grouped importance (risk vs marketing vs controls).
- [ ] Freeze the design → log it → **unlock the test window** (log date) → one run on 2022-01 → 2025-06.
- [ ] Limitations section (RESEARCH_DESIGN.md §Limitations).

## Phase 7: website
- [ ] `website/` Vite + React + TS + Plotly; data from `results/json/` copied by `src/viz/export.py` into `website/public/data/`. Sections exactly as listed in brief §7. `npm run dev` and `npm run build`.
- [ ] Playwright: screenshots at 1440 and 390 px, console free of errors, fixes applied.

## Phase 8: final QA
- [ ] `python run_all.py` from raw caches reproduces results and site data; `pytest` is green.
- [ ] Independent reviewer subagent: leakage, bugs, code-vs-site number consistency; fix all findings.
- [ ] README (overview, how to run, repo map, headline results); final summary to the user.
