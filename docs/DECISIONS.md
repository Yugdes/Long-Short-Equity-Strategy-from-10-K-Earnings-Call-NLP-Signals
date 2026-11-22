# Decision log

Format: **ID — date — decision**. Each entry gives the alternatives considered and the rationale. Newest entries go at the bottom of each section.

---

## A. Audit of earlier work (2026-11-22)

The repo held a complete-looking pipeline (`src/`, Streamlit app, README with results). The audit found it unreliable:

| # | Finding | Severity |
|---|---|---|
| A1 | No SIC codes were ever fetched, so every firm sat in FF48 = 48 ("Other"). The "industry-neutral" primary signal was really OpRisk minus the year median. | High |
| A2 | No market cap was computed, so "value-weighted" portfolios silently fell back to equal weights. | High |
| A3 | Filing dates came from EDGAR joined on a heuristic fiscal year ("filed Jan–Jun ⇒ FY−1"). The dataset's own `Filing Date` column was ignored, and its `FYEAR` is really the filing year (DATA_AUDIT §2.2), so December-FYE firms were mis-joined by a year. | High |
| A4 | Cross-sectional ranks and industry medians were taken **within FYEAR across all firms**, including firms that filed later in the year. That is look-ahead in the signal. | High |
| A5 | Marketing constructs were built from 3/5/3 components instead of the authors' 8/8/3. Multi-GVKEY rows and link dates were ignored, and missing values were imputed with the month median and then 0. | Medium |
| A6 | Returns covered current tickers only (survivorship), and returns outside [−95%, +1000%] were dropped (real crashes were deleted). | Medium |
| A7 | README misreads its own table. L/S = Q1(low) − Q5(high) with α = −6.0%/yr means **high-risk firms out-performed**, the opposite of the README's "penalty" claim. A t of −1.85 was described as significant. | High (presentation) |
| A8 | The README's ML claims (IC 0.064, IC-IR 0.50) cannot be reproduced. The committed `panel_stock_month.parquet` has no return columns. An IC of 0.064 from slow annual text signals is implausible and should be treated as a bug until shown otherwise. ElasticNetCV used random K-fold, and LightGBM ran with `min_child_samples=20` despite a config of 500. | High |
| A9 | `scripts/generate_dummy_data.py` creates synthetic data. That is a fabrication risk in a research repo. | Medium |
| A10 | The tests mostly re-assert implementation details. There was no test proving no signal is used before its release date on the real panel. | Medium |

**Keep / rewrite / delete**

| Path | Decision | Why |
|---|---|---|
| `data/raw/**` (risk, marketing, link table) | **Keep** | Source data |
| `data/raw/dataset.xlsx` (root copy) | **Delete** | Byte-identical duplicate of `data/raw/risk/dataset.xlsx` |
| `data/interim/edgar_idx/*.parquet` | **Keep** | Used to validate filing dates |
| `data/interim/{daily_prices,sec_tickers}.parquet`, `data/processed/*` | **Delete/rebuild** | Old-pipeline products; leaky or incomplete |
| `results/models`, `results/figures`, `results/tables/factor_regressions.csv` | **Delete** | Products of the leaky pipeline |
| `src/utils/helpers.py` | **Rewrite (slim)** | Config/IO helpers reused; unused helpers dropped |
| `src/data/ingest_risk.py` | **Rewrite** | A3: use `Filing Date`; zero-word rows become missing |
| `src/data/ingest_marketing.py` | **Rewrite** | A5 |
| `src/data/filings.py` | **Reduce** | Keep only the EDGAR-index loader for validation; heuristic fiscal-year mapping deleted |
| `src/data/security_master.py`, `src/data/prices.py` | **Rewrite** | A1, A2, A6 |
| `src/data/factors.py` | **Rewrite (simplify)** | Direct Ken French download (pandas-datareader was flaky); same output |
| `src/features/*` | **Rewrite** | A4; the full CIK×month cartesian grid wasted memory |
| `src/backtest/metrics.py`, `factor_regs.py` (NW regression core) | **Keep after review + tests** | Formulas are standard. Re-verified by unit tests against statsmodels and hand examples |
| `src/backtest/portfolios.py` | **Rewrite** | A2; turnover was measured in names not weights; costs never applied; `min_names=1` in run_all |
| `src/ml/*`, `src/analysis/*` | **Rewrite** into `src/models`, `src/evaluation` | A8; brief's layout |
| `app/streamlit_app.py` | **Delete** | Superseded by the static website (Phase 7) |
| `scripts/generate_dummy_data.py` | **Delete** | A9 |
| `tests/test_stage1_verification.py` | **Replace** | A10; a couple of its ideas (missing month ⇒ NaN forward return; RF alignment) are carried into new tests |
| `README.md`, `docs/ML_METHODOLOGY.md` | **Rewrite at the end** | A7, A8 |
| `Makefile` | **Replace** by `run_all.py` (one command), with a thin Makefile alias | Brief: one-command rebuild |

---

## B. Phase 0 decisions

**D-001 — 2026-11-22 — Read both papers in full, as the brief instructs.** The INFORMS PDF carries a notice restricting use with AI tools, and the earlier plan had said not to feed the PDFs to the agent. The user's current brief explicitly asks for a full reading, and it takes precedence. Mitigation: the text conversions live only in gitignored `internal_files/papers/`. Docs and the site summarise and cite the papers; they don't reproduce them. *Flag for the user:* check that this use is acceptable under your library's licence terms.

**D-002 — 2026-11-22 — Reuse the existing `venv/` (Python 3.10.11) and pin it.** The alternative was a fresh venv or conda env. The existing one is already isolated and holds every needed package. Pinning it (`requirements.txt` direct, `requirements-lock.txt` full) gives the same reproducibility without ~1 GB of re-downloads.

**D-003 — 2026-11-22 — Free data path (yfinance + SEC XBRL/submissions + Ken French).** No CRSP or Compustat files are present (the brief's trigger for the free path). `cik_gvkey.csv` suggests the user may have WRDS access, but I won't log into an external account with the user's credentials. A CRSP/Compustat adapter is listed as future work. The consequences (survivorship bias; size data only from ~2009–2011) are designed around below and stated in Limitations.

**D-004 — 2026-11-22 — Key all risk-signal timing on the dataset's `Filing Date`.** Validated: 131,917/131,920 rows match an original 10-K in EDGAR on the same date. The rejected alternative was the old EDGAR fiscal-year heuristic (A3).

**D-005 — 2026-11-22 — Risk-signal availability = filing date + 2 business days. The first month a signal may be used is the first full calendar month after that date.** Example: a filing on 2019-02-27 becomes available on 2019-03-01. Its availability month is March, so it is first used to form the April 2019 portfolio. A filing on 2019-02-12 becomes available on 2019-02-14 and is first used for March 2019. Conservative: the availability month is always skipped, even when the signal arrives on its first day. Alternatives: use from the next day (too aggressive for monthly data), or a June-only annual rebalance (kept as a robustness variant).

**D-006 — 2026-11-22 — Marketing-score availability.** The quarter is the call's calendar quarter (verified on 7 IPO firms; DATA_AUDIT §3), so a score is public by the quarter's end. A one-month buffer is added: **first usable month = quarter-end month + 2** (Q1 → May, Q2 → Aug, Q3 → Nov, Q4 → Feb). Alternatives: + 1 (no buffer, as aggressive as the data allows) and + 3 (extra-conservative); both run as robustness lags.

**D-007 — 2026-11-22 — Staleness caps.** A risk signal expires **15 months after its filing month**, covering the normal 12-month cycle plus late filers; after that it is missing. A marketing score expires **6 months after its first usable month**, so it survives one skipped call. Without caps, dead or late filers would carry stale information indefinitely.

**D-008 — 2026-11-22 — Sample windows.**
- Risk signals start with 2006 filings (2005 filings carry no Item 1A; DATA_AUDIT §2.4).
- **Primary sample: return months 2010-01 → 2025-06.** It starts when XBRL market caps begin to exist (needed for value weights, size filters and size controls). Free-data survivorship is also milder after 2010 (coverage rises from ~28% of firm-years pre-2010 to 60–76% by the 2020s). It ends because risk filings stop in 2024 (D-007 cap ⇒ broad coverage until ~mid-2025).
- **Extended EW sample: 2006-07 → 2025-06** for single-signal equal-weighted sorts only, without size filters. It exists so the 2008 crisis can be studied (brief Phase 6), and it is labelled survivor-heavy.
- **Locked final test: 2022-01 → 2025-06 (42 months).** It is not looked at (no metric computed or printed on it) until the model design is frozen. The unlock will be recorded here.
- Development OOS: 2014-01 → 2021-12 (walk-forward). The initial training window is 2010-01 → 2013-12.

**D-009 — 2026-11-22 — Primary universe filters, evaluated at formation using month-(t−1) data:**
- U.S. common equity listed on NYSE or Nasdaq per SEC's exchange file. OTC is dropped: illiquid, with noisy Yahoo data.
- Price ≥ $5. Removes penny stocks, where bid-ask bounce and data errors dominate (standard; Jegadeesh & Titman 2001).
- Market cap ≥ $100M nominal. Removes nano-caps that are untradeable at any meaningful size, where free-data errors concentrate.
- At least 12 months of prior returns (for momentum and volatility controls).
- Financials (SIC 6000–6999) and utilities (4900–4949) are excluded. Book-based controls and the operational-cost mechanism behind the risk signal don't carry over to banks or regulated utilities. They are included again as a robustness check.

Robustness variants: an NYSE-20th-percentile microcap cut (using this sample's NYSE firms), no size filter, and price ≥ $1.

**D-010 — 2026-11-22 — All cross-sectional transforms (ranks, z-scores, industry demeaning, winsorising) are computed per month over the stocks eligible that month.** This replaces A4, which ranked within FYEAR over firms that had not yet filed. A feature at month *t* uses only information public before month *t* begins.

**D-011 — 2026-11-22 — Industries: current SIC from SEC `submissions.zip`, mapped to FF48 and FF12 with Ken French's official `Siccodes48`/`Siccodes12` files.** Historical SIC (CRSP/Compustat) is unavailable on the free path. Current SIC is a standard approximation, and the limitation is logged.

**D-012 — 2026-11-22 — Market cap = split-adjusted close × split-adjusted XBRL shares outstanding.** Shares come from `dei:EntityCommonStockSharesOutstanding`, falling back to `us-gaap:CommonStockSharesOutstanding`, and are used only after their `filed` date + 1 month. A split ratio between the share-count date and month *t* is applied using Yahoo split events. The alternative of back-filling pre-XBRL shares from later reports was rejected because it puts future data into weights and filters.

**D-013 — 2026-11-22 — SEC requests send a generic User-Agent (`MS499-research-pipeline research@example.org`), overridable by the `SEC_USER_AGENT` environment variable.** The user's real email isn't sent to an external service. SEC fair access (≤10 req/s) is respected, and bulk zips are used to keep the request count tiny.

**D-014 — 2026-11-22 — The small MLP uses scikit-learn `MLPRegressor`.** PyTorch (CPU or CUDA) is the alternative. The tabular MLP is small, scikit-learn is already pinned, and the GTX 1650 offers no meaningful gain for a model this size.

**D-015 — 2026-11-22 — Repo layout follows the brief:** `src/{data,features,models,backtest,evaluation,viz,utils}`, `run_all.py` at the root, `website/`, `docs/`. The old `src/ml` and `src/analysis` are folded into `models` and `evaluation`.

**D-016 — 2026-11-22 — Git: local commits only.** Never push or add remotes; an `origin` remote already exists and is left untouched. Generated data (`data/interim`, `data/processed`, downloaded caches) is gitignored. Raw source data stays tracked via LFS as before. Small result JSON/CSV files that feed the site are tracked.

**D-017 — 2026-11-22 — Execute the plan natively (superpowers:executing-plans), using subagents only for independent work: the literature review, and the leakage reviewer in Phase 8.** The brief forbids interactive Q&A, so the writing-plans hand-off question is skipped. Subagent-per-task was considered and rejected: the data tasks share interfaces tightly, and running them sequentially in one context avoids re-deriving context.

## C. Phase 1–2 decisions

**D-018 — 2026-11-22 — Amends D-009: the primary universe does *not* filter on current exchange.** Yahoo is pulled for the primary ticker of every risk CIK with any current SEC ticker (4,535 firms, 764 of them currently OTC). Many of today's OTC names traded on NYSE or Nasdaq earlier and were demoted after poor performance; their Yahoo history covers those years. Filtering on *current* exchange would delete exactly these losers and worsen survivorship bias. The price ≥ $5 and market cap ≥ $100M screens do the liquidity work instead. A current-NYSE/Nasdaq-only universe is run as a robustness check.

**D-019 — 2026-11-22 — Phase 1 frozen: hypotheses H1–H5, the primary specification and the hurdles are in RESEARCH_DESIGN.md, written before any return-based result.** The primary strategy is a value-weighted decile L/S of the ML ensemble, net of 25 bps, rather than EW. This is the more conservative choice: text signals tend to be stronger in small stocks, and VW limits that.

**D-020 — 2026-11-22 — Return cleaning (Yahoo data errors).** A monthly return is set to missing if it is < −100% (impossible), > +1,000%, or part of a spike-and-reversal (> +300% followed by < −75%: a bad tick). This blanks 548 of 750,424 ticker-months (0.07%; `results/tables/return_cleaning.csv`). Real crashes (e.g. −90%) are kept. The rejected alternative was the old pipeline's [−95%, +1,000%] truncation, which deleted genuine crashes.

**D-021 — 2026-11-22 — Market-cap sanity checks.** Free XBRL shares × Yahoo prices produced impossible caps: unit errors (e.g. Oracle 2013 ×1,000) and reverse splits that Yahoo applies in `Close` but omits from its split list. Rules, using past data or same-month data only:
1. Drop a share count that jumps more than 10× (or below 0.1×) against the median of the firm's previous 3 counts.
2. Market cap is missing if it exceeds $5T, the raw price exceeds $100k, monthly dollar-volume turnover is outside [0.1%, 500%], or the cap exceeds 500× book assets.
3. Price filters use the **raw** price, reconstructed as `Close` × the product of later splits. The split-adjusted price wrongly passed reverse-split names.

Verified caps: NVDA May-2024 $2.74T, AAPL Jun-2015 $723B, MSFT Dec-2012 $225B. The residual errors I know of (ADS-ratio issuers such as BeiGene, a missing DuPont reverse split) are a few names, bounded at about 3–13×.

**D-022 — 2026-11-22 — "Value-weighted" means capped value weights.** Market cap is winsorised each month at the 80th percentile of currently NYSE-listed names in the universe (Jensen, Kelly & Pedersen 2023 use NYSE 80th-percentile caps). Two reasons: (1) it limits mega-cap concentration (MSFT alone is 21% of the universe's cap in 2025-06); (2) it bounds the influence of the residual cap errors in D-021. Uncapped VW is reported as a robustness check.

**D-023 — 2026-11-22 — ML grid and compute budget.**
- *Feature sets* (nested, identical rows, so matched by construction): C (10 controls) ⊂ C+R (+ risk text) ⊂ C+R+M (+ marketing; imputed 0 + `has_mkt` flag) ⊂ ALL (+ interactions). TEXT = R+M+X, without controls.
- *Learners:* ElasticNet (α × l1 grid of 9), LightGBM (num_leaves 15/63, min_child 1000, lr 0.03, early stopping on validation), XGBoost (depth 3/6, same style), MLP (32-16, L2 2 values, epochs ≤ 25 chosen on validation, 2-seed average), and an equal-weight mean ensemble.
- *Scope:* ENet and LGBM run on all 5 sets. XGB, MLP and the ensemble run on C and ALL, which is the comparison H5 needs. This saves ~1.5 h of CPU for comparisons no hypothesis needs.
- *Protocol:* tuning uses the last 2 years of each training window (1-month gap); the model is then refit on the whole window. Annual refits; test years 2014–2021; target = centred rank of the month-*t* excess return.

**D-024 — 2026-11-22 — H1's realised-volatility test uses formation months up to 2020-12 only.** The forward 12-month window then ends by 2021-12 and never reads the locked test window.

**D-025 — 2026-11-22 — Reduced grids outside the primary horizon.** The 3-month-horizon robustness run uses ENet (ALL) and LGBM (C vs ALL) only. The final locked-window run uses all 4 learners on C and ALL plus the ensemble: what the frozen primary strategy and the H5 comparison need, and nothing else, so the test window is evaluated on as few specs as possible.

**D-026 — 2026-11-22 — Diagnostic: is the marketing block's IC gain the language or the coverage flag?** Prompted by the dev-window LightGBM ladder (C 0.039 → C+R 0.036 → C+R+M 0.047). `has_mkt` (the firm holds earnings calls) could proxy for size or coverage, so three LGBM sets were run and logged as specs:
- C+flag: IC 0.041;
- C+marketing without the flag: IC 0.0505 (t 4.7);
- C+marketing with the flag: IC 0.0508.

Conclusion: the gain comes from the call-language scores, not the flag. These diagnostics also show that the IC gains do **not** produce factor-adjusted long-short alpha (VW decile FF5+Mom α t ≈ −0.2 to 0.2). This finding is reported; it does not change the pre-specified primary strategy.

## D. Design freeze and test unlock

**D-027 — 2026-11-22 — Design frozen; locked test window unlocked.** Development work is complete:
- single signals and Fama-MacBeth;
- the full walk-forward ladder;
- 3-month horizon, alternative lags and the marketing-flag diagnostic;
- tear sheet, placebo, shuffled-label retrain, PBO, DSR and SHAP, all on 2010/2014–2021 only.

No result from 2022-01 → 2025-06 has been computed or viewed before this entry. Frozen choices: the features, learners, grids and walk-forward protocol (D-023, D-025); the primary strategy (ensemble of ENet/LGBM/XGB/MLP on ALL features, capped-VW decile long-short, net of 25 bps); and the universe and timing rules (D-005–D-009, D-018, D-021, D-022). The dev results did **not** change the primary specification, even though it performed worst among the construction variants (VW decile gross Sharpe 0.08 vs EW 0.46). Switching to the best-looking variant now would be selection on dev results; EW is reported alongside as a secondary variant. Action: set `config.test.unlocked = true` and run the frozen pipeline once on test years 2022–2025 (`python -m src.models.run_ml --final`). The models keep expanding annually inside the test window, as in development: the 2023 model trains on data to 2022-11.

**D-028 — 2026-11-22 — Data bug found through the test-window red flag; fixed; pipeline rerun end to end.**

*What was seen.* The first locked-window run (D-027) produced implausible numbers: VW decile Sharpe 1.5–2.0 and FF5+Mom α t up to 5.3, vs ≤ 0.36 and ≤ 1.1 in development. Controls-only models did best. Following the plan's red-flag rule (Sharpe > 1.5 or α t > 4 ⇒ presume a bug), I inspected the bottom decile.

*Cause.* It held distressed firms with impossible caps: AGL at $127B with a $313 price, IRLA at $2,524/share, FOXO at $1,202. Yahoo was queried with `end=2025-12-31`. Its adjusted `Close` reflects every split up to the download date (Nov 2026), but split *events* are returned only inside the requested window. Reverse splits executed in 2026 (AGL 1:25 on 2026-03-31; IRLA 1:10 and 1:20; FOXO 1:3,000) were therefore missing. Raw prices and market caps were inflated by the split ratio, true penny stocks passed the $5 / $100M screens and received large value weights. Because the inflation comes from a *future* corporate action, this is look-ahead, and it concentrated soon-to-collapse firms in the short leg. The D-021 checks missed it: the errors (10–3,000×) still produced turnover above 0.1% and caps below 500× assets.

*Fix.* Download with no end date (`src/data/prices.py`, `END = None`) so the split list and the price adjustment cover the same period. The flawed download is kept in `data/raw/yahoo_bad_end2025/` for audit. Every downstream stage is rebuilt: prices, panel, features, single signals, the dev ML ladder, diagnostics, alternative lags, h=3, the final window and the analysis.

*Integrity of the test window.* The test window has now been evaluated once on flawed data. The correction is a mechanical data fix, not a design change: features, learners, grids, universe rules and the primary specification are unchanged from D-027, and nothing was tuned to the test results. The corrected test run is reported. The flawed run's summary is kept here for transparency (`results/tables/ml_summary_final_h1_BUGGY_pre_D028.csv`).

**D-029 — 2026-11-22 — Fixes from the independent reviewer audit (Phase 8 reviewer, run early).** The reviewer confirmed no look-ahead in panel timing, features, walk-forward or backtest arithmetic, and raised the items below. All are applied before the full rerun that D-028 requires.
1. *Time-varying survivorship (critical, cannot be fixed on free data).* Every sampled stock survives to late 2025, so a 2014 stock had to survive ~11 more years but a 2023 stock only ~2. Losers are therefore missing mostly from the *development* window, which flatters the dev short leg; dev and test are not comparable. Responses: stated as the lead caveat on the site and in Limitations; long and short legs reported separately; a new alpha model adds a low-volatility factor built inside this same sample (FF5+Mom+LowVol), so alpha against it isolates anything beyond the low-vol anomaly under identical survivorship. H5 (ALL vs C on identical rows) is the comparison least affected.
2. *Ensemble.* The code averaged raw predictions, so the MLP (largest prediction spread) dominated. RESEARCH_DESIGN §3 pre-registered an equal-weight *rank* ensemble, so the code is corrected to match the pre-registration (rank within month, then average). OOS R² is undefined for a rank ensemble and is reported as n/a.
3. `_oos` crashed when unlocked (no TEXT final predictions): final predictions are appended only if they exist.
4. *Sample end 2025-06 → 2025-05.* In 2025-06 the universe halves (727 names vs ~1,600), because FY2024 10-Ks are absent and the 15-month cap expires the Feb-2024 filers. The test window becomes 2022-01 → 2025-05 (41 months).
5. *Return cleaning.* The spike-and-reversal rule used ret(t+1) to decide whether ret(t) exists, so it is dropped. The impossible (< −100%) and > +1,000% rules remain.
6. *h=3.* Dev formation months stop at 2021-10 (labels never read 2022), and the fit/validation gap is now horizon + embargo.
7. *DSR family = every dev-window spec in the log* (single sorts, ML ladder, diagnostics, alternative-lag runs, tear-sheet construction variants; test-window specs excluded). All trials are gross, so the tested primary series is gross too. The site's spec count uses the same N.
8. *Tear sheet.* Dev and test statistics come from one continuous portfolio history (no artificial rebuild cost at 2022-01).
9. *ZCA look-ahead suspicion.* The authors' ZCA whitening was likely fit on the full 2003–2025 sample. Robustness: LGBM with the 19 *raw* components in place of the ZCA ones (`C+R+Mraw`, logged).
10. *Weight cap uses the current exchange (NYSE).* Kept, and noted in Limitations.

**D-030 — 2026-11-22 — Second independent review (site and docs vs code) and errata.** All JSON copies and exported tables matched their CSVs, and the spec count matched the DSR trial count. The fixes were in prose and verdict logic:
- Numbers in site prose are now read from `results/json`: coverage share, prob–count correlations (with their sample named), scored-filing count (122,600 non-empty sections, not 131,920 rows), feature count (83), test-window length, construct correlations.
- *H1 verdict.* It now leads with the pre-registered volatility test (operations risk and aggregate risk with controls: t = 0.10 and −1.04, so **not** supported). The 8-factor volatility count is labelled exploratory. The no-alpha part holds, so the tag is "Partly supported".
- *Verdict tiers.* Only the |t| ≥ 3 hurdle is pre-registered; the 2–3 tier is now labelled "Below hurdle" (H3: t = 2.84, the best of 10 statistics).
- *Hero verdict.* It requires a positive t ≥ 3 for any "partly" claim and now says the text "adds little beyond a controls-only model". The exported `fm_forecast` shows the ensemble forecast keeps FM t ≈ 3.3 after controls, but the paired IC gain over the controls-only model is t = 1.6.
- *Labels.* Gross/net and window labels added; SHAP model described as the 2022-fold model (trained to 2021-11); DSR stated to use the gross series; "survived to late 2026" (download date), not 2025.
- *Errata for superseded figures* (pre-D-028 data and the raw-mean ensemble):
  - D-026's ICs (C 0.039, C+flag 0.041, C+M 0.0508) are now C 0.032, C+flag 0.035, C+M_noflag 0.042, C+M 0.042.
  - D-027's "VW decile gross Sharpe 0.08 vs EW 0.46" is now 0.25 vs 0.70.
  - The conclusions of both entries are unchanged.

Current values: `results/tables/ml_summary_dev_h1.csv`, `ml_diagnostics_dev.csv`, `results/json/strategy.json`.

**D-031 — 2026-11-23 — H4 3×3 conditional double sort run (pre-registered, previously omitted).** RESEARCH_DESIGN §2 listed "3×3 conditional double sorts" as one of H4's deciding tests, but only the Fama-MacBeth interaction and the ML interaction features had been run. It was added after the user asked whether marketing offsets the return effect of high risk.
- *Construction* (`src/backtest/portfolios.py::double_sort`, `src/evaluation/single_signals.py::run_double_sort`):
  - Each month, on marketing-covered rows in the dev window (2010–2021), stocks are first split into terciles of operations risk.
  - The risk measure is either the level (`r_operations_prob`) or the change versus the previous 10-K (`r_d_operations_prob`).
  - Within each risk tercile, stocks are split into terciles of a marketing construct (orientation, capabilities, excellence).
  - Portfolios are EW and capped VW, using month-t excess returns. Cells with fewer than 20 stocks are dropped.
- *Test statistic.* The "cushion" is (high minus low marketing within the high-risk tercile) minus (the same within the low-risk tercile). H4 predicts it is positive.
- *Trials.* 24 trials (the high-risk marketing spread and the cushion, for each of the 12 specs) are logged to `spec_log.csv` under `single_sort`. The DSR family grows from 99 to 123 trials, and the deflated Sharpe probability of the primary strategy falls from 0.11 to 0.08. No other number changed.
- *Result* (value-weighted, FF5+Mom α t; `results/tables/double_sort_dev.csv`):
  - Cushion on the risk *level*: −2.20 to −1.04, the wrong sign.
  - Cushion on the risk *change*: +1.70 to +2.26.
  - Among firms whose operations risk rose most, high-marketing firms beat low-marketing ones by 0.26–0.29% a month (α t 2.1–2.3).
  - Rising risk itself carries no significant return penalty.
  - Nothing clears |t| ≥ 3.
- *Verdict.* The site's H4 verdict now uses the larger of the FM interaction t and the VW cushion α t, and moves from "Not supported" to "Below hurdle" (2.26, right sign).
- *Window.* The double sort is a development-window hypothesis test like the other single-signal sorts; it is not run on the test window.
