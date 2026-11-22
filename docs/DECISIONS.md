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
