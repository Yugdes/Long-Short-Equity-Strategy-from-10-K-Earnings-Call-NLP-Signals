# Data audit (Phase 0)

Every number below comes from `python scripts/profile_raw_data.py`, which writes `results/tables/data_audit/` (`summary.json` plus per-table CSVs). Audit date: 2026-11-22.

## 1. File inventory

### Raw inputs (`data/raw/`), kept

| File | Format | Rows × cols | Identifier | Date fields | Notes |
|---|---|---|---|---|---|
| `risk/dataset.xlsx` | Excel | 131,920 × 30 | `CIK` | `FYEAR`, `Filing Date` (int YYYYMMDD) | Astvansh & Simpson (2026) OSF release. 8 dictionary counts, 8 transformer probabilities and 8 binary classifications per firm-year, plus Item 1A word count and an optional-section flag. Column names have stray double spaces (e.g. `Accounting  Probability`). |
| `risk/dataset_dictionary.xlsx` | Excel | 29 variables | | | The authors' data dictionary (converted to `.md` beside it). It documents `Filing Date` as "Date the 10-K or 10-K/A was filed". |
| `risk/01_prepare_data.py` | Python | | | | The authors' train/val/test split script for the 761 synthetic sentences. It imports a missing `00_config`, so it isn't runnable. Kept for provenance only. |
| `marketing/MarketingEmphasisScores_Executive.csv` | CSV | 337,599 × 42 | `GVKEY` | `year`, `quarter` (calendar) | Damavandi, Mai & Astvansh (2025), **v2 (Aug 2026)**, coverage 2003Q1–2025Q4. Raw and ZCA-whitened values of the 19 components, plus `document_length`. |
| `marketing/MarketingEmphasisScores_Analyst.csv` | CSV | 254,338 × 6 | `GVKEY` | `year`, `quarter` | Analysts' customer orientation (raw and ZCA). |
| `marketing/README.pdf`, `VERSION.txt` | PDF / txt | | | | The authors' codebook and version history (converted to `.md`). |
| `cik_gvkey.csv` | CSV | 1,003,087 × 25 | `cik`, `gvkey` | `link_start_date`, `link_end_date`, `sec_start/end_date` | WRDS SEC Analytics "WRDS SEC Linking Table". 136,580 rows carry a GVKEY (41,364 CIKs ↔ 38,575 GVKEYs). Sources: CRSP/Compustat Merged 44,549; Capital IQ 43,915; Compustat Company 31,340; Compustat Security 16,776. 5,082 GVKEYs map to more than one CIK, and 2,328 CIKs map to more than one GVKEY, so links must be resolved by link dates. Gitignored (WRDS licence). |
| `dataset.xlsx` (in the `raw/` root) | Excel | | | | **Byte-identical duplicate** of `risk/dataset.xlsx`. Untracked, to be deleted. |

### Previously generated files, to be rebuilt or deleted (see DECISIONS.md)

| File | Rows | Status |
|---|---|---|
| `data/interim/edgar_idx/*.parquet` (80 files) | 189,074 10-K / 10-K/A index rows, 2005Q1–2024Q4 | **Kept**: used to validate filing dates (§2.3). |
| `data/interim/sec_tickers.parquet` | 8,016 | Superseded by a fresh `company_tickers_exchange.json` download. |
| `data/interim/daily_prices.parquet` | 28,630 | A test sample of a few tickers. Delete. |
| `data/processed/returns_monthly.parquet` | 1,148,411 (7,607 tickers, 2004-02 → 2026-10) | Old yfinance pull: dividend-adjusted closes, no market cap, no split data. Rebuild. |
| `data/processed/panel_stock_month.parquet` | 1,241,695 × 28 | **Contains no return columns**, so the README's backtest and ML numbers cannot be reproduced from it. Rebuild. |
| `data/processed/{risk_scores, marketing_scores, filings, security_master, factors_monthly, ml_predictions}.parquet` | various | Products of the old pipeline. Rebuild. |
| `results/models/*.joblib` (14), `results/figures/*`, `results/tables/factor_regressions.csv` | | Products of the old pipeline. Delete. |

## 2. Risk scores (Astvansh & Simpson 2026)

### 2.1 Coverage and keys
- 131,920 firm-years, 16,959 CIKs; no duplicate (CIK, FYEAR); no missing values in any column.
- Filing dates run from **2005-01-03 to 2025-01-07**. Only 3 rows are from 2025, so the dataset effectively ends with 10-Ks filed in calendar 2024.

### 2.2 `FYEAR` is the *filing* year, not the Compustat fiscal year
`FYEAR == year(Filing Date)` for **100% of rows**. Examples: Amazon's `FYEAR=2024` row was filed 2024-02-02, which is its FY2023 10-K. Apple's `FYEAR=2024` row was filed 2024-11-01 (FY ending Sept 2024). Consequences:
1. Point-in-time logic must key on `Filing Date`, never on `FYEAR`. The old pipeline instead merged EDGAR dates on a "Jan–Jun → year − 1" heuristic, which misassigns most December-FYE filers by one year.
2. "Year-over-year change" means change versus the firm's previous filing. Consecutive filings are a median of 364 days apart (IQR 362–368).

### 2.3 Filing-date validation against EDGAR
Joining on (CIK, filing date) to the cached EDGAR `form.idx` files matched **131,917 of 131,920 rows to an original `10-K`** (zero matched only to a `10-K/A`). The 3 unmatched rows are the January 2025 filings, which fall outside the cached index range. The dataset's filing date is therefore a reliable public-release date.

### 2.4 Item 1A availability by filing year (`results/tables/data_audit/risk_by_fyear.csv`)

| Filing year | Firm-years | Median Item 1A words | Share < 1,000 words | Item 1A optional |
|---|---|---|---|---|
| 2005 | 5,815 | 0 | 94.3% | 0.3% |
| 2006 | 5,748 | 5,298 | 12.5% | 1.3% |
| 2010 | 7,857 | 5,887 | 18.8% | 9.1% |
| 2015 | 7,034 | 9,636 | 16.2% | 13.0% |
| 2020 | 5,691 | 15,166 | 12.1% | 11.8% |
| 2024 | 5,818 | 21,016 | 9.0% | 10.4% |

- Item 1A became mandatory for 10-Ks filed after 1 Dec 2005, so **2005 filings carry almost no risk text** (94% have fewer than 1,000 words). Signals start with 2006 filings.
- 9,320 rows have 0 words; their 8 probabilities are all exactly 0. These are missing values, not "zero risk".
- Item 1A length roughly quadrupled between 2006 and 2024 (boilerplate growth, and the 2020 SEC Item 105 amendments). Scores must therefore be compared **within** each cross-section, never across years in levels.

### 2.5 Score distributions (match the paper's Table 4)
Means of the probabilities: accounting 0.160, finance 0.285, international 0.053, legal 0.200, management 0.063, marketing 0.047, **operations 0.177**, technology 0.051. These match Paper 1 Table 4, Panel A to three decimals. The binary classifications are degenerate (means 0.0000–0.14; international is never 1), so they are ignored.

### 2.6 Early validity flags (expanded in Phase 1 diagnostics)
- Probability vs. dictionary-count correlation (Pearson / Spearman): operations 0.14 / 0.38, finance 0.55 / 0.71, technology 0.18 / 0.52. The transformer probability is **only loosely related** to the dictionary count it was trained to imitate.
- Among filings with at least 1,000 words, operations probability is **negatively** rank-correlated with Item 1A length (ρ = −0.38). The probability behaves like a *share* or *density* of risk language, not an amount.
- Cross-factor correlations of the probabilities reach 0.54 (operations vs. international). The 8 dimensions are far from orthogonal.

## 3. Marketing-emphasis scores (Damavandi, Mai & Astvansh 2025)

- Executive file: 337,599 firm-quarters, 11,296 GVKEYs, 2003Q1–2025Q4; no duplicate (GVKEY, year, quarter); no missing values; `document_length` > 0 everywhere. Coverage grows from 882 rows in 2003 to ~27–29k rows per year in 2023–2025.
- **1,158 rows (all 2023–2025, from the v2 extension) carry multiple GVKEYs** in one cell (e.g. `002001,GV_002002`). They are exploded and resolved through the CIK link (§4).
- **The `quarter` is the calendar quarter of the earnings call** (README: "Calendar year / Calendar quarter"). Verified on IPO firms with known first-call dates:

  | Firm | First call | First labelled quarter |
  |---|---|---|
  | Uber | May 2019 | 2019Q2 |
  | Lyft | May 2019 | 2019Q2 |
  | Airbnb | Feb 2021 | 2021Q1 |
  | Rivian | Dec 2021 | 2021Q4 |
  | Snowflake | Dec 2020 | 2020Q4 |
  | Coinbase | May 2021 | 2021Q2 |
  | Facebook | Jul 2012 | 2012Q3 |

  A quarter's score is therefore fully public by that quarter's end. A conservative one-month lag is added on top (DECISIONS.md).
- No call dates are provided, so the lag rule above is necessary.
- The 19 components are listed in the authors' README: market orientation has 8, marketing capabilities 8 and marketing excellence 3. The old pipeline averaged only 3, 5 and 3 of them, which was **wrong**. It will be rebuilt.
- Analyst file: 254,338 firm-quarters, 9,258 GVKEYs.

## 4. Identifier overlap

| Link | Count |
|---|---|
| Risk CIKs present in the WRDS CIK–GVKEY table | 14,476 of 16,959 (85%) |
| Risk CIKs linked to a GVKEY that has marketing scores | 5,183 |
| Marketing GVKEYs linked to a risk CIK | 5,013 of 11,296 |

The combined risk × marketing universe is therefore ~5k firms. Marketing-feature experiments use matched samples (RESEARCH_DESIGN.md).

## 5. Returns, prices and fundamentals

- **No CRSP or Compustat return or fundamentals files are in the folder.** `cik_gvkey.csv` is only a link table. Following the brief, a **free pipeline** is used:
  - Prices: `yfinance` daily OHLCV with dividends and splits, for every current SEC ticker (`company_tickers_exchange.json`, which includes the exchange). Monthly total returns come from adjusted closes.
  - Shares outstanding and book equity: SEC XBRL bulk `companyfacts.zip`, which carries the true `filed` date of every fact, so it is point-in-time. XBRL starts in 2009 for the largest filers and reaches every filer by mid-2011. **Market cap, and hence value weighting and size filters, is only available from ~2009–2011.** Quarterly `frames` were tested and are too sparse (~3k firms per quarter).
  - Industry: current SIC from SEC bulk `submissions.zip`, mapped to FF48/FF12 with Ken French's `Siccodes48.zip`.
  - Factors: Ken French FF5 (2×3), momentum and RF, monthly.
- **Survivorship bias, quantified.** Yahoo only serves currently listed tickers, so firms that delisted, merged or went bankrupt before 2026 are missing. Using the old current-ticker pull, the share of risk firm-years with any return history is **26% for 2005 filings, 29% for 2008, 31% for 2012, 43% for 2016, 60% for 2020 and 76% for 2024** (42% overall; `returns_coverage_by_fyear.csv`). Part of the gap is registrants that never had listed common equity (shells, trusts, debt-only filers), which a free source cannot separate out. The remainder is survivorship. The rebuilt pipeline recomputes this funnel. Implications are carried into the design (start of the primary sample, delisting-bias sensitivity, explicit Limitations).
- Effective sample end: risk filings stop at calendar 2024 (FY2024 10-Ks filed in 2025 are not in the dataset). With a 15-month staleness cap, the last month with broad risk-signal coverage is about **2025-06**. Return months after that are not used.
