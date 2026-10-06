# Disclosed Operational Risk as an Equity Signal

**A systematic long-short equity strategy built on NLP-derived 10-K risk disclosures, with an optional marketing emphasis overlay.**

> *This project investigates whether text-derived operational risk scores from annual SEC 10-K filings are priced by the stock market, net of known risk factors.*

---

## Executive Summary

U.S. public firms must describe their material risks in Item 1A of the annual report (Form 10-K). Recent research (Astvansh & Simpson, 2026; Damavandi et al., 2025) used transformer models to score over 130,000 firm-year filings on eight specific risk factors.

This project extends that work into the domain of quantitative finance by asking: **Does sorting firms on disclosed operational risk produce risk-adjusted return information beyond known factors?**

### Key Results

The pipeline successfully ran a 20-year backtest (2004–2025) over a 500-stock universe, producing the following Fama-French factor attribution on the Long-Short (Q1 − Q5) spread:

| Factor Model | Annualized Alpha (α) | t-statistic | Adj R² |
| :--- | :--- | :--- | :--- |
| CAPM | −4.50% | −1.42 | 0.014 |
| Fama-French 3-Factor | −4.04% | −1.27 | 0.011 |
| Fama-French 5-Factor | −5.76% | −1.71 | 0.022 |
| **FF5 + Momentum** | **−5.92%** | **−1.74** | **0.019** |

**Interpretation:** Firms disclosing higher operational risk systematically underperform, yielding a ~6% annual penalty after controlling for Size, Value, Profitability, Investment, and Momentum factors.

### Key Deliverables
1. **Backtesting Engine**: A rigorous, point-in-time, calendar-time long-short backtest with `min_names=20` diversification constraint.
2. **Factor Attribution**: Fama-French 5-factor + Momentum regressions with Newey-West standard errors.
3. **Machine Learning Layer**: An expanding-window walk-forward ML pipeline (LightGBM) combining 16 operational risk signal features.
4. **Interactive Benchmark Tool**: A Streamlit application for analyzing the operational risk trajectory of individual companies versus their peers.

---

## Repository Structure

```text
├── config/                  # Configuration (sample windows, filters, hyperparams)
├── src/
│   ├── data/                # Data ingestion (risk scores, marketing, EDGAR, yfinance, FF factors)
│   ├── features/            # Signal construction and point-in-time panel generation
│   ├── backtest/            # Portfolio construction, metrics, factor regressions
│   ├── ml/                  # Walk-forward validation, LightGBM models
│   ├── analysis/            # Text diagnostics (autocorrelation, prob-count correlations)
│   └── utils/               # Helpers, config loader, finance-style plotting
├── app/                     # Streamlit dashboard (reads real panel data)
├── data/
│   ├── raw/                 # Raw input datasets (not committed)
│   ├── interim/             # Intermediate cached parquets
│   └── processed/           # Final pipeline outputs (panel, signals, returns)
├── results/                 # Output tables and figures
│   ├── figures/             # Heatmaps, correlation plots
│   └── tables/              # Factor regression CSVs
├── tests/                   # Automated quality assurance
├── venv/                    # Python virtual environment (not committed)
├── Makefile                 # Build orchestration
└── requirements.txt         # Python dependencies
```

---

## Methodology & Rigor

This project strictly adheres to institutional quantitative research standards:
- **No Look-Ahead Bias**: Signals are constructed point-in-time based on actual EDGAR filing dates, with a conservative availability lag. Verified programmatically (`first_hold_start ≤ month_start`).
- **Out-of-Sample Validation**: ML models use strict expanding-window walk-forward validation with an embargo period.
- **Memory-Optimized Panel Construction**: The stock-month panel (14,388 CIKs × 273 months ≈ 4M rows) is built using a Cartesian product grid with `pd.merge_asof` for efficient backward-looking signal alignment, replacing the original iterative approach that caused out-of-memory errors.
- **Diversification Constraint**: Quintile portfolios require at least 20 names per leg to ensure statistical reliability.

---

## Getting Started

### Prerequisites
- Python 3.10+
- Windows, macOS, or Linux

### Setup
```bash
# Clone the repository
git clone https://github.com/Yugdes/Long-Short-Equity-Strategy-from-10-K-Earnings-Call-NLP-Signals.git
cd Long-Short-Equity-Strategy-from-10-K-Earnings-Call-NLP-Signals

# Create and activate virtual environment
python -m venv venv
.\venv\Scripts\activate        # Windows
# source venv/bin/activate     # macOS/Linux

# Install dependencies
pip install -r requirements.txt
```

### Data Sources
Place the following raw data in `data/raw/`:
1. **Risk Scores**: `dataset.xlsx` from Astvansh & Simpson (2026) — [OSF Repository](https://osf.io/gz93b/).
2. **Marketing Emphasis** *(optional)*: `MarketingEmphasisScores_Executive.csv` from Damavandi et al. (2025) — [GitHub](https://marketing-measures.github.io/). Place in `data/raw/marketing/`.

> **Note on Marketing Data**: The marketing dataset uses Compustat `GVKEY` identifiers while the risk dataset uses SEC `CIK`. Merging them requires a WRDS `gvkey-cik` crosswalk table. Without WRDS access, the pipeline gracefully skips the marketing merge and trains purely on the 16 operational risk features.

*SEC EDGAR filing indices, daily prices (via yfinance), and Fama-French factors are downloaded automatically.*

### Execution

**Run the full pipeline (recommended):**
```bash
python -m src.run_all
```

**Force re-run (ignore cached data):**
```bash
python -m src.run_all --force
```

**Launch the Streamlit dashboard:**
```bash
streamlit run app/streamlit_app.py
```

The pipeline runs in 5 phases:
1. **Data Ingestion** — Load risk scores, marketing data, EDGAR filings, stock prices, and Fama-French factors.
2. **Text Diagnostics** — Compute prob-count correlations and year-over-year autocorrelation of the OpRisk signal.
3. **Signals & Panel** — Construct 16 NLP signal features and build the 1.15M-row stock-month panel with point-in-time alignment.
4. **Baseline Backtest** — Sort stocks into quintile portfolios on industry-adjusted OpRisk and run CAPM/FF3/FF5/FF5+Mom factor regressions on the L/S spread.
5. **Machine Learning** — Walk-forward LightGBM predictions using expanding training windows.

---

## Authors & Citations

Developed by **Yug Mitulkumar Desai** and **Arin Mehta** as part of MS 499 (Independent Research).

**Foundational Papers (Data Sources):**
* Astvansh, V. & Simpson, J. J. (2026). *A Firm's Operational Risk: Data Set and Empirical Evidence*. Manufacturing & Service Operations Management.
* Damavandi, H., Mai, F. & Astvansh, V. (2025). *A new technique for measuring a firm's marketing emphasis*. Marketing Letters.

*Disclaimer: This repository is for educational and research purposes only and does not constitute investment advice.*
