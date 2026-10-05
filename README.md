# Disclosed Operational Risk as an Equity Signal

**A systematic long-short equity strategy and risk-intelligence toolkit built on NLP-derived 10-K risk disclosures and earnings-call marketing emphasis.**

> *This project investigates whether text-derived operational risk scores from annual SEC 10-K filings are priced by the stock market, net of transaction costs and known risk factors.*

---

## Executive Summary

U.S. public firms must describe their material risks in Item 1A of the annual report (Form 10-K). Recent research (Astvansh & Simpson, 2026; Damavandi et al., 2025) used transformer models to score over 130,000 firm-year filings on eight specific risk factors. 

This project extends that work into the domain of quantitative finance by asking: **Does sorting firms on disclosed operational risk (and on an ML combination with marketing emphasis) produce return, risk-adjusted-return, or risk-prediction information beyond known factors?**

### Key Deliverables
1. **Backtesting Engine**: A rigorous, point-in-time, industry-neutral long-short calendar-time backtest.
2. **Factor Attribution**: Fama-French 5-factor + Momentum regressions with Newey-West standard errors and Fama-MacBeth cross-sectional tests.
3. **Machine Learning Layer**: An expanding-window walk-forward ML pipeline (LightGBM & ElasticNet) combining risk scores, YoY changes, and marketing-emphasis scores.
4. **Risk Prediction**: Analysis of whether disclosed risk predicts next-year volatility and drawdowns (useful for risk advisory and consulting).
5. **Interactive Benchmark Tool**: A Streamlit application for analyzing the operational risk trajectory of individual companies versus their industry peers.

---

## Repository Structure

```text
├── config/                  # Configuration (sample windows, filters, hyperparams)
├── src/
│   ├── data/                # Data ingestion pipelines (OSF, EDGAR, yfinance, Kenneth French)
│   ├── features/            # Signal construction and point-in-time panel generation
│   ├── backtest/            # Portfolio construction, metrics, factor regressions
│   ├── ml/                  # Walk-forward validation, LightGBM/ElasticNet models, SHAP
│   ├── analysis/            # Risk prediction and text diagnostics
│   └── utils/               # Helpers and finance-style plotting
├── app/                     # Streamlit dashboard
├── tests/                   # Automated quality assurance (leakage tests, etc.)
├── results/                 # Output tables and figures
└── Makefile                 # Build orchestration
```

---

## Methodology & Rigor

This project strictly adheres to institutional quantitative research standards:
- **No Look-Ahead Bias**: Signals are constructed point-in-time based on actual EDGAR filing dates, with a conservative availability lag.
- **Out-of-Sample Validation**: ML models use strict expanding-window walk-forward validation with an embargo period. The final 3 years of data act as a completely untouched holdout.
- **Realistic Costs**: Portfolio returns are evaluated net of size-tiered transaction costs (turnover penalty).
- **Multiple Testing Discipline**: Implementation of the Deflated Sharpe Ratio to correct for selection bias across multiple strategy specifications.
- **Industry Neutrality**: Core signals are cross-sectionally ranked and neutralized within Fama-French 48 industries to isolate idiosyncratic operational risk.

---

## Getting Started

### Prerequisites
- Python 3.11+
- Install dependencies: `pip install -r requirements.txt`

### Data Sources
To run the full pipeline, you will need to place the following raw data in `data/raw/`:
1. **Risk Scores**: `data set.xlsx` from Astvansh & Simpson (2026) [OSF Repository](https://osf.io/gz93b/).
2. **Marketing Emphasis**: CSV files from Damavandi et al. (2025) [GitHub](https://marketing-measures.github.io/).

*Note: The code handles the downloading of SEC EDGAR indices, daily prices via yfinance, and Fama-French factors automatically.*

### Execution
Run the entire pipeline via the Makefile:
```bash
make all
```
Or run individual phases:
```bash
make data        # Ingest raw data and build security master
make features    # Build signals and stock-month panel
make backtest    # Run portfolio backtests and factor attributions
make ml          # Run machine learning pipeline
```

Launch the Streamlit App:
```bash
make app
```

---

## Authors & Citations

Developed as part of MS 499 (Independent Research).

**Foundational Papers (Data Sources):**
* Astvansh, V. & Simpson, J. J. (2026). *A Firm's Operational Risk: Data Set and Empirical Evidence*. Manufacturing & Service Operations Management.
* Damavandi, H., Mai, F. & Astvansh, V. (2025). *A new technique for measuring a firm's marketing emphasis*. Marketing Letters.

*Disclaimer: This repository is for educational and research purposes only and does not constitute investment advice.*
