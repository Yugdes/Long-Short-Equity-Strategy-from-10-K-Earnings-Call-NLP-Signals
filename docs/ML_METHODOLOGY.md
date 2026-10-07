# Machine Learning Methodology: Long-Short Equity Strategy

This document details the machine learning approach used to construct predictive models from the 10-K NLP risk disclosures and marketing emphasis signals.

## 1. Problem Formulation
Our goal is to predict the relative cross-sectional performance of stocks in the next month. We use **Ranked Next-Month Excess Return (`rank_excess_ret_next`)** as our target variable.

- **Why Ranks?**: In quantitative finance, predicting absolute returns is notoriously noisy. By predicting the cross-sectional rank (from 0 to 1) of excess returns, the model focuses on distinguishing winners from losers, which directly translates to a long-short portfolio construction.

## 2. Walk-Forward Cross-Validation
Financial data is non-stationary and has a strict temporal ordering. Standard K-Fold Cross Validation introduces **lookahead bias** (training on future data to predict the past). 

To prevent this, we use an **Expanding Window Walk-Forward Validation** scheme:
- **Initial Training Window**: 7 years of data.
- **Refit Frequency**: Every 12 months.
- **Embargo Period**: 1 month between the end of the training set and the start of the test set. This ensures that any overlap in return calculation (e.g., month-end pricing) does not leak into the test set.

*Example Split*:
- Train: Jan 2004 - Dec 2010 -> Embargo: Jan 2011 -> Test: Feb 2011 - Jan 2012
- Train: Jan 2004 - Dec 2011 -> Embargo: Jan 2012 -> Test: Feb 2012 - Jan 2013

## 3. Features
The model uses 16 NLP-derived operational risk signals (`S1_OpRisk`, `S2_NonOpRisk`, etc.) extracted from Item 1A of 10-K filings. 
Additionally, we incorporate **Marketing Emphasis Scores**:
- `market_orientation`
- `marketing_capabilities`
- `marketing_excellence`

The hypothesis is that firms with high operational risk might be penalized less by the market if they possess strong marketing capabilities (e.g., strong brand equity acts as a buffer). The ML model can capture these non-linear interaction effects.

## 4. The Model: LightGBM
We selected **LightGBM**, a gradient boosting framework that uses tree-based learning algorithms.
- **Why LightGBM?**: It naturally handles non-linear relationships and interactions between features (e.g., High Risk + Low Marketing vs. High Risk + High Marketing). It is also extremely fast and robust to monotonic transformations of features.
- **Hyperparameters**: We use `num_leaves=31`, `learning_rate=0.03`, and apply early stopping based on an internal validation set (the last 2 years of the training block) to prevent overfitting.

## 5. Evaluation Metrics
We evaluate the out-of-sample predictions using standard quantitative finance metrics:
1. **Information Coefficient (IC)**: The Spearman rank correlation between the model's predicted ranks and the actual future return ranks. A positive IC indicates predictive power.
2. **Information Ratio of IC (IC-IR)**: The mean IC divided by the standard deviation of ICs across time. It measures the consistency of the alpha signal.
3. **Out-of-Sample R² (OOS R²)**: Measures the reduction in mean squared error relative to a naive forecast of zero mean excess return. (Based on Gu, Kelly, Xiu 2020).

## 6. Model Persistence and Reproducibility
All trained models across every walk-forward split are serialized and saved to `results/models/` as `.joblib` files. The out-of-sample predictions are saved to `data/processed/ml_predictions.parquet` to ensure full reproducibility and to power the Streamlit dashboard visualizations.
