# Research design

Written 2026-11-22, **before any return-based result was computed**. Changes after this date are logged in DECISIONS.md, together with the result that motivated them.

## 1. What the two papers give us

### 1.1 Astvansh & Simpson (2026, *MSOM* 28(1):326–341): risk factors from 10-K Item 1A
- **Construction.**
  1. Item 1A is extracted from 133,394 10-Ks filed 2005–2024 for 21,821 Compustat CIKs. The section runs from the "Item 1A" marker to "Item 1B" or "Item 2" (Schnatterly et al. 2021 procedure). Filings without Item 1A and duplicate CIK-years are removed, leaving 131,920 firm-years and 16,959 firms.
  2. Eight risk dictionaries (Schnatterly et al. 2021: accounting, finance, international, legal, management, marketing, operations, technology) give **count** measures.
  3. **Transformer probabilities:** ChatGPT generated 761 sentences containing dictionary terms, and Python code labelled them by dictionary presence. Eight architectures × 8 factors = 64 fine-tuned transformers. The best model per factor (by the average of 8 metrics) was kept: BERT-base for 6 factors, DeBERTa-base for management, RoBERTa-base for technology. Applied to each Item 1A, it returns a 0–1 probability that the section "mentions" the factor.
- **Validity reported.** Near-perfect test metrics on the synthetic sentences (operations: accuracy 0.96–0.99 for 6 of 8 architectures). Two architectures (legal-BERT, RoBERTa-large) failed completely. There is no human-labelled validation on real filings. Probability means match our data to 3 decimals (DATA_AUDIT §2.5).
- **Economic result.** Operational risk at *t* is positively associated with ln(COGS) at *t*+1, with firm and year fixed effects: β = 0.247 (SE 0.041), N = 67,727. Non-operational risk (the sum of the other 7 probabilities) amplifies it: interaction β = 0.298 (SE 0.125). This survives count measures (moderation not significant there), alternative fixed effects, and 2SLS using the industry-year mean as an instrument.
- **Returns.** The paper makes **no return or valuation claims**; every return hypothesis here is ours. It cites evidence that supply-chain disruptions lower returns and raise risk (Hendricks & Singhal 2003, 2005, 2014).
- **Issues we handle:**
  - `FYEAR` is the filing year (DATA_AUDIT §2.2).
  - Probabilities correlate only weakly with counts (operations ρ_s = 0.38).
  - Probability falls with Item 1A length (ρ_s = −0.38), so it behaves like a *share* of risk language.
  - Strong cross-factor correlation (up to 0.54).
  - Large secular growth in Item 1A length.
  - The binary labels are degenerate.
  - Internal inconsistencies in the paper (the text says "five" non-operational factors but lists seven; firm size is defined as ln(revenue) in the text but ln(AT) in Table 5).

### 1.2 Damavandi, Mai & Astvansh (2025, *Marketing Letters* 36:823–837): marketing emphasis from earnings calls
- **Construction.**
  1. FactSet XML transcripts are parsed. Management presentations and answers (executive file) are separated from analyst questions (analyst file).
  2. Each paragraph is embedded with SBERT `all-mpnet-base-v2` (768 dimensions).
  3. Each of 19 lower-order components is defined by the centroid of its dictionary-term embeddings.
  4. A component score is the cosine similarity of each paragraph with the centroid, averaged over the call. ZCA whitening is applied within each higher-order group to counter anisotropy.
  5. The higher-order constructs are market orientation (8 components), marketing capabilities (8) and marketing excellence (3).
- **Validity reported.**
  - Cronbach's α of 0.95 / 0.98 / 0.97.
  - Convergent correlations with dictionary counts of only 0.17 / 0.21 / 0.21.
  - Nomological validity: adding the 19 scores lowers 5-fold-CV MSE for next-year advertising intensity in all 7 ML models (e.g. random forest 34 → 24 ×10⁻⁵). The gains are weaker and mixed for R&D, profit and Tobin's q.
  - Their CV uses random folds on a firm-year panel, so the same firm appears in training and test sets.
- **Returns.** **None reported.** Their discussion suggests that financial stakeholders "would value" the scores.
- **Issues we handle:**
  - The raw constructs are almost collinear in our data (ρ = 0.89–0.97), so the ZCA components carry the distinct information.
  - v2 extends coverage to 2025Q4, with multi-GVKEY cells.
  - The quarter is the call's calendar quarter.

## 2. Hypotheses (fixed now; signs pre-specified)

The literature review (docs/LITERATURE.md) informs the priors: Campbell et al. 2014; Kravet & Muslu 2013; Cohen, Malloy & Nguyen 2020; Price et al. 2012; Edeling & Fischer 2016; Gu, Kelly & Xiu 2020; Harvey, Liu & Zhu 2016; McLean & Pontiff 2016.

| ID | Hypothesis | Predicted sign | Deciding test (dev window) |
|---|---|---|---|
| **H1** Risk levels are priced as risk, not mispricing | The *level* of disclosed risk (operations probability; aggregate risk rank) predicts higher future return volatility, and no factor-adjusted alpha | Volatility: + ; FF5+Mom alpha of the level L/S ≈ 0 (two-sided) | (a) FM regression of next-12-month realised volatility on OpRisk with controls; (b) decile L/S on FF48-demeaned OpRisk rank: FF5+Mom α, NW t |
| **H2** Risk *increases* predict lower returns (Lazy-Prices logic) | Firms whose risk scores rise versus their previous 10-K underperform over the following months | High-Δ minus low-Δ < 0 | Decile L/S on ΔOpRisk and on Δ(aggregate risk rank); FM coefficient with controls; \|t\| ≥ 3 hurdle |
| **H3** Marketing emphasis signals growth and pricing power | Higher marketing capabilities and excellence (and increases in them) in calls predict higher returns | + | Decile L/S on the construct rank and the YoY change; FM with controls, on the marketing-matched sample |
| **H4** Risk × marketing carries information neither carries alone | Marketing emphasis attenuates the (negative) return implication of high or rising operational risk | Interaction > 0 for OpRisk × marketing | 3×3 conditional double sorts; FM with the interaction term; ML incremental IC of the interaction features (matched sample) |
| **H5** An ML combination adds out-of-sample value | A walk-forward model on text + control features beats a controls-only model on the same rows, and its long-short survives costs | ΔIC > 0; net α > 0 | Monthly IC difference (paired t, NW); decile L/S of the ensemble: VW, net of 25 bps, FF5+Mom α |

All hypotheses are evaluated on the **development window**. The **locked test window (2022-01 → 2025-06)** is run once, after the design freezes, for the primary strategy and the H1–H5 headline tests. Both sets of numbers are reported, whether or not they agree.

## 3. Primary specification (fixed before results)

- **Universe (D-009):** price ≥ $5 and market cap ≥ $100M at the end of *t*−1; ≥ 12 months of return history; excluding financials and utilities; a live risk signal (D-007). Return months 2010-01 → 2025-06 (D-008).
- **Target:** month-*t* excess return, ranked cross-sectionally to [−0.5, 0.5] for ML training. Robustness: the cumulative 3-month excess return *t* … *t*+2.
- **Features at formation** (each known before month *t* begins):
  - *Controls:* log size, log B/M, momentum 12-2, 1-month reversal, 12-month volatility, CAPM β (36 months), Amihud illiquidity, asset growth, gross profitability.
  - *Risk text:* ranks of the 8 probabilities, count density, log words, aggregate non-operational rank, Δ vs the previous filing, surprise vs the firm's own history, FF48-demeaned OpRisk.
  - *Marketing:* ranks of the 3 constructs and the 19 ZCA components, QoQ and YoY Δ, analysts' customer orientation.
  - *Interactions:* OpRisk × non-operational risk; OpRisk × each marketing construct.
- **Models:** ElasticNet, LightGBM, XGBoost, MLP, and an equal-weight rank ensemble. Expanding-window walk-forward with annual refits: first test year 2014, training from 2010. A 1-month embargo; hyperparameters tuned on the last 2 years of each training window only.
- **Primary strategy:** value-weighted **decile** long-short (top minus bottom decile of the ensemble score), monthly rebalance, **net of 25 bps one-way**, reported with FF5+Mom α and Newey-West(6) t. EW, quintile and FF12-sector-neutral versions are secondary.
- **Hurdles:** |t| ≥ 3.0 for headline alphas and FM premia (Harvey, Liu & Zhu 2016); deflated Sharpe ratio using the number of specs in `results/spec_log.csv`; PBO by CSCV.

## 4. Multiple-testing budget
Every portfolio or model specification evaluated on the dev window is appended to `results/spec_log.csv` (id, date, signal, universe, weighting, metric values), and the DSR uses that count. "Exploratory" specs are not dropped from the count.

## 5. Known limitations (expanded in Phase 6)
- **Survivorship:** returns exist only for firms with a current SEC ticker (coverage of risk firm-years: ~28% pre-2010, rising to 60–76% by the 2020s). There are no delisting returns.
- Current, not historical, SIC codes and tickers.
- Market cap and book equity come from XBRL and start ~2009–2011.
- Yahoo data is unofficial and occasionally revised; the downloads are frozen in `data/raw/yahoo/`.
- The risk scores come from transformers trained on synthetic, dictionary-labelled sentences, with no human validation on real 10-Ks. The marketing scores are cosine similarities with near-collinear constructs.

---
## Erratum (2026-11-22, after the design freeze; see DECISIONS D-029/D-030)
- The locked test window ends **2025-05**, not 2025-06. In 2025-06 the universe halves because FY2024 10-Ks are absent from the dataset (D-029).
- The prob–count and prob–length figures in §1.1 were computed on all rows, including empty Item 1A sections coded as zero. On sections of at least 1,000 words, the operations probability's rank correlation is −0.07 with its dictionary count and −0.38 with length (`results/json/explorer.json`, `text_diag`).
- The body of this document is left as written before results, as the pre-registration.
