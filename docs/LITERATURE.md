# Literature Review

## Introduction

This project asks whether two text-derived firm characteristics predict next-month cross-sectional US stock returns once known factors are controlled for. It then asks whether a long-short strategy built on them survives realistic backtest hygiene. The first signal set is the transformer-based 10-K Item 1A risk-factor scores of Astvansh and Simpson (2026): eight risk categories (accounting, finance, international, legal, management, marketing, operations, technology) for 131,920 firm-years from 2005 to 2024. The second is the SBERT-based earnings-call marketing-emphasis scores of Damavandi, Mai and Astvansh (2025): 19 components aggregated into market orientation, marketing capabilities and marketing excellence.

The review is organised in six strands: machine learning in asset pricing; textual analysis of filings; risk-factor disclosure; earnings-call language; marketing and firm value; and methodology and backtest hygiene. Each entry gives a citation verified in this session, the key finding, and how the work bears on our design. A final section draws out what the literature predicts for our hypotheses.

**Data-source papers (verified, not reviewed further).**

- Astvansh, V., & Simpson, J. J. (2026). A firm's operational risk: Data set and empirical evidence. *Manufacturing & Service Operations Management*, 28(1), 326–341. https://doi.org/10.1287/msom.2025.0134. Verified via Crossref API (`api.crossref.org/works/10.1287/msom.2025.0134`) and https://pubsonline.informs.org/doi/fpi/10.1287/msom.2025.0134. The paper links higher disclosed operational risk to higher operational cost in the following year. That is evidence that the scores carry fundamental content, but it says nothing about returns.
- Damavandi, H., Mai, F., & Astvansh, V. (2025). A new technique for measuring a firm's marketing emphasis. *Marketing Letters*, 36(4), 823–837. https://doi.org/10.1007/s11002-025-09781-7. Verified via doi.org/Crossref metadata and https://ideas.repec.org/a/kap/mktlet/v36y2025i4d10.1007_s11002-025-09781-7.html.

---

## 1. Machine learning in asset pricing

**Gu, S., Kelly, B., & Xiu, D. (2020). Empirical asset pricing via machine learning. *The Review of Financial Studies*, 33(5), 2223–2273. https://doi.org/10.1093/rfs/hhaa009**
Verified: https://econpapers.repec.org/RePEc:oup:rfinst:v:33:y:2020:i:5:p:2223-2273. Numbers below are from the NBER revision (w25398, Sept. 2019): https://www.nber.org/system/files/working_papers/w25398/w25398.pdf

*Key finding.* The paper compares linear, penalised, tree and neural-network forecasts of individual stock returns built from a large panel of characteristics.
- Trees and neural networks win, because they capture nonlinear interactions among predictors.
- The dominant signals are variants of momentum, liquidity and volatility.
- Monthly stock-level out-of-sample R² is small even for the best models: 0.33–0.40% for neural networks (peak 0.40% for NN3), 0.11% for elastic net, and −3.46% for unregularised OLS on all predictors.
- Even so, a value-weighted long-short decile spread on neural-network forecasts earns an annualised out-of-sample Sharpe ratio of about 1.35 (2.45 equal-weighted).
- Turnover for the neural-network strategies runs at 110–130% per month.

*Relevance.* This is the template for our walk-forward design: expanding or rolling training windows, separate validation and test periods, and an ElasticNet / LightGBM / XGBoost / MLP model menu. It also sets realistic expectations. Monthly predictive R² well below 1% is normal and still economically meaningful. Because GKX's gains come mostly from price-based signals, we must measure the incremental contribution of the text features (models with versus without them), not raw ML performance.

---

## 2. Textual analysis of financial filings

**Loughran, T., & McDonald, B. (2011). When is a liability not a liability? Textual analysis, dictionaries, and 10-Ks. *The Journal of Finance*, 66(1), 35–65. https://doi.org/10.1111/j.1540-6261.2010.01625.x**
Verified: https://ideas.repec.org/a/bla/jfinan/v66y2011i1p35-65.html. Abstract read from https://www.uts.edu.au/globalassets/sites/default/files/adg_cons2015_loughran-mcdonald-je-2011.pdf

*Key finding.* In 10-Ks filed from 1994 to 2008, almost three-fourths of the words the Harvard IV-4 dictionary classifies as negative are typically not negative in financial contexts. The authors build a finance-specific negative list and five other lists. These lists relate to filing-period returns, trading volume, return volatility, fraud, material weakness and unexpected earnings.

*Relevance.* Domain-specific measurement matters. This is the main justification for using purpose-built transformer and SBERT scores rather than generic sentiment. Loughran–McDonald negative-tone and uncertainty measures are natural baseline text controls: our scores should add information beyond them.

**Cohen, L., Malloy, C., & Nguyen, Q. (2020). Lazy prices. *The Journal of Finance*, 75(3), 1371–1415. https://doi.org/10.1111/jofi.12885**
Verified: https://ideas.repec.org/a/bla/jfinan/v75y2020i3p1371-1415.html

*Key finding.* Year-over-year changes in the language and construction of 10-K/10-Q filings predict future returns, earnings, profitability, news and bankruptcies. Shorting "changers" and buying "non-changers" earns up to 188 basis points per month in alpha (over 22% per year). There is no announcement effect, which the authors read as investor inattention.

*Relevance.* This is the most direct precedent for a slow-diffusion text signal at monthly horizons. It motivates testing year-over-year changes in risk scores as well as levels, with the expected sign negative for increases. The absence of an announcement effect supports skipping the filing month without losing the signal. The predictability should appear in the months after the 10-K becomes public.

---

## 3. Risk-factor disclosure (Item 1A)

**Campbell, J. L., Chen, H., Dhaliwal, D. S., Lu, H., & Steele, L. B. (2014). The information content of mandatory risk factor disclosures in corporate filings. *Review of Accounting Studies*, 19(1), 396–455. https://doi.org/10.1007/s11142-013-9258-3**
Verified: https://experts.arizona.edu/en/publications/the-information-content-of-mandatory-risk-factor-disclosures-in-c/ and https://ideas.repec.org/a/spr/reaccs/v19y2014i1d10.1007_s11142-013-9258-3.html

*Key finding.* The study covers the risk-factor section the SEC mandated from 2005. Firms facing greater risk disclose more risk factors, and the type of risk a firm faces shapes how much of its disclosure goes to that risk type. The disclosed information is reflected in systematic risk, idiosyncratic risk, information asymmetry and firm value. Disclosures therefore appear firm-specific rather than pure boilerplate.

*Relevance.* Our sample starts in 2005, the first mandate year. The finding that disclosure *levels* load on systematic and idiosyncratic risk means a level signal could earn a risk premium, not an anomaly return. Level portfolios must therefore be tested against market beta and idiosyncratic volatility, not only against FF5 plus momentum. The category-specific finding supports keeping the eight categories separate rather than collapsing them into one total-risk score.

**Kravet, T., & Muslu, V. (2013). Textual risk disclosures and investors' risk perceptions. *Review of Accounting Studies*, 18(4), 1088–1122. https://doi.org/10.1007/s11142-013-9228-9**
Verified: https://ideas.repec.org/a/spr/reaccs/v18y2013i4d10.1007_s11142-013-9228-9.html and doi.org/Crossref metadata

*Key finding.* Annual *increases* in textual risk disclosure go with higher return volatility and trading volume around and after the filing, and with more dispersed analyst forecast revisions. Risk disclosures therefore raise investors' risk perceptions rather than resolve uncertainty. The effects are weaker for firm-level disclosure that deviates from industry–year peers, which the authors read as consistent with critics' view that firm-level risk disclosures tend to be boilerplate.

*Relevance.* This supports constructing *change* signals and checking whether they predict volatility as well as mean returns. A risk-perception channel could mean higher required returns (positive sign) or temporary overreaction (negative, then reversal). Industry-year demeaning of the scores is a natural robustness check.

---

## 4. Earnings-call language and returns

**Price, S. M., Doran, J. S., Peterson, D. R., & Bliss, B. A. (2012). Earnings conference calls and stock returns: The incremental informativeness of textual tone. *Journal of Banking & Finance*, 36(4), 992–1011. https://doi.org/10.1016/j.jbankfin.2011.10.013**
Verified: https://ideas.repec.org/a/eee/jbfina/v36y2012i4p992-1011.html and https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1625863

*Key findings.*
- Conference-call tone significantly predicts abnormal returns and trading volume.
- Over the 60 trading days after the call, tone dominates the earnings surprise.
- The Q&A portion adds explanatory power for post-earnings-announcement drift, mainly among non-dividend payers.
- A context-specific dictionary outperforms the Harvard IV-4 dictionary.

*Relevance.* Call language moves prices beyond the numbers, and part of the response is delayed over roughly a quarter. That supports a next-month horizon for call-based signals. Our marketing-emphasis scores are about *content* (strategic orientation), not tone, so tone and the earnings surprise (SUE) should be included as controls. That way any marketing-emphasis effect is not just a repackaged PEAD or tone effect. The dividend-payer split suggests testing heterogeneity by cash-flow uncertainty.

---

## 5. Marketing and firm value

**Srinivasan, S., & Hanssens, D. M. (2009). Marketing and firm value: Metrics, methods, findings, and future directions. *Journal of Marketing Research*, 46(3), 293–312. https://doi.org/10.1509/jmkr.46.3.293**
Verified: https://journals.sagepub.com/doi/10.1509/jmkr.46.3.293 and doi.org/Crossref metadata

*Key finding.* The review frames the marketing–finance interface. It surveys investor-response metrics and the analytical models used to translate marketing resource allocations and their performance consequences into firm-value effects, and it sets out the open research questions.

*Relevance.* This gives the conceptual case for why marketing emphasis could carry information about intangible assets that investors price slowly. It also distinguishes stock-return response models from valuation-level models. Our design is a return-response test, so the right null is that marketing information is already priced.

**Edeling, A., & Fischer, M. (2016). Marketing's impact on firm value: Generalizations from a meta-analysis. *Journal of Marketing Research*, 53(4), 515–534. https://doi.org/10.1509/jmr.14.0046**
Verified: https://www.jstor.org/stable/44134929 and https://journals.sagepub.com/doi/abs/10.1509/jmr.14.0046

*Key finding.* Across 488 elasticities from 83 studies:
- The mean firm-value elasticity is 0.04 for advertising expenditure and 0.54 for marketing assets (0.72 for customer-related assets, 0.33 for brand-related assets).
- Advertising elasticities are lower in more concentrated industries, and marketing-asset elasticities are higher in recessions.
- Design choices strongly affect estimates: the firm-value metric, omitted controls, and failure to address endogeneity.

*Relevance.* Capability- and asset-like constructs (our marketing capabilities and excellence) should matter more than spending-like ones. Effects may be state-dependent, which justifies recession subsample tests. The sensitivity to omitted controls argues for including size, book-to-market, profitability, investment and industry fixed effects in the Fama–MacBeth regressions.

---

## 6. Methodology and backtest hygiene

**Fama, E. F., & MacBeth, J. D. (1973). Risk, return, and equilibrium: Empirical tests. *Journal of Political Economy*, 81(3), 607–636. https://doi.org/10.1086/260061**
Verified: https://ideas.repec.org/a/ucp/jpolec/v81y1973i3p607-36.html and doi.org/Crossref metadata

*Key finding.* The paper tests the two-parameter (CAPM) risk–return relation for NYSE stocks by running cross-sectional regressions each period and averaging the slopes over time. The time-series variability of those slopes supplies the standard errors.

*Relevance.* We use this two-pass procedure to estimate the monthly return premium on each text score, controlling jointly for firm characteristics. It handles cross-sectional correlation of residuals, but it does not handle persistence in the slope series. That is why we pair it with Newey–West errors.

**Newey, W. K., & West, K. D. (1987). A simple, positive semi-definite, heteroskedasticity and autocorrelation consistent covariance matrix. *Econometrica*, 55(3), 703–708. https://doi.org/10.2307/1913610**
Verified: https://www.econometricsociety.org/publications/econometrica/1987/05/01/notes-and-comments-simple-positive-semi-definite and https://www.jstor.org/stable/1913610

*Key finding.* The paper gives a covariance estimator that stays positive semi-definite and is consistent under heteroskedasticity and autocorrelation of unknown form, using Bartlett-kernel weights on autocovariances.

*Relevance.* Text scores update annually (10-K) or quarterly (calls), so monthly Fama–MacBeth slopes and long-short returns built on them will be autocorrelated. HAC standard errors with a lag length covering the update cycle are needed so we do not overstate t-statistics.

**Carhart, M. M. (1997). On persistence in mutual fund performance. *The Journal of Finance*, 52(1), 57–82. https://doi.org/10.1111/j.1540-6261.1997.tb03808.x**
Verified: https://ideas.repec.org/a/bla/jfinan/v52y1997i1p57-82.html and https://onlinelibrary.wiley.com/doi/10.1111/j.1540-6261.1997.tb03808.x

*Key finding.* In a survivor-bias-free sample, common factors in stock returns (including a one-year momentum factor) and investment expenses almost completely explain persistence in mutual fund returns. The results do not support skilled or informed managers.

*Relevance.* This is the source of the momentum (UMD) control. Text signals released around filings and calls may correlate with recent returns, so alphas must be reported net of momentum.

**Fama, E. F., & French, K. R. (2015). A five-factor asset pricing model. *Journal of Financial Economics*, 116(1), 1–22. https://doi.org/10.1016/j.jfineco.2014.10.010**
Verified: https://econpapers.repec.org/RePEc:eee:jfinec:v:116:y:2015:i:1:p:1-22 and doi.org/Crossref metadata

*Key finding.* Adding profitability (RMW) and investment (CMA) factors to the market, size and value factors captures average-return patterns better than the three-factor model, although some anomalies remain.

*Relevance.* FF5 plus UMD is our benchmark factor model for portfolio alphas. Risk disclosures plausibly correlate with profitability and investment (riskier, more aggressive investors may disclose more), so a raw-return spread that disappears under FF5 would be reclassified as known-factor exposure rather than new information.

**Harvey, C. R., Liu, Y., & Zhu, H. (2016). … and the cross-section of expected returns. *The Review of Financial Studies*, 29(1), 5–68. https://doi.org/10.1093/rfs/hhv059**
Verified: https://academic.oup.com/rfs/article/29/1/5/1843824, doi.org/Crossref metadata, and abstract at https://www.nber.org/papers/w20592

*Key finding.* Hundreds of factors have been proposed (at least 316 tested). Given the extensive data mining, a t-ratio of 2.0 is an inadequate hurdle. A multiple-testing framework that allows for correlated tests implies a newly proposed factor today should clear a t-ratio of about 3.0. The authors conclude that many published findings in financial economics are likely false.

*Relevance.* We adopt |t| ≥ 3.0 as the primary significance hurdle for any single text factor. We also pre-register the signal list (eight risk categories, three marketing aggregates, levels and changes) so the size of the testing family is explicit.

**Bailey, D. H., & López de Prado, M. (2014). The deflated Sharpe ratio: Correcting for selection bias, backtest overfitting, and non-normality. *The Journal of Portfolio Management*, 40(5), 94–107. https://doi.org/10.3905/jpm.2014.40.5.094**
Verified: https://www.pm-research.com/content/iijpormgmt/40/5/94 and https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2460551

*Key finding.* When many strategy configurations are tried, the expected maximum Sharpe ratio is positive even if every true Sharpe ratio is zero. The deflated Sharpe ratio (DSR) is a probabilistic Sharpe ratio whose benchmark is raised to that expected maximum, given the number of independent trials and the variance of trial Sharpe ratios. It also adjusts for sample length, skewness and kurtosis.

*Relevance.* We will log every strategy variant tried (signal × weighting × model × hyperparameter grid). We will then report the DSR of the selected long-short strategy, not only its raw Sharpe ratio.

**Bailey, D. H., Borwein, J., López de Prado, M., & Zhu, Q. J. (2017). The probability of backtest overfitting. *The Journal of Computational Finance*, 20(4). https://doi.org/10.21314/JCF.2016.322**
Verified: https://www.risk.net/journal-of-computational-finance/2471206/the-probability-of-backtest-overfitting (Vol. 20, No. 4, April 2017), Crossref API, and https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2326253. The page range could not be verified in this session.

*Key finding.* Standard holdout validation is unreliable for investment backtests. The paper defines the probability of backtest overfitting (PBO) as the probability that the strategy that is optimal in-sample underperforms the median of the trials out of sample. It estimates PBO non-parametrically by combinatorially symmetric cross-validation (CSCV).

*Relevance.* Applying CSCV to the matrix of monthly returns across all configurations tests whether our model and hyperparameter selection generalises. This complements the walk-forward test set.

**McLean, R. D., & Pontiff, J. (2016). Does academic research destroy stock return predictability? *The Journal of Finance*, 71(1), 5–32. https://doi.org/10.1111/jofi.12365**
Verified: https://onlinelibrary.wiley.com/doi/abs/10.1111/jofi.12365 and doi.org/Crossref metadata. Published-abstract figures confirmed via https://www.gwern.net/doc/economics/2016-mclean.pdf. Working-paper version: https://www.fmg.ac.uk/sites/default/files/2020-08/Jeffrey-Pontiff.pdf

*Key finding.* Across 97 published cross-sectional predictors:
- Portfolio returns are 26% lower out-of-sample and 58% lower after publication. The out-of-sample drop is an upper bound on data-mining bias, which leaves roughly 32% attributable to publication-informed trading.
- Post-publication declines are larger for predictors that are cheaper to arbitrage: large, liquid, low-idiosyncratic-risk stocks.

*Relevance.* Both datasets are recent publications, and the underlying constructs (risk disclosure, 10-K text change, call tone) have been in the literature since about 2011–2014. We should expect, and test for, weaker performance in the later sample. Concretely, we will split at the Lazy Prices / Campbell et al. publication dates and examine 2020–2024 separately. We will also report results separately for large, liquid stocks.

---


**Jensen, T. I., Kelly, B., & Pedersen, L. H. (2023). Is there a replication crisis in finance? *The Journal of Finance*, 78(5), 2465–2518. https://doi.org/10.1111/jofi.13249**
Verified (added 2026-11-22 by the main session): https://research.cbs.dk/en/publications/is-there-a-replication-crisis-in-finance/ and https://econpapers.repec.org/RePEc:nbr:nberwo:28432
They build 153 factors in 93 countries with a common methodology and a Bayesian model that accounts for publication bias and multiple testing; most factors replicate. *Use here:* their capped value weights (market caps winsorised at the NYSE 80th percentile) are the template for D-022.

## Implications for our hypotheses

**(i) Levels versus changes, and direction.**
- *Levels.* Campbell et al. (2014) show that disclosed risk levels load on systematic and idiosyncratic risk and on firm value. Under a risk-based reading, higher risk-factor scores should predict higher, not lower, expected returns, but much of that premium should be absorbed by beta, idiosyncratic volatility and FF5 exposures. We therefore expect level signals to show weak raw spreads and near-zero factor-adjusted alphas.
- *Changes.* Cohen et al. (2020) and Kravet and Muslu (2013) point to changes as the informative object. Increases in disclosed risk raise perceived risk and volatility. Firms that change their filings subsequently underperform, with no announcement-period reaction. Our main directional prediction is that year-over-year *increases* in risk scores predict *negative* next-month (and following-months) abnormal returns. Our own conjecture, not established by the literature, is that the effect is strongest for categories tied to operating fundamentals (operations, finance, legal), consistent with Astvansh and Simpson's link between disclosed operational risk and next-year operating cost.
- *Marketing emphasis.* Edeling and Fischer (2016) find positive firm-value effects concentrated in marketing assets and capabilities. If investors under-react to qualitative call content, as Price et al. (2012) find for tone, higher marketing capabilities and excellence should predict positive returns. The prior on magnitude is small.

**(ii) Survival after factor controls and post-publication decay.**
- The Lazy Prices alpha survives five-factor adjustment, and Price et al.'s tone effect survives controls for the earnings surprise, so text signals can carry information that standard factors miss.
- Campbell et al. and Fama–French (2015) imply that part of any risk-score spread is profitability, investment or risk exposure, so we expect meaningful shrinkage from raw to FF5 + UMD alphas.
- McLean and Pontiff (2016) imply a further decline of about one-half after publication, larger among liquid large caps. A plausible expectation is that any alpha is concentrated in earlier years and in smaller, less liquid stocks, which raises transaction-cost concerns for the implementable strategy.

**(iii) Expected magnitude of monthly rank IC.**
None of the verified works reports rank information coefficients for text signals directly, so this is an inference, not a cited range.
- Gu, Kelly and Xiu report pooled monthly stock-level out-of-sample R² of about 0.3–0.4% for their best models, which use hundreds of predictors dominated by momentum, liquidity and volatility. An R² of 0.4% corresponds to a forecast–return correlation of roughly √0.004 ≈ 0.06.
- A single slow-moving fundamental or text characteristic should sit well below that ceiling. We therefore treat a mean monthly cross-sectional rank IC of about 0.01–0.03 as a realistic target for individual text scores, with ML composites possibly reaching 0.03–0.05.
- At those magnitudes, an IC series over roughly 200 test months needs to be persistent and positive in most months to clear the multiple-testing hurdle below. Low turnover relative to GKX's 110–130% per month is a practical advantage of annual and quarterly signals.

**(iv) Multiple-testing hurdles.**
- The family of tests is large: 8 risk categories and 3 marketing aggregates (plus 19 components), in levels and changes, across equal and value weighting, four ML models with hyperparameter grids, and several factor models.
- Following Harvey, Liu and Zhu (2016), single-factor Fama–MacBeth premia and long-short alphas should clear |t| ≥ 3.0 using Newey–West standard errors, not the conventional 2.0.
- For the final strategy, we will report the deflated Sharpe ratio (Bailey & López de Prado, 2014) using the logged count of trials, and the probability of backtest overfitting via CSCV (Bailey et al., 2017). A PBO well below 0.5 and a DSR above 0.95 are the evidence standards we commit to before examining test-period results.
