"""
Factor attribution — FF5+Momentum alpha regressions with Newey-West HAC
standard errors, and Fama-MacBeth cross-sectional regressions.
"""
import pandas as pd
import numpy as np
import statsmodels.api as sm
from typing import Dict, Optional, List, Tuple

from src.utils.helpers import load_config, setup_logging

logger = setup_logging("factor_regs")

# Standard factor model specifications
FACTOR_MODELS = {
    "CAPM": ["Mkt-RF"],
    "FF3": ["Mkt-RF", "SMB", "HML"],
    "FF5": ["Mkt-RF", "SMB", "HML", "RMW", "CMA"],
    "FF5+Mom": ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "Mom"],
}


def run_factor_regression(
    portfolio_returns: pd.Series,
    factors: pd.DataFrame,
    model: str = "FF5+Mom",
    nw_lags: int = 6,
) -> Dict:
    """
    Run a time-series factor regression with Newey-West HAC standard errors.

    Parameters
    ----------
    portfolio_returns : Series
        Monthly portfolio returns (or L/S spread for zero-investment)
    factors : DataFrame
        Must contain columns for the chosen model + 'RF' + 'month'
    model : str
        One of "CAPM", "FF3", "FF5", "FF5+Mom"
    nw_lags : int
        Number of Newey-West lags (default 6; also test 12)

    Returns
    -------
    dict with alpha, t_alpha, factor loadings, adj_r2, and full results object
    """
    factor_cols = FACTOR_MODELS[model]

    # Align dates
    if hasattr(portfolio_returns.index, "to_timestamp"):
        portfolio_returns.index = portfolio_returns.index.to_timestamp()
    if "month" in factors.columns:
        factors = factors.set_index("month")
        if hasattr(factors.index, "to_timestamp"):
            factors.index = factors.index.to_timestamp()

    # Merge and drop NAs
    merged = pd.DataFrame({"port_ret": portfolio_returns}).join(factors[factor_cols], how="inner")
    merged = merged.dropna()

    if len(merged) < 24:
        logger.warning(f"Only {len(merged)} observations — regression may be unreliable")

    # Dependent variable (for L/S, use raw returns; for long-only, subtract RF)
    y = merged["port_ret"]
    X = sm.add_constant(merged[factor_cols])

    # OLS with Newey-West HAC standard errors
    result = sm.OLS(y, X, missing="drop").fit(
        cov_type="HAC",
        cov_kwds={"maxlags": nw_lags},
    )

    # Extract results
    alpha_monthly = result.params["const"]
    alpha_annual = alpha_monthly * 12
    t_alpha = result.tvalues["const"]
    p_alpha = result.pvalues["const"]

    loadings = {col: result.params[col] for col in factor_cols}
    t_stats = {f"{col}_t": result.tvalues[col] for col in factor_cols}

    output = {
        "model": model,
        "alpha_monthly": alpha_monthly,
        "alpha_annual": alpha_annual,
        "t_alpha": t_alpha,
        "p_alpha": p_alpha,
        "adj_r2": result.rsquared_adj,
        "n_obs": result.nobs,
        "nw_lags": nw_lags,
        **loadings,
        **t_stats,
        "result_obj": result,
    }

    logger.info(
        f"{model}: α = {alpha_monthly:.4f}/mo ({alpha_annual:.2%}/yr), "
        f"t = {t_alpha:.2f}, adj R² = {result.rsquared_adj:.3f}"
    )

    return output


def run_all_factor_models(
    portfolio_returns: pd.Series,
    factors: pd.DataFrame,
    nw_lags: int = 6,
) -> pd.DataFrame:
    """
    Run CAPM, FF3, FF5, and FF5+Mom regressions.
    Returns a summary DataFrame.
    """
    results = []
    for model_name in FACTOR_MODELS:
        try:
            res = run_factor_regression(
                portfolio_returns, factors, model=model_name, nw_lags=nw_lags
            )
            # Remove the statsmodels object for serialization
            res.pop("result_obj", None)
            results.append(res)
        except Exception as e:
            logger.warning(f"Failed {model_name}: {e}")

    return pd.DataFrame(results)


def fama_macbeth_regression(
    panel: pd.DataFrame,
    return_col: str = "ret_next",
    signal_col: str = "S5_OpRisk_ind",
    controls: List[str] = None,
    nw_lags: int = 6,
) -> Dict:
    """
    Run Fama-MacBeth (1973) cross-sectional regressions.

    1. Each month: regress next-month returns on signals + controls
    2. Compute time-series average of the monthly coefficients
    3. Newey-West standard errors on the coefficient time series

    Parameters
    ----------
    panel : DataFrame
        Stock-month panel
    return_col : str
        Forward return column
    signal_col : str
        Primary signal column
    controls : list
        Additional control variables

    Returns
    -------
    dict with coefficient estimates, t-stats, and monthly coefficient series
    """
    if controls is None:
        controls = []

    regressors = [signal_col] + controls
    df = panel.dropna(subset=[return_col] + regressors).copy()

    # Standardize regressors within each month
    for col in regressors:
        df[col] = df.groupby("month")[col].transform(
            lambda x: (x - x.mean()) / x.std() if x.std() > 0 else 0
        )

    # Monthly cross-sectional regressions
    months = sorted(df["month"].unique())
    monthly_coefs = []

    for month in months:
        month_data = df[df["month"] == month]
        if len(month_data) < 30:
            continue

        y = month_data[return_col].values
        X = sm.add_constant(month_data[regressors].values)

        try:
            result = sm.OLS(y, X, missing="drop").fit()
            coef = {"month": month, "const": result.params[0]}
            for i, reg in enumerate(regressors):
                coef[reg] = result.params[i + 1]
            monthly_coefs.append(coef)
        except Exception:
            continue

    if not monthly_coefs:
        return {"error": "No valid monthly regressions"}

    coef_df = pd.DataFrame(monthly_coefs).set_index("month")

    # Time-series statistics of monthly coefficients
    results = {}
    for col in coef_df.columns:
        coef_series = coef_df[col]
        mean_coef = coef_series.mean()

        # Newey-West standard error
        try:
            nw_result = sm.OLS(
                coef_series.values,
                np.ones(len(coef_series)),
            ).fit(cov_type="HAC", cov_kwds={"maxlags": nw_lags})
            t_stat = nw_result.tvalues[0]
            se = nw_result.bse[0]
        except Exception:
            t_stat = mean_coef / (coef_series.std() / np.sqrt(len(coef_series)))
            se = coef_series.std() / np.sqrt(len(coef_series))

        results[col] = {
            "mean": mean_coef,
            "t_stat": t_stat,
            "se": se,
            "n_months": len(coef_series),
        }

    logger.info(f"Fama-MacBeth: signal = {signal_col}, "
                f"coef = {results[signal_col]['mean']:.4f}, "
                f"t = {results[signal_col]['t_stat']:.2f} "
                f"({len(monthly_coefs)} months)")

    return {
        "coefficients": results,
        "monthly_coefs": coef_df,
    }


def bootstrap_sharpe_ci(
    returns: pd.Series,
    n_draws: int = 5000,
    block_size: int = 6,
    confidence: float = 0.95,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """
    Stationary bootstrap confidence interval for the Sharpe ratio.

    Parameters
    ----------
    returns : Series
        Monthly returns
    n_draws : int
        Number of bootstrap replications
    block_size : int
        Expected block size for the stationary bootstrap
    confidence : float
        Confidence level (0.95 for 95% CI)

    Returns
    -------
    sharpe_point : float
        Point estimate
    ci_lower : float
        Lower bound of CI
    ci_upper : float
        Upper bound of CI
    """
    np.random.seed(seed)
    n = len(returns)
    values = returns.values
    prob = 1.0 / block_size  # Probability of starting a new block

    sharpes = []
    for _ in range(n_draws):
        # Stationary bootstrap: random block starting points with geometric block lengths
        indices = np.zeros(n, dtype=int)
        indices[0] = np.random.randint(0, n)
        for t in range(1, n):
            if np.random.random() < prob:
                indices[t] = np.random.randint(0, n)
            else:
                indices[t] = (indices[t - 1] + 1) % n

        boot_rets = values[indices]
        ann_ret = np.mean(boot_rets) * 12
        ann_vol = np.std(boot_rets, ddof=1) * np.sqrt(12)
        if ann_vol > 0:
            sharpes.append(ann_ret / ann_vol)

    sharpes = np.array(sharpes)
    alpha = (1 - confidence) / 2
    ci_lower = np.percentile(sharpes, alpha * 100)
    ci_upper = np.percentile(sharpes, (1 - alpha) * 100)
    sharpe_point = np.mean(sharpes)

    return sharpe_point, ci_lower, ci_upper


def deflated_sharpe_ratio(
    sharpe_observed: float,
    n_months: int,
    n_specs_tried: int,
    skew: float = 0.0,
    kurtosis: float = 3.0,
) -> float:
    """
    Deflated Sharpe Ratio (Bailey & López de Prado, 2014).
    Adjusts for multiple testing by computing the probability that
    the observed Sharpe exceeds what would be expected by chance
    given the number of specifications tried.

    Returns: probability that the Sharpe is genuine (0–1).
    """
    from scipy import stats

    # Expected maximum Sharpe from n_specs tries under the null
    # Using the Euler-Mascheroni approximation
    euler_mascheroni = 0.5772156649
    if n_specs_tried > 1:
        expected_max = np.sqrt(2 * np.log(n_specs_tried)) - (
            np.log(np.pi) + euler_mascheroni
        ) / (2 * np.sqrt(2 * np.log(n_specs_tried)))
    else:
        expected_max = 0

    # Standard error of the Sharpe ratio (accounting for non-normality)
    se_sharpe = np.sqrt(
        (1 + 0.5 * sharpe_observed**2 - skew * sharpe_observed +
         (kurtosis - 3) / 4 * sharpe_observed**2) / (n_months - 1)
    )

    if se_sharpe == 0:
        return 0.0

    # Probability that observed Sharpe exceeds the expected max under null
    z = (sharpe_observed - expected_max) / se_sharpe
    dsr = stats.norm.cdf(z)

    return dsr


if __name__ == "__main__":
    print("Factor regressions — run via src.run_all or Phase 4 pipeline")
