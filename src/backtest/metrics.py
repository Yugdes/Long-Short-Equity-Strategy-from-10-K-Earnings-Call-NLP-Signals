"""Performance metrics for monthly return series (arithmetic annualisation, the convention for Sharpe)."""
import numpy as np
import pandas as pd
from scipy import stats

PPY = 12


def nw_tstat(x: pd.Series, lags: int = 6) -> float:
    """Newey-West t-stat of the mean of x."""
    import statsmodels.api as sm
    x = pd.Series(x).dropna()
    res = sm.OLS(x.values, np.ones(len(x))).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
    return float(res.tvalues[0])


def max_drawdown(r: pd.Series) -> tuple[float, object, object]:
    wealth = (1 + r.fillna(0)).cumprod()
    dd = wealth / wealth.cummax() - 1
    trough = dd.idxmin()
    peak = wealth.loc[:trough].idxmax()
    return float(dd.min()), peak, trough


def drawdown_series(r: pd.Series) -> pd.Series:
    wealth = (1 + r.fillna(0)).cumprod()
    return wealth / wealth.cummax() - 1


def perf_stats(r: pd.Series, nw_lags: int = 6) -> dict:
    """r: monthly excess (or zero-investment) returns."""
    r = r.dropna()
    mu, sd = r.mean(), r.std(ddof=1)
    downside = np.sqrt(np.mean(np.minimum(r, 0) ** 2))
    mdd, peak, trough = max_drawdown(r)
    cagr = (1 + r).prod() ** (PPY / len(r)) - 1
    return {
        "n_months": len(r), "ann_mean": mu * PPY, "ann_vol": sd * np.sqrt(PPY),
        "sharpe": mu / sd * np.sqrt(PPY) if sd > 0 else np.nan,
        "sortino": mu / downside * np.sqrt(PPY) if downside > 0 else np.nan,
        "t_mean_nw": nw_tstat(r, nw_lags), "cagr": cagr, "max_dd": mdd,
        "calmar": cagr / abs(mdd) if mdd < 0 else np.nan,
        "hit_rate": float((r > 0).mean()), "skew": float(stats.skew(r)), "kurt": float(stats.kurtosis(r, fisher=False)),
        "worst_month": float(r.min()), "best_month": float(r.max()),
        "dd_peak": str(peak), "dd_trough": str(trough),
    }


def deflated_sharpe(sr: float, n_obs: int, skew: float, kurt: float, sr_trials: np.ndarray) -> dict:
    """Bailey & López de Prado (2014). Every Sharpe here is PER PERIOD (monthly, not annualised).

    sr_trials: Sharpe ratios of all strategy variants tried (the multiple-testing family).
    Returns the expected max Sharpe under the null (SR0) and DSR = P(true SR > SR0)."""
    sr_trials = np.asarray(sr_trials, dtype=float)
    n = len(sr_trials)
    var = np.var(sr_trials, ddof=1) if n > 1 else 0.0
    g = 0.5772156649
    sr0 = np.sqrt(var) * ((1 - g) * stats.norm.ppf(1 - 1 / n) + g * stats.norm.ppf(1 - 1 / (n * np.e))) if n > 1 else 0.0
    se = np.sqrt((1 - skew * sr + (kurt - 1) / 4 * sr ** 2) / (n_obs - 1))
    return {"sr0_monthly": float(sr0), "n_trials": n, "dsr": float(stats.norm.cdf((sr - sr0) / se))}
