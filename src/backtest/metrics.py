"""
Performance metrics for portfolio backtests.
Sharpe, Sortino, max drawdown, Calmar, hit rate, and related statistics.
"""
import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple


def annualized_return(returns: pd.Series, periods_per_year: int = 12) -> float:
    """Compute annualized return from a series of periodic returns."""
    total = (1 + returns).prod()
    n_years = len(returns) / periods_per_year
    if n_years <= 0:
        return 0.0
    return total ** (1 / n_years) - 1


def annualized_volatility(returns: pd.Series, periods_per_year: int = 12) -> float:
    """Compute annualized volatility."""
    return returns.std() * np.sqrt(periods_per_year)


def sharpe_ratio(
    returns: pd.Series,
    rf: Optional[pd.Series] = None,
    periods_per_year: int = 12,
) -> float:
    """
    Compute annualized Sharpe ratio.
    For zero-investment (L/S) portfolios, rf should be None (returns are already excess).
    """
    if rf is not None:
        excess = returns - rf
    else:
        excess = returns

    ann_ret = annualized_return(excess, periods_per_year)
    ann_vol = annualized_volatility(excess, periods_per_year)

    if ann_vol == 0:
        return 0.0
    return ann_ret / ann_vol


def sortino_ratio(
    returns: pd.Series,
    rf: Optional[pd.Series] = None,
    periods_per_year: int = 12,
    mar: float = 0.0,
) -> float:
    """Compute annualized Sortino ratio (downside deviation only)."""
    if rf is not None:
        excess = returns - rf
    else:
        excess = returns

    ann_ret = annualized_return(excess, periods_per_year)
    downside = excess[excess < mar]
    downside_std = downside.std() * np.sqrt(periods_per_year)

    if downside_std == 0:
        return np.inf if ann_ret > 0 else 0.0
    return ann_ret / downside_std


def max_drawdown(returns: pd.Series) -> Tuple[float, object, object, Optional[int]]:
    """
    Compute maximum drawdown and its characteristics.

    Returns
    -------
    mdd : float
        Maximum drawdown (negative number)
    peak_date : index value
        Date of the peak before the drawdown
    trough_date : index value
        Date of the trough
    recovery_months : int or None
        Months to recover (None if not recovered)
    """
    cum = (1 + returns).cumprod()
    running_max = cum.cummax()
    drawdown = (cum - running_max) / running_max

    mdd = drawdown.min()
    trough_idx = drawdown.idxmin()

    # Find peak date (last time at the running max before trough)
    peak_mask = cum[:trough_idx] == running_max[:trough_idx]
    peak_idx = peak_mask[peak_mask].index[-1] if peak_mask.any() else returns.index[0]

    # Find recovery date
    recovery_idx = None
    recovery_months = None
    if trough_idx is not None:
        post_trough = cum[trough_idx:]
        recovery_mask = post_trough >= running_max[trough_idx]
        if recovery_mask.any():
            recovery_idx = recovery_mask[recovery_mask].index[0]
            # Approximate months to recovery
            try:
                trough_pos = returns.index.get_loc(trough_idx)
                recovery_pos = returns.index.get_loc(recovery_idx)
                recovery_months = recovery_pos - trough_pos
            except Exception:
                recovery_months = None

    return mdd, peak_idx, trough_idx, recovery_months


def calmar_ratio(returns: pd.Series, periods_per_year: int = 12) -> float:
    """Compute Calmar ratio (annualized return / abs max drawdown)."""
    ann_ret = annualized_return(returns, periods_per_year)
    mdd, _, _, _ = max_drawdown(returns)
    if mdd == 0:
        return np.inf if ann_ret > 0 else 0.0
    return ann_ret / abs(mdd)


def hit_rate(returns: pd.Series) -> float:
    """Fraction of months with positive returns."""
    return (returns > 0).mean()


def compute_full_metrics(
    returns: pd.Series,
    rf: Optional[pd.Series] = None,
    periods_per_year: int = 12,
    label: str = "",
) -> Dict:
    """
    Compute a full suite of performance metrics.
    Returns a dict suitable for display in a summary table.
    """
    mdd, peak, trough, recovery = max_drawdown(returns)

    metrics = {
        "label": label,
        "ann_return": annualized_return(returns, periods_per_year),
        "ann_volatility": annualized_volatility(returns, periods_per_year),
        "sharpe": sharpe_ratio(returns, rf, periods_per_year),
        "sortino": sortino_ratio(returns, rf, periods_per_year),
        "max_drawdown": mdd,
        "mdd_peak": peak,
        "mdd_trough": trough,
        "mdd_recovery_months": recovery,
        "calmar": calmar_ratio(returns, periods_per_year),
        "hit_rate": hit_rate(returns),
        "skewness": returns.skew(),
        "kurtosis": returns.kurtosis(),
        "worst_month": returns.min(),
        "best_month": returns.max(),
        "avg_monthly_return": returns.mean(),
        "n_months": len(returns),
        "total_return": (1 + returns).prod() - 1,
    }

    return metrics


def metrics_table(
    returns_dict: Dict[str, pd.Series],
    rf: Optional[pd.Series] = None,
) -> pd.DataFrame:
    """
    Compute metrics for multiple return series and return as a formatted table.
    """
    rows = []
    for label, rets in returns_dict.items():
        m = compute_full_metrics(rets, rf=rf, label=label)
        rows.append(m)

    df = pd.DataFrame(rows).set_index("label")

    # Format for display
    pct_cols = ["ann_return", "ann_volatility", "max_drawdown", "hit_rate",
                "worst_month", "best_month", "avg_monthly_return", "total_return"]

    return df


def subperiod_metrics(
    returns: pd.Series,
    periods: Dict[str, Tuple[str, str]],
    rf: Optional[pd.Series] = None,
) -> pd.DataFrame:
    """
    Compute metrics for pre-defined subperiods.

    Parameters
    ----------
    periods : dict
        {"period_name": ("start_date", "end_date"), ...}
    """
    rows = []
    for name, (start, end) in periods.items():
        mask = (returns.index >= start) & (returns.index <= end)
        sub = returns[mask]
        if len(sub) > 0:
            m = compute_full_metrics(sub, rf=rf, label=name)
            rows.append(m)

    return pd.DataFrame(rows).set_index("label")


if __name__ == "__main__":
    # Test with random returns
    np.random.seed(42)
    test_rets = pd.Series(
        np.random.normal(0.005, 0.04, 120),
        index=pd.period_range("2010-01", periods=120, freq="M"),
    )
    metrics = compute_full_metrics(test_rets, label="Test")
    for k, v in metrics.items():
        print(f"  {k}: {v}")
