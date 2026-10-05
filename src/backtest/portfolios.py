"""
Portfolio construction — quintile sorts, calendar-time portfolios,
value-weighted and equal-weighted, with transaction cost estimation.

Implements the Jegadeesh-Titman style calendar-time methodology.
"""
import pandas as pd
import numpy as np
from typing import Tuple, Optional, Dict

from src.utils.helpers import load_config, setup_logging

logger = setup_logging("portfolios")


def assign_quantiles(
    panel: pd.DataFrame,
    signal_col: str,
    n_quantiles: int = 5,
    group_col: str = "month",
) -> pd.Series:
    """
    Assign quantile labels (1 = lowest, N = highest) within each cross-section.
    Uses conditional quantile labels to handle tied values gracefully.
    """
    def _qcut(group):
        try:
            return pd.qcut(group, n_quantiles, labels=range(1, n_quantiles + 1))
        except ValueError:
            # Too few unique values — fall back to rank-based assignment
            ranks = group.rank(method="first", pct=True)
            bins = np.linspace(0, 1, n_quantiles + 1)
            return pd.cut(ranks, bins=bins, labels=range(1, n_quantiles + 1), include_lowest=True)

    return panel.groupby(group_col)[signal_col].transform(_qcut).astype(int)


def compute_portfolio_returns(
    panel: pd.DataFrame,
    signal_col: str,
    return_col: str = "ret_next",
    n_quantiles: int = 5,
    weighting: str = "value",
    weight_cap: float = 0.05,
    min_names: int = 20,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute calendar-time quintile portfolio returns.

    Parameters
    ----------
    panel : DataFrame
        Stock-month panel with signals and returns
    signal_col : str
        Column name of the signal to sort on
    return_col : str
        Column name of forward returns
    n_quantiles : int
        Number of portfolios (default 5 for quintiles)
    weighting : str
        "value" for value-weighted, "equal" for equal-weighted
    weight_cap : float
        Maximum weight for any single stock (VW only)
    min_names : int
        Minimum number of names per leg

    Returns
    -------
    portfolio_returns : DataFrame
        Monthly returns for each quintile and the L/S spread
    portfolio_stats : DataFrame
        Per-month statistics (names, turnover, etc.)
    """
    df = panel.dropna(subset=[signal_col, return_col]).copy()

    # Assign quintiles
    df["quintile"] = assign_quantiles(df, signal_col, n_quantiles)

    # Compute portfolio returns per month
    port_returns = {}
    port_stats = []

    months = sorted(df["month"].unique())

    for month in months:
        month_data = df[df["month"] == month]
        month_rets = {}
        month_stat = {"month": month}

        for q in range(1, n_quantiles + 1):
            q_data = month_data[month_data["quintile"] == q]

            if len(q_data) < min_names:
                month_rets[f"Q{q}"] = np.nan
                month_stat[f"n_Q{q}"] = len(q_data)
                continue

            if weighting == "value" and "mktcap" in q_data.columns:
                # Value-weighted
                weights = q_data["mktcap"] / q_data["mktcap"].sum()
                # Cap individual weights
                weights = weights.clip(upper=weight_cap)
                weights = weights / weights.sum()  # Renormalize
                month_rets[f"Q{q}"] = (weights * q_data[return_col]).sum()
            else:
                # Equal-weighted
                month_rets[f"Q{q}"] = q_data[return_col].mean()

            month_stat[f"n_Q{q}"] = len(q_data)

        # Long-short spread: Q1 (low risk) minus Q5 (high risk)
        if f"Q1" in month_rets and f"Q{n_quantiles}" in month_rets:
            q1_ret = month_rets.get("Q1", np.nan)
            q5_ret = month_rets.get(f"Q{n_quantiles}", np.nan)
            if not np.isnan(q1_ret) and not np.isnan(q5_ret):
                month_rets["L/S"] = q1_ret - q5_ret

        port_returns[month] = month_rets
        port_stats.append(month_stat)

    # Convert to DataFrames
    ret_df = pd.DataFrame.from_dict(port_returns, orient="index")
    ret_df.index.name = "month"
    ret_df = ret_df.sort_index()

    stats_df = pd.DataFrame(port_stats)

    logger.info(f"Portfolio returns: {len(ret_df)} months, "
                f"{ret_df.columns.tolist()}")

    return ret_df, stats_df


def compute_double_sort_returns(
    panel: pd.DataFrame,
    signal1_col: str,
    signal2_col: str,
    return_col: str = "ret_next",
    n1: int = 3,
    n2: int = 3,
    weighting: str = "equal",
) -> pd.DataFrame:
    """
    Compute double-sorted portfolio returns (e.g., OpRisk × NonOp 3×3).
    Returns a DataFrame of average returns for each cell.
    """
    df = panel.dropna(subset=[signal1_col, signal2_col, return_col]).copy()

    df["q1"] = assign_quantiles(df, signal1_col, n1)
    df["q2"] = assign_quantiles(df, signal2_col, n2)

    if weighting == "equal":
        result = df.groupby(["q1", "q2"])[return_col].mean().unstack()
    else:
        # Value-weighted double sort
        result = df.groupby(["q1", "q2"]).apply(
            lambda g: np.average(g[return_col], weights=g.get("mktcap", np.ones(len(g))))
        ).unstack()

    return result


def compute_turnover(
    panel: pd.DataFrame,
    signal_col: str,
    n_quantiles: int = 5,
) -> pd.DataFrame:
    """
    Compute monthly one-way turnover for each quintile portfolio.
    Turnover = fraction of portfolio that changes each month.
    """
    df = panel.dropna(subset=[signal_col]).copy()
    df["quintile"] = assign_quantiles(df, signal_col, n_quantiles)

    months = sorted(df["month"].unique())
    turnover_data = []

    prev_holdings = {}  # q -> set of CIKs

    for month in months:
        month_data = df[df["month"] == month]
        month_turnover = {"month": month}

        for q in range(1, n_quantiles + 1):
            current_ciks = set(month_data[month_data["quintile"] == q]["cik"])

            if q in prev_holdings and prev_holdings[q]:
                prev_ciks = prev_holdings[q]
                if len(current_ciks) > 0:
                    # One-way turnover: fraction that entered or exited
                    entered = len(current_ciks - prev_ciks)
                    exited = len(prev_ciks - current_ciks)
                    avg_size = (len(current_ciks) + len(prev_ciks)) / 2
                    month_turnover[f"turnover_Q{q}"] = (entered + exited) / (2 * avg_size)
                else:
                    month_turnover[f"turnover_Q{q}"] = np.nan
            else:
                month_turnover[f"turnover_Q{q}"] = np.nan

            prev_holdings[q] = current_ciks

        turnover_data.append(month_turnover)

    return pd.DataFrame(turnover_data)


def apply_transaction_costs(
    gross_returns: pd.Series,
    turnover: pd.Series,
    cost_bps: float = 25,
) -> pd.Series:
    """
    Apply transaction costs to gross portfolio returns.
    Net return = gross return - (turnover × 2 × cost_bps / 10000)
    Factor of 2 for round-trip (buy + sell).
    """
    cost_per_dollar = turnover * 2 * cost_bps / 10000
    return gross_returns - cost_per_dollar


def compute_breakeven_cost(
    gross_alpha_monthly: float,
    avg_turnover_monthly: float,
) -> float:
    """
    Compute the break-even one-way transaction cost (in bps)
    that sets net alpha to zero.
    """
    if avg_turnover_monthly <= 0:
        return np.inf
    # gross_alpha = turnover * 2 * cost / 10000
    return gross_alpha_monthly / (avg_turnover_monthly * 2) * 10000


if __name__ == "__main__":
    print("Portfolio construction — run via src.run_all or Phase 4 pipeline")
