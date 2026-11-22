"""Point-in-time release rules (DECISIONS D-005, D-006).

A signal's *first month* is the first holding month whose portfolio may use it.
Returns are those of that month, and the portfolio is formed at its start.
"""
import pandas as pd


def risk_first_month(filing_date: pd.Series, lag_bdays: int = 2) -> pd.Series:
    """Filing date + `lag_bdays` business days; the first full calendar month after that."""
    avail = pd.to_datetime(filing_date) + pd.offsets.BDay(lag_bdays)
    return avail.dt.to_period("M") + 1


def marketing_first_month(year: pd.Series, quarter: pd.Series, buffer_months: int = 1) -> pd.Series:
    """Calendar-quarter call scores are public by quarter end; usable from the month after plus a buffer."""
    q_end = pd.PeriodIndex.from_fields(year=year.astype(int), month=quarter.astype(int) * 3, freq="M")
    return pd.Series(q_end + 1 + buffer_months, index=year.index)


def months_since(later: pd.Series, earlier: pd.Series) -> pd.Series:
    """Integer month difference between two Period[M] series."""
    return pd.Series(pd.PeriodIndex(later, freq="M").asi8 - pd.PeriodIndex(earlier, freq="M").asi8,
                     index=later.index)
