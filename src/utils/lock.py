"""Locked final-test window guard (brief rule 5).

Until config.test.unlocked is true, analysis code must pass data through
`dev_only` before computing any metric, and code paths that evaluate the test
window must call `assert_unlocked` first.
"""
import pandas as pd
from src.utils.config import load_config


def test_start() -> pd.Period:
    return pd.Period(load_config()["test"]["start"], freq="M")


def is_unlocked() -> bool:
    return bool(load_config()["test"]["unlocked"])


def assert_unlocked() -> None:
    if not is_unlocked():
        raise PermissionError("Final test window is locked (config.test.unlocked=false). "
                              "Freeze the design and log the unlock in DECISIONS.md first.")


def dev_only(df: pd.DataFrame, month_col: str = "month") -> pd.DataFrame:
    """Rows strictly before the locked test window. Works on a column or a PeriodIndex."""
    months = df.index if month_col is None else df[month_col]
    months = pd.PeriodIndex(months, freq="M")
    return df[months < test_start()]
