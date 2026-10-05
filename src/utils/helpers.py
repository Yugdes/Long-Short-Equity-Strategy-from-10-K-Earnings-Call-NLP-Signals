"""
Utility functions — I/O, config loading, logging, and path management.
"""
import yaml
import logging
from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np

# ── Project root ─────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"


def load_config(path: Optional[Path] = None) -> dict:
    """Load the YAML configuration file."""
    path = path or CONFIG_PATH
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    # Convert relative paths to absolute
    for key in ("raw", "interim", "processed", "results"):
        cfg["paths"][key] = str(PROJECT_ROOT / cfg["paths"][key])
    return cfg


def get_path(cfg: dict, key: str) -> Path:
    """Get an absolute Path from the config paths dict."""
    return Path(cfg["paths"][key])


# ── Logging ──────────────────────────────────────────────────────────────────
def setup_logging(name: str = "risk_signal", level: int = logging.INFO) -> logging.Logger:
    """Configure and return a project logger."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s | %(name)s | %(levelname)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


# ── Parquet helpers ──────────────────────────────────────────────────────────
def save_parquet(df: pd.DataFrame, path: Path, **kwargs) -> None:
    """Save a DataFrame to parquet with zstd compression."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, engine="pyarrow", compression="zstd", **kwargs)


def load_parquet(path: Path) -> pd.DataFrame:
    """Load a parquet file into a DataFrame."""
    return pd.read_parquet(Path(path), engine="pyarrow")


def parquet_exists(path: Path) -> bool:
    """Check if a cached parquet file exists."""
    return Path(path).exists()


# ── Date utilities ───────────────────────────────────────────────────────────
def to_month_end(dates: pd.Series) -> pd.Series:
    """Convert dates to their month-end."""
    return pd.to_datetime(dates) + pd.offsets.MonthEnd(0)


def to_month_start(dates: pd.Series) -> pd.Series:
    """Convert dates to their month-start."""
    return pd.to_datetime(dates).dt.to_period("M").dt.to_timestamp()


def add_business_days(dates: pd.Series, n: int) -> pd.Series:
    """Add n business days to a Series of dates."""
    return dates.apply(lambda d: d + pd.offsets.BDay(n))


def get_first_holding_month(signal_dates: pd.Series) -> pd.Series:
    """
    Given signal availability dates, return the first full calendar month
    that can be used for holding (the month AFTER the signal month).
    """
    month_end = pd.to_datetime(signal_dates) + pd.offsets.MonthEnd(0)
    first_hold = (month_end + pd.offsets.MonthBegin(1)).to_period("M")
    return first_hold


# ── Cross-sectional transforms ──────────────────────────────────────────────
def rank_within_group(
    df: pd.DataFrame,
    value_col: str,
    group_col: str = "month",
    output_col: Optional[str] = None,
) -> pd.Series:
    """
    Compute percentile rank (0–1) of value_col within each group (default: month).
    """
    output_col = output_col or f"{value_col}_rank"
    return df.groupby(group_col)[value_col].rank(pct=True)


def winsorize(
    series: pd.Series, lower: float = 0.01, upper: float = 0.99
) -> pd.Series:
    """Winsorize a Series at the given percentiles."""
    lo = series.quantile(lower)
    hi = series.quantile(upper)
    return series.clip(lo, hi)


def winsorize_within_group(
    df: pd.DataFrame,
    value_col: str,
    group_col: str = "month",
    lower: float = 0.01,
    upper: float = 0.99,
) -> pd.Series:
    """Winsorize within each cross-section (e.g., month)."""
    def _clip(s):
        lo, hi = s.quantile(lower), s.quantile(upper)
        return s.clip(lo, hi)
    return df.groupby(group_col)[value_col].transform(_clip)


# ── Seed management ─────────────────────────────────────────────────────────
def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    try:
        import random
        random.seed(seed)
    except ImportError:
        pass
