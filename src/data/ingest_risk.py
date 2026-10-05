"""
Ingest the Astvansh & Simpson (2026) operational-risk dataset from OSF.
Converts `data set.xlsx` → `risk_scores.parquet`.

Expected columns from the OSF Excel file:
  CIK, Year, Item_1A_optional, Item_1A_word_count,
  8 dictionary counts (accounting_count, ..., technology_count),
  8 transformer probabilities (accounting_prob, ..., technology_prob),
  8 binary labels (accounting_binary, ..., technology_binary).

Reference: Astvansh & Simpson (2026, MSOM). OSF: https://osf.io/gz93b/
"""
import pandas as pd
import numpy as np
from pathlib import Path

from src.utils.helpers import (
    load_config, get_path, setup_logging, save_parquet, load_parquet, parquet_exists
)

logger = setup_logging("ingest_risk")

# The eight risk factors from Schnatterly et al. (2021)
RISK_FACTORS = [
    "accounting", "finance", "international", "legal",
    "management", "marketing", "operations", "technology",
]

# Expected column name patterns (will be normalized on read)
PROB_COLS = [f"{f}_prob" for f in RISK_FACTORS]
COUNT_COLS = [f"{f}_count" for f in RISK_FACTORS]
BINARY_COLS = [f"{f}_binary" for f in RISK_FACTORS]


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize column names: lowercase, strip, replace spaces with underscores."""
    df.columns = (
        df.columns.str.strip()
        .str.lower()
        .str.replace(" ", "_", regex=False)
        .str.replace("-", "_", regex=False)
    )
    return df


def ingest_risk_scores(
    raw_path: Path = None,
    output_path: Path = None,
    force: bool = False,
) -> pd.DataFrame:
    """
    Read the OSF risk-score Excel file and convert to a clean parquet.

    Parameters
    ----------
    raw_path : Path
        Path to `data set.xlsx` in data/raw/.
    output_path : Path
        Where to write `risk_scores.parquet`.
    force : bool
        If True, re-read from Excel even if parquet exists.

    Returns
    -------
    pd.DataFrame
        Cleaned risk scores with columns: cik, year, item_1a_optional,
        item_1a_word_count, {factor}_prob, {factor}_count for all 8 factors.
    """
    cfg = load_config()
    raw_dir = Path(raw_path or get_path(cfg, "raw"))
    out_dir = Path(output_path or get_path(cfg, "processed"))
    out_file = out_dir / "risk_scores.parquet"

    # Cache check
    if parquet_exists(out_file) and not force:
        logger.info(f"Loading cached risk scores from {out_file}")
        return load_parquet(out_file)

    # Find the Excel file
    excel_candidates = list(raw_dir.glob("*data*set*.xlsx")) + list(raw_dir.glob("*data*set*.xls"))
    if not excel_candidates:
        raise FileNotFoundError(
            f"No 'data set.xlsx' found in {raw_dir}. "
            f"Download from https://osf.io/gz93b/ and place in {raw_dir}/"
        )
    excel_path = excel_candidates[0]
    logger.info(f"Reading risk dataset from {excel_path} (this may take 1–3 minutes)...")

    # Read — openpyxl for .xlsx
    df = pd.read_excel(excel_path, engine="openpyxl")
    df = _normalize_columns(df)
    logger.info(f"Raw shape: {df.shape}. Columns: {list(df.columns)}")

    # ── Validate expected columns ────────────────────────────────────────
    required = {"cik", "year"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Attempt to identify probability and count columns even if naming varies
    found_prob = [c for c in PROB_COLS if c in df.columns]
    found_count = [c for c in COUNT_COLS if c in df.columns]
    logger.info(f"Found {len(found_prob)}/8 probability cols, {len(found_count)}/8 count cols")

    # ── Basic cleaning ───────────────────────────────────────────────────
    df["cik"] = df["cik"].astype(int)
    df["year"] = df["year"].astype(int)

    # Ensure probability columns are float
    for col in found_prob:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # ── Derived measures ─────────────────────────────────────────────────
    # Operational risk = operations factor probability
    if "operations_prob" in df.columns:
        df["op_risk"] = df["operations_prob"]

    # Non-operational risk = mean of the other 7 factor probabilities
    non_op_factors = [f"{f}_prob" for f in RISK_FACTORS if f != "operations"]
    non_op_available = [c for c in non_op_factors if c in df.columns]
    if non_op_available:
        df["non_op_risk_mean"] = df[non_op_available].mean(axis=1)
        df["non_op_risk_sum"] = df[non_op_available].sum(axis=1)

    # ── Drop binary columns (degenerate, per D2) ────────────────────────
    binary_to_drop = [c for c in BINARY_COLS if c in df.columns]
    if binary_to_drop:
        logger.info(f"Dropping {len(binary_to_drop)} degenerate binary columns (D2)")
        df = df.drop(columns=binary_to_drop)

    # ── Item 1A metadata ─────────────────────────────────────────────────
    for col_name in ["item_1a_optional", "item_1a_word_count"]:
        if col_name in df.columns:
            df[col_name] = pd.to_numeric(df[col_name], errors="coerce")

    # ── Summary statistics ───────────────────────────────────────────────
    logger.info(f"Final shape: {df.shape}")
    logger.info(f"Year range: {df['year'].min()} – {df['year'].max()}")
    logger.info(f"Unique CIKs: {df['cik'].nunique()}")
    if "op_risk" in df.columns:
        logger.info(f"Operations prob — mean: {df['op_risk'].mean():.4f}, "
                     f"std: {df['op_risk'].std():.4f}")

    # Save
    save_parquet(df, out_file)
    logger.info(f"Saved risk_scores.parquet to {out_file}")
    return df


def get_risk_descriptives(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute descriptive statistics matching Paper A Table 4.
    Returns a DataFrame with mean, std, min, p25, p50, p75, max for each score.
    """
    prob_cols = [c for c in PROB_COLS if c in df.columns]
    count_cols = [c for c in COUNT_COLS if c in df.columns]
    cols = prob_cols + count_cols
    if "item_1a_word_count" in df.columns:
        cols.append("item_1a_word_count")
    return df[cols].describe().T


if __name__ == "__main__":
    df = ingest_risk_scores(force=True)
    print(get_risk_descriptives(df))
