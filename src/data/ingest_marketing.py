"""
Ingest marketing-emphasis scores from Damavandi, Mai & Astvansh (2025).
Source: https://marketing-measures.github.io/

Expected: CSV files with firm-quarter marketing emphasis scores
(19 lower-order components, 3 higher-order constructs).
"""
import pandas as pd
import numpy as np
from pathlib import Path

from src.utils.helpers import (
    load_config, get_path, setup_logging, save_parquet, load_parquet, parquet_exists
)

logger = setup_logging("ingest_marketing")

# Higher-order constructs
HIGHER_ORDER = ["market_orientation", "marketing_capabilities", "marketing_excellence"]


def ingest_marketing_scores(
    raw_path: Path = None,
    output_path: Path = None,
    force: bool = False,
) -> pd.DataFrame:
    """
    Ingest marketing-emphasis CSV files → marketing_scores.parquet.

    Steps:
    1. Read raw CSV(s) from data/raw/marketing/
    2. Identify firm identifiers (GVKEY, ticker, CUSIP, or name)
    3. Normalize column names
    4. Compute assumed availability date (quarter-end + 60 days conservative lag)
    5. Save to parquet
    """
    cfg = load_config()
    raw_dir = Path(raw_path or get_path(cfg, "raw")) / "marketing"
    out_dir = Path(output_path or get_path(cfg, "processed"))
    out_file = out_dir / "marketing_scores.parquet"

    if parquet_exists(out_file) and not force:
        logger.info(f"Loading cached marketing scores from {out_file}")
        return load_parquet(out_file)

    # Find CSV files
    csv_files = list(raw_dir.glob("*.csv"))
    if not csv_files:
        logger.warning(
            f"No marketing CSV files found in {raw_dir}. "
            f"Download from https://marketing-measures.github.io/ "
            f"and place in {raw_dir}/"
        )
        return pd.DataFrame()

    # Read and concatenate
    dfs = []
    for f in csv_files:
        logger.info(f"Reading {f.name}...")
        df = pd.read_csv(f, low_memory=False)
        df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_", regex=False)
        dfs.append(df)
        logger.info(f"  → {df.shape[0]} rows, columns: {list(df.columns)[:10]}...")

    df = pd.concat(dfs, ignore_index=True) if len(dfs) > 1 else dfs[0]

    # ── Identify the firm identifier ─────────────────────────────────────
    id_candidates = {"gvkey": "gvkey", "ticker": "ticker", "cik": "cik"}
    found_ids = {k: v for k, v in id_candidates.items() if k in df.columns}
    logger.info(f"Found identifiers: {list(found_ids.keys())}")
    
    # WRDS Link Table Integration
    link_file = raw_dir.parent / "cik_gvkey.csv"
    if "cik" not in df.columns and "gvkey" in df.columns and link_file.exists():
        logger.info(f"Found WRDS Link Table: {link_file.name}. Merging GVKEY to CIK...")
        link_df = pd.read_csv(link_file, usecols=["cik", "gvkey"]).dropna()
        
        # Format gvkey consistently as 6-digit string
        link_df["gvkey"] = link_df["gvkey"].astype(float).astype("Int64").astype(str).str.zfill(6)
        link_df = link_df.drop_duplicates(subset=["gvkey"])
        
        df["gvkey"] = df["gvkey"].astype(str).str.zfill(6)
        df = df.merge(link_df, on="gvkey", how="left")
        
        mapped_count = df["cik"].notna().sum()
        logger.info(f"Successfully mapped {mapped_count} / {len(df)} rows to CIK.")
        
        df = df.dropna(subset=["cik"])
        df["cik"] = df["cik"].astype("int64")
        found_ids["cik"] = "cik"
    
    for col in found_ids:
        if col != "cik":
            df[col] = df[col].astype(str)

    # ── Time dimension ───────────────────────────────────────────────────
    # Look for quarter/year columns
    if "quarter" in df.columns and "year" in df.columns:
        df["year"] = df["year"].astype(int)
        df["quarter"] = df["quarter"].astype(int)
    elif "fyearq" in df.columns and "fqtr" in df.columns:
        df["year"] = df["fyearq"].astype(int)
        df["quarter"] = df["fqtr"].astype(int)
    elif "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        df["year"] = df["date"].dt.year
        df["quarter"] = df["date"].dt.quarter

    # Compute quarter-end date and conservative availability date
    if "year" in df.columns and "quarter" in df.columns:
        df["quarter_end"] = pd.to_datetime(
            df["year"].astype(str) + "-" + (df["quarter"] * 3).astype(str) + "-01"
        ) + pd.offsets.MonthEnd(0)
        # Conservative: available 60 days after quarter end
        df["avail_date"] = df["quarter_end"] + pd.Timedelta(days=60)

    # ── Compute Higher-Order Constructs ──────────────────────────────────
    if "customer_orientation" in df.columns:
        df["market_orientation"] = df[["customer_orientation", "competitor_orientation", "interfunctional_coordination"]].mean(axis=1)
        df["marketing_capabilities"] = df[["pricing_capabilities", "product_development", "channel_management", "communication", "selling_capabilities"]].mean(axis=1)
        df["marketing_excellence"] = df[["marketing_ecosystem", "end_user", "marketing_agility"]].mean(axis=1)

    # ── Summary ──────────────────────────────────────────────────────────
    logger.info(f"Marketing scores shape: {df.shape}")
    if "year" in df.columns:
        logger.info(f"Year range: {df['year'].min()} – {df['year'].max()}")

    save_parquet(df, out_file)
    logger.info(f"Saved marketing_scores.parquet to {out_file}")
    return df


if __name__ == "__main__":
    df = ingest_marketing_scores(force=True)
    if not df.empty:
        print(df.describe())
