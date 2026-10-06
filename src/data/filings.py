"""
Parse SEC EDGAR full-index files to extract 10-K filing dates.
Maps CIK × Year → actual filing date for point-in-time signal construction.

EDGAR etiquette: User-Agent header required, ≤10 requests/second, cache everything.
"""
import pandas as pd
import numpy as np
import requests
import time
from pathlib import Path
from typing import Optional

from src.utils.helpers import (
    load_config, get_path, setup_logging, save_parquet, load_parquet, parquet_exists
)

logger = setup_logging("filings")

EDGAR_BASE = "https://www.sec.gov/Archives/edgar/full-index"
FORM_TYPES_10K = {"10-K", "10-K405", "10-K/A"}


def _get_headers(cfg: dict) -> dict:
    """Build EDGAR-compliant request headers."""
    return {
        "User-Agent": cfg.get("sec_user_agent", "Research research@example.com"),
        "Accept-Encoding": "gzip, deflate",
    }


def download_form_index(
    year: int,
    quarter: int,
    cache_dir: Path,
    cfg: dict,
) -> Optional[pd.DataFrame]:
    """
    Download and parse a single EDGAR quarterly form.idx file.
    Returns DataFrame with columns: cik, company_name, form_type, date_filed, filename.
    """
    cache_file = cache_dir / f"form_idx_{year}_Q{quarter}.parquet"
    if cache_file.exists():
        df = load_parquet(cache_file)
        # Filter cache to prevent OOM
        return df[df["form_type"].isin(FORM_TYPES_10K)].copy()

    url = f"{EDGAR_BASE}/{year}/QTR{quarter}/form.idx"
    headers = _get_headers(cfg)

    try:
        time.sleep(0.15)  # Stay well under 10 req/s
        resp = requests.get(url, headers=headers, timeout=30)
        resp.raise_for_status()
    except requests.RequestException as e:
        logger.warning(f"Failed to download {url}: {e}")
        return None

    lines = resp.text.split("\n")
    # Skip header lines (typically first 9–11 lines with dashes)
    data_start = 0
    for i, line in enumerate(lines):
        if line.startswith("---"):
            data_start = i + 1
            break

    records = []
    for line in lines[data_start:]:
        if len(line.strip()) < 10:
            continue
        # Fixed-width format: Form Type | Company Name | CIK | Date Filed | Filename
        parts = line.split()
        if len(parts) < 5:
            continue
        try:
            # Form type can be multi-word (e.g., "10-K/A")
            # The last 3 fields are: CIK, Date Filed, Filename (URL path)
            filename = parts[-1]
            date_filed = parts[-2]
            cik = parts[-3]

            # Everything before CIK is form_type + company_name
            # We need to identify form_type (first token(s))
            form_type = parts[0]
            if len(parts) > 5 and parts[1] in ("/A",):
                form_type = f"{parts[0]}{parts[1]}"

            records.append({
                "cik": int(cik),
                "date_filed": date_filed,
                "form_type": form_type,
                "filename": filename,
            })
        except (ValueError, IndexError):
            continue

    if not records:
        logger.warning(f"No records parsed from {url}")
        return None

    df = pd.DataFrame(records)
    df["date_filed"] = pd.to_datetime(df["date_filed"], errors="coerce")
    df = df.dropna(subset=["date_filed"])
    
    # Filter before saving to disk to prevent huge cache files and OOM issues
    df = df[df["form_type"].isin(FORM_TYPES_10K)].copy()

    save_parquet(df, cache_file)
    logger.info(f"Parsed {len(df)} filings from {year} Q{quarter}")
    return df


def build_filings_table(
    start_year: int = 2005,
    end_year: int = 2024,
    force: bool = False,
) -> pd.DataFrame:
    """
    Build the full filings table by parsing all EDGAR quarterly indices.
    Keeps only 10-K and 10-K/A forms.

    Returns DataFrame: cik, date_filed, form_type, fiscal_year_approx, is_amendment
    """
    cfg = load_config()
    out_file = Path(get_path(cfg, "processed")) / "filings.parquet"
    cache_dir = Path(get_path(cfg, "interim")) / "edgar_idx"
    cache_dir.mkdir(parents=True, exist_ok=True)

    if parquet_exists(out_file) and not force:
        logger.info(f"Loading cached filings from {out_file}")
        return load_parquet(out_file)

    all_dfs = []
    for year in range(start_year, end_year + 1):
        for qtr in range(1, 5):
            df = download_form_index(year, qtr, cache_dir, cfg)
            if df is not None:
                all_dfs.append(df)

    if not all_dfs:
        raise RuntimeError("No EDGAR index files successfully downloaded")

    filings = pd.concat(all_dfs, ignore_index=True)

    # Keep only 10-K family
    filings = filings[filings["form_type"].isin(FORM_TYPES_10K)].copy()
    filings["is_amendment"] = filings["form_type"].str.contains("/A", regex=False)

    # Approximate fiscal year: if filed Jan–Jun → fiscal year = filing year - 1;
    # if filed Jul–Dec → fiscal year = filing year.
    # This is a heuristic; the D6 reconciliation step will validate it.
    filings["filing_month"] = filings["date_filed"].dt.month
    filings["fiscal_year_approx"] = np.where(
        filings["filing_month"] <= 6,
        filings["date_filed"].dt.year - 1,
        filings["date_filed"].dt.year,
    )

    # For each CIK-fiscal_year, keep the earliest original 10-K
    originals = filings[~filings["is_amendment"]].copy()
    originals = originals.sort_values("date_filed").groupby(
        ["cik", "fiscal_year_approx"]
    ).first().reset_index()

    logger.info(f"Total 10-K filings: {len(filings)}")
    logger.info(f"Unique CIK-fiscal_year pairs (originals): {len(originals)}")

    save_parquet(originals, out_file)
    logger.info(f"Saved filings.parquet to {out_file}")
    return originals


def reconcile_filing_years(
    filings: pd.DataFrame,
    risk_scores: pd.DataFrame,
    sample_ciks: list = None,
) -> pd.DataFrame:
    """
    Reconcile the `Year` column in risk_scores with actual EDGAR filing dates.
    Tests on a sample of well-known CIKs to determine if Year = fiscal year
    or filing year.

    Returns a DataFrame with reconciliation results.
    """
    if sample_ciks is None:
        # Well-known firms: Apple, Microsoft, Amazon, Google, JPMorgan, ExxonMobil
        sample_ciks = [320193, 789019, 1018724, 1652044, 19617, 34088]

    results = []
    for cik in sample_ciks:
        risk_years = risk_scores[risk_scores["cik"] == cik]["year"].sort_values().tolist()
        filing_dates = filings[filings["cik"] == cik][
            ["date_filed", "fiscal_year_approx"]
        ].sort_values("date_filed")

        results.append({
            "cik": cik,
            "risk_years": risk_years,
            "filing_dates": filing_dates.to_dict("records"),
        })

    return pd.DataFrame(results)


if __name__ == "__main__":
    filings = build_filings_table(force=True)
    print(filings.head(20))
