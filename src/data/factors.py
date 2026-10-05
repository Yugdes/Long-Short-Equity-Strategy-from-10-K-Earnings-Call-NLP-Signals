"""
Download and parse Fama-French factor data from Kenneth French's data library.
FF5 + Momentum monthly factors and the 48-industry classification.
"""
import pandas as pd
import numpy as np
from pathlib import Path

from src.utils.helpers import (
    load_config, get_path, setup_logging, save_parquet, load_parquet, parquet_exists
)

logger = setup_logging("factors")


def download_ff_factors(force: bool = False) -> pd.DataFrame:
    """
    Download monthly Fama-French 5 factors + Momentum.
    Uses pandas_datareader to fetch from Kenneth French Data Library.

    Returns DataFrame indexed by month with columns:
    Mkt-RF, SMB, HML, RMW, CMA, RF, Mom
    """
    cfg = load_config()
    out_file = Path(get_path(cfg, "processed")) / "factors_monthly.parquet"

    if parquet_exists(out_file) and not force:
        logger.info("Loading cached factor data")
        return load_parquet(out_file)

    try:
        import pandas_datareader.data as web

        # Fama-French 5 Factors (monthly)
        ff5 = web.DataReader(
            "F-F_Research_Data_5_Factors_2x3",
            "famafrench",
            start="2004-01",
        )[0]  # [0] is the monthly table
        ff5 = ff5 / 100  # Convert from percentage to decimal

        # Momentum factor
        mom = web.DataReader(
            "F-F_Momentum_Factor",
            "famafrench",
            start="2004-01",
        )[0]
        mom = mom / 100
        mom.columns = ["Mom"]

        # Merge
        factors = ff5.join(mom, how="left")
        factors.index = factors.index.to_timestamp()
        factors.index.name = "month"
        factors = factors.reset_index()
        factors["month"] = pd.to_datetime(factors["month"]).dt.to_period("M")

        logger.info(f"Factor data: {len(factors)} months, "
                     f"{factors['month'].min()} – {factors['month'].max()}")

    except Exception as e:
        logger.warning(f"pandas_datareader failed: {e}. Trying direct download...")
        factors = _download_ff_direct()

    save_parquet(factors, out_file)
    return factors


def _download_ff_direct() -> pd.DataFrame:
    """Fallback: download factor CSVs directly from Kenneth French's website."""
    import requests
    import zipfile
    import io

    base_url = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"

    # FF5 factors
    ff5_url = base_url + "F-F_Research_Data_5_Factors_2x3_CSV.zip"
    resp = requests.get(ff5_url, timeout=30)
    resp.raise_for_status()

    with zipfile.ZipFile(io.BytesIO(resp.content)) as z:
        csv_name = [n for n in z.namelist() if n.endswith(".CSV") or n.endswith(".csv")][0]
        with z.open(csv_name) as f:
            lines = f.read().decode("utf-8").split("\n")

    # Find the start of monthly data (skip header lines)
    data_start = 0
    for i, line in enumerate(lines):
        if line.strip().startswith("19") or line.strip().startswith("20"):
            data_start = i
            break

    # Parse monthly data until we hit annual data or empty lines
    records = []
    for line in lines[data_start:]:
        parts = line.strip().split(",")
        if len(parts) < 7:
            break
        try:
            ym = parts[0].strip()
            if len(ym) != 6:
                break
            records.append({
                "month": pd.Period(f"{ym[:4]}-{ym[4:]}", freq="M"),
                "Mkt-RF": float(parts[1]) / 100,
                "SMB": float(parts[2]) / 100,
                "HML": float(parts[3]) / 100,
                "RMW": float(parts[4]) / 100,
                "CMA": float(parts[5]) / 100,
                "RF": float(parts[6]) / 100,
            })
        except (ValueError, IndexError):
            break

    ff5 = pd.DataFrame(records)

    # Momentum factor (similar process)
    mom_url = base_url + "F-F_Momentum_Factor_CSV.zip"
    try:
        resp = requests.get(mom_url, timeout=30)
        resp.raise_for_status()
        with zipfile.ZipFile(io.BytesIO(resp.content)) as z:
            csv_name = [n for n in z.namelist() if ".csv" in n.lower()][0]
            with z.open(csv_name) as f:
                mom_lines = f.read().decode("utf-8").split("\n")

        mom_start = 0
        for i, line in enumerate(mom_lines):
            if line.strip().startswith("19") or line.strip().startswith("20"):
                mom_start = i
                break

        mom_records = []
        for line in mom_lines[mom_start:]:
            parts = line.strip().split(",")
            if len(parts) < 2:
                break
            try:
                ym = parts[0].strip()
                if len(ym) != 6:
                    break
                mom_records.append({
                    "month": pd.Period(f"{ym[:4]}-{ym[4:]}", freq="M"),
                    "Mom": float(parts[1]) / 100,
                })
            except (ValueError, IndexError):
                break

        mom_df = pd.DataFrame(mom_records)
        ff5 = ff5.merge(mom_df, on="month", how="left")
    except Exception as e:
        logger.warning(f"Momentum download failed: {e}")
        ff5["Mom"] = np.nan

    logger.info(f"Direct download: {len(ff5)} months of factor data")
    return ff5


def download_ff48_industries() -> dict:
    """
    Download FF48 industry SIC-code mapping from Kenneth French's website.
    Returns dict: industry_number → (name, [(sic_lo, sic_hi), ...])
    """
    # Use the built-in mapping from security_master for now
    from src.data.security_master import FF48_SIC_RANGES
    return FF48_SIC_RANGES


if __name__ == "__main__":
    factors = download_ff_factors(force=True)
    print(factors.tail(20))
    print(f"\nColumns: {list(factors.columns)}")
