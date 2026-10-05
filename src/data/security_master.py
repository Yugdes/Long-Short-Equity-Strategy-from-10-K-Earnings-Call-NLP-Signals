"""
Security master — maps CIK ↔ Ticker ↔ SIC ↔ Fama-French industry codes.

Uses SEC's company_tickers.json for current mappings, with
provisions for historical ticker changes (known gap for Path B).
"""
import pandas as pd
import numpy as np
import requests
import json
from pathlib import Path
from typing import Optional

from src.utils.helpers import (
    load_config, get_path, setup_logging, save_parquet, load_parquet, parquet_exists
)

logger = setup_logging("security_master")

SEC_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"

# Fama-French 48-industry SIC mapping (abridged — top-level boundaries)
# Full mapping loaded from Kenneth French data library
FF48_SIC_RANGES = {
    1: ("Agric", [(100, 299), (700, 799), (910, 919), (2048, 2048)]),
    2: ("Food", [(2000, 2046), (2050, 2063), (2070, 2079), (2090, 2099)]),
    3: ("Soda", [(2064, 2068), (2086, 2086), (2087, 2087), (2096, 2097)]),
    4: ("Beer", [(2080, 2085)]),
    5: ("Smoke", [(2100, 2199)]),
    6: ("Toys", [(920, 999), (3650, 3651), (3732, 3732), (3930, 3931), (3940, 3949)]),
    7: ("Fun", [(7800, 7833), (7840, 7841), (7900, 7999)]),
    8: ("Books", [(2710, 2719), (2720, 2729), (2730, 2739), (2740, 2749), (2770, 2771),
                  (2780, 2799), (3993, 3993)]),
    9: ("Hshld", [(2047, 2047), (2391, 2392), (2510, 2519), (2590, 2599), (2840, 2844),
                  (3160, 3161), (3170, 3171), (3172, 3172), (3190, 3199), (3229, 3229),
                  (3260, 3260), (3262, 3263), (3269, 3269), (3589, 3589), (3630, 3639),
                  (3750, 3751), (3800, 3800), (3860, 3861), (3870, 3873), (3910, 3911),
                  (3914, 3914), (3915, 3915), (3960, 3962), (3991, 3991), (3995, 3995)]),
    30: ("Oil", [(1200, 1399), (2900, 2912), (2990, 2999)]),
    31: ("Util", [(4900, 4942)]),
    32: ("Telcm", [(4800, 4899)]),
    44: ("Banks", [(6000, 6000), (6010, 6019), (6020, 6025), (6030, 6036),
                   (6040, 6059), (6060, 6062), (6080, 6082), (6090, 6099),
                   (6100, 6111), (6120, 6129), (6130, 6139), (6140, 6149),
                   (6150, 6159), (6160, 6169), (6170, 6179), (6190, 6199)]),
    45: ("Insur", [(6300, 6300), (6310, 6319), (6320, 6329), (6330, 6331),
                   (6350, 6351), (6360, 6361), (6370, 6379), (6390, 6399),
                   (6400, 6411)]),
    46: ("RlEst", [(6500, 6500), (6510, 6510), (6512, 6515), (6517, 6519),
                   (6520, 6529), (6530, 6531), (6532, 6532), (6540, 6541),
                   (6550, 6553), (6590, 6599), (6610, 6611)]),
    47: ("Fin", [(6200, 6299), (6700, 6700), (6710, 6719), (6720, 6726),
                 (6730, 6733), (6740, 6779), (6790, 6795), (6798, 6798)]),
    48: ("Other", []),  # everything else
}


def _sic_to_ff48(sic: int) -> int:
    """Map a 4-digit SIC code to its FF48 industry number."""
    if pd.isna(sic) or sic == 0:
        return 48  # Other
    sic = int(sic)
    for ind_num, (_, ranges) in FF48_SIC_RANGES.items():
        for lo, hi in ranges:
            if lo <= sic <= hi:
                return ind_num
    return 48  # Other


def download_sec_tickers(cache_dir: Path, cfg: dict) -> pd.DataFrame:
    """Download SEC company_tickers.json → DataFrame with cik, ticker, title."""
    cache_file = cache_dir / "sec_tickers.parquet"
    if cache_file.exists():
        return load_parquet(cache_file)

    headers = {
        "User-Agent": cfg.get("sec_user_agent", "Research research@example.com")
    }
    resp = requests.get(SEC_TICKERS_URL, headers=headers, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    records = []
    for key, entry in data.items():
        records.append({
            "cik": int(entry.get("cik_str", entry.get("cik", 0))),
            "ticker": entry.get("ticker", ""),
            "title": entry.get("title", ""),
        })

    df = pd.DataFrame(records)
    df = df[df["cik"] > 0].drop_duplicates(subset=["cik"])
    save_parquet(df, cache_file)
    logger.info(f"Downloaded {len(df)} ticker mappings from SEC")
    return df


def build_security_master(
    risk_ciks: pd.Series = None,
    force: bool = False,
) -> pd.DataFrame:
    """
    Build security_master.parquet: CIK ↔ ticker ↔ SIC ↔ FF48 industry.

    Parameters
    ----------
    risk_ciks : pd.Series
        Series of CIKs from the risk dataset (to focus the mapping).
    """
    cfg = load_config()
    out_file = Path(get_path(cfg, "processed")) / "security_master.parquet"
    cache_dir = Path(get_path(cfg, "interim"))
    cache_dir.mkdir(parents=True, exist_ok=True)

    if parquet_exists(out_file) and not force:
        return load_parquet(out_file)

    # Get SEC tickers
    sec_df = download_sec_tickers(cache_dir, cfg)

    # If we have risk CIKs, flag which ones we matched
    if risk_ciks is not None:
        unique_risk_ciks = risk_ciks.unique()
        matched = sec_df[sec_df["cik"].isin(unique_risk_ciks)]
        match_rate = len(matched) / len(unique_risk_ciks) * 100
        logger.info(
            f"CIK match rate: {len(matched)}/{len(unique_risk_ciks)} "
            f"({match_rate:.1f}%) risk CIKs found in SEC tickers"
        )
        # Use matched as the base, but keep all SEC entries for future use
        sec_df["in_risk_dataset"] = sec_df["cik"].isin(unique_risk_ciks)

    # Try to get SIC codes from SEC submissions API (first 100 as sample)
    # For full build, this queries data.sec.gov for each CIK
    sec_df["sic"] = np.nan  # Will be populated in the SIC lookup step
    sec_df["ff48"] = 48  # Default to "Other"
    sec_df["exchange"] = ""

    save_parquet(sec_df, out_file)
    logger.info(f"Saved security_master.parquet with {len(sec_df)} entries")
    return sec_df


def enrich_sic_codes(
    security_master: pd.DataFrame,
    max_requests: int = 500,
) -> pd.DataFrame:
    """
    Enrich the security master with SIC codes from SEC submissions API.
    Queries: https://data.sec.gov/submissions/CIK{cik:010d}.json
    """
    cfg = load_config()
    headers = {
        "User-Agent": cfg.get("sec_user_agent", "Research research@example.com")
    }

    needs_sic = security_master[security_master["sic"].isna()].head(max_requests)
    logger.info(f"Enriching SIC codes for {len(needs_sic)} CIKs...")

    updates = {}
    for i, (idx, row) in enumerate(needs_sic.iterrows()):
        cik_padded = str(int(row["cik"])).zfill(10)
        url = f"https://data.sec.gov/submissions/CIK{cik_padded}.json"

        try:
            if i > 0 and i % 10 == 0:
                time.sleep(1.0)  # Rate limiting

            import time
            time.sleep(0.12)
            resp = requests.get(url, headers=headers, timeout=15)
            if resp.status_code == 200:
                data = resp.json()
                sic = data.get("sic", None)
                exchange = data.get("exchanges", [""])[0] if data.get("exchanges") else ""
                if sic:
                    updates[idx] = {
                        "sic": int(sic),
                        "ff48": _sic_to_ff48(int(sic)),
                        "exchange": exchange,
                    }
        except Exception as e:
            logger.debug(f"Failed for CIK {row['cik']}: {e}")
            continue

        if (i + 1) % 100 == 0:
            logger.info(f"  Processed {i + 1}/{len(needs_sic)} CIKs...")

    # Apply updates
    for idx, vals in updates.items():
        for col, val in vals.items():
            security_master.loc[idx, col] = val

    logger.info(f"Updated SIC codes for {len(updates)} CIKs")
    return security_master


if __name__ == "__main__":
    sm = build_security_master(force=True)
    print(sm.head(20))
    print(f"\nTotal entries: {len(sm)}")
