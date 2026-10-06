"""
Panel construction — merge signals, returns, characteristics, and factors
into the final modelling panel (panel_stock_month.parquet).

Implements calendar-time eligibility per Jegadeesh-Titman methodology.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional

from src.utils.helpers import (
    load_config, get_path, setup_logging, save_parquet, load_parquet, parquet_exists,
    winsorize_within_group
)

logger = setup_logging("panel")


def build_stock_month_panel(
    signals: pd.DataFrame,
    returns: pd.DataFrame,
    factors: pd.DataFrame,
    security_master: pd.DataFrame,
    marketing: pd.DataFrame = None,
    cfg: dict = None,
    force: bool = False,
) -> pd.DataFrame:
    """
    Build the master stock-month panel for backtesting and ML.

    Steps:
    1. Create monthly calendar grid
    2. For each month, find the latest available signal per CIK (within 12 months)
    3. Merge with monthly returns
    4. Compute forward returns and characteristics
    5. Merge factor data

    Returns: panel_stock_month DataFrame
    """
    if cfg is None:
        cfg = load_config()

    out_file = Path(get_path(cfg, "processed")) / "panel_stock_month.parquet"
    if parquet_exists(out_file) and not force:
        logger.info("Loading cached panel")
        return load_parquet(out_file)

    # ── Step 1: Create signal table with availability dates ──────────────
    sig = signals.copy()
    sig["first_hold_start"] = pd.to_datetime(sig["first_hold_month"].dt.to_timestamp()).astype('datetime64[ns]')

    # ── Step 2: Create the monthly grid ──────────────────────────────────
    # Create full CIK × month grid (cartesian product)
    from itertools import product
    if returns.empty:
        logger.warning("No return data available — building panel from signals only")
        all_months = pd.period_range(start=cfg.get("sample", {}).get("start", "2006-01"), end="2024-12", freq="M")
    else:
        all_months = sorted(returns["month"].unique())

    holding_months = cfg.get("signal", {}).get("holding_months", 12)
    unique_ciks = sig["cik"].unique()
    logger.info(f"Building panel: {len(unique_ciks)} CIKs × {len(all_months)} months")

    full_grid = pd.DataFrame(list(product(unique_ciks, all_months)), columns=["cik", "month"])
    full_grid["month_start"] = pd.to_datetime(
        full_grid["month"].apply(lambda m: m.to_timestamp() if hasattr(m, "to_timestamp") else pd.Timestamp(m))
    ).astype('datetime64[ns]')

    # Single efficient merge_asof
    full_grid = full_grid.sort_values("month_start")
    cik_sig = sig[["cik", "first_hold_start"] + [c for c in sig.columns if c.startswith("S")]].sort_values("first_hold_start")

    merged = pd.merge_asof(
        full_grid,
        cik_sig,
        left_on="month_start",
        right_on="first_hold_start",
        by="cik",
        direction="backward",
        tolerance=pd.Timedelta(days=365 * holding_months / 12 + 30),
    )

    panel = merged.dropna(subset=["first_hold_start"]).reset_index(drop=True)

    if panel.empty:
        logger.error("No panel rows created — check signal/return data alignment")
        return pd.DataFrame()

    logger.info(f"Raw panel: {len(panel)} stock-month rows")

    # ── Step 2b: Merge marketing scores point-in-time ────────────────────
    if marketing is not None and not marketing.empty and "avail_date" in marketing.columns:
        logger.info("Merging marketing data point-in-time")
        
        if "cik" not in marketing.columns:
            logger.warning("Marketing data lacks 'cik' column! Cannot merge. You need a WRDS gvkey-cik link table.")
        else:
            mktg = marketing.sort_values("avail_date")
            panel = panel.sort_values("month_start")
            
            merged_m = pd.merge_asof(
                panel,
                mktg[["cik", "avail_date", "market_orientation", "marketing_capabilities", "marketing_excellence"]].sort_values("avail_date"),
                left_on="month_start",
                right_on="avail_date",
                by="cik",
                direction="backward"
            )
            panel = merged_m
            
            # Fill NA marketing scores with median or 0 for ML
            for col in ["market_orientation", "marketing_capabilities", "marketing_excellence"]:
                if col in panel.columns:
                    panel[col] = panel.groupby("month")[col].transform(lambda x: x.fillna(x.median()))
                    panel[col] = panel[col].fillna(0) # Global fallback
                
    # ── Step 3: Merge returns ────────────────────────────────────────────
    if not returns.empty:
        # Map CIK → ticker through security master
        if not security_master.empty:
            cik_ticker = security_master[["cik", "ticker"]].drop_duplicates(subset=["cik"])
            panel = panel.merge(cik_ticker, on="cik", how="left")

            # Merge returns on ticker × month
            ret_cols = ["ticker", "month", "ret", "price_end", "volume_avg"]
            available_ret_cols = [c for c in ret_cols if c in returns.columns]
            panel = panel.merge(
                returns[available_ret_cols],
                on=["ticker", "month"],
                how="left",
            )

    # ── Step 4: Forward returns ──────────────────────────────────────────
    if "ret" in panel.columns:
        panel = panel.sort_values(["cik", "month"])
        panel["ret_next"] = panel.groupby("cik")["ret"].shift(-1)

        # Excess return (subtract RF)
        if not factors.empty:
            panel = panel.merge(
                factors[["month", "RF"]],
                on="month",
                how="left",
            )
            panel["excess_ret_next"] = panel["ret_next"] - panel["RF"]

            # Rank-transform for ML target
            panel["rank_excess_ret_next"] = panel.groupby("month")[
                "excess_ret_next"
            ].rank(pct=True) - 0.5

    # ── Step 5: Characteristics ──────────────────────────────────────────
    if "price_end" in panel.columns and "volume_avg" in panel.columns:
        panel["log_price"] = np.log(panel["price_end"].clip(lower=0.01))

        # Momentum 12-1 (simplified — needs 12 months of past returns)
        panel = panel.sort_values(["cik", "month"])
        panel["mom_12_1"] = panel.groupby("cik")["ret"].transform(
            lambda x: x.rolling(12, min_periods=10).apply(
                lambda r: (1 + r[:-1]).prod() - 1 if len(r) > 1 else np.nan,
                raw=True,
            )
        )

        # 1-month reversal
        panel["reversal_1"] = panel["ret"]

        # Realized volatility (12-month rolling)
        panel["vol_12m"] = panel.groupby("cik")["ret"].transform(
            lambda x: x.rolling(12, min_periods=6).std() * np.sqrt(12)
        )

    # ── Step 6: Winsorize continuous features ────────────────────────────
    winsorize_cols = [c for c in panel.columns if c.startswith("S") or c in
                      ["mom_12_1", "reversal_1", "vol_12m"]]
    for col in winsorize_cols:
        if panel[col].dtype in ["float64", "float32"]:
            panel[col] = winsorize_within_group(panel, col, "month")

    # ── Step 7: Point-in-time assertion ──────────────────────────────────
    if "first_hold_start" in panel.columns and "month_start" in panel.columns:
        violations = (panel["first_hold_start"] > panel["month_start"]).sum()
        if violations > 0:
            logger.error(f"LOOK-AHEAD BIAS DETECTED: {violations} rows violate "
                         f"point-in-time constraint!")
        else:
            logger.info("✓ Point-in-time constraint satisfied")

    save_parquet(panel, out_file)
    logger.info(f"Final panel: {len(panel)} rows, {panel['cik'].nunique()} firms")
    return panel


def compute_coverage_funnel(
    risk_scores: pd.DataFrame,
    signals: pd.DataFrame,
    panel: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute the coverage funnel by year — how many firms survive each filter.
    """
    years = sorted(risk_scores["year"].unique())

    funnel = []
    for year in years:
        row = {"year": year}
        row["raw_text"] = (risk_scores["year"] == year).sum()

        if not signals.empty:
            row["after_filters"] = (signals["year"] == year).sum()

        if not panel.empty:
            # Approximate: count unique CIKs in the panel for this year
            year_panel = panel[panel["month"].apply(
                lambda m: m.year if hasattr(m, "year") else int(str(m)[:4])
            ) == year]
            row["in_panel"] = year_panel["cik"].nunique() if not year_panel.empty else 0

        funnel.append(row)

    return pd.DataFrame(funnel)


if __name__ == "__main__":
    print("Panel construction — run via src.run_all or Phase 3 pipeline")
