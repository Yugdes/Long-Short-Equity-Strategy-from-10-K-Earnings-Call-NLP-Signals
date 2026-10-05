"""
Signal construction — builds all trading signals from risk scores.
Implements S1–S6 from the research design (Section 4.2).

Signals:
  S1: OpRisk — operations-factor probability
  S1c: OpRisk-count — dictionary count version
  S2: NonOpRisk — mean of percentile ranks of other 7 factors
  S3: OpRisk × NonOp — interaction signal
  S4: ΔOpRisk — year-over-year change
  S5: OpRisk_ind — industry-neutral (primary signal)
  S6: OpRisk_resid — residualized on log word count and log size
"""
import pandas as pd
import numpy as np
from typing import Optional

from src.utils.helpers import (
    load_config, setup_logging, rank_within_group, winsorize_within_group
)

logger = setup_logging("signals")

RISK_FACTORS = [
    "accounting", "finance", "international", "legal",
    "management", "marketing", "operations", "technology",
]


def build_signals(
    risk_scores: pd.DataFrame,
    filings: pd.DataFrame,
    security_master: pd.DataFrame,
    cfg: dict = None,
) -> pd.DataFrame:
    """
    Build all pre-specified signals from risk scores.

    Parameters
    ----------
    risk_scores : DataFrame
        From ingest_risk — columns include cik, year, {factor}_prob, item_1a_word_count
    filings : DataFrame
        Filing dates — cik, date_filed, fiscal_year_approx
    security_master : DataFrame
        CIK → ticker, SIC, FF48 mapping

    Returns
    -------
    DataFrame with cik, year, filing_date, signal_date, first_hold_month,
    and all signal columns (S1–S6).
    """
    if cfg is None:
        cfg = load_config()

    df = risk_scores.copy()

    # ── Merge filing dates ───────────────────────────────────────────────
    if not filings.empty:
        df = df.merge(
            filings[["cik", "fiscal_year_approx", "date_filed"]].rename(
                columns={"fiscal_year_approx": "year"}
            ),
            on=["cik", "year"],
            how="left",
        )
    else:
        # Fallback: assume filed ~90 days after fiscal year end
        df["date_filed"] = pd.to_datetime(df["year"].astype(str) + "-03-31")

    # ── Point-in-time availability ───────────────────────────────────────
    lag_bdays = cfg.get("signal", {}).get("availability_lag_bdays", 2)
    df["date_filed"] = pd.to_datetime(df["date_filed"])
    df["signal_date"] = df["date_filed"] + pd.offsets.BDay(lag_bdays)

    # First holding month = first full calendar month after signal date
    month_end = df["signal_date"] + pd.offsets.MonthEnd(0)
    df["first_hold_month"] = (month_end + pd.offsets.MonthBegin(1)).dt.to_period("M")

    # ── Merge industry codes ─────────────────────────────────────────────
    if not security_master.empty and "ff48" in security_master.columns:
        df = df.merge(
            security_master[["cik", "ff48", "sic"]].drop_duplicates(subset=["cik"]),
            on="cik",
            how="left",
        )
    else:
        df["ff48"] = 48  # Other
        df["sic"] = np.nan

    # ── S1: OpRisk (operations probability) ──────────────────────────────
    if "operations_prob" in df.columns:
        df["S1_OpRisk"] = df["operations_prob"]
    elif "op_risk" in df.columns:
        df["S1_OpRisk"] = df["op_risk"]

    # ── S1c: OpRisk count version ────────────────────────────────────────
    if "operations_count" in df.columns:
        df["S1c_OpRisk_count"] = df["operations_count"]

    # ── S2: NonOpRisk ────────────────────────────────────────────────────
    non_op = [f"{f}_prob" for f in RISK_FACTORS if f != "operations"]
    available_non_op = [c for c in non_op if c in df.columns]
    if available_non_op:
        # Rank each factor within year, then average ranks
        for col in available_non_op:
            df[f"{col}_rank"] = df.groupby("year")[col].rank(pct=True)

        rank_cols = [f"{c}_rank" for c in available_non_op]
        df["S2_NonOpRisk"] = df[rank_cols].mean(axis=1)

        # Also keep the raw sum version (Paper A's definition)
        df["S2_NonOpRisk_sum"] = df[available_non_op].sum(axis=1)

        # Clean up temp rank columns
        df = df.drop(columns=rank_cols)

    # ── S3: Interaction (OpRisk × NonOp) ─────────────────────────────────
    if "S1_OpRisk" in df.columns and "S2_NonOpRisk" in df.columns:
        # Rank both within year first
        df["S1_rank_year"] = df.groupby("year")["S1_OpRisk"].rank(pct=True)
        df["S2_rank_year"] = df.groupby("year")["S2_NonOpRisk"].rank(pct=True)
        df["S3_Interaction"] = df["S1_rank_year"] * df["S2_rank_year"]
        df = df.drop(columns=["S1_rank_year", "S2_rank_year"])

    # ── S4: ΔOpRisk (year-over-year change) ──────────────────────────────
    if "S1_OpRisk" in df.columns:
        df = df.sort_values(["cik", "year"])
        df["S1_OpRisk_lag"] = df.groupby("cik")["S1_OpRisk"].shift(1)
        df["year_lag"] = df.groupby("cik")["year"].shift(1)
        df["S4_DeltaOpRisk"] = df["S1_OpRisk"] - df["S1_OpRisk_lag"]

        # Only valid if consecutive years (≤ 18 months apart)
        df.loc[df["year"] - df["year_lag"] > 1.5, "S4_DeltaOpRisk"] = np.nan
        df = df.drop(columns=["S1_OpRisk_lag", "year_lag"])

    # ── S5: Industry-neutral OpRisk (PRIMARY SIGNAL) ─────────────────────
    if "S1_OpRisk" in df.columns:
        # Subtract FF48 industry-year median
        industry_median = df.groupby(["ff48", "year"])["S1_OpRisk"].transform("median")
        df["S5_OpRisk_ind"] = df["S1_OpRisk"] - industry_median

    # ── S6: Residualized OpRisk ──────────────────────────────────────────
    if "S1_OpRisk" in df.columns and "item_1a_word_count" in df.columns:
        df["log_words"] = np.log1p(df["item_1a_word_count"])
        # Will residualize within year in panel construction (needs size)
        df["S6_OpRisk_resid"] = np.nan  # Placeholder — computed in panel.py

    # ── Per-factor signals (S8, exploratory) ─────────────────────────────
    for factor in RISK_FACTORS:
        prob_col = f"{factor}_prob"
        if prob_col in df.columns:
            # Industry-neutral version
            ind_med = df.groupby(["ff48", "year"])[prob_col].transform("median")
            df[f"S8_{factor}_ind"] = df[prob_col] - ind_med

    # ── Apply universe filters ───────────────────────────────────────────
    min_words = cfg.get("universe", {}).get("min_item1a_words", 1000)
    drop_optional = cfg.get("universe", {}).get("drop_item1a_optional", True)

    n_before = len(df)
    if "item_1a_word_count" in df.columns:
        df = df[df["item_1a_word_count"] >= min_words]
    if drop_optional and "item_1a_optional" in df.columns:
        df = df[df["item_1a_optional"] != 1]

    # Exclude financials and utilities in primary specification
    exclude_sic = cfg.get("universe", {}).get("exclude_sic", [[6000, 6999], [4900, 4999]])
    if exclude_sic and "sic" in df.columns:
        for lo, hi in exclude_sic:
            df = df[~df["sic"].between(lo, hi)]

    logger.info(f"Signals built: {n_before} → {len(df)} rows after filters")
    logger.info(f"Signal columns: {[c for c in df.columns if c.startswith('S')]}")

    return df


def compute_signal_diagnostics(signals: pd.DataFrame) -> dict:
    """
    Compute key diagnostics for text signals (Phase 2).
    Returns dict with:
    - prob_count_correlations: correlation between probability and count versions
    - persistence: year-over-year autocorrelation of OpRisk
    - cross_factor_corr: correlation matrix of all 8 factor probabilities
    """
    diagnostics = {}

    # Probability vs. count correlation per factor
    prob_count_corr = {}
    for factor in RISK_FACTORS:
        prob_col = f"{factor}_prob"
        count_col = f"{factor}_count"
        if prob_col in signals.columns and count_col in signals.columns:
            corr = signals[[prob_col, count_col]].corr().iloc[0, 1]
            prob_count_corr[factor] = corr
    diagnostics["prob_count_correlations"] = prob_count_corr

    # Persistence (autocorrelation)
    if "S1_OpRisk" in signals.columns:
        df_sorted = signals.sort_values(["cik", "year"])
        df_sorted["OpRisk_lag1"] = df_sorted.groupby("cik")["S1_OpRisk"].shift(1)
        autocorr = df_sorted[["S1_OpRisk", "OpRisk_lag1"]].corr().iloc[0, 1]
        diagnostics["oprisk_autocorrelation"] = autocorr

    # Cross-factor correlations
    prob_cols = [f"{f}_prob" for f in RISK_FACTORS if f"{f}_prob" in signals.columns]
    if prob_cols:
        diagnostics["cross_factor_corr"] = signals[prob_cols].corr()

    return diagnostics


if __name__ == "__main__":
    print("Signal construction module — run via src.run_all or Phase 3 pipeline")
