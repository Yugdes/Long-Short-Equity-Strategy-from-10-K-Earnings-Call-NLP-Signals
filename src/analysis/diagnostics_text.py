"""
Diagnostics for the text-based risk scores (Phase 2).
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from src.utils.helpers import setup_logging
from src.utils.plotting import save_figure, plot_heatmap

logger = setup_logging("diagnostics")

def run_text_diagnostics(
    risk_scores: pd.DataFrame,
    security_master: pd.DataFrame,
    out_dir: Path
):
    """
    Run the full suite of text diagnostics.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    
    df = risk_scores.copy()
    
    # Merge industry
    if not security_master.empty and "ff48" in security_master.columns:
        df = df.merge(
            security_master[["cik", "ff48"]].drop_duplicates(),
            on="cik",
            how="left"
        )
        
    # 1. Heatmap of OpRisk by Industry-Year
    if "operations_prob" in df.columns and "ff48" in df.columns:
        ind_year_mean = df.groupby(["ff48", "year"])["operations_prob"].mean().unstack()
        plot_heatmap(
            ind_year_mean, 
            title="Mean Operational Risk Probability by Industry-Year",
            save_path=out_dir / "oprisk_heatmap.png"
        )
        
    # 2. Probability vs Count Correlation
    factors = ["operations", "finance", "legal"]
    corrs = []
    for f in factors:
        p_col, c_col = f"{f}_prob", f"{f}_count"
        if p_col in df.columns and c_col in df.columns:
            clean = df[[p_col, c_col]].dropna()
            if not clean.empty:
                r = clean.corr().iloc[0, 1]
                corrs.append({"factor": f, "correlation": r})
                
    corr_df = pd.DataFrame(corrs)
    corr_df.to_csv(out_dir / "prob_count_corr.csv", index=False)
    logger.info(f"Prob-Count correlations:\n{corr_df}")
    
    # 3. Persistence
    if "operations_prob" in df.columns:
        df_sort = df.sort_values(["cik", "year"])
        df_sort["op_lag"] = df_sort.groupby("cik")["operations_prob"].shift(1)
        clean = df_sort[["operations_prob", "op_lag"]].dropna()
        if not clean.empty:
            pers = clean.corr().iloc[0, 1]
            logger.info(f"Year-over-year autocorrelation of OpRisk: {pers:.4f}")
            
    logger.info("Text diagnostics complete.")
