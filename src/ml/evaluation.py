"""
ML evaluation metrics — IC, IC-IR, and Out-of-Sample R^2.
"""
import pandas as pd
import numpy as np
from typing import Dict

def compute_ic(
    panel: pd.DataFrame,
    pred_col: str = "ml_prediction",
    target_col: str = "rank_excess_ret_next",
    group_col: str = "month"
) -> pd.DataFrame:
    """
    Compute Information Coefficient (Spearman rank correlation) per month.
    """
    df = panel.dropna(subset=[pred_col, target_col]).copy()
    
    # Calculate rank of predictions within each month
    df["pred_rank"] = df.groupby(group_col)[pred_col].rank(pct=True)
    df["target_rank"] = df.groupby(group_col)[target_col].rank(pct=True)
    
    # Calculate Spearman correlation per month
    def _spearman(g):
        return g["pred_rank"].corr(g["target_rank"])
        
    ic_series = df.groupby(group_col).apply(_spearman)
    
    return pd.DataFrame({
        "IC": ic_series
    })

def evaluate_predictions(
    panel: pd.DataFrame,
    pred_col: str = "ml_prediction",
    ret_col: str = "excess_ret_next",
    target_col: str = "rank_excess_ret_next",
) -> Dict[str, float]:
    """
    Evaluate ML predictions with standard quant metrics.
    
    Returns:
    - mean_ic: Average Information Coefficient
    - ic_ir: Information Ratio of the IC (mean / std)
    - oos_r2: Out-of-Sample R^2 (Gu, Kelly, Xiu 2020 definition)
    """
    df = panel.dropna(subset=[pred_col, target_col, ret_col]).copy()
    
    if df.empty:
        return {"mean_ic": np.nan, "ic_ir": np.nan, "oos_r2": np.nan}
        
    # IC
    ic_df = compute_ic(df, pred_col=pred_col, target_col=target_col)
    mean_ic = ic_df["IC"].mean()
    ic_ir = mean_ic / ic_df["IC"].std() if ic_df["IC"].std() > 0 else np.nan
    
    # OOS R2 (vs. naive forecast of zero for excess returns)
    # R^2 = 1 - sum((y - y_hat)^2) / sum(y^2)
    # Note: ML predicts rank, so we can't directly compute R2 of returns unless the target was returns.
    # If the target was ranks, OOS R2 on returns is ill-defined. We report R2 on the target itself.
    
    ss_res = ((df[target_col] - df[pred_col]) ** 2).sum()
    # Baseline: predict the mean rank (which is 0 since we centered it)
    ss_tot = ((df[target_col] - 0) ** 2).sum() 
    
    oos_r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else np.nan
    
    return {
        "mean_ic": mean_ic,
        "ic_ir": ic_ir,
        "oos_r2": oos_r2,
        "n_obs": len(df),
        "n_months": df["month"].nunique()
    }
