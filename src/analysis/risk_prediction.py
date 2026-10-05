"""
Risk Prediction Analysis (H4).
Tests whether disclosed operational risk predicts next-year realized volatility
and maximum drawdown, independently of return alpha.
"""
import pandas as pd
import numpy as np
import statsmodels.api as sm
from typing import Dict, List, Tuple

from src.utils.helpers import setup_logging
from src.backtest.metrics import max_drawdown

logger = setup_logging("risk_prediction")

def compute_forward_risk_metrics(
    returns_df: pd.DataFrame, 
    min_months: int = 10,
    horizon_months: int = 12
) -> pd.DataFrame:
    """
    Compute forward 12-month realized volatility and max drawdown for each stock-month.
    This is computationally heavy, so in practice we might only run it annually 
    (e.g., matching the signal release month) rather than every single month.
    """
    logger.info("Computing forward risk metrics (this may take a moment)...")
    
    # We need a time-series approach per stock
    # For simplicity in this skeleton, we'll return a placeholder that 
    # assumes the panel already has `vol_12m` (backward) and we just shift it.
    # In a full implementation, you'd calculate forward rolling windows.
    
    df = returns_df.copy()
    df = df.sort_values(["cik", "month"])
    
    # Very rough approximation for the skeleton: shift backward 12m vol to get forward vol
    df["fwd_vol_12m"] = df.groupby("cik")["vol_12m"].shift(-horizon_months)
    
    # Rough proxy for forward drawdown: worst single month in next 12
    df["fwd_worst_ret"] = df.groupby("cik")["ret"].transform(
        lambda x: x.rolling(horizon_months).min().shift(-horizon_months)
    )
    
    return df

def run_risk_regressions(
    panel: pd.DataFrame,
    signal_col: str = "S1_OpRisk",
    controls: List[str] = None
) -> Dict:
    """
    Regress forward risk metrics on the risk signal and controls.
    """
    if controls is None:
        controls = ["log_size"] # basic control
        
    df = panel.dropna(subset=[signal_col, "fwd_vol_12m"] + controls).copy()
    
    # Volatility regression
    y_vol = df["fwd_vol_12m"]
    X = sm.add_constant(df[[signal_col] + controls])
    
    try:
        model_vol = sm.OLS(y_vol, X).fit(cov_type="cluster", cov_kwds={"groups": df["cik"]})
        logger.info(f"Vol Regression - Signal Coef: {model_vol.params[signal_col]:.4f} "
                    f"(t={model_vol.tvalues[signal_col]:.2f})")
    except Exception as e:
        logger.error(f"Vol regression failed: {e}")
        model_vol = None
        
    return {
        "volatility_model": model_vol
    }
