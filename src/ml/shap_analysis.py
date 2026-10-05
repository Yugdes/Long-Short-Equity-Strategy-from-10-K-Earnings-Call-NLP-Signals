"""
SHAP analysis for LightGBM model interpretation.
"""
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional

from src.utils.helpers import setup_logging
from src.utils.plotting import save_figure

logger = setup_logging("shap_analysis")

def plot_shap_summary(
    model, 
    X_sample: pd.DataFrame, 
    max_display: int = 20,
    save_path: Optional[Path] = None
):
    """
    Generate and save a SHAP summary plot.
    """
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample)
        
        plt.figure(figsize=(10, 8))
        shap.summary_plot(
            shap_values, 
            X_sample, 
            max_display=max_display, 
            show=False,
            plot_type="dot"
        )
        
        if save_path:
            save_figure(plt.gcf(), save_path, tight=False)
            
        return shap_values
    except Exception as e:
        logger.error(f"Failed to generate SHAP plot: {e}")
        return None
