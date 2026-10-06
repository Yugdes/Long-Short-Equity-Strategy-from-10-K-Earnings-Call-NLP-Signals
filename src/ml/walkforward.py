"""
Walk-forward validation logic for ML models.
Implements expanding window with an optional embargo period to prevent leakage.
"""
import pandas as pd
import numpy as np
from typing import List, Dict, Callable, Optional, Tuple
import time

from src.utils.helpers import setup_logging

logger = setup_logging("walkforward")


def get_walkforward_splits(
    dates: pd.Series,
    initial_train_years: int = 7,
    refit_every_months: int = 12,
    embargo_months: int = 1,
) -> List[Dict[str, pd.Timestamp]]:
    """
    Generate train/test split dates for an expanding-window walk-forward validation.

    Parameters
    ----------
    dates : Series
        All available dates in the panel
    initial_train_years : int
        Size of the first training window
    refit_every_months : int
        How often to retrain the model (size of the test block)
    embargo_months : int
        Number of months to drop between train and test to prevent leakage

    Returns
    -------
    list of dicts with 'train_start', 'train_end', 'test_start', 'test_end'
    """
    unique_dates = pd.Series(dates.unique()).sort_values()
    if unique_dates.empty:
        return []

    start_date = unique_dates.min()
    end_date = unique_dates.max()

    # Initial train end
    current_train_end = start_date + pd.DateOffset(years=initial_train_years)

    splits = []
    while current_train_end < end_date:
        # Test start is after embargo
        test_start = current_train_end + pd.DateOffset(months=embargo_months)

        # If test_start is beyond our data, we're done
        if test_start > end_date:
            break

        # Test block size
        test_end = test_start + pd.DateOffset(months=refit_every_months) - pd.DateOffset(days=1)
        if test_end > end_date:
            test_end = end_date

        splits.append({
            "train_start": start_date,
            "train_end": current_train_end,
            "test_start": test_start,
            "test_end": test_end,
        })

        # Expanding window: train end moves forward by the refit step
        current_train_end += pd.DateOffset(months=refit_every_months)

    return splits


def walkforward_cv(
    panel: pd.DataFrame,
    features: List[str],
    target: str,
    model_func: Callable,
    initial_train_years: int = 7,
    refit_every_months: int = 12,
    embargo_months: int = 1,
    date_col: str = "month_start",
    **model_kwargs
) -> Tuple[pd.DataFrame, List[object]]:
    """
    Execute walk-forward cross-validation.

    Parameters
    ----------
    panel : DataFrame
        The full stock-month panel
    features : list
        List of feature column names
    target : str
        Target column name
    model_func : callable
        Function that takes (X_train, y_train, X_val, y_val, **kwargs) and returns a trained model
        and its predictions on X_val.
    ...
    Returns
    -------
    predictions : DataFrame
        Panel rows from the test periods with a new 'prediction' column
    models : list
        List of trained model objects (one per split)
    """
    # Ensure dates are timestamps
    if not np.issubdtype(panel[date_col].dtype, np.datetime64):
        panel[date_col] = pd.to_datetime(panel[date_col].apply(
            lambda x: x.to_timestamp() if hasattr(x, "to_timestamp") else x
        ))

    splits = get_walkforward_splits(
        panel[date_col],
        initial_train_years=initial_train_years,
        refit_every_months=refit_every_months,
        embargo_months=embargo_months,
    )

    logger.info(f"Generated {len(splits)} walk-forward splits")

    all_preds = []
    models = []

    for i, split in enumerate(splits):
        logger.info(f"Split {i+1}/{len(splits)}: "
                    f"Train [{split['train_start'].date()} – {split['train_end'].date()}], "
                    f"Test [{split['test_start'].date()} – {split['test_end'].date()}]")

        train_mask = (panel[date_col] >= split["train_start"]) & (panel[date_col] <= split["train_end"])
        test_mask = (panel[date_col] >= split["test_start"]) & (panel[date_col] <= split["test_end"])

        train_df = panel[train_mask].dropna(subset=[target])
        test_df = panel[test_mask].dropna(subset=[target])

        if train_df.empty or test_df.empty:
            logger.warning(f"Empty train or test set for split {i+1}. Skipping.")
            continue

        X_train = train_df[features]
        y_train = train_df[target]
        X_test = test_df[features]
        y_test = test_df[target]

        # Use the last 2 years of the training window as an internal validation set
        # (Useful for early stopping in LightGBM/XGBoost)
        val_start = split["train_end"] - pd.DateOffset(years=2)
        val_mask_internal = (train_df[date_col] > val_start)
        
        if val_mask_internal.sum() > 1000:
            X_tr, y_tr = X_train[~val_mask_internal], y_train[~val_mask_internal]
            X_val, y_val = X_train[val_mask_internal], y_train[val_mask_internal]
        else:
            # If not enough data, just use the whole train set and don't do early stopping
            X_tr, y_tr = X_train, y_train
            X_val, y_val = None, None

        start_time = time.time()
        
        # Train model
        model, test_preds = model_func(
            X_train=X_tr, 
            y_train=y_tr, 
            X_val=X_val, 
            y_val=y_val, 
            X_test=X_test,
            **model_kwargs
        )
        
        train_time = time.time() - start_time
        logger.info(f"  Trained in {train_time:.1f}s")

        test_df = test_df.copy()
        test_df["ml_prediction"] = test_preds
        all_preds.append(test_df)
        models.append(model)

    if not all_preds:
        return pd.DataFrame(), []

    final_preds = pd.concat(all_preds, ignore_index=True)
    return final_preds, models

if __name__ == "__main__":
    pass
