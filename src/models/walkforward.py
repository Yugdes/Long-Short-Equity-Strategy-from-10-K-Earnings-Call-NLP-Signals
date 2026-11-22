"""Expanding-window walk-forward with annual refits, embargo and in-window tuning (brief Phase 4).

Fold for test year Y:
  train  = months <= train_end, where train_end = Dec(Y-1) - embargo - (horizon - 1), so every training
           label is realised before the embargo month and the test year begins;
  fit/val split inside train: val = the last `val_years` years of train (tuning and early stopping),
           with a 1-month gap; after selection the model is refit on all of train;
  test   = Jan..Dec of Y.
"""
import time

import numpy as np
import pandas as pd

from src.utils.lock import assert_unlocked, test_start


def splits(months, first_test_year: int, last_test_year: int, embargo: int = 1, horizon: int = 1,
           val_years: int = 2) -> list[dict]:
    months = pd.PeriodIndex(sorted(set(months)), freq="M")
    out = []
    for y in range(first_test_year, last_test_year + 1):
        ts = pd.Period(f"{y}-01", "M")
        te = pd.Period(f"{y}-12", "M")
        train_end = ts - 1 - embargo - (horizon - 1)
        val_start = train_end - 12 * val_years + 1
        out.append({"year": y, "train_start": months.min(), "train_end": train_end,
                    "fit_end": val_start - 2, "val_start": val_start, "val_end": train_end,
                    "test_start": ts, "test_end": te})
    return out


def run_walkforward(feat: pd.DataFrame, features: list, target: str, learner, first_test_year: int,
                    last_test_year: int, horizon: int = 1, embargo: int = 1, log=print) -> pd.DataFrame:
    """learner(X_fit, y_fit, X_val, y_val) -> fitted object with .predict and .refit(X, y).
    Returns out-of-sample predictions [cik, month, pred, fold]."""
    if pd.Period(f"{last_test_year}-12", "M") >= test_start():
        assert_unlocked()
    preds = []
    for f in splits(feat["month"].unique(), first_test_year, last_test_year, embargo, horizon):
        tr = feat[(feat["month"] <= f["train_end"]) & feat[target].notna()]
        fit = tr[tr["month"] <= f["fit_end"]]
        val = tr[tr["month"] >= f["val_start"]]
        te = feat[(feat["month"] >= f["test_start"]) & (feat["month"] <= f["test_end"])]
        t0 = time.time()
        model = learner(fit[features].values, fit[target].values, val[features].values, val[target].values)
        model.refit(tr[features].values, tr[target].values)
        p = model.predict(te[features].values)
        preds.append(pd.DataFrame({"cik": te["cik"].values, "month": te["month"].values, "pred": p, "fold": f["year"]}))
        log(f"  fold {f['year']}: train {len(tr):,} rows -> test {len(te):,}  [{time.time() - t0:.0f}s] {getattr(model, 'chosen', '')}")
    return pd.concat(preds, ignore_index=True)
