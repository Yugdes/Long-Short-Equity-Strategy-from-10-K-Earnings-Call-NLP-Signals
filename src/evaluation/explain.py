"""SHAP explainability for the frozen LightGBM (ALL features) that predicts the first test year.

The model is tuned and fit exactly like the walk-forward fold for 2022 (train <= 2021-11), so it is the
model actually used out of sample. It never sees the locked window. Explained on <= 50k sampled training rows.
"""
import numpy as np
import pandas as pd

from src.features.signals import FEATURE_GROUPS
from src.models.learners import LGBM
from src.models.run_ml import SETS
from src.models.walkforward import splits
from src.utils.config import path

GROUP_OF = {f: g for g, fs in FEATURE_GROUPS.items() for f in fs}


def frozen_lgbm(feat: pd.DataFrame, features: list):
    f = splits(feat["month"].unique(), 2022, 2022)[0]
    tr = feat[(feat["month"] <= f["train_end"]) & feat["y_rank"].notna()]
    fit, val = tr[tr["month"] <= f["fit_end"]], tr[tr["month"] >= f["val_start"]]
    m = LGBM(fit[features].values, fit["y_rank"].values, val[features].values, val["y_rank"].values)
    m.refit(tr[features].values, tr["y_rank"].values)
    return m, tr


def shap_analysis(feat: pd.DataFrame, n_sample: int = 50_000, top_k: int = 6, seed: int = 0) -> dict:
    import shap
    features = SETS["ALL"]
    model, tr = frozen_lgbm(feat, features)
    X = tr[features].sample(min(n_sample, len(tr)), random_state=seed)
    sv = shap.TreeExplainer(model.m).shap_values(X.values)
    imp = pd.DataFrame({"feature": features, "mean_abs_shap": np.abs(sv).mean(axis=0)})
    imp["group"] = imp["feature"].map(GROUP_OF)
    imp = imp.sort_values("mean_abs_shap", ascending=False)
    groups = imp.groupby("group")["mean_abs_shap"].sum().sort_values(ascending=False)
    dep = {}
    for f in imp["feature"].head(top_k):
        j = features.index(f)
        b = pd.qcut(X[f].rank(method="first"), 20, labels=False)
        dep[f] = pd.DataFrame({"x": X[f].groupby(b).mean(), "shap": pd.Series(sv[:, j], index=X.index).groupby(b).mean()})
    imp.to_csv(path("results") / "tables" / "shap_importance.csv", index=False)
    return {"importance": imp, "groups": groups, "dependence": dep, "chosen": model.chosen}
