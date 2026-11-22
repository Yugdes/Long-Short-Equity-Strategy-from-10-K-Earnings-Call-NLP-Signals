"""Cross-sectional features. Every transform runs per month over that month's eligible rows (D-010).

Features are centred percentile ranks in [-0.5, 0.5]; a missing value becomes 0, the cross-sectional
median of that month (so imputation uses no other month and no future data). Missing-marketing rows
get a `has_mkt` flag. Groups are exported in FEATURE_GROUPS for the nested model experiments.
"""
import numpy as np
import pandas as pd

from src.data.marketing import CONSTRUCTS, ZCA
from src.data.risk import FACTORS

CONTROL_VARS = ["size", "bm", "mom_12_2", "rev_1", "vol_12", "beta_36", "illiq_12", "ldvol_3", "ag", "gp"]
RISK_LEVEL = [f"{f}_prob" for f in FACTORS] + ["nonop_prob", "dens_ops", "log_words"]
RISK_VARS = RISK_LEVEL + [f"d_{c}" for c in RISK_LEVEL] + [f"s_{c}" for c in RISK_LEVEL]
MKT_VARS = list(CONSTRUCTS) + list(ZCA.values()) + ["an_cust"] + \
           [f"dq_{c}" for c in CONSTRUCTS] + [f"dy_{c}" for c in CONSTRUCTS]


def _rank(df: pd.DataFrame, cols: list, by: str = "month") -> pd.DataFrame:
    return df.groupby(by)[cols].rank(pct=True) - 0.5


def make_features(panel: pd.DataFrame, mask_col: str = "elig") -> pd.DataFrame:
    p = panel[panel[mask_col]].copy()
    # D-022: capped value weights (Jensen, Kelly & Pedersen 2023): mcap winsorised at the monthly 80th
    # percentile of currently NYSE-listed names in the universe
    if {"mcap_l", "exchange"} <= set(p.columns):
        cap = p[p["exchange"] == "NYSE"].groupby("month")["mcap_l"].quantile(0.8)
        p["mcap_cap"] = np.minimum(p["mcap_l"], p["month"].map(cap).fillna(p["mcap_l"].max()))
    rk = _rank(p, CONTROL_VARS + RISK_VARS + MKT_VARS)
    rk.columns = ["r_" + c for c in rk.columns]
    p = pd.concat([p, rk], axis=1)
    p["has_mkt"] = p["mkt_orient"].notna().astype("float32")
    # composites (computed before imputation; NaN stays NaN in the inputs' mean)
    p["r_risk_all"] = p[[f"r_{f}_prob" for f in FACTORS]].mean(axis=1)
    p["r_d_risk_all"] = p[[f"r_d_{f}_prob" for f in FACTORS]].mean(axis=1)
    p["r_ops_ind"] = p["r_operations_prob"] - p.groupby(["month", "ff48"])["r_operations_prob"].transform("mean")
    feat = [c for c in p.columns if c.startswith("r_")]
    p[feat] = p[feat].fillna(0.0)
    # interactions of centred ranks
    p["x_ops_nonop"] = p["r_operations_prob"] * p["r_nonop_prob"]
    for c in CONSTRUCTS:
        p[f"x_ops_{c}"] = p["r_operations_prob"] * p[f"r_{c}"]
        p[f"x_dops_{c}"] = p["r_d_operations_prob"] * p[f"r_{c}"]
    # target: centred cross-sectional rank of the month-t excess return
    p["y_rank"] = p.groupby("month")["exret"].rank(pct=True) - 0.5
    p["y_rank_3m"] = p.groupby("month")["exret_3m"].rank(pct=True) - 0.5
    fcols = [c for c in p.columns if c.startswith(("r_", "x_"))] + ["has_mkt"]
    p[fcols] = p[fcols].astype("float32")
    return p


FEATURE_GROUPS = {
    "controls": [f"r_{c}" for c in CONTROL_VARS],
    "risk": [f"r_{c}" for c in RISK_VARS] + ["r_risk_all", "r_d_risk_all", "r_ops_ind"],
    "marketing": [f"r_{c}" for c in MKT_VARS] + ["has_mkt"],
    "interactions": ["x_ops_nonop"] + [f"x_ops_{c}" for c in CONSTRUCTS] + [f"x_dops_{c}" for c in CONSTRUCTS],
}
