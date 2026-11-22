"""Time-series factor alphas (Newey-West) and Fama-MacBeth cross-sectional regressions."""
import numpy as np
import pandas as pd
import statsmodels.api as sm

MODELS = {
    "CAPM": ["mktrf"],
    "FF3": ["mktrf", "smb", "hml"],
    "FF5": ["mktrf", "smb", "hml", "rmw", "cma"],
    "FF5+Mom": ["mktrf", "smb", "hml", "rmw", "cma", "mom"],
}


def factor_alpha(r: pd.Series, factors: pd.DataFrame, model: str = "FF5+Mom", nw_lags: int = 6,
                 excess: bool = False) -> dict:
    """r indexed by Period[M]. Zero-investment returns are used as is; for a long-only leg pass
    excess=True to subtract RF."""
    f = factors.set_index("month")
    d = pd.DataFrame({"r": r}).join(f, how="inner").dropna(subset=["r"] + MODELS[model])
    y = d["r"] - d["rf"] if excess else d["r"]
    X = sm.add_constant(d[MODELS[model]])
    res = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": nw_lags})
    out = {"model": model, "alpha_m": res.params["const"], "alpha_ann": res.params["const"] * 12,
           "t_alpha": res.tvalues["const"], "r2_adj": res.rsquared_adj, "n": int(res.nobs)}
    for c in MODELS[model]:
        out[f"b_{c}"], out[f"t_{c}"] = res.params[c], res.tvalues[c]
    return out


def all_alphas(r: pd.Series, factors: pd.DataFrame, nw_lags: int = 6, excess: bool = False) -> pd.DataFrame:
    return pd.DataFrame([factor_alpha(r, factors, m, nw_lags, excess) for m in MODELS])


def fama_macbeth(df: pd.DataFrame, y: str, xs: list, nw_lags: int = 6, min_obs: int = 50) -> pd.DataFrame:
    """Monthly OLS of y on xs (plus constant); NW t-stats on the time series of slopes."""
    coefs = []
    for m, g in df.dropna(subset=[y] + xs).groupby("month"):
        if len(g) < min_obs:
            continue
        X = np.column_stack([np.ones(len(g)), g[xs].values])
        b, *_ = np.linalg.lstsq(X, g[y].values, rcond=None)
        coefs.append(pd.Series(b, index=["const"] + xs, name=m))
    c = pd.DataFrame(coefs)
    rows = []
    for col in c.columns:
        res = sm.OLS(c[col].values, np.ones(len(c))).fit(cov_type="HAC", cov_kwds={"maxlags": nw_lags})
        rows.append({"var": col, "coef": c[col].mean(), "t_nw": res.tvalues[0], "n_months": len(c)})
    return pd.DataFrame(rows).set_index("var")
