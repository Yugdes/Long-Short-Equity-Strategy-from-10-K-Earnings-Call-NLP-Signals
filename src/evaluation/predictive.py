"""Out-of-sample predictive metrics."""
import numpy as np
import pandas as pd

from src.backtest.metrics import nw_tstat


def ic_series(df: pd.DataFrame, pred: str, y: str) -> pd.Series:
    """Monthly Spearman rank correlation between prediction and realised outcome."""
    d = df.dropna(subset=[pred, y])
    rp = d.groupby("month")[pred].rank()
    ry = d.groupby("month")[y].rank()
    return pd.concat([rp, ry, d["month"]], axis=1, keys=["p", "y", "month"]).groupby("month").apply(
        lambda g: g["p"].corr(g["y"]))


def oos_r2(y: pd.Series, yhat: pd.Series) -> float:
    """Gu-Kelly-Xiu out-of-sample R^2 against a zero forecast (here on centred-rank targets)."""
    return float(1 - ((y - yhat) ** 2).sum() / (y ** 2).sum())


def hit_rate(yhat: pd.Series, y: pd.Series) -> float:
    """Share of stock-months where the prediction is on the right side of the cross-sectional median
    (both are centred ranks, so the sign comparison is the median comparison)."""
    return float((np.sign(yhat) == np.sign(y)).mean())


def summarize(df: pd.DataFrame, pred: str = "pred", y: str = "y_rank") -> dict:
    ic = ic_series(df, pred, y)
    prank = df.groupby("month")[pred].rank(pct=True) - 0.5
    return {"ic_mean": ic.mean(), "ic_std": ic.std(), "ic_ir": ic.mean() / ic.std() * np.sqrt(12),
            "ic_t_nw": nw_tstat(ic), "ic_pos_months": float((ic > 0).mean()), "n_months": len(ic),
            "oos_r2_rank": oos_r2(df[y], df[pred]), "hit_rate": hit_rate(prank, df[y])}
