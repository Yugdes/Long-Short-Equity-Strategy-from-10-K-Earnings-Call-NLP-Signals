"""Monthly-rebalanced quantile portfolios with turnover and transaction costs.

Input rows are (cik, month=t) with a score known before t, the month-t return `ret` and the
end-of-(t-1) market cap. Value weights use `size_col` (default `mcap_cap`, D-022 capped weights). P1 = lowest score, Pn = highest; LS = Pn - P1.
"""
import numpy as np
import pandas as pd


def assign_bins(df: pd.DataFrame, score: str, n: int, neutral: str | None = None) -> pd.Series:
    keys = ["month"] + ([neutral] if neutral else [])
    pct = df.groupby(keys)[score].rank(pct=True, method="first")
    return np.ceil(pct * n).clip(1, n).astype("Int64")


def sort_portfolios(df: pd.DataFrame, score: str, n: int = 10, weight: str = "ew",
                    neutral: str | None = None, min_names: int = 20, ret_col: str = "ret",
                    size_col: str = "mcap_cap"):
    """-> (rets: DataFrame[month x P1..Pn, LS], holdings: DataFrame[month, cik, bin, w, ret])."""
    d = df.dropna(subset=[score, ret_col]).copy()
    if weight == "vw":
        d = d[d[size_col] > 0]
    d["bin"] = assign_bins(d, score, n, neutral)
    d["w"] = d[size_col] if weight == "vw" else 1.0
    d["w"] = d["w"] / d.groupby(["month", "bin"])["w"].transform("sum")
    cnt = d.groupby(["month", "bin"])["cik"].transform("size")
    d = d[cnt >= min_names]
    d["wr"] = d["w"] * d[ret_col]
    rets = d.groupby(["month", "bin"])["wr"].sum().unstack().reindex(columns=range(1, n + 1))
    rets.columns = [f"P{int(c)}" for c in rets.columns]
    rets["LS"] = rets[f"P{n}"] - rets["P1"]
    return rets.sort_index(), d[["month", "cik", "bin", "w", ret_col]].rename(columns={ret_col: "ret"})


def double_sort(df: pd.DataFrame, first: str, second: str, n: int = 3, weight: str = "ew",
                min_names: int = 20, ret_col: str = "ret", size_col: str = "mcap_cap") -> pd.DataFrame:
    """Conditional (dependent) n x n sort: bins of `first` each month, then bins of `second` within each
    `first` bin. -> DataFrame[month x MultiIndex (i, j)] of cell returns; i = first bin, j = second bin
    (1 = lowest). Cells with fewer than `min_names` stocks are NaN."""
    d = df.dropna(subset=[first, second, ret_col]).copy()
    if weight == "vw":
        d = d[d[size_col] > 0]
    d["i"] = assign_bins(d, first, n)
    d["j"] = assign_bins(d, second, n, neutral="i")
    d["w"] = d[size_col] if weight == "vw" else 1.0
    g = d.groupby(["month", "i", "j"])
    d["w"] = d["w"] / g["w"].transform("sum")
    d["wr"] = d["w"] * d[ret_col]
    cells = d.groupby(["month", "i", "j"]).agg(r=("wr", "sum"), k=("wr", "size"))
    cells.loc[cells["k"] < min_names, "r"] = np.nan
    out = cells["r"].unstack(["i", "j"])
    full = pd.MultiIndex.from_product([range(1, n + 1)] * 2, names=["i", "j"])
    return out.reindex(columns=full).sort_index()


def leg_turnover(h: pd.DataFrame) -> pd.Series:
    """Sum_i |w_it - drifted w_i,t-1| per month for one leg (2 x one-way turnover; 1.0 in the first month)."""
    h = h.copy()
    h["drift"] = h["w"] * (1 + h["ret"].fillna(0))
    h["drift"] = h["drift"] / h.groupby("month")["drift"].transform("sum")
    prev = h[["month", "cik", "drift"]].copy()
    prev["month"] = prev["month"] + 1
    m = h[["month", "cik", "w"]].merge(prev, on=["month", "cik"], how="outer").fillna({"w": 0.0, "drift": 0.0})
    m = m[m["month"].isin(h["month"].unique())]
    return m.assign(a=(m["w"] - m["drift"]).abs()).groupby("month")["a"].sum()


def ls_turnover(holdings: pd.DataFrame, n: int) -> pd.DataFrame:
    """Traded fraction of each leg's capital per month: long (Pn), short (P1) and their sum."""
    tl = leg_turnover(holdings[holdings["bin"] == n])
    ts = leg_turnover(holdings[holdings["bin"] == 1])
    return pd.DataFrame({"long": tl, "short": ts}).assign(total=lambda x: x["long"] + x["short"])


def net_of_costs(ls: pd.Series, turnover_total: pd.Series, bps: float) -> pd.Series:
    """LS return minus one-way cost `bps` on every dollar traded in both legs."""
    return ls - turnover_total.reindex(ls.index).fillna(0) * bps / 1e4
