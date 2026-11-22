"""Control characteristics, measured with information available at the END of month m.

`panel.py` shifts them to holding month t = m + 1, so a row for month t only carries data known
before t begins.
"""
import numpy as np
import pandas as pd


def asof_join(left: pd.DataFrame, right: pd.DataFrame, key: str, cols: list) -> pd.DataFrame:
    """For each (cik, month) in `left`, the last `right` row (same cik) with right[key] <= month.

    Both `month` and `right[key]` are Period[M]; ties in `key` resolve to the later row in `right`'s order.
    """
    l = left[["cik", "month"]].copy()
    l["_k"] = l["month"].astype("int64") if l["month"].dtype == "int64" else pd.PeriodIndex(l["month"]).asi8
    r = right[["cik", key] + cols].copy()
    f64 = r[cols].select_dtypes("float64").columns
    r[f64] = r[f64].astype("float32")          # memory: the panel stores float32 anyway
    r["_k"] = pd.PeriodIndex(r[key]).asi8
    l = l.reset_index().sort_values("_k")
    r = r.drop(columns=key).sort_values("_k", kind="stable")
    m = pd.merge_asof(l, r, on="_k", by="cik", direction="backward")
    return m.set_index("index").sort_index()[cols]


def full_monthly_grid(pm: pd.DataFrame) -> pd.DataFrame:
    """Reindex each ticker to a gap-free monthly grid between its first and last month."""
    rng = pm.groupby("ticker")["month"].agg(["min", "max"])
    lens = (pd.PeriodIndex(rng["max"]).asi8 - pd.PeriodIndex(rng["min"]).asi8 + 1)
    tick = np.repeat(rng.index.values, lens)
    start = np.repeat(pd.PeriodIndex(rng["min"]).asi8, lens)
    offs = np.concatenate([np.arange(n) for n in lens])
    grid = pd.DataFrame({"ticker": tick, "month": pd.PeriodIndex.from_ordinals(start + offs, freq="M")})
    return grid.merge(pm, on=["ticker", "month"], how="left")


def return_characteristics(g: pd.DataFrame, mkt: pd.Series) -> pd.DataFrame:
    """g: gap-free ticker-month grid with ret, prc, dvol, amihud, vol_d. Adds (end-of-month m):
    mom_12_2 (cum. ret m-11..m-1, i.e. t-12..t-2 for t=m+1), rev_1 (ret_m), vol_12 (std ret m-11..m),
    beta_36 (36m, min 24), illiq_12 (log mean Amihud m-11..m), ldvol_3 (log mean $vol m-2..m),
    hist (count of returns observed up to m)."""
    g = g.sort_values(["ticker", "month"]).reset_index(drop=True)
    by = g.groupby("ticker", sort=False)
    lr = np.log1p(g["ret"])
    g["rev_1"] = g["ret"]
    s = lr.groupby(g["ticker"]).shift(1)
    g["mom_12_2"] = np.expm1(s.groupby(g["ticker"]).rolling(11, min_periods=8).sum().reset_index(level=0, drop=True))
    g["vol_12"] = by["ret"].rolling(12, min_periods=8).std().reset_index(level=0, drop=True)
    ami = by["amihud"].rolling(12, min_periods=6).mean().reset_index(level=0, drop=True)
    g["illiq_12"] = np.log(ami.where(ami > 0))
    g["ldvol_3"] = np.log(by["dvol"].rolling(3, min_periods=1).mean().reset_index(level=0, drop=True).clip(lower=1))
    g["hist"] = by["ret"].transform(lambda x: x.notna().cumsum())
    g["mkt"] = g["month"].map(mkt)
    xy = g["ret"] * g["mkt"]
    cnt = (g["ret"].notna() & g["mkt"].notna()).astype(float)
    def roll(x):
        return x.groupby(g["ticker"]).rolling(36, min_periods=24).sum().reset_index(level=0, drop=True)
    rx = g["ret"].where(cnt > 0)
    mx = g["mkt"].where(cnt > 0)
    n = roll(cnt)
    cov = roll(xy.where(cnt > 0)) / n - roll(rx) / n * roll(mx) / n
    var = roll(mx ** 2) / n - (roll(mx) / n) ** 2
    g["beta_36"] = (cov / var).where(n >= 24)
    return g.drop(columns=["mkt"])


def latest_facts(facts: pd.DataFrame, value_cols: list, lag_months: int) -> pd.DataFrame:
    """Make XBRL facts point-in-time: usable from month(filed) + lag. Within a CIK, a later-usable
    fact with an older period `end` (e.g. an amendment of an old period) never overrides a newer one."""
    f = facts.copy()
    f["usable"] = f["filed"].dt.to_period("M") + lag_months
    f = f.sort_values(["cik", "usable", "end"])
    f["max_end"] = f.groupby("cik")["end"].cummax()
    f = f[f["end"] >= f["max_end"]]
    return f[["cik", "usable", "end"] + value_cols]


def shares_table(sh: pd.DataFrame) -> pd.DataFrame:
    """One share count per (cik, end, filed): cover-page (dei) values summed across share classes;
    us-gaap balance-sheet value only when no dei value exists for that filing."""
    sh = sh.drop_duplicates(["cik", "end", "filed", "src", "shares"])
    agg = sh.groupby(["cik", "end", "filed", "src"], as_index=False)["shares"].sum()
    agg = agg.sort_values("src").drop_duplicates(["cik", "end", "filed"])
    return agg.drop(columns="src")


def annual_fundamentals(fu: pd.DataFrame) -> pd.DataFrame:
    """Annual (10-K, FY) values per (cik, fiscal period end, filed): be, at, rev, cogs, oi, plus
    at_prev (previous fiscal year's assets, 10-14 months earlier) -> asset growth and gross profit."""
    f = fu[fu["form"].isin(["10-K", "10-K/A"]) & (fu["fp"] == "FY")].copy()
    flow = f["item"].isin(["rev", "cogs", "oi"])
    dur = (f["end"] - f["start"]).dt.days
    f = f[~flow | dur.between(330, 400)]
    # original filing for each period, highest-priority concept
    f = f.sort_values(["cik", "item", "end", "pri", "filed"]).drop_duplicates(["cik", "item", "end"])
    w = f.pivot_table(index=["cik", "end"], columns="item", values="val", aggfunc="first")
    filed = f.groupby(["cik", "end"])["filed"].max()
    w = w.join(filed).reset_index().sort_values(["cik", "end"])
    for c in ["be", "at", "rev", "cogs", "oi"]:
        if c not in w:
            w[c] = np.nan
    prev = w.groupby("cik")[["end", "at"]].shift(1)
    gap = (w["end"] - prev["end"]).dt.days
    w["at_prev"] = prev["at"].where(gap.between(300, 430))
    return w
