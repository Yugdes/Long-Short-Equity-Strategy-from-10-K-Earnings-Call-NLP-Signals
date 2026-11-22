"""Yahoo Finance prices (free path, D-003).

download_daily(): raw daily bars per batch of 50 tickers -> data/raw/yahoo/batch_XXXX.parquet
                  (resumable; a finished batch is never re-downloaded, which freezes the data).
monthly_from_daily(): one row per (ticker, month) with total return, split-adjusted close,
                  dollar volume, Amihud illiquidity, daily-return volatility and split factors.
"""
from pathlib import Path
import time

import numpy as np
import pandas as pd

from src.utils.config import path

START, END = "2004-12-01", "2025-12-31"
BENCH = ["SPY"]


def yahoo_dir() -> Path:
    d = path("raw") / "yahoo"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _to_long(df: pd.DataFrame, batch: list) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        long = df.stack(level=1, future_stack=True).rename_axis(["date", "ticker"]).reset_index()
    else:
        long = df.reset_index().assign(ticker=batch[0])
    long.columns = [str(c).lower().replace(" ", "_") for c in long.columns]
    long = long.rename(columns={"adj_close": "adj", "stock_splits": "split", "dividends": "div"})
    keep = ["date", "ticker", "adj", "close", "volume", "div", "split"]
    long = long[[c for c in keep if c in long.columns]].dropna(subset=["adj", "close"])
    for c in ("adj", "close", "volume", "div", "split"):
        if c in long:
            long[c] = long[c].astype("float64")
    return long


def download_daily(tickers: list, batch_size: int = 50) -> None:
    import yfinance as yf
    tickers = sorted(set(tickers))
    pd.Series(tickers).to_csv(yahoo_dir() / "tickers_requested.csv", index=False, header=["ticker"])
    for b, i in enumerate(range(0, len(tickers), batch_size)):
        out = yahoo_dir() / f"batch_{b:04d}.parquet"
        if out.exists():
            continue
        batch = tickers[i:i + batch_size]
        for attempt in range(4):
            try:
                df = yf.download(batch, start=START, end=END, auto_adjust=False, actions=True,
                                 threads=True, progress=False)
                break
            except Exception as e:  # network/rate-limit errors: back off and retry
                print(f"batch {b} error {e}; retry {attempt + 1}")
                time.sleep(30 * (attempt + 1))
        else:
            continue
        long = _to_long(df, batch) if not df.empty else pd.DataFrame()
        long.to_parquet(out, index=False)
        print(f"batch {b}: {len(batch)} tickers, {long['ticker'].nunique() if len(long) else 0} with data")
        time.sleep(1.0)


def monthly_from_daily(daily: pd.DataFrame) -> pd.DataFrame:
    """Aggregate one batch of daily bars to ticker-months.

    ret       : total return from adjusted closes (dividends and splits included)
    prc       : month-end close, split-adjusted to today's share basis (Yahoo 'Close')
    dvol      : sum of close*volume (split effects cancel, so ~ raw dollar volume)
    amihud    : mean(|r_d| / dollar volume_d) * 1e6
    vol_d     : std of daily total returns within the month
    """
    d = daily.sort_values(["ticker", "date"]).copy()
    d["date"] = pd.to_datetime(d["date"])
    d["r"] = d.groupby("ticker")["adj"].pct_change()
    d["dv"] = d["close"] * d["volume"]
    d["month"] = d["date"].dt.to_period("M")
    d["illiq"] = np.where(d["dv"] > 0, d["r"].abs() / d["dv"] * 1e6, np.nan)
    g = d.groupby(["ticker", "month"])
    m = g.agg(adj=("adj", "last"), prc=("close", "last"), dvol=("dv", "sum"), amihud=("illiq", "mean"),
              vol_d=("r", "std"), ndays=("r", "count")).reset_index()
    m["ret"] = m.groupby("ticker")["adj"].pct_change()
    # a month gap (missing month) invalidates the return
    gap = m.groupby("ticker")["month"].diff().apply(lambda x: x.n if pd.notna(x) else np.nan)
    m.loc[gap != 1, "ret"] = np.nan
    return m.drop(columns="adj")


def clean_returns(m: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """D-020: blank out returns that are data errors, not market moves.
    (a) ret < -100% (impossible); (b) ret > +1000%; (c) spike-and-reversal: ret_t > +300% followed by
    ret_t+1 < -75% (classic bad tick) -> both months. Prices are kept; only `ret` is set to NaN."""
    m = m.sort_values(["ticker", "month"]).copy()
    nxt = m.groupby("ticker")["ret"].shift(-1)
    spike = (m["ret"] > 3) & (nxt < -0.75)
    rev = spike.groupby(m["ticker"]).shift(1, fill_value=False)
    flags = {"lt_minus100": m["ret"] < -1, "gt_1000": m["ret"] > 10, "spike_reversal": spike | rev}
    bad = flags["lt_minus100"] | flags["gt_1000"] | flags["spike_reversal"]
    stats = {k: int(v.sum()) for k, v in flags.items()} | {"total_blanked": int(bad.sum()), "rows": len(m)}
    m.loc[bad, "ret"] = np.nan
    return m, stats


def build_monthly() -> tuple[pd.DataFrame, pd.DataFrame]:
    """-> (prices_monthly, split events [ticker, date, ratio])."""
    out, out_s = path("interim") / "prices_monthly.parquet", path("interim") / "splits.parquet"
    if out.exists() and out_s.exists():
        return pd.read_parquet(out), pd.read_parquet(out_s)
    parts, splits = [], []
    for f in sorted(yahoo_dir().glob("batch_*.parquet")):
        d = pd.read_parquet(f)
        if d.empty:
            continue
        parts.append(monthly_from_daily(d))
        s = d[(d["split"] > 0) & (d["split"] != 1)][["ticker", "date", "split"]]
        splits.append(s.rename(columns={"split": "ratio"}))
    m, stats = clean_returns(pd.concat(parts, ignore_index=True))
    pd.Series(stats).to_csv(path("results") / "tables" / "return_cleaning.csv", header=["n"])
    s = pd.concat(splits, ignore_index=True)
    m.to_parquet(out, index=False)
    s.to_parquet(out_s, index=False)
    return m, s


def split_adjust_shares(shares: pd.Series, asof: pd.Series, tickers: pd.Series,
                        splits: pd.DataFrame) -> pd.Series:
    """Put a share count reported as of `asof` on Yahoo's split-adjusted (today's) basis.

    Yahoo 'Close' = raw price / (product of splits with ex-date > t). A count of N raw shares
    as of date d therefore equals N * prod(splits with ex-date > d) shares on today's basis,
    so mcap(t) = Close(t) * shares_adj for any t (issuance aside).
    """
    out = shares.astype("float64").copy()
    sp = splits.sort_values("date")
    asof = pd.to_datetime(asof)
    for tk, grp in sp.groupby("ticker"):
        idx = tickers.index[tickers == tk]
        if len(idx) == 0:
            continue
        dates = grp["date"].values.astype("datetime64[ns]")
        # product of ratios strictly after each asof date
        rev_cum = np.cumprod(grp["ratio"].values[::-1])[::-1]           # prod of ratios[i:]
        pos = np.searchsorted(dates, asof.loc[idx].values.astype("datetime64[ns]"), side="right")
        factor = np.append(rev_cum, 1.0)[pos]
        out.loc[idx] = out.loc[idx].values * factor
    return out
