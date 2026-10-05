"""
Price & return data adapter — supports Path B (yfinance, free).
Downloads adjusted prices, computes monthly returns, market cap, and volume.

Path A (WRDS/CRSP) adapter is a stub — implement if institutional access is available.
"""
import pandas as pd
import numpy as np
import time
from pathlib import Path
from typing import List, Optional

from src.utils.helpers import (
    load_config, get_path, setup_logging, save_parquet, load_parquet, parquet_exists
)

logger = setup_logging("prices")


# =============================================================================
# Path B — yfinance (free)
# =============================================================================

def _download_yfinance_batch(
    tickers: List[str],
    start: str = "2004-01-01",
    end: str = None,
    batch_size: int = 50,
) -> pd.DataFrame:
    """
    Download daily adjusted prices from yfinance in batches.
    Returns: DataFrame with columns [date, ticker, adj_close, volume, close, high, low].
    """
    import yfinance as yf

    all_data = []
    n_batches = (len(tickers) + batch_size - 1) // batch_size

    for i in range(0, len(tickers), batch_size):
        batch = tickers[i:i + batch_size]
        batch_num = i // batch_size + 1
        logger.info(f"Downloading batch {batch_num}/{n_batches} ({len(batch)} tickers)...")

        try:
            df = yf.download(
                batch,
                start=start,
                end=end,
                auto_adjust=True,
                threads=True,
                progress=False,
            )

            if df.empty:
                logger.warning(f"Empty data for batch {batch_num}")
                continue

            # Handle single vs multi-ticker format
            if isinstance(df.columns, pd.MultiIndex):
                # Multi-ticker: columns are (Price, Ticker)
                for ticker in batch:
                    if ticker in df.columns.get_level_values(1):
                        ticker_data = df.xs(ticker, axis=1, level=1).copy()
                        ticker_data["ticker"] = ticker
                        ticker_data = ticker_data.reset_index()
                        ticker_data.columns = ticker_data.columns.str.lower()
                        all_data.append(ticker_data)
            else:
                # Single ticker
                df = df.copy()
                df["ticker"] = batch[0]
                df = df.reset_index()
                df.columns = df.columns.str.lower()
                all_data.append(df)

        except Exception as e:
            logger.warning(f"Batch {batch_num} failed: {e}")

        # Rate limiting
        time.sleep(1.0)

    if not all_data:
        return pd.DataFrame()

    result = pd.concat(all_data, ignore_index=True)
    return result


def compute_monthly_returns(daily_prices: pd.DataFrame) -> pd.DataFrame:
    """
    Compute monthly returns from daily adjusted prices.

    Returns DataFrame: ticker, month, ret, price_end, volume_avg, mktcap_proxy
    """
    df = daily_prices.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["month"] = df["date"].dt.to_period("M")

    # Monthly: last price, average volume
    monthly = df.groupby(["ticker", "month"]).agg(
        price_end=("close", "last"),
        volume_avg=("volume", "mean"),
        n_days=("close", "count"),
    ).reset_index()

    # Compute returns
    monthly = monthly.sort_values(["ticker", "month"])
    monthly["price_prev"] = monthly.groupby("ticker")["price_end"].shift(1)
    monthly["ret"] = monthly["price_end"] / monthly["price_prev"] - 1

    # Drop first month per ticker (no return)
    monthly = monthly.dropna(subset=["ret"])

    # Filter extreme returns (likely data errors)
    monthly = monthly[monthly["ret"].between(-0.95, 10.0)]

    monthly["month_start"] = monthly["month"].dt.to_timestamp()

    return monthly


def build_returns_table(
    tickers: List[str],
    start: str = "2004-01-01",
    end: str = None,
    force: bool = False,
) -> pd.DataFrame:
    """
    Full pipeline: download daily prices → compute monthly returns.
    Caches the raw daily data and the final monthly returns.
    """
    cfg = load_config()
    out_file = Path(get_path(cfg, "processed")) / "returns_monthly.parquet"
    raw_cache = Path(get_path(cfg, "interim")) / "daily_prices.parquet"

    if parquet_exists(out_file) and not force:
        logger.info(f"Loading cached returns from {out_file}")
        return load_parquet(out_file)

    # Download or load cached daily prices
    if parquet_exists(raw_cache) and not force:
        logger.info("Loading cached daily prices...")
        daily = load_parquet(raw_cache)
    else:
        logger.info(f"Downloading prices for {len(tickers)} tickers...")
        daily = _download_yfinance_batch(tickers, start=start, end=end)
        if daily.empty:
            raise RuntimeError("No price data downloaded")
        save_parquet(daily, raw_cache)
        logger.info(f"Cached {len(daily)} daily price rows")

    # Compute monthly returns
    monthly = compute_monthly_returns(daily)
    logger.info(f"Monthly returns: {len(monthly)} rows, "
                f"{monthly['ticker'].nunique()} tickers, "
                f"{monthly['month'].min()} – {monthly['month'].max()}")

    save_parquet(monthly, out_file)
    return monthly


# =============================================================================
# Path A — WRDS/CRSP (stub — requires institutional access)
# =============================================================================

def build_returns_wrds(force: bool = False) -> pd.DataFrame:
    """
    Download monthly stock returns from CRSP via WRDS.
    Requires: `pip install wrds` and WRDS account credentials.

    Returns standardized table matching Path B output.
    """
    try:
        import wrds
    except ImportError:
        raise ImportError("WRDS Python library not installed. Run: pip install wrds")

    cfg = load_config()
    out_file = Path(get_path(cfg, "processed")) / "returns_monthly.parquet"

    if parquet_exists(out_file) and not force:
        return load_parquet(out_file)

    db = wrds.Connection()

    query = """
    SELECT a.permno, a.date, a.ret, a.retx, a.prc, a.shrout, a.vol,
           a.shrcd, a.exchcd,
           b.siccd AS sic,
           ABS(a.prc) * a.shrout / 1000 AS mktcap
    FROM crsp.msf AS a
    LEFT JOIN crsp.msenames AS b
        ON a.permno = b.permno
        AND a.date BETWEEN b.namedt AND b.nameendt
    WHERE a.date >= '2004-01-01'
      AND a.shrcd IN (10, 11)
      AND a.exchcd IN (1, 2, 3)
    ORDER BY a.permno, a.date
    """

    df = db.raw_sql(query)
    db.close()

    # Handle delisting returns
    delist_query = """
    SELECT permno, dlstdt AS date, dlret, dlstcd
    FROM crsp.msedelist
    WHERE dlstdt >= '2004-01-01'
    """
    # Merge delisting returns... (simplified)

    df["month"] = pd.to_datetime(df["date"]).dt.to_period("M")
    save_parquet(df, out_file)
    return df


if __name__ == "__main__":
    # Test with a small set of tickers
    test_tickers = ["AAPL", "MSFT", "XOM", "JPM", "WMT"]
    df = build_returns_table(test_tickers, start="2020-01-01", force=True)
    print(df.head(20))
    print(f"\nShape: {df.shape}")
