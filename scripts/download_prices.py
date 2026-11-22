"""Download Yahoo daily bars for the primary ticker of every risk-dataset CIK with a current SEC ticker."""
import pandas as pd
from src.data.sec import tickers_exchange
from src.data.prices import download_daily, BENCH
from src.data.risk import load_risk_scores

if __name__ == "__main__":
    t = tickers_exchange()
    ciks = set(load_risk_scores()["cik"])
    first = t[t["cik"].isin(ciks)].sort_values("ticker_rank").drop_duplicates("cik")
    tickers = first["ticker"].dropna().tolist() + BENCH
    print(len(tickers), "tickers")
    download_daily(tickers)
