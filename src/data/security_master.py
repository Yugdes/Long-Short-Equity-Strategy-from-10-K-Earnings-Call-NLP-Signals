"""Security master: risk CIK -> primary current ticker, current exchange, current SIC, FF48/FF12."""
import pandas as pd

from src.data.factors import sic_to_industry
from src.data.risk import load_risk_scores
from src.data.sec import parse_submissions, tickers_exchange
from src.utils.config import path


def build_security_master() -> pd.DataFrame:
    out = path("interim") / "security_master.parquet"
    if out.exists():
        return pd.read_parquet(out)
    ciks = set(load_risk_scores()["cik"])
    t = tickers_exchange()
    t = t[t["cik"].isin(ciks)].sort_values("ticker_rank").drop_duplicates("cik")
    sub = parse_submissions(ciks)
    sm = t[["cik", "ticker", "exchange", "name"]].merge(sub[["cik", "sic", "sic_desc", "entity_type"]], on="cik", how="left")
    sm["ff48"] = sic_to_industry(sm["sic"], "sic48")
    sm["ff12"] = sic_to_industry(sm["sic"], "sic12")
    sm["fin_util"] = sm["sic"].between(6000, 6999) | sm["sic"].between(4900, 4949)
    assert not sm["ticker"].duplicated().any() and not sm["cik"].duplicated().any()
    sm.to_parquet(out, index=False)
    return sm
