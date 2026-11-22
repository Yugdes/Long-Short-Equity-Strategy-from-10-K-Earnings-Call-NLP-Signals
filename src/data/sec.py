"""SEC data: ticker/exchange file, bulk submissions (SIC) and bulk XBRL companyfacts.

Every downloaded file is cached in data/raw/sec/ and never re-downloaded unless missing.
Bulk zips are streamed member-by-member; only the CIKs we need are parsed.
"""
import json
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from src.utils.config import path, sec_headers

URLS = {
    "tickers": "https://www.sec.gov/files/company_tickers_exchange.json",
    "submissions": "https://www.sec.gov/Archives/edgar/daily-index/bulkdata/submissions.zip",
    "companyfacts": "https://www.sec.gov/Archives/edgar/daily-index/xbrl/companyfacts.zip",
}
# XBRL concepts. Shares: cover page first, balance sheet fallback (D-012).
SHARE_CONCEPTS = [("dei", "EntityCommonStockSharesOutstanding"), ("us-gaap", "CommonStockSharesOutstanding")]
FUND_CONCEPTS = {
    "be": ["StockholdersEquity", "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"],
    "at": ["Assets"],
    "rev": ["Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax", "SalesRevenueNet"],
    "cogs": ["CostOfRevenue", "CostOfGoodsAndServicesSold", "CostOfGoodsSold"],
    "oi": ["OperatingIncomeLoss"],
}


def sec_dir() -> Path:
    d = path("raw") / "sec"
    d.mkdir(parents=True, exist_ok=True)
    return d


def download(name: str) -> Path:
    url = URLS[name]
    dest = sec_dir() / url.rsplit("/", 1)[1]
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    tmp = dest.with_suffix(dest.suffix + ".part")
    for attempt in range(5):
        try:
            with requests.get(url, headers=sec_headers(), stream=True, timeout=120) as r:
                r.raise_for_status()
                with open(tmp, "wb") as f:
                    for chunk in r.iter_content(chunk_size=1 << 20):
                        f.write(chunk)
            tmp.replace(dest)
            return dest
        except requests.RequestException as e:  # retry with backoff; SEC throttles bursts
            print(f"download {name} failed ({e}); retry {attempt + 1}")
            time.sleep(10 * (attempt + 1))
    raise RuntimeError(f"could not download {url}")


def tickers_exchange() -> pd.DataFrame:
    """Current SEC ticker file -> [cik, name, ticker, exchange, ticker_rank] (rank 0 = listed first)."""
    with open(download("tickers"), encoding="utf-8") as f:
        js = json.load(f)
    df = pd.DataFrame(js["data"], columns=js["fields"]).rename(columns={"cik": "cik"})
    df["cik"] = df["cik"].astype("int64")
    df["ticker_rank"] = df.groupby("cik").cumcount()
    return df


def _zip_members(zpath: Path, ciks) -> dict:
    with zipfile.ZipFile(zpath) as z:
        names = {n for n in z.namelist()}
    want = {}
    for c in ciks:
        n = f"CIK{int(c):010d}.json"
        if n in names:
            want[int(c)] = n
    return want


def parse_submissions(ciks) -> pd.DataFrame:
    """[cik, sic, sic_desc, name, exchanges, tickers] from bulk submissions.zip (current values)."""
    out_f = path("interim") / "sec_submissions.parquet"
    if out_f.exists():
        return pd.read_parquet(out_f)
    zpath = download("submissions")
    want = _zip_members(zpath, ciks)
    rows = []
    with zipfile.ZipFile(zpath) as z:
        for cik, n in want.items():
            js = json.loads(z.read(n))
            sic = js.get("sic") or None
            rows.append({"cik": cik, "sic": float(sic) if sic else np.nan, "sic_desc": js.get("sicDescription"),
                         "name": js.get("name"), "exchanges": ",".join(e or "" for e in js.get("exchanges", [])),
                         "tickers": ",".join(js.get("tickers", [])), "entity_type": js.get("entityType")})
    df = pd.DataFrame(rows)
    df.to_parquet(out_f, index=False)
    return df


def _facts(js: dict, taxonomy: str, concept: str, unit: str):
    try:
        return js["facts"][taxonomy][concept]["units"][unit]
    except KeyError:
        return []


def parse_companyfacts(ciks) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Point-in-time XBRL facts.

    shares: [cik, end, filed, form, shares, src]  (instant values)
    fund:   [cik, item, end, start, filed, form, fp, val]  (10-K / 10-Q facts for FUND_CONCEPTS)
    """
    sh_f, fu_f = path("interim") / "xbrl_shares.parquet", path("interim") / "xbrl_fund.parquet"
    if sh_f.exists() and fu_f.exists():
        return pd.read_parquet(sh_f), pd.read_parquet(fu_f)
    zpath = download("companyfacts")
    want = _zip_members(zpath, ciks)
    sh_rows, fu_rows = [], []
    with zipfile.ZipFile(zpath) as z:
        for i, (cik, n) in enumerate(want.items()):
            js = json.loads(z.read(n))
            for k, (tax, con) in enumerate(SHARE_CONCEPTS):
                for f in _facts(js, tax, con, "shares"):
                    sh_rows.append((cik, f.get("end"), f.get("filed"), f.get("form"), f.get("val"), k))
            for item, concepts in FUND_CONCEPTS.items():
                for k, con in enumerate(concepts):
                    for f in _facts(js, "us-gaap", con, "USD"):
                        if f.get("form") in ("10-K", "10-Q", "10-K/A", "10-Q/A"):
                            fu_rows.append((cik, item, k, f.get("start"), f.get("end"), f.get("filed"),
                                            f.get("form"), f.get("fp"), f.get("val")))
            if i % 2000 == 0:
                print(f"companyfacts: {i}/{len(want)}")
    sh = pd.DataFrame(sh_rows, columns=["cik", "end", "filed", "form", "shares", "src"])
    fu = pd.DataFrame(fu_rows, columns=["cik", "item", "pri", "start", "end", "filed", "form", "fp", "val"])
    for d in (sh, fu):
        for c in ("end", "filed", "start"):
            if c in d:
                d[c] = pd.to_datetime(d[c], errors="coerce")
    sh = sh.dropna(subset=["end", "filed", "shares"])
    sh = sh[sh["shares"] > 0]
    fu = fu.dropna(subset=["end", "filed", "val"])
    sh.to_parquet(sh_f, index=False)
    fu.to_parquet(fu_f, index=False)
    return sh, fu


if __name__ == "__main__":
    for k in ("tickers", "submissions", "companyfacts"):
        print(k, download(k))
