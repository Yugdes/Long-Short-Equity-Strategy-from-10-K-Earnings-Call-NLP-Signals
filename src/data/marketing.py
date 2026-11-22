"""Damavandi, Mai & Astvansh (2025) marketing-emphasis scores -> CIK-quarter table.

- `quarter` is the calendar quarter of the earnings call (DATA_AUDIT §3); availability per D-006.
- GVKEY -> CIK through the WRDS SEC link table, using links valid at the quarter end.
  Cells listing several GVKEYs (v2 extension) are exploded first. A quarter mapping to more
  than one risk CIK is ambiguous and dropped (never force-matched). If several GVKEYs map to
  the same CIK-quarter, the row with the longest transcript text is kept.
- Higher-order constructs = mean of the raw (cosine-similarity) components, 8/8/3 per the
  authors' README (the earlier pipeline used 3/5/3: DECISIONS A5).
"""
import numpy as np
import pandas as pd

from src.features.timing import marketing_first_month
from src.utils.config import path, load_config

MO = ["customer_orientation", "competitor_orientation", "interfunctional_coordination", "longterm_focus",
      "profit_focus", "intelligence_generation", "intelligence_dissemination", "responsiveness"]
MC = ["info_management", "planning", "implementation", "pricing_capabilities", "product_development",
      "channel_management", "communication", "selling_capabilities"]
ME = ["marketing_ecosystem", "end_user", "marketing_agility"]
COMPONENTS = MO + MC + ME
ZCA = {c: f"zca_{'interfunc_coordination' if c == 'interfunctional_coordination' else c}" for c in COMPONENTS}
CONSTRUCTS = {"mkt_orient": MO, "mkt_capab": MC, "mkt_excel": ME}


def _links() -> pd.DataFrame:
    lk = pd.read_csv(path("raw") / "cik_gvkey.csv", usecols=["cik", "gvkey", "link_start_date", "link_end_date"])
    lk = lk.dropna(subset=["gvkey"])
    lk["gvkey"] = lk["gvkey"].astype("int64")
    lk["start"] = pd.to_datetime(lk["link_start_date"].replace("B", "19000101"), format="%Y%m%d")
    lk["end"] = pd.to_datetime(lk["link_end_date"].replace("E", "21001231"), format="%Y%m%d")
    return lk[["cik", "gvkey", "start", "end"]]


def link_to_cik(df: pd.DataFrame, risk_ciks: set) -> tuple[pd.DataFrame, dict]:
    """df has gvkey_list (list[int]), year, quarter. Returns rows with `cik` plus match stats."""
    x = df.explode("gvkey_list").rename(columns={"gvkey_list": "gvkey"})
    x["gvkey"] = x["gvkey"].astype("int64")
    x["q_end"] = pd.PeriodIndex.from_fields(year=x["year"], month=x["quarter"] * 3, freq="M").to_timestamp(how="end").normalize()
    lk = _links()
    lk = lk[lk["cik"].isin(risk_ciks)]
    m = x.reset_index().merge(lk, on="gvkey", how="inner")
    m = m[(m["q_end"] >= m["start"]) & (m["q_end"] <= m["end"])]
    m = m.drop_duplicates(["index", "cik"])
    n_cik = m.groupby("index")["cik"].transform("nunique")
    stats = {"rows_in": len(df), "rows_linked": int(m["index"].nunique()),
             "rows_ambiguous_dropped": int(m.loc[n_cik > 1, "index"].nunique())}
    m = m[n_cik == 1]
    m = m.sort_values("document_length", ascending=False).drop_duplicates(["cik", "year", "quarter"])
    stats["cik_quarters"] = len(m)
    return m.drop(columns=["index", "start", "end", "gvkey"]), stats


def load_marketing(risk_ciks: set) -> pd.DataFrame:
    out = path("interim") / "marketing.parquet"
    if out.exists():
        return pd.read_parquet(out)
    ex = pd.read_csv(path("raw") / "marketing" / "MarketingEmphasisScores_Executive.csv", dtype={"GVKEY": str})
    an = pd.read_csv(path("raw") / "marketing" / "MarketingEmphasisScores_Analyst.csv", dtype={"GVKEY": str})
    an = an.rename(columns={"customer_orientation": "an_cust", "zca_customer_orientation": "an_cust_zca"})
    ex = ex.merge(an[["GVKEY", "year", "quarter", "an_cust", "an_cust_zca"]], on=["GVKEY", "year", "quarter"], how="left")
    ex["gvkey_list"] = ex["GVKEY"].str.replace("GV_", "", regex=False).str.split(",").apply(lambda l: [int(v) for v in l])
    m, stats = link_to_cik(ex.drop(columns="GVKEY"), risk_ciks)
    for name, comps in CONSTRUCTS.items():
        m[name] = m[comps].mean(axis=1)
    m["first_month"] = marketing_first_month(m["year"], m["quarter"], load_config()["timing"]["mkt_buffer_months"])
    keep = ["cik", "year", "quarter", "first_month", "document_length", *CONSTRUCTS, *COMPONENTS,
            *ZCA.values(), "an_cust", "an_cust_zca"]
    m = m[keep].reset_index(drop=True)
    pd.Series(stats).to_csv(path("results") / "tables" / "marketing_link_stats.csv", header=["value"])
    m.to_parquet(out, index=False)
    return m
