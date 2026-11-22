"""Astvansh & Simpson (2026) risk-factor scores -> tidy firm-filing table.

Output columns: cik, filing_date, file_year, words, optional, {f}_prob, {f}_cnt (8 factors).
`file_year` is the dataset's FYEAR, which equals the filing calendar year (DATA_AUDIT §2.2).
Rows with an empty Item 1A (0 words) get NaN scores: absent text is missing, not zero risk.
"""
import pandas as pd

from src.utils.config import path

FACTORS = ["accounting", "finance", "international", "legal",
           "management", "marketing", "operations", "technology"]


def _raw() -> pd.DataFrame:
    cache = path("interim") / "risk_raw.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    df = pd.read_excel(path("raw") / "risk" / "dataset.xlsx", engine="openpyxl")
    df.to_parquet(cache, index=False)
    return df


def load_risk_scores() -> pd.DataFrame:
    out = path("interim") / "risk.parquet"
    if out.exists():
        return pd.read_parquet(out)
    raw = _raw()
    raw.columns = [" ".join(c.split()) for c in raw.columns]     # collapse stray double spaces
    df = pd.DataFrame({
        "cik": raw["CIK"].astype("int64"),
        "filing_date": pd.to_datetime(raw["Filing Date"].astype(str), format="%Y%m%d"),
        "file_year": raw["FYEAR"].astype("int64"),
        "words": raw["Item 1A Word Count"].astype("float64"),
        "optional": raw["Item 1A Optional"].astype("int8"),
    })
    for f in FACTORS:
        name = f.capitalize()
        df[f"{f}_prob"] = raw[f"{name} Probability"].astype("float64")
        df[f"{f}_cnt"] = raw[name].astype("float64")
    empty = df["words"] <= 0
    score_cols = [c for c in df.columns if c.endswith(("_prob", "_cnt"))]
    df.loc[empty, score_cols] = float("nan")
    df = df.sort_values(["cik", "filing_date"]).reset_index(drop=True)
    assert not df.duplicated(["cik", "filing_date"]).any()
    df.to_parquet(out, index=False)
    return df
