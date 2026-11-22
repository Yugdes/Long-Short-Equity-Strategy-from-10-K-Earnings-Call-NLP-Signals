"""Phase 0 data audit: profile every raw input and write summary tables.

Run:  python scripts/profile_raw_data.py
Out:  results/tables/data_audit/*.csv  (numbers quoted in docs/DATA_AUDIT.md)
"""
from pathlib import Path
import json
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
INTERIM = ROOT / "data" / "interim"
OUT = ROOT / "results" / "tables" / "data_audit"
OUT.mkdir(parents=True, exist_ok=True)
summary = {}


def missing_table(df, name):
    t = pd.DataFrame({"dtype": df.dtypes.astype(str), "n_missing": df.isna().sum(),
                      "pct_missing": df.isna().mean().round(4)})
    t.to_csv(OUT / f"{name}_columns.csv")
    return t


# ---------------- 1. Risk scores (Astvansh & Simpson) ----------------
risk_cache = INTERIM / "risk_raw.parquet"
if risk_cache.exists():
    risk = pd.read_parquet(risk_cache)
else:
    risk = pd.read_excel(RAW / "risk" / "dataset.xlsx", engine="openpyxl")
    risk.to_parquet(risk_cache, index=False)
missing_table(risk, "risk")
risk["Filing Date"] = pd.to_datetime(risk["Filing Date"].astype(str), format="%Y%m%d", errors="coerce")  # stored as int YYYYMMDD
risk["filing_year"] = risk["Filing Date"].dt.year
dup = risk.duplicated(["CIK", "FYEAR"]).sum()
lag_years = (risk["filing_year"] - risk["FYEAR"]).value_counts().sort_index()
by_year = risk.groupby("FYEAR").agg(n=("CIK", "size"), firms=("CIK", "nunique"),
                                    optional=("Item 1A Optional", "mean"),
                                    words_med=("Item 1A Word Count", "median"),
                                    words_lt1000=("Item 1A Word Count", lambda s: (s < 1000).mean()),
                                    ops_prob_mean=("Operations Probability", "mean"))
by_year.round(4).to_csv(OUT / "risk_by_fyear.csv")
prob_cols = [c for c in risk.columns if c.endswith("Probability")]
desc = risk[prob_cols + ["Item 1A Word Count", "Item 1A Optional"]].describe().T
desc.round(4).to_csv(OUT / "risk_describe.csv")
summary["risk"] = {
    "rows": len(risk), "firms": int(risk["CIK"].nunique()),
    "fyear_range": [int(risk["FYEAR"].min()), int(risk["FYEAR"].max())],
    "filing_date_range": [str(risk["Filing Date"].min().date()), str(risk["Filing Date"].max().date())],
    "filing_date_missing": int(risk["Filing Date"].isna().sum()),
    "dup_cik_fyear": int(dup),
    "filing_year_minus_fyear": {int(k): int(v) for k, v in lag_years.items()},
    "word_count_zero": int((risk["Item 1A Word Count"] == 0).sum()),
    "word_count_lt1000": int((risk["Item 1A Word Count"] < 1000).sum()),
    "optional_share": float(risk["Item 1A Optional"].mean()),
    "columns": list(risk.columns),
}

# ---------------- 2. Marketing scores (Damavandi, Mai & Astvansh) ----------------
mk = pd.read_csv(RAW / "marketing" / "MarketingEmphasisScores_Executive.csv", low_memory=False)
an = pd.read_csv(RAW / "marketing" / "MarketingEmphasisScores_Analyst.csv", low_memory=False)
missing_table(mk, "marketing_exec")
mk_by = mk.groupby("year").agg(n=("GVKEY", "size"), firms=("GVKEY", "nunique"),
                               doc_len_med=("document_length", "median"))
mk_by.to_csv(OUT / "marketing_by_year.csv")
summary["marketing_exec"] = {
    "rows": len(mk), "firms": int(mk["GVKEY"].nunique()),
    "year_range": [int(mk["year"].min()), int(mk["year"].max())],
    "dup_gvkey_yq": int(mk.duplicated(["GVKEY", "year", "quarter"]).sum()),
    "multi_gvkey_rows": int(mk["GVKEY"].astype(str).str.contains(",").sum()),
    "quarters": sorted(mk["quarter"].unique().tolist()),
    "doc_len_zero": int((mk["document_length"] == 0).sum()),
    "n_columns": mk.shape[1],
}
summary["marketing_analyst"] = {"rows": len(an), "firms": int(an["GVKEY"].nunique()),
                                "dup_gvkey_yq": int(an.duplicated(["GVKEY", "year", "quarter"]).sum())}

# ---------------- 3. WRDS SEC CIK-GVKEY link table ----------------
link = pd.read_csv(RAW / "cik_gvkey.csv", low_memory=False)
missing_table(link, "cik_gvkey")
lk = link.dropna(subset=["gvkey"])
summary["cik_gvkey"] = {
    "rows": len(link), "rows_with_gvkey": len(lk),
    "unique_cik_with_gvkey": int(lk["cik"].nunique()), "unique_gvkey": int(lk["gvkey"].nunique()),
    "gvkeys_with_multiple_ciks": int((lk.groupby("gvkey")["cik"].nunique() > 1).sum()),
    "ciks_with_multiple_gvkeys": int((lk.groupby("cik")["gvkey"].nunique() > 1).sum()),
    "source_counts": link["source"].value_counts().to_dict(),
}
risk_ciks = set(risk["CIK"].unique())
mk_gv = set(mk["GVKEY"].astype(str).str.replace("GV_", "").str.split(",").explode().astype(int).unique())
cik_to_gv = lk[lk["cik"].isin(risk_ciks)]
summary["overlap"] = {
    "risk_ciks_in_link": int(cik_to_gv["cik"].nunique()),
    "risk_ciks_linked_to_marketing_gvkey": int(cik_to_gv[cik_to_gv["gvkey"].astype(int).isin(mk_gv)]["cik"].nunique()),
    "marketing_gvkeys_linked_to_risk_cik": int(len(set(cik_to_gv["gvkey"].astype(int)) & mk_gv)),
}

# ---------------- 4. Free returns coverage (existing yfinance pull, current tickers) ----------------
tick = pd.read_parquet(INTERIM / "sec_tickers.parquet")
summary["sec_tickers"] = {"rows": len(tick), "risk_ciks_with_current_ticker": int(tick["cik"].isin(risk_ciks).sum())}
rets_f = ROOT / "data" / "processed" / "returns_monthly.parquet"
if rets_f.exists():
    rets = pd.read_parquet(rets_f, columns=["ticker", "month", "ret"])
    have = set(rets["ticker"].unique())
    tick["has_returns"] = tick["ticker"].isin(have)
    m = risk.merge(tick[["cik", "has_returns"]], left_on="CIK", right_on="cik", how="left")
    cov = m.groupby("FYEAR").agg(firm_years=("CIK", "size"),
                                 with_ticker=("cik", lambda s: s.notna().sum()),
                                 with_returns=("has_returns", lambda s: s.fillna(False).astype(bool).sum()))
    cov["pct_with_returns"] = (cov["with_returns"] / cov["firm_years"]).round(4)
    cov.to_csv(OUT / "returns_coverage_by_fyear.csv")
    summary["returns_existing"] = {"rows": len(rets), "tickers": int(rets["ticker"].nunique()),
                                   "month_range": [str(rets["month"].min()), str(rets["month"].max())],
                                   "firm_years_with_returns": int(cov["with_returns"].sum()),
                                   "pct_firm_years_with_returns": float(cov["with_returns"].sum() / len(risk))}

(OUT / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
print(json.dumps(summary, indent=2, default=str))
print(by_year.round(3).to_string())
print(desc.round(3).to_string())
print(mk_by.to_string())
if rets_f.exists():
    print(cov.to_string())
