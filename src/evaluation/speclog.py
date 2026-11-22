"""Multiple-testing log: every strategy/model spec evaluated on the dev window is appended here.
The deflated Sharpe ratio (Phase 6) uses the monthly Sharpe of every logged spec."""
import pandas as pd

from src.utils.config import path

FILE = "spec_log.csv"


def log_spec(stage: str, spec: str, ls: pd.Series, **extra) -> dict:
    r = ls.dropna()
    row = {"stage": stage, "spec": spec, "n_months": len(r), "mean_m": r.mean(),
           "sr_m": r.mean() / r.std() if r.std() > 0 else float("nan"), **extra}
    f = path("results") / FILE
    df = pd.read_csv(f) if f.exists() else pd.DataFrame()
    df = df[~((df.get("stage") == stage) & (df.get("spec") == spec))] if len(df) else df
    pd.concat([df, pd.DataFrame([row])], ignore_index=True).to_csv(f, index=False)
    return row


def load_specs() -> pd.DataFrame:
    f = path("results") / FILE
    return pd.read_csv(f) if f.exists() else pd.DataFrame()
