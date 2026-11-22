"""Diagnostic walk-forward runs (dev window): is the marketing block's gain the *language* or just
the earnings-call coverage flag? D-026. Logged in the spec log like every other spec."""
import pandas as pd

from src.data.factors import load_ff_factors
from src.features.signals import FEATURE_GROUPS as G
from src.models.learners import LEARNERS
from src.models.run_ml import evaluate
from src.models.walkforward import run_walkforward
from src.utils.config import load_config, path

MKT_NO_FLAG = [f for f in G["marketing"] if f != "has_mkt"]
DIAG_SETS = {"C+flag": G["controls"] + ["has_mkt"], "C+M_noflag": G["controls"] + MKT_NO_FLAG,
             "C+M": G["controls"] + G["marketing"]}


def main():
    cfg = load_config()
    feat = pd.read_parquet(path("processed") / "features.parquet")
    ff = load_ff_factors()
    rows = []
    for s, cols in DIAG_SETS.items():
        f = path("processed") / "preds" / f"{s}__lgbm__h1.parquet"
        if not f.exists():
            run_walkforward(feat, cols, "y_rank", LEARNERS["lgbm"], cfg["sample"]["first_test_year"],
                            cfg["sample"]["last_dev_year"]).to_parquet(f, index=False)
        rows.append({"set": s, "model": "lgbm", **evaluate(feat, pd.read_parquet(f), f"{s}|lgbm|h1", ff, "ml_diag")})
    out = pd.DataFrame(rows)
    out.to_csv(path("results") / "tables" / "ml_diagnostics_dev.csv", index=False)
    print(out.round(4).to_string())
    return out


if __name__ == "__main__":
    main()
