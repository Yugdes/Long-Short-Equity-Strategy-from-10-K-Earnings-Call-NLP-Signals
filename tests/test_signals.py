import numpy as np
import pandas as pd

from src.features.panel import risk_filing_features
from src.features.signals import make_features, CONTROL_VARS, RISK_VARS, MKT_VARS


def _toy_panel():
    rng = np.random.default_rng(1)
    months = pd.PeriodIndex(["2015-01"] * 4 + ["2015-02"] * 4, freq="M")
    df = pd.DataFrame({"cik": [1, 2, 3, 4] * 2, "month": months, "ff48": [1, 1, 2, 2] * 2,
                       "elig": [True, True, True, False, True, True, True, True],
                       "exret": rng.normal(size=8), "exret_3m": rng.normal(size=8)})
    for c in CONTROL_VARS + RISK_VARS + MKT_VARS:
        df[c] = rng.normal(size=8)
    df.loc[3, "operations_prob"] = 99.0            # extreme value in an ineligible row
    return df


def test_rank_uses_only_same_month_eligible():
    f = make_features(_toy_panel())
    jan = f[f["month"] == pd.Period("2015-01", "M")]
    assert len(jan) == 3                                   # ineligible row excluded
    assert np.allclose(sorted(jan["r_operations_prob"] + 0.5), [1 / 3, 2 / 3, 1.0], atol=1e-6)
    assert f["r_operations_prob"].between(-0.5, 0.5).all()


def test_missing_feature_imputed_with_month_median():
    p = _toy_panel()
    p.loc[0, "size"] = np.nan
    f = make_features(p)
    assert f.loc[0, "r_size"] == 0.0


def test_delta_and_surprise_use_only_past_filings():
    r = pd.DataFrame({"cik": [1, 1, 1, 1], "filing_date": pd.to_datetime(["2010-03-01", "2011-03-01", "2013-03-01", "2014-03-01"]),
                      "words": 1000.0, "operations_cnt": 10.0})
    for f in ["accounting", "finance", "international", "legal", "management", "marketing", "operations", "technology"]:
        r[f"{f}_prob"] = [0.1, 0.2, 0.4, 0.3]
    out = risk_filing_features(r)
    d = out["d_operations_prob"].tolist()
    assert np.isnan(d[0]) and np.isclose(d[1], 0.1) and np.isnan(d[2]) and np.isclose(d[3], -0.1)  # 2-year gap -> NaN
    s = out["s_operations_prob"].tolist()
    assert np.isnan(s[0]) and np.isnan(s[1]) and np.isclose(s[2], 0.4 - 0.15) and np.isclose(s[3], 0.3 - (0.7 / 3))
