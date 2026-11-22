import numpy as np
import pandas as pd
from scipy import stats

from src.models.walkforward import splits
from src.evaluation.predictive import ic_series, oos_r2, hit_rate


def test_train_label_month_before_test_start():
    months = pd.period_range("2010-01", "2021-12", freq="M")
    for f in splits(months, first_test_year=2014, last_test_year=2021, embargo=1, horizon=1):
        assert f["train_end"] < f["test_start"] - 1                 # >= 1 embargo month between them
        assert f["val_start"] > f["fit_end"]                        # validation strictly after fit
        assert f["val_end"] == f["train_end"]
        assert f["test_end"] == pd.Period(f"{f['test_start'].year}-12", "M")


def test_horizon_extends_embargo():
    months = pd.period_range("2010-01", "2015-12", freq="M")
    f = splits(months, 2014, 2014, embargo=1, horizon=3)[0]
    assert f["train_end"] == pd.Period("2013-09", "M")              # 3m label of Sep ends Nov; Dec embargo


def test_ic_matches_scipy():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"month": np.repeat(pd.period_range("2015-01", periods=3, freq="M"), 50),
                       "pred": rng.normal(size=150), "y": rng.normal(size=150)})
    ic = ic_series(df, "pred", "y")
    m0 = df[df["month"] == df["month"].iloc[0]]
    assert np.isclose(ic.iloc[0], stats.spearmanr(m0["pred"], m0["y"]).statistic)


def test_oos_r2_and_hit_rate():
    y = pd.Series([0.1, -0.2, 0.3])
    assert np.isclose(oos_r2(y, y), 1.0) and np.isclose(oos_r2(y, pd.Series([0.0, 0.0, 0.0])), 0.0)
    assert hit_rate(pd.Series([1, -1, 1.0]), pd.Series([2, 1, 3.0])) == 2 / 3


def test_preprocess_fit_on_train_only():
    from src.models.learners import ENet
    rng = np.random.default_rng(0)
    Xf, Xv = rng.normal(0, 1, (300, 3)), rng.normal(5, 1, (100, 3))
    y = Xf[:, 0] * 0.1
    m = ENet(Xf, y, Xv, Xv[:, 0] * 0.1).refit(Xf, y)
    assert np.allclose(m.scaler.mean_, Xf.mean(axis=0))      # never sees validation/test rows
