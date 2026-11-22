import pandas as pd
import pytest
from src.utils import lock


def test_dev_only_drops_locked_months():
    df = pd.DataFrame({"month": pd.PeriodIndex(["2021-12", "2022-01", "2023-05"], freq="M"), "x": [1, 2, 3]})
    assert lock.dev_only(df)["x"].tolist() == [1]


def test_dev_only_on_index():
    s = pd.DataFrame({"x": [1, 2]}, index=pd.PeriodIndex(["2021-11", "2022-02"], freq="M"))
    assert lock.dev_only(s, month_col=None)["x"].tolist() == [1]


def test_guard_raises_on_locked_months():
    if lock.is_unlocked():
        pytest.skip("test window already unlocked")
    with pytest.raises(PermissionError):
        lock.assert_unlocked()
