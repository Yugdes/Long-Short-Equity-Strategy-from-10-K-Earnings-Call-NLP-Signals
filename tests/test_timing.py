import pandas as pd
import pytest
from src.features.timing import risk_first_month, marketing_first_month, months_since


def _fm(dates):
    return [str(p) for p in risk_first_month(pd.Series(pd.to_datetime(dates)))]


def test_filing_mid_month_used_next_month():
    assert _fm(["2019-02-12"]) == ["2019-03"]      # avail 02-14 (Feb) -> March


def test_filing_near_month_end_skips_month_of_availability():
    assert _fm(["2019-02-27"]) == ["2019-04"]      # avail 03-01 (Mar) -> April


def test_month_end_filing_skips_next_partial():
    assert _fm(["2019-02-28"]) == ["2019-04"]      # Thu -> avail Mon 03-04 -> April


def test_friday_filing_lag_crosses_weekend():
    assert _fm(["2021-01-29"]) == ["2021-03"]      # Fri -> avail Tue 02-02 -> March


def test_marketing_first_month():
    out = marketing_first_month(pd.Series([2020, 2020, 2020]), pd.Series([1, 2, 4]), buffer_months=1)
    assert [str(p) for p in out] == ["2020-05", "2020-08", "2021-02"]


def test_marketing_zero_buffer():
    out = marketing_first_month(pd.Series([2020]), pd.Series([1]), buffer_months=0)
    assert str(out.iloc[0]) == "2020-04"


def test_months_since():
    a = pd.Series(pd.PeriodIndex(["2020-03", "2021-01"], freq="M"))
    b = pd.Series(pd.PeriodIndex(["2020-01", "2020-01"], freq="M"))
    assert months_since(a, b).tolist() == [2, 12]
