import numpy as np
import pandas as pd

from src.evaluation.event_study import event_windows
from src.evaluation.survivorship import classify_fate

T = pd.Timestamp


def test_classify_fate_priority():
    h = pd.DataFrame({
        "last_filing": [T("2026-03-01"), T("2020-06-01"), T("2020-06-01"), T("2020-06-01"), T("2020-06-01")],
        "bk_dates": [[T("2019-01-01")], [T("2019-01-01")], [], [T("2010-01-01")], []],
        "last_merger_form": [pd.NaT, T("2020-01-01"), T("2019-01-01"), pd.NaT, T("2016-01-01")],
    })
    # still filing -> survivor even after a bankruptcy; bankruptcy beats merger; old bankruptcy / old merger form -> other
    assert classify_fate(h).tolist() == ["survivor", "bankrupt", "merged", "exit_other", "exit_other"]


def test_event_day0_is_next_trading_day_and_windows():
    dates = pd.bdate_range("2019-01-01", periods=400)
    ar = pd.DataFrame({"ticker": "X", "date": dates, "ar": 0.0})
    i_sat = 300                                   # filing on the Saturday after trading day 300's Friday
    fri = dates[i_sat]
    while fri.weekday() != 4:
        i_sat += 1
        fri = dates[i_sat]
    ar.loc[i_sat + 1, "ar"] = 0.02                # Monday = day 0
    ar.loc[i_sat + 2, "ar"] = 0.01                # day +1
    ar.loc[i_sat + 3, "ar"] = 0.05                # day +2 belongs to the drift window, not the reaction
    ev = pd.DataFrame({"eid": [0], "ticker": ["X"], "filing_date": [fri + pd.Timedelta(days=1)]})
    out = event_windows(ar, ev).iloc[0]
    assert np.isclose(out["car01"], 0.03) and np.isclose(out["car2_60"], 0.05)


def test_event_needs_pre_window():
    dates = pd.bdate_range("2019-01-01", periods=300)
    ar = pd.DataFrame({"ticker": "X", "date": dates, "ar": 0.0})
    ev = pd.DataFrame({"eid": [0], "ticker": ["X"], "filing_date": [dates[100]]})   # < 250 days of history
    assert event_windows(ar, ev).empty
