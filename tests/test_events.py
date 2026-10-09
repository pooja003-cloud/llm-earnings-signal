"""The look-ahead guards. If these fail, every backtest number is suspect."""
import numpy as np
import pandas as pd
import pytest

from earnsig.events import assign_t0, event_study, season_label

DAYS = pd.bdate_range("2024-01-01", "2024-03-29").drop(pd.Timestamp("2024-01-15"))  # MLK holiday


@pytest.mark.parametrize("published, expected", [
    ("2024-01-10 07:00", "2024-01-10"),  # before the open -> same day's close
    ("2024-01-10 12:30", "2024-01-10"),  # during the session -> same day's close
    ("2024-01-10 15:59", "2024-01-10"),
    ("2024-01-10 16:00", "2024-01-11"),  # at/after the close -> next day
    ("2024-01-10 16:05", "2024-01-11"),
    ("2024-01-12 16:30", "2024-01-16"),  # Friday after close -> skips weekend and holiday
    ("2024-01-13 10:00", "2024-01-16"),  # Saturday
])
def test_t0_is_first_close_after_publication(published, expected):
    pos = assign_t0(pd.Series([pd.Timestamp(published)]), DAYS)
    assert DAYS[int(pos.iloc[0])] == pd.Timestamp(expected)


def test_entry_lag_shifts_further():
    pos = assign_t0(pd.Series([pd.Timestamp("2024-01-10 07:00")]), DAYS, entry_lag_days=2)
    assert DAYS[int(pos.iloc[0])] == pd.Timestamp("2024-01-12")


def _cfg(H=5):
    return {"event": {"benchmark": "market", "market_ticker": "SPY", "hold_days": H, "entry_lag_days": 0},
            "portfolio": {"rebalance_months": [3, 6, 9, 12]}}


def test_day0_jump_is_not_in_the_tradable_car():
    """A +10% move on the announcement day must not show up in CAR[+1,+H]."""
    days = pd.bdate_range("2024-01-01", periods=40)
    r = pd.DataFrame(0.0, index=days, columns=["AAA", "SPY"])
    t0 = days[10]
    r.loc[t0, "AAA"] = 0.10
    r.loc[days[11]:days[15], "AAA"] = 0.01  # drift we *can* capture
    prices = (1 + r).cumprod()
    ev = pd.DataFrame({"event_id": ["e1"], "ticker": ["AAA"],
                       "published_at": [t0 + pd.Timedelta(hours=7)]})
    uni = pd.DataFrame({"ticker": ["AAA"], "sector_etf": ["SPY"]})
    out, paths = event_study(ev, prices, uni, _cfg(5))
    assert out["ar_day0"].iloc[0] == pytest.approx(0.10)
    assert out["car_1_5"].iloc[0] == pytest.approx(0.05)
    assert paths.iloc[0, 0] == 0


def test_after_close_release_moves_t0_and_reaction():
    days = pd.bdate_range("2024-01-01", periods=40)
    r = pd.DataFrame(0.0, index=days, columns=["AAA", "SPY"])
    r.loc[days[11], "AAA"] = -0.08  # reaction lands the day after an after-close release
    prices = (1 + r).cumprod()
    ev = pd.DataFrame({"event_id": ["e1"], "ticker": ["AAA"],
                       "published_at": [days[10] + pd.Timedelta(hours=16, minutes=10)]})
    out, _ = event_study(ev, prices, pd.DataFrame({"ticker": ["AAA"], "sector_etf": ["SPY"]}), _cfg(5))
    assert out["t0"].iloc[0] == days[11]
    assert out["ar_day0"].iloc[0] == pytest.approx(-0.08)
    assert out["car_1_5"].iloc[0] == pytest.approx(0.0)


def test_abnormal_return_subtracts_benchmark():
    days = pd.bdate_range("2024-01-01", periods=40)
    r = pd.DataFrame(0.01, index=days, columns=["AAA", "SPY"])
    prices = (1 + r).cumprod()
    ev = pd.DataFrame({"event_id": ["e1"], "ticker": ["AAA"], "published_at": [days[5] + pd.Timedelta(hours=7)]})
    out, _ = event_study(ev, prices, pd.DataFrame({"ticker": ["AAA"], "sector_etf": ["SPY"]}), _cfg(5))
    assert out["car_1_5"].iloc[0] == pytest.approx(0.0, abs=1e-12)


def test_events_too_close_to_end_get_no_car():
    days = pd.bdate_range("2024-01-01", periods=20)
    prices = pd.DataFrame(100.0, index=days, columns=["AAA", "SPY"])
    ev = pd.DataFrame({"event_id": ["e1"], "ticker": ["AAA"], "published_at": [days[17] + pd.Timedelta(hours=7)]})
    out, _ = event_study(ev, prices, pd.DataFrame({"ticker": ["AAA"], "sector_etf": ["SPY"]}), _cfg(5))
    assert np.isnan(out["car_1_5"].iloc[0])


def test_season_label_maps_to_next_rebalance():
    s = season_label(pd.Series(pd.to_datetime(["2024-01-25", "2024-02-28", "2024-03-01", "2024-11-10", "2024-12-15"])),
                     [3, 6, 9, 12])
    assert s.tolist() == ["2024-03", "2024-03", "2024-06", "2024-12", "2025-03"]
