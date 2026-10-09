import numpy as np
import pandas as pd
import pytest

from earnsig.portfolio import event_time_long_short, historical_midrank, rebalance_dates, seasonal_long_short


def _cfg(**kw):
    pc = {"rebalance_months": [3, 6, 9, 12], "lookback_days": 100, "quantiles": 5, "min_names_per_leg": 1,
          "min_history": 5, "cost_bps": 0, "borrow_bps_annual": 0}
    pc.update(kw)
    return {"portfolio": pc, "event": {"hold_days": 5}}


def _panel(n=10, start="2024-01-01", periods=200, seed=0):
    days = pd.bdate_range(start, periods=periods)
    rng = np.random.default_rng(seed)
    tick = [f"S{i}" for i in range(n)]
    return days, tick, pd.DataFrame(rng.normal(0, 0.01, (periods, n)), index=days, columns=tick)


def test_rebalance_dates_are_first_trading_day_of_month():
    days = pd.bdate_range("2024-01-01", "2024-12-31")
    r = rebalance_dates(days, [3, 6, 9, 12], "2024-01-01", "2024-12-31")
    assert list(r) == [pd.Timestamp(d) for d in ("2024-03-01", "2024-06-03", "2024-09-02", "2024-12-02")]


def test_seasonal_uses_only_events_before_rebalance_day():
    days, tick, rets = _panel()
    R = pd.Timestamp("2024-03-01")
    # 10 events in February with signal = i; plus one event *on* the rebalance day with a huge signal
    ev = pd.DataFrame({"event_id": [f"e{i}" for i in range(10)], "ticker": tick,
                       "t0": pd.Timestamp("2024-02-15"), "sig": np.arange(10.0)})
    late = pd.DataFrame({"event_id": ["late"], "ticker": ["S0"], "t0": [R], "sig": [99.0]})
    out = seasonal_long_short(pd.concat([ev, late]), "sig", rets, _cfg())
    first = out["holdings"].iloc[0]
    assert first["rebalance"] == R
    assert first["longs"].split() == ["S8", "S9"]
    assert first["shorts"].split() == ["S0", "S1"]  # S0's same-day 99 was NOT used


def test_seasonal_returns_start_after_rebalance_close():
    days, tick, rets = _panel()
    R = pd.Timestamp("2024-03-01")
    rets.loc[R] = 0.5  # a huge move on the rebalance day itself must not be earned
    ev = pd.DataFrame({"event_id": [f"e{i}" for i in range(10)], "ticker": tick,
                       "t0": pd.Timestamp("2024-02-15"), "sig": np.arange(10.0)})
    out = seasonal_long_short(ev, "sig", rets, _cfg())
    assert out["returns"].index.min() > R


def test_seasonal_long_short_return_math():
    days, tick, rets = _panel()
    rets[:] = 0.0
    rets[["S8", "S9"]] = 0.002
    rets[["S0", "S1"]] = -0.001
    ev = pd.DataFrame({"event_id": [f"e{i}" for i in range(10)], "ticker": tick,
                       "t0": pd.Timestamp("2024-02-15"), "sig": np.arange(10.0)})
    out = seasonal_long_short(ev, "sig", rets, _cfg(cost_bps=10))
    r = out["returns"]
    assert r["gross"].iloc[1] == pytest.approx(0.003)
    assert r["cost"].iloc[0] == pytest.approx(2.0 * 10 / 1e4)  # initial turnover = 200% of gross
    assert r["cost"].iloc[1] == 0


def test_historical_midrank_ignores_same_day_and_future():
    ev = pd.DataFrame({"t0": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-02", "2024-01-03"]),
                       "s": [1.0, 5.0, -5.0, 3.0]})
    pct = historical_midrank(ev, "s")
    assert np.isnan(pct.iloc[0])
    assert pct.iloc[1] == 1.0 and pct.iloc[2] == 0.0  # only the 2024-01-01 event is history
    assert pct.iloc[3] == pytest.approx(2 / 3)


def test_event_time_holds_days_1_to_H_only():
    days = pd.bdate_range("2024-01-01", periods=60)
    rets = pd.DataFrame(0.0, index=days, columns=["L", "S", "SPY"])
    rets.loc[days[30], "L"] = 0.2  # day 0 of the trade event: not earnable
    rets.loc[days[31], "L"] = 0.01
    rets.loc[days[36], "L"] = 0.3  # day 6: after the 5-day hold
    hist = pd.DataFrame({"event_id": [f"h{i}" for i in range(10)], "ticker": "S",
                         "t0": days[:10], "t0_pos": range(10), "sig": np.linspace(-1, 1, 10)})
    trade = pd.DataFrame({"event_id": ["t"], "ticker": ["L"], "t0": [days[30]], "t0_pos": [30], "sig": [5.0]})
    out = event_time_long_short(pd.concat([hist, trade], ignore_index=True), "sig", rets, rets["SPY"], _cfg())
    r = out["returns"]["gross"]
    assert r.get(days[30], 0) == 0
    assert r[days[31]] == pytest.approx(0.01)
    assert r.get(days[36], 0) == 0
