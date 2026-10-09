"""Event alignment and abnormal returns.

Timing convention (the core look-ahead guard)
---------------------------------------------
``published_at`` is a US/Eastern wall-clock time. Day 0 (``t0``) is the first
trading day whose 16:00 close comes *after* publication:

* released 07:00 before the open on day d   -> t0 = d
* released 12:00 during the session on d    -> t0 = d
* released 16:05 after the close on d       -> t0 = next trading day

We assume we can only trade at the close of t0 (plus ``entry_lag_days``). The
day-0 move - the bulk of the earnings reaction - is therefore *not* captured;
the strategy only earns returns from day +1 onwards (post-announcement drift).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

MARKET_CLOSE = pd.Timedelta(hours=16)


def assign_t0(published_at: pd.Series, trading_days: pd.DatetimeIndex, entry_lag_days: int = 0) -> pd.Series:
    """Return the integer position (in ``trading_days``) of each event's entry day."""
    ts = pd.to_datetime(published_at)
    day = ts.dt.normalize()
    after_close = (ts - day) >= MARKET_CLOSE
    candidate = day + pd.to_timedelta(after_close.astype(int), unit="D")
    pos = trading_days.searchsorted(candidate.values, side="left") + entry_lag_days
    pos = pd.Series(pos, index=published_at.index)
    return pos.where(pos < len(trading_days))


def returns_from_prices(prices: pd.DataFrame) -> pd.DataFrame:
    return prices.sort_index().pct_change(fill_method=None)


def benchmark_returns(events: pd.DataFrame, rets: pd.DataFrame, universe: pd.DataFrame, cfg: dict) -> dict:
    mode = cfg["event"]["benchmark"]
    mkt = cfg["event"]["market_ticker"]
    if mode == "market":
        return {t: mkt for t in events["ticker"].unique()}
    if mode == "sector":
        m = dict(zip(universe["ticker"], universe["sector_etf"]))
        return {t: m.get(t, mkt) if m.get(t, mkt) in rets else mkt for t in events["ticker"].unique()}
    raise ValueError(f"benchmark must be market or sector, got {mode}")


def event_study(events: pd.DataFrame, prices: pd.DataFrame, universe: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute day-0 reaction and CAR[+1, +H] for each event.

    Returns (events with t0/date/car columns, path matrix of cumulative
    abnormal returns for days 0..H, one row per event).
    """
    H = cfg["event"]["hold_days"]
    rets = returns_from_prices(prices)
    days = rets.index
    bench = benchmark_returns(events, rets, universe, cfg)
    ev = events.copy()
    ev["t0_pos"] = assign_t0(ev["published_at"], days, cfg["event"].get("entry_lag_days", 0))
    ev = ev[ev["t0_pos"].notna() & ev["ticker"].isin(rets.columns)].copy()
    ev["t0_pos"] = ev["t0_pos"].astype(int)
    ev["t0"] = days[ev["t0_pos"].values]

    car = np.full(len(ev), np.nan)
    day0 = np.full(len(ev), np.nan)
    paths = np.full((len(ev), H + 1), np.nan)
    R = rets.to_numpy()
    col = {c: i for i, c in enumerate(rets.columns)}
    for k, (tk, p) in enumerate(zip(ev["ticker"], ev["t0_pos"])):
        if p + H >= len(days):
            continue
        ar = R[p:p + H + 1, col[tk]] - R[p:p + H + 1, col[bench[tk]]]
        if np.isnan(ar[1:]).mean() > 0.2:  # too much missing data
            continue
        day0[k] = ar[0]
        car[k] = np.nansum(ar[1:])
        paths[k, 0] = 0.0
        paths[k, 1:] = np.nancumsum(ar[1:])
    ev["ar_day0"] = day0
    ev[f"car_1_{H}"] = car
    ev["season"] = season_label(ev["t0"], cfg["portfolio"]["rebalance_months"])
    path_df = pd.DataFrame(paths, index=ev["event_id"].values, columns=range(H + 1))
    return ev.reset_index(drop=True), path_df


def season_label(dates: pd.Series, rebalance_months: list[int]) -> pd.Series:
    """Label each date with the rebalance it feeds, e.g. events in Dec-Feb -> '2024-03'."""
    months = sorted(rebalance_months)
    d = pd.to_datetime(dates)
    labels = []
    for ts in d:
        nxt = next((m for m in months if m > ts.month), None)
        year = ts.year if nxt else ts.year + 1
        labels.append(f"{year}-{(nxt or months[0]):02d}")
    return pd.Series(labels, index=dates.index)
