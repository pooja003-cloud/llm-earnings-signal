"""Long/short portfolio construction.

Two constructions, both free of look-ahead:

1. ``seasonal_long_short`` (the main test). On the first trading day of each
   rebalance month (default Mar/Jun/Sep/Dec, i.e. after each earnings season)
   rank stocks by the signal from their most recent document published
   *before* that day. Buy the top quintile, short the bottom quintile, equal
   weight, hold until the next rebalance.

2. ``event_time_long_short`` (robustness). Each document opens a position at
   the close of its day 0 and holds it for days +1..+H. Long/short membership
   uses breakpoints from documents published *earlier* only (expanding
   window), since later reporters in the same season are not yet known.
"""
from __future__ import annotations

import bisect
import hashlib

import numpy as np
import pandas as pd


def _tiebreak(event_ids: pd.Series) -> pd.Series:
    """Deterministic pseudo-random tie-breaker (stable across runs)."""
    return event_ids.map(lambda e: int(hashlib.md5(e.encode()).hexdigest()[:8], 16))


def rebalance_dates(days: pd.DatetimeIndex, months: list[int], start, end) -> pd.DatetimeIndex:
    days = days[(days >= pd.Timestamp(start)) & (days <= pd.Timestamp(end))]
    s = pd.Series(days, index=days)
    firsts = s.groupby([days.year, days.month]).min()
    return pd.DatetimeIndex([d for d in firsts if d.month in months])


def seasonal_long_short(ev: pd.DataFrame, signal: str, rets: pd.DataFrame, cfg: dict) -> dict:
    pc = cfg["portfolio"]
    q = pc["quantiles"]
    min_leg = pc.get("min_names_per_leg", 3)
    days = rets.index
    ev = ev.dropna(subset=[signal])
    rdates = rebalance_dates(days, pc["rebalance_months"], ev["t0"].min(), days[-1])
    daily, holdings = [], []
    prev_w = pd.Series(dtype=float)
    for k, R in enumerate(rdates):
        window = ev[(ev["t0"] < R) & (ev["t0"] >= R - pd.Timedelta(days=pc["lookback_days"]))]
        latest = window.sort_values("t0").groupby("ticker").tail(1)
        n_leg = len(latest) // q
        end = rdates[k + 1] if k + 1 < len(rdates) else days[-1]
        hold = days[(days > R) & (days <= end)]
        if n_leg < min_leg or len(hold) == 0:
            continue
        ranked = latest.assign(_tb=_tiebreak(latest["event_id"])).sort_values([signal, "_tb"])
        shorts, longs = ranked["ticker"].iloc[:n_leg].tolist(), ranked["ticker"].iloc[-n_leg:].tolist()
        w = pd.Series({**{t: -1 / n_leg for t in shorts}, **{t: 1 / n_leg for t in longs}})
        turnover = w.subtract(prev_w, fill_value=0).abs().sum()
        prev_w = w
        r_long = rets.loc[hold, longs].mean(axis=1)
        r_short = rets.loc[hold, shorts].mean(axis=1)
        gross = (r_long - r_short).fillna(0.0)
        cost = pd.Series(0.0, index=hold)
        cost.iloc[0] = turnover * pc["cost_bps"] / 1e4
        cost += pc["borrow_bps_annual"] / 1e4 / 252
        daily.append(pd.DataFrame({"long": r_long, "short": r_short, "gross": gross,
                                   "cost": cost, "net": gross - cost}))
        holdings.append({"rebalance": R, "end": end, "n_per_leg": n_leg, "turnover": turnover,
                         "longs": " ".join(longs), "shorts": " ".join(shorts),
                         "period_net": float((1 + (gross - cost)).prod() - 1)})
    returns = pd.concat(daily) if daily else pd.DataFrame(columns=["long", "short", "gross", "cost", "net"])
    return {"returns": returns, "holdings": pd.DataFrame(holdings)}


def historical_midrank(ev: pd.DataFrame, signal: str) -> pd.Series:
    """Percentile of each event's signal among events with strictly earlier t0."""
    ev = ev.sort_values("t0")
    hist: list[float] = []
    out = pd.Series(np.nan, index=ev.index)
    for t0, grp in ev.groupby("t0", sort=True):
        if hist:
            for i, v in grp[signal].items():
                lo = bisect.bisect_left(hist, v)
                hi = bisect.bisect_right(hist, v)
                out[i] = (lo + 0.5 * (hi - lo)) / len(hist)
        for v in grp[signal]:
            bisect.insort(hist, v)
    return out


def event_time_long_short(ev: pd.DataFrame, signal: str, rets: pd.DataFrame, bench: pd.Series, cfg: dict) -> dict:
    pc = cfg["portfolio"]
    H = cfg["event"]["hold_days"]
    q = pc["quantiles"]
    ev = ev.dropna(subset=[signal]).sort_values("t0").copy()
    ev["hist_pct"] = historical_midrank(ev, signal)
    counts = ev.groupby("t0").size().cumsum().shift(fill_value=0)
    ev["n_prior"] = ev["t0"].map(counts)
    ev["leg"] = 0
    ok = ev["n_prior"] >= pc["min_history"]
    ev.loc[ok & (ev["hist_pct"] >= 1 - 1 / q), "leg"] = 1
    ev.loc[ok & (ev["hist_pct"] <= 1 / q), "leg"] = -1

    T = len(rets)
    R = rets.to_numpy()
    col = {c: i for i, c in enumerate(rets.columns)}
    sums = {1: np.zeros(T), -1: np.zeros(T)}
    cnts = {1: np.zeros(T), -1: np.zeros(T)}
    flows = {1: np.zeros(T), -1: np.zeros(T)}  # entries + exits, for costs
    for tk, p, leg in zip(ev["ticker"], ev["t0_pos"], ev["leg"]):
        if leg == 0:
            continue
        a, b = p + 1, min(p + H, T - 1)
        r = R[a:b + 1, col[tk]]
        valid = ~np.isnan(r)
        sums[leg][a:b + 1] += np.where(valid, r, 0.0)
        cnts[leg][a:b + 1] += valid
        flows[leg][a] += 1
        if b + 1 < T:
            flows[leg][b + 1] += 1
    with np.errstate(invalid="ignore", divide="ignore"):
        rl = np.where(cnts[1] > 0, sums[1] / cnts[1], np.nan)
        rs = np.where(cnts[-1] > 0, sums[-1] / cnts[-1], np.nan)
        tl = np.where(cnts[1] > 0, flows[1] / np.maximum(cnts[1], 1), 0)
        ts = np.where(cnts[-1] > 0, flows[-1] / np.maximum(cnts[-1], 1), 0)
    b = bench.reindex(rets.index).to_numpy()
    # if one leg is empty, hedge the other with the market
    gross = np.where(~np.isnan(rl) & ~np.isnan(rs), rl - rs,
             np.where(~np.isnan(rl), rl - b, np.where(~np.isnan(rs), b - rs, 0.0)))
    gross = np.nan_to_num(gross)
    active = (cnts[1] + cnts[-1]) > 0
    cost = (tl + ts) * pc["cost_bps"] / 1e4 + active * pc["borrow_bps_annual"] / 1e4 / 252
    df = pd.DataFrame({"long": rl, "short": rs, "gross": gross, "cost": cost, "net": gross - cost},
                      index=rets.index)
    first = np.argmax(active) if active.any() else T
    last = T - 1 - np.argmax(active[::-1]) if active.any() else -1
    return {"returns": df.iloc[first:last + 1], "events": ev}
