"""Signal and portfolio statistics."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

TRADING_DAYS = 252


def perf_stats(r: pd.Series) -> dict:
    r = r.dropna()
    if len(r) < 2:
        return {}
    wealth = (1 + r).cumprod()
    dd = wealth / wealth.cummax() - 1
    ann_ret = r.mean() * TRADING_DAYS
    ann_vol = r.std(ddof=1) * np.sqrt(TRADING_DAYS)
    years = len(r) / TRADING_DAYS
    return {
        "ann_return": ann_ret,
        "cagr": wealth.iloc[-1] ** (1 / years) - 1 if years > 0 else np.nan,
        "ann_vol": ann_vol,
        "sharpe": ann_ret / ann_vol if ann_vol > 0 else np.nan,
        "max_drawdown": dd.min(),
        "total_return": wealth.iloc[-1] - 1,
        "days": len(r),
    }


def ic_by_season(ev: pd.DataFrame, signal: str, target: str, min_n: int = 10) -> pd.Series:
    """Cross-sectional Spearman rank correlation per earnings season."""
    out = {}
    for season, g in ev.dropna(subset=[signal, target]).groupby("season"):
        if len(g) >= min_n and g[signal].nunique() > 1:
            out[season] = stats.spearmanr(g[signal], g[target]).statistic
    return pd.Series(out, name=signal).sort_index()


def ic_summary(ics: pd.Series) -> dict:
    n = len(ics)
    if n == 0:
        return {"mean_ic": np.nan, "ic_t": np.nan, "ic_pos_share": np.nan, "seasons": 0}
    sd = ics.std(ddof=1) if n > 1 else np.nan
    return {
        "mean_ic": ics.mean(),
        "ic_t": ics.mean() / sd * np.sqrt(n) if sd and sd > 0 else np.nan,
        "ic_pos_share": (ics > 0).mean(),
        "seasons": n,
    }


def quantile_in_season(ev: pd.DataFrame, signal: str, q: int = 5) -> pd.Series:
    """Ex-post quantile (1 = most negative) of each event within its season.

    For evaluation only (hit rate, CAR-by-quintile chart), not for trading.
    """
    def f(s):
        r = s.rank(method="first")
        return np.ceil(r / len(s) * q).clip(1, q)
    return ev.dropna(subset=[signal]).groupby("season")[signal].transform(f)


def hit_rate(ev: pd.DataFrame, signal: str, target: str, q: int = 5) -> dict:
    """Share of extreme-quintile events whose abnormal return had the predicted sign."""
    e = ev.dropna(subset=[signal, target]).copy()
    e["qt"] = quantile_in_season(e, signal, q)
    top, bot = e[e["qt"] == q], e[e["qt"] == 1]
    hits = (top[target] > 0).sum() + (bot[target] < 0).sum()
    n = len(top) + len(bot)
    return {"hit_rate": hits / n if n else np.nan, "hit_n": n,
            "top_q_car": top[target].mean(), "bottom_q_car": bot[target].mean()}


def factor_regression(r: pd.Series, factors: pd.DataFrame | None, lags: int = 5) -> dict:
    """OLS of daily long/short returns on Fama-French 5 factors + momentum.

    The long/short book is self-financing, so returns are not reduced by the
    risk-free rate. Standard errors are Newey-West (HAC).
    """
    if factors is None:
        return {}
    import statsmodels.api as sm

    cols = [c for c in ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "Mom"] if c in factors]
    df = pd.concat([r.rename("y"), factors[cols]], axis=1, join="inner").dropna()
    if len(df) < 60:
        return {}
    X = sm.add_constant(df[cols])
    fit = sm.OLS(df["y"], X).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
    out = {"alpha_ann": fit.params["const"] * TRADING_DAYS, "alpha_t": fit.tvalues["const"],
           "r2": fit.rsquared, "n": int(fit.nobs)}
    for c in cols:
        out[f"beta_{c}"] = fit.params[c]
        out[f"t_{c}"] = fit.tvalues[c]
    return out
