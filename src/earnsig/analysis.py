"""Run the full evaluation: event study, IC, hit rate, portfolios, factor regressions."""
from __future__ import annotations

import json
import logging

import numpy as np
import pandas as pd

from . import metrics as M
from .config import load_universe
from .events import event_study, returns_from_prices
from .llm_score import scorer_label
from .market_data import load_factors, load_prices
from .portfolio import event_time_long_short, seasonal_long_short

log = logging.getLogger(__name__)


def load_panel(cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    p = cfg["paths"]
    events = pd.read_csv(p.events, parse_dates=["published_at"])
    universe = load_universe(cfg)
    prices = load_prices(cfg)
    ev, paths = event_study(events, prices, universe, cfg)
    if p.llm_scores.exists():
        ev = ev.merge(pd.read_csv(p.llm_scores), on="event_id", how="left")
    if p.lm_scores.exists():
        ev = ev.merge(pd.read_csv(p.lm_scores), on="event_id", how="left")
    cutoff = pd.Timestamp(cfg["llm"]["training_cutoff"])
    ev["post_cutoff"] = ev["t0"] > cutoff
    return ev, paths, prices


def evaluate_signal(ev: pd.DataFrame, signal: str, rets: pd.DataFrame, bench: pd.Series,
                    factors: pd.DataFrame | None, cfg: dict) -> dict:
    H = cfg["event"]["hold_days"]
    target = f"car_1_{H}"
    q = cfg["portfolio"]["quantiles"]
    e = ev.dropna(subset=[signal, target])
    ics = M.ic_by_season(e, signal, target)
    seas = seasonal_long_short(e, signal, rets, cfg)
    evt = event_time_long_short(e, signal, rets, bench, cfg)
    sr, er = seas["returns"], evt["returns"]
    post = e[e["post_cutoff"]]
    res = {
        "signal": signal,
        "n_events": len(e),
        **M.ic_summary(ics),
        "pooled_ic": e[[signal, target]].corr(method="spearman").iloc[0, 1],
        **M.hit_rate(e, signal, target, q),
        "post_cutoff_n": len(post),
        "post_cutoff_pooled_ic": post[[signal, target]].corr(method="spearman").iloc[0, 1] if len(post) > 10 else np.nan,
    }
    if len(sr):
        res.update({f"seasonal_net_{k}": v for k, v in M.perf_stats(sr["net"]).items()})
        res.update({f"seasonal_gross_{k}": v for k, v in M.perf_stats(sr["gross"]).items()})
        res["seasonal_turnover"] = seas["holdings"]["turnover"].mean()
        res["seasonal_periods_positive"] = (seas["holdings"]["period_net"] > 0).mean()
        res["seasonal_periods"] = len(seas["holdings"])
        res.update({f"ff_{k}": v for k, v in M.factor_regression(sr["net"], factors).items()})
    if len(er):
        res.update({f"event_net_{k}": v for k, v in M.perf_stats(er["net"]).items()})
        res.update({f"event_gross_{k}": v for k, v in M.perf_stats(er["gross"]).items()})
    return {"summary": res, "ics": ics, "seasonal": seas, "event_time": evt}


def run_backtest(cfg: dict) -> dict:
    p = cfg["paths"]
    ev, paths, prices = load_panel(cfg)
    rets = returns_from_prices(prices)
    bench = rets[cfg["event"]["market_ticker"]]
    factors = load_factors(cfg)
    sig_cfg = cfg["signals"]
    signals = [s for s in [sig_cfg["primary"], sig_cfg["baseline"], *sig_cfg.get("topics", [])] if s in ev]
    if not signals:
        raise SystemExit("No scores found. Run `earnsig score` and `earnsig baseline` first.")
    out = {s: evaluate_signal(ev, s, rets, bench, factors, cfg) for s in signals}

    res_dir = p.results
    summary = pd.DataFrame([o["summary"] for o in out.values()]).set_index("signal")
    summary.to_csv(res_dir / "summary.csv")
    pd.DataFrame({s: o["ics"] for s, o in out.items()}).to_csv(res_dir / "ic_by_season.csv")
    daily = pd.DataFrame({f"{s}_{kind}_{col}": o[kind]["returns"][col]
                          for s, o in out.items() for kind in ("seasonal", "event_time")
                          for col in ("gross", "net")})
    daily[f"market_{cfg['event']['market_ticker']}"] = bench.reindex(daily.index)
    daily.to_csv(res_dir / "daily_returns.csv")
    for s in (sig_cfg["primary"], sig_cfg["baseline"]):
        if s in out:
            out[s]["seasonal"]["holdings"].to_csv(res_dir / f"holdings_{s}.csv", index=False)
    cols = ["event_id", "ticker", "published_at", "t0", "season", "post_cutoff", "ar_day0",
            f"car_1_{cfg['event']['hold_days']}", *signals]
    if "llm_reason" in ev:
        cols.append("llm_reason")
    ev[cols].to_csv(res_dir / "events_scored.csv", index=False)
    meta = {"n_events": int(len(ev)), "n_tickers": int(ev["ticker"].nunique()),
            "first_event": str(ev["t0"].min().date()), "last_event": str(ev["t0"].max().date()),
            "provider": cfg["llm"]["provider"], "model": scorer_label(cfg),
            "benchmark": cfg["event"]["benchmark"], "hold_days": cfg["event"]["hold_days"],
            "cost_bps": cfg["portfolio"]["cost_bps"], "borrow_bps": cfg["portfolio"]["borrow_bps_annual"],
            "training_cutoff": cfg["llm"]["training_cutoff"], "has_factors": factors is not None}
    (res_dir / "meta.json").write_text(json.dumps(meta, indent=2))

    from .figures import make_figures
    make_figures(cfg, out, ev, paths, bench)
    log.info("Backtest done: %s", res_dir)
    return {"summary": summary, "meta": meta}
