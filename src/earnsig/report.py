"""Write the results section of README.md from results/ (between marker comments)."""
from __future__ import annotations

import json
import re

import numpy as np
import pandas as pd

from .config import ROOT
from .figures import LABELS

START, END = "<!-- RESULTS:START -->", "<!-- RESULTS:END -->"


def _pct(x, d=1):
    return "–" if x is None or pd.isna(x) else f"{x * 100:.{d}f}%"


def _num(x, d=2, sign=False):
    return "–" if x is None or pd.isna(x) else (f"{x:+.{d}f}" if sign else f"{x:.{d}f}")


def stats_table(s: pd.DataFrame, cols: list[str], H: int) -> str:
    g = lambda c, k: s.loc[c].get(k, np.nan)  # noqa: E731
    rows = [
        ("Documents with a score", lambda c: f"{int(g(c, 'n_events')):,}"),
        (f"Mean IC (Spearman vs. CAR[+1,+{H}])", lambda c: _num(g(c, "mean_ic"), 3, True)),
        ("IC t-stat (across seasons)", lambda c: _num(g(c, "ic_t"))),
        ("Seasons with positive IC", lambda c: f"{_pct(g(c, 'ic_pos_share'), 0)} of {int(g(c, 'seasons'))}"),
        ("Hit rate (extreme quintiles)", lambda c: _pct(g(c, "hit_rate"))),
        (f"Mean CAR, top / bottom quintile", lambda c: f"{_pct(g(c, 'top_q_car'), 2)} / {_pct(g(c, 'bottom_q_car'), 2)}"),
        ("**Seasonal L/S, net:** annual return", lambda c: _pct(g(c, "seasonal_net_ann_return"))),
        ("Annual volatility", lambda c: _pct(g(c, "seasonal_net_ann_vol"))),
        ("Sharpe ratio, net (gross)", lambda c: f"{_num(g(c, 'seasonal_net_sharpe'))} ({_num(g(c, 'seasonal_gross_sharpe'))})"),
        ("Max drawdown", lambda c: _pct(g(c, "seasonal_net_max_drawdown"))),
        ("Turnover per rebalance (full swap = 400%)", lambda c: _pct(g(c, "seasonal_turnover"), 0)),
        ("Holding periods with a gain", lambda c: _pct(g(c, "seasonal_periods_positive"), 0)),
        (f"**Event-time L/S (days +1..+{H}), net:** Sharpe", lambda c: _num(g(c, "event_net_sharpe"))),
        ("Max drawdown", lambda c: _pct(g(c, "event_net_max_drawdown"))),
        ("**FF5 + momentum:** alpha, annual (t)", lambda c: f"{_pct(g(c, 'ff_alpha_ann'))} ({_num(g(c, 'ff_alpha_t'))})"),
        ("Market beta", lambda c: _num(g(c, "ff_beta_Mkt-RF"), 2, True)),
        ("SMB / HML / RMW / CMA", lambda c: " / ".join(_num(g(c, f"ff_beta_{f}"), 2, True) for f in ("SMB", "HML", "RMW", "CMA"))),
        ("Momentum beta", lambda c: _num(g(c, "ff_beta_Mom"), 2, True)),
        ("R² of factor regression", lambda c: _num(g(c, "ff_r2"))),
        ("Pooled IC after LLM training cutoff (n)", lambda c: (
            f"{_num(g(c, 'post_cutoff_pooled_ic'), 3, True)} ({int(g(c, 'post_cutoff_n'))})"
            if g(c, "post_cutoff_n") > 0 else "none: all documents predate the cutoff")),
    ]
    head = "| Metric | " + " | ".join(LABELS.get(c, c) for c in cols) + " |\n|---|" + "---:|" * len(cols)
    body = "\n".join(f"| {name} | " + " | ".join(fn(c) for c in cols) + " |" for name, fn in rows)
    return head + "\n" + body


def topic_table(s: pd.DataFrame, topics: list[str]) -> str:
    lines = ["| Signal | Mean IC | IC t-stat | Hit rate | Seasonal L/S Sharpe (net) | Event-time L/S Sharpe (net) |",
             "|---|---:|---:|---:|---:|---:|"]
    for t in topics:
        if t in s.index:
            r = s.loc[t]
            lines.append(f"| {LABELS.get(t, t)} | {_num(r['mean_ic'], 3, True)} | {_num(r['ic_t'])} | "
                         f"{_pct(r['hit_rate'])} | {_num(r.get('seasonal_net_sharpe', np.nan))} | "
                         f"{_num(r.get('event_net_sharpe', np.nan))} |")
    return "\n".join(lines)


def consistency_table(c: dict) -> str:
    lines = [f"Scored {c['n']} documents twice with identical inputs.", "",
             "| Score | Exact agreement | Within ±1 | Spearman | Weighted kappa |", "|---|---:|---:|---:|---:|"]
    for k in ("guidance_tone", "overall_tone", "margins_tone", "demand_tone"):
        v = c[k]
        lines.append(f"| {k.replace('_', ' ')} | {_pct(v['exact_agreement'], 0)} | {_pct(v['within_one'], 0)} | "
                     f"{_num(v['spearman'])} | {_num(v['weighted_kappa'])} |")
    return "\n".join(lines)


def build_section(cfg: dict) -> str:
    rd = cfg["paths"].results
    s = pd.read_csv(rd / "summary.csv", index_col=0)
    meta = json.loads((rd / "meta.json").read_text())
    rel = rd.relative_to(ROOT).as_posix()
    prim, base = cfg["signals"]["primary"], cfg["signals"]["baseline"]
    cols = [c for c in (prim, base) if c in s.index]
    H = meta["hold_days"]
    demo = meta["provider"] == "mock"
    parts = [START, ""]
    if demo:
        parts += ["> [!WARNING]",
                  "> **These numbers come from the synthetic demo** (`earnsig demo`): fake prices, fake documents and a",
                  "> mock scorer that peeks at a planted signal. They show what the report looks like and prove the",
                  "> plumbing works. They say nothing about real markets. Run the real pipeline to replace this section.", ""]
    post_n = int(s.loc[prim, "post_cutoff_n"]) if prim in s.index else 0
    if not demo and post_n == 0:
        parts += ["> [!CAUTION]",
                  f"> **Every document in this sample predates the scorer's training cutoff "
                  f"({meta['training_cutoff']}).** The model may have read news about how these stocks moved",
                  "> after each release, so a positive result here can reflect memory rather than reading skill.",
                  "> Treat it as an upper bound. A clean test needs a model trained before the sample period",
                  "> (see *Biases* below).", ""]
    elif not demo and post_n < meta["n_events"]:
        parts += [f"> [!NOTE]\n> {meta['n_events'] - post_n} of {meta['n_events']} documents predate the scorer's "
                  f"training cutoff ({meta['training_cutoff']}); the last row of the table uses only the {post_n} after it.", ""]
    parts += [f"_Sample: {meta['n_events']:,} documents, {meta['n_tickers']} stocks, {meta['first_event']} to "
              f"{meta['last_event']}. Scorer: `{meta['model']}`. Abnormal returns vs. {meta['benchmark']} benchmark. "
              f"Costs: {meta['cost_bps']} bps one-way + {meta['borrow_bps']} bps/yr borrow._", "",
              f"![Cumulative return of the long/short portfolio]({rel}/figures/cumulative_long_short.png)", "",
              "### Summary statistics", "", stats_table(s, cols, H), ""]
    if not meta.get("has_factors"):
        parts += ["_Factor rows are empty: run `earnsig factors` to download Fama-French data._", ""]
    parts += [f"![Drift by quintile]({rel}/figures/car_by_quintile.png)", "",
              "### Which topic matters most? (stretch goal)", "",
              topic_table(s, [prim, *cfg["signals"].get("topics", []), base]), "",
              f"![IC by signal]({rel}/figures/ic_by_signal.png)", ""]
    cons = rd / "consistency.json"
    if cons.exists():
        parts += ["### Scoring consistency (same document, scored twice)", "",
                  consistency_table(json.loads(cons.read_text())), ""]
    parts += [END]
    return "\n".join(parts)


def update_readme(cfg: dict, readme=None) -> None:
    readme = readme or ROOT / "README.md"
    text = readme.read_text()
    section = build_section(cfg)
    if START in text and END in text:
        text = re.sub(re.escape(START) + r".*?" + re.escape(END), lambda m: section, text, flags=re.S)
    else:
        text += "\n" + section + "\n"
    readme.write_text(text)
