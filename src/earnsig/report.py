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
        ("Reports with a score", lambda c: f"{int(g(c, 'n_events')):,}"),
        (f"Information coefficient: rank correlation of score with the {H}-day return above the market",
         lambda c: _num(g(c, "mean_ic"), 3, True)),
        ("t-statistic of the information coefficient across seasons (about 2 or more = unlikely to be luck)",
         lambda c: _num(g(c, "ic_t"))),
        ("Earnings seasons where the correlation was positive",
         lambda c: f"{_pct(g(c, 'ic_pos_share'), 0)} of {int(g(c, 'seasons'))}"),
        ("Hit rate: top and bottom fifth that moved the predicted way", lambda c: _pct(g(c, "hit_rate"))),
        (f"Average {H}-day return above the market, most upbeat fifth / most gloomy fifth",
         lambda c: f"{_pct(g(c, 'top_q_car'), 2)} / {_pct(g(c, 'bottom_q_car'), 2)}"),
        ("**Long/short portfolio, rebalanced each season, after costs:** yearly return",
         lambda c: _pct(g(c, "seasonal_net_ann_return"))),
        ("Yearly volatility (typical size of ups and downs)", lambda c: _pct(g(c, "seasonal_net_ann_vol"))),
        ("Sharpe ratio (return per unit of risk), after costs (before costs)",
         lambda c: f"{_num(g(c, 'seasonal_net_sharpe'))} ({_num(g(c, 'seasonal_gross_sharpe'))})"),
        ("Maximum drawdown (worst fall from a peak)", lambda c: _pct(g(c, "seasonal_net_max_drawdown"))),
        ("Turnover per rebalance (replacing every holding = 400%)", lambda c: _pct(g(c, "seasonal_turnover"), 0)),
        ("Holding periods that made money", lambda c: _pct(g(c, "seasonal_periods_positive"), 0)),
        (f"**Long/short portfolio, each report held {H} trading days, after costs:** Sharpe ratio",
         lambda c: _num(g(c, "event_net_sharpe"))),
        ("Maximum drawdown", lambda c: _pct(g(c, "event_net_max_drawdown"))),
        ("**Fama-French five factors plus momentum:** alpha, yearly return not explained by the factors "
         "(t-statistic)", lambda c: f"{_pct(g(c, 'ff_alpha_ann'))} ({_num(g(c, 'ff_alpha_t'))})"),
        ("Market beta (sensitivity to the overall stock market)", lambda c: _num(g(c, "ff_beta_Mkt-RF"), 2, True)),
        ("Size / value / profitability / investment betas",
         lambda c: " / ".join(_num(g(c, f"ff_beta_{f}"), 2, True) for f in ("SMB", "HML", "RMW", "CMA"))),
        ("Momentum beta (tendency to hold recent winners)", lambda c: _num(g(c, "ff_beta_Mom"), 2, True)),
        ("R-squared (share of the ups and downs explained by the factors)", lambda c: _num(g(c, "ff_r2"))),
        ("Information coefficient using only reports after the model's training cutoff (number of reports)",
         lambda c: (f"{_num(g(c, 'post_cutoff_pooled_ic'), 3, True)} ({int(g(c, 'post_cutoff_n'))})"
                    if g(c, "post_cutoff_n") > 0 else "none: every report is older than the cutoff")),
    ]
    head = "| Measure | " + " | ".join(LABELS.get(c, c) for c in cols) + " |\n|---|" + "---:|" * len(cols)
    body = "\n".join(f"| {name} | " + " | ".join(fn(c) for c in cols) + " |" for name, fn in rows)
    return head + "\n" + body


def topic_table(s: pd.DataFrame, topics: list[str]) -> str:
    lines = ["| Score | Information coefficient | t-statistic | Hit rate | Sharpe ratio, seasonal portfolio | "
             "Sharpe ratio, 20-day portfolio |",
             "|---|---:|---:|---:|---:|---:|"]
    for t in topics:
        if t in s.index:
            r = s.loc[t]
            lines.append(f"| {LABELS.get(t, t)} | {_num(r['mean_ic'], 3, True)} | {_num(r['ic_t'])} | "
                         f"{_pct(r['hit_rate'])} | {_num(r.get('seasonal_net_sharpe', np.nan))} | "
                         f"{_num(r.get('event_net_sharpe', np.nan))} |")
    return "\n".join(lines)


def consistency_table(c: dict) -> str:
    lines = [f"The same {c['n']} reports were scored twice with identical inputs.", "",
             "| Score | Same score both times | Within one step | Rank correlation | "
             "Agreement beyond chance (weighted kappa, 1 = perfect) |",
             "|---|---:|---:|---:|---:|"]
    for k in ("guidance_tone", "overall_tone", "margins_tone", "demand_tone"):
        v = c[k]
        lines.append(f"| {k.replace('_', ' ').capitalize()} | {_pct(v['exact_agreement'], 0)} | "
                     f"{_pct(v['within_one'], 0)} | {_num(v['spearman'])} | {_num(v['weighted_kappa'])} |")
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
                  f"> **Every report in this sample is older than the language model's training cutoff "
                  f"({meta['training_cutoff']}).** The model may have read news about how these stocks moved",
                  "> after each report, so a good result here can come from memory rather than reading skill.",
                  "> Treat it as a best case. A clean test needs a model trained before the reports were published",
                  "> (see *Biases and limitations* below).", ""]
    elif not demo and post_n < meta["n_events"]:
        parts += [f"> [!NOTE]\n> {meta['n_events'] - post_n} of {meta['n_events']} reports are older than the language model's "
                  f"training cutoff ({meta['training_cutoff']}); the last row of the table uses only the {post_n} after it.", ""]
    bench = "the S&P 500 index fund (SPY)" if meta["benchmark"] == "market" else "each stock's sector fund"
    parts += [f"_Sample: {meta['n_events']:,} earnings reports from {meta['n_tickers']} companies, "
              f"{meta['first_event']} to {meta['last_event']}. Scored by: {meta['model']}. "
              f"Returns are measured above {bench}. Trading costs: {meta['cost_bps'] / 100:.2f}% per trade "
              f"plus {meta['borrow_bps'] / 100:.2f}% a year to borrow shares for selling short._", "",
              f"![Growth of $1 in the long/short portfolios]({rel}/figures/cumulative_long_short.png)", "",
              "### All measures", "", stats_table(s, cols, H), ""]
    if not meta.get("has_factors"):
        parts += ["_Factor rows are empty: run `earnsig factors` to download the Fama-French factor data._", ""]
    parts += [f"![Return after the report, by tone group]({rel}/figures/car_by_quintile.png)", "",
              "### Which topic matters most?", "",
              "Each report also got separate scores for what management said about future guidance, profit "
              "margins and customer demand.", "",
              topic_table(s, [prim, *cfg["signals"].get("topics", []), base]), "",
              f"![Information coefficient by score]({rel}/figures/ic_by_signal.png)", ""]
    cons = rd / "consistency.json"
    if cons.exists():
        parts += ["### Does the model give the same answer twice?", "",
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
