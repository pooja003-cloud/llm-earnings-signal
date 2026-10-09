"""README figures (static PNGs, light theme)."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .metrics import quantile_in_season  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e6e5e1"
BLUE, ORANGE = "#2a78d6", "#eb6834"
# diverging red <-> gray <-> blue for ordered quintiles (Q1 most negative)
DIVERGING = ["#c23434", "#f0a3a2", "#8f8e8a", "#9cc0ee", "#1f5fae"]
LABELS = {"llm": "Language model tone", "lm": "Finance word list", "llm_guidance": "Language model: guidance",
          "llm_margins": "Language model: margins", "llm_demand": "Language model: demand",
          "llm_overall": "Language model: overall tone", "llm_delta": "Language model: change since last quarter"}


def _style(ax, title: str, subtitle: str | None = None):
    ax.set_facecolor(SURFACE)
    ax.figure.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9, length=0)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_title(title, loc="left", fontsize=12, color=INK, fontweight="bold", pad=22 if subtitle else 10)
    if subtitle:
        ax.text(0, 1.02, subtitle, transform=ax.transAxes, fontsize=9, color=INK2, va="bottom")


def make_figures(cfg: dict, out: dict, ev: pd.DataFrame, paths: pd.DataFrame, bench: pd.Series):
    fig_dir = cfg["paths"].results / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    demo = cfg["llm"]["provider"] == "mock"
    tag = "SYNTHETIC DEMO DATA - not a real result. " if demo else ""
    prim, base = cfg["signals"]["primary"], cfg["signals"]["baseline"]
    H = cfg["event"]["hold_days"]

    # 1. cumulative return of both long/short constructions, net of costs, LLM vs. baseline
    fig, axes = plt.subplots(2, 1, figsize=(9, 7.4), dpi=150, sharex=True)
    titles = {"seasonal": "Rebalanced once after each earnings season",
              "event_time": f"Each report held for {H} trading days"}
    for ax, kind in zip(axes, ("seasonal", "event_time")):
        series = {s: out[s][kind]["returns"]["net"] for s in (prim, base)
                  if s in out and len(out[s][kind]["returns"])}
        if not series:
            continue
        df = pd.DataFrame(series).dropna(how="all").fillna(0.0)
        wealth = (1 + df).cumprod()
        for s, color in zip(df.columns, (BLUE, ORANGE)):
            ax.plot(wealth.index, wealth[s], color=color, lw=2, label=LABELS.get(s, s))
            ax.annotate(f"{LABELS.get(s, s)}  {wealth[s].iloc[-1]:.2f}", (wealth.index[-1], wealth[s].iloc[-1]),
                        xytext=(6, 0), textcoords="offset points", va="center", fontsize=9, color=INK)
        ax.axhline(1, color=INK2, lw=0.8, ls=":")
        ax.set_xlim(wealth.index[0], wealth.index[-1] + (wealth.index[-1] - wealth.index[0]) * 0.2)
        ax.legend(frameon=False, loc="upper left", fontsize=9, labelcolor=INK)
        ax.set_ylabel("Growth of $1", color=INK2, fontsize=9)
        _style(ax, titles[kind])
        ax.title.set_fontsize(10.5)
    fig.suptitle(("SYNTHETIC DEMO DATA - not a real result\n" if demo else "") + "Buy the most upbeat fifth of reports, sell short the most gloomy fifth\n"
                 f"Growth of $1 after {cfg['portfolio']['cost_bps'] / 100:.2f}% per trade and "
                 f"{cfg['portfolio']['borrow_bps_annual'] / 100:.2f}% a year borrowing fee",
                 x=0.01, ha="left", fontsize=12, fontweight="bold", color=INK)
    fig.tight_layout()
    fig.savefig(fig_dir / "cumulative_long_short.png", facecolor=SURFACE)
    plt.close(fig)

    # 2. average cumulative abnormal return by LLM quintile, days 0..H
    if prim in ev:
        e = ev.dropna(subset=[prim, f"car_1_{H}"]).copy()
        e["qt"] = quantile_in_season(e, prim, cfg["portfolio"]["quantiles"])
        P = paths.loc[e["event_id"]]
        fig, ax = plt.subplots(figsize=(9, 4.6), dpi=150)
        for qv, color in zip(range(1, 6), DIVERGING):
            m = P[(e["qt"] == qv).values].mean() * 100
            name = {1: "Group 1: most gloomy fifth", 5: "Group 5: most upbeat fifth"}.get(qv, f"Group {qv}")
            ax.plot(m.index, m.values, color=color, lw=2, label=name)
            ax.annotate(f"{qv}", (m.index[-1], m.values[-1]), xytext=(5, 0), textcoords="offset points",
                        va="center", fontsize=9, color=INK)
        ax.axhline(0, color=INK2, lw=0.8)
        ax.set_xticks(range(0, H + 1, 5 if H >= 10 else 1))
        ax.set_xlabel("Trading days after the first chance to trade (day 0)", color=INK2, fontsize=9)
        ax.set_ylabel("Return above the market, % (cumulative)", color=INK2, fontsize=9)
        ax.legend(frameon=False, fontsize=9, loc="upper left", labelcolor=INK)
        _style(ax, f"{tag}Return after the report, by language model tone",
               "Average return above the market; reports split into five equal groups within each earnings season")
        fig.tight_layout()
        fig.savefig(fig_dir / "car_by_quintile.png", facecolor=SURFACE)
        plt.close(fig)

    # 3. information coefficient by signal (stretch goal: which topic matters)
    rows = []
    for s, o in out.items():
        ics = o["ics"]
        if len(ics) > 1:
            rows.append((LABELS.get(s, s), ics.mean(), 1.96 * ics.std(ddof=1) / np.sqrt(len(ics))))
    if rows:
        rows.sort(key=lambda r: r[1])
        fig, ax = plt.subplots(figsize=(9, 0.5 * len(rows) + 1.6), dpi=150)
        y = np.arange(len(rows))
        ax.barh(y, [r[1] for r in rows], xerr=[r[2] for r in rows], color=BLUE, height=0.55,
                error_kw={"ecolor": INK2, "elinewidth": 1, "capsize": 3})
        ax.set_yticks(y, [r[0] for r in rows])
        ax.axvline(0, color=INK2, lw=0.8)
        for yi, r in zip(y, rows):
            ax.annotate(f"{r[1]:+.3f}", (max(r[1] + r[2], 0), yi), xytext=(6, 0), textcoords="offset points",
                        va="center", fontsize=9, color=INK)
        _style(ax, f"{'Demo data: ' if demo else ''}How well each score ranks the next {H} days' returns",
               "Information coefficient (rank correlation), averaged over seasons; lines show the 95% confidence range")
        ax.grid(axis="y", visible=False)
        ax.grid(axis="x", color=GRID, linewidth=0.8)
        fig.tight_layout()
        fig.savefig(fig_dir / "ic_by_signal.png", facecolor=SURFACE)
        plt.close(fig)
