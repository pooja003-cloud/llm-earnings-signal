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
        ("t-statistic of the information coefficient across seasons", lambda c: _num(g(c, "ic_t"))),
        ("p-value", lambda c: _num(g(c, "ic_p"), 2)),
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
        ("Standard error of that Sharpe ratio", lambda c: _num(g(c, "seasonal_net_sharpe_se"))),
        ("Maximum drawdown (worst fall from a peak)", lambda c: _pct(g(c, "seasonal_net_max_drawdown"))),
        ("Turnover per rebalance (replacing every holding = 400%)", lambda c: _pct(g(c, "seasonal_turnover"), 0)),
        ("Holding periods that made money", lambda c: _pct(g(c, "seasonal_periods_positive"), 0)),
        (f"**Long/short portfolio, each report held {H} trading days, after costs:** Sharpe ratio "
         "(± one standard error)",
         lambda c: f"{_num(g(c, 'event_net_sharpe'))} (± {_num(g(c, 'event_net_sharpe_se'))})"),
        ("Maximum drawdown", lambda c: _pct(g(c, "event_net_max_drawdown"))),
        ("Stocks held on a typical day, long / short", lambda c: (
            f"{_num(g(c, 'event_median_names_long'), 0)} / {_num(g(c, 'event_median_names_short'), 0)}")),
        ("Days with only one side held (other side hedged with the market fund)",
         lambda c: _pct(g(c, "event_share_one_leg"), 0)),
        (f"{H}-day portfolio: alpha against the Fama-French five factors plus momentum, yearly (t-statistic)",
         lambda c: f"{_pct(g(c, 'ffe_alpha_ann'))} ({_num(g(c, 'ffe_alpha_t'))})"),
        ("**Fama-French five factors plus momentum, seasonal portfolio:** alpha, yearly return not explained "
         "by the factors (t-statistic)", lambda c: f"{_pct(g(c, 'ff_alpha_ann'))} ({_num(g(c, 'ff_alpha_t'))})"),
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


def main_test_table(s: pd.DataFrame, cols: list[str], H: int) -> str:
    g = lambda c, k: s.loc[c].get(k, np.nan)  # noqa: E731
    crit = g(cols[0], "ic_t_crit")
    rows = [
        (f"Information coefficient: rank correlation of the score with the {H}-day return above the market",
         lambda c: _num(g(c, "mean_ic"), 3, True)),
        (f"t-statistic across {int(g(cols[0], 'seasons'))} seasons (5% significance needs about {_num(crit)} "
         f"with this few seasons)", lambda c: _num(g(c, "ic_t"))),
        ("p-value (chance of a result this strong if there were no real link; below 0.05 is the usual bar)",
         lambda c: _num(g(c, "ic_p"), 2)),
        ("Earnings seasons where the correlation was positive",
         lambda c: f"{_pct(g(c, 'ic_pos_share'), 0)} of {int(g(c, 'seasons'))}"),
        ("Portfolio rebalanced each season, after costs: Sharpe ratio (± one standard error)",
         lambda c: f"{_num(g(c, 'seasonal_net_sharpe'))} (± {_num(g(c, 'seasonal_net_sharpe_se'))})"),
    ]
    head = "| Main test | " + " | ".join(LABELS.get(c, c) for c in cols) + " |\n|---|" + "---:|" * len(cols)
    return head + "\n" + "\n".join(f"| {n} | " + " | ".join(fn(c) for c in cols) + " |" for n, fn in rows)


def main_test_verdict(s: pd.DataFrame, prim: str) -> str:
    r = s.loc[prim]
    p, ic, sharpe = r.get("ic_p", np.nan), r["mean_ic"], r.get("seasonal_net_sharpe", np.nan)
    if pd.isna(p):
        return ""
    sig = ("statistically significant at the usual 5% level" if p < 0.05 else
           f"**not** statistically significant at the usual 5% level (p = {p:.2f})")
    port = ("and the pre-chosen seasonal portfolio made little after costs"
            if pd.isna(sharpe) or sharpe < 0.5 else "and the pre-chosen seasonal portfolio was profitable after costs")
    return (f"**Verdict on the main test:** the language model's score was positively linked to later returns "
            f"(information coefficient {ic:+.3f}), but the link is {sig}, {port}. "
            "Treat the result as suggestive, not proven.")


def thin_portfolio_note(s: pd.DataFrame, prim: str, H: int) -> str:
    r = s.loc[prim]
    nl, ns, one, se = (r.get("event_median_names_long"), r.get("event_median_names_short"),
                       r.get("event_share_one_leg"), r.get("event_net_sharpe_se"))
    if pd.isna(nl):
        return ""
    return (f"**The {H}-day portfolio is very thin.** Holding each report for {H} trading days looks much better "
            f"than the main portfolio, but on a typical day it holds only {nl:.0f} stocks long and {ns:.0f} short, "
            f"and on {one:.0%} of days only one side (the other side is then the market fund). A single stock can "
            "swing the result, which is why the delay table below jumps around and why its Sharpe ratio of "
            f"{r.get('event_net_sharpe', np.nan):.2f} has a margin of error of about ±{se:.2f}.")


def noise_note(s: pd.DataFrame, cols: list[str], H: int) -> str:
    """Flag 'significant-looking' alphas: with this many numbers, some |t| > 2 are expected by chance."""
    hits = []
    for c in cols:
        for key, what in (("ff_alpha_t", "seasonal portfolio alpha"), ("ffe_alpha_t", f"{H}-day portfolio alpha")):
            t = s.loc[c].get(key, np.nan)
            if not pd.isna(t) and abs(t) > 2:
                hits.append(f"{LABELS.get(c, c).lower()} {what} (t = {t:.2f})")
    if not hits:
        return ""
    return ("_A note on noise: this page reports dozens of statistics, so one or two with a t-statistic above 2 are "
            "expected by chance alone. Treat the " + "; ".join(hits) +
            " as likely noise, not a finding; it was not part of the main test._")


def entry_delay_table(path) -> str:
    if not path.exists():
        return ""
    d = pd.read_csv(path)
    head = "| Trades start this many trading days later | " + " | ".join(str(int(x)) for x in d["delay_days"]) + " |"
    sep = "|---|" + "---:|" * len(d)
    row = "| Sharpe ratio after costs | " + " | ".join(_num(x) for x in d["sharpe_net"]) + " |"
    return "\n".join([head, sep, row])


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


def build_section(cfg: dict, include_chart: bool = True) -> str:
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
    version = (f", exact model `{meta['model_version']}`"
               if meta.get("model_version") and meta["model_version"] not in meta["model"] else "")
    lag = meta.get("seasonal_median_days_to_trade")
    parts += [f"_Sample: {meta['n_events']:,} earnings reports from {meta['n_tickers']} companies, "
              f"{meta['first_event']} to {meta['last_event']}. Scored by: {meta['model']}{version}. "
              f"Returns are measured above {bench}. Trading costs: {meta['cost_bps'] / 100:.2f}% per trade "
              f"plus {meta['borrow_bps'] / 100:.2f}% a year to borrow shares for selling short._", "",
              "### Main test (chosen before the data were analysed)", "",
              f"The `{prim}` score (future-guidance tone, with the other topics as a tie-breaker), its "
              f"information coefficient against the {H}-day return above the market, and a long/short portfolio "
              "rebalanced after each earnings season. These choices were fixed in the settings before any real "
              "report was scored.", "",
              main_test_table(s, cols, H), "", main_test_verdict(s, prim), "",
              *([f"![Growth of $1 in the long/short portfolios]({rel}/figures/cumulative_long_short.png)", ""]
                if include_chart else []),
              "### Exploratory results", "",
              "Everything below was examined after seeing the data. With this many variations, some will look "
              "good by luck, so none of it should be read as a finding on its own.", "",
              thin_portfolio_note(s, prim, H), "",
              "**Why the two portfolios differ.** The link between tone and returns shows up in the first few "
              "days after a report. Starting the same trades later shows how fast it fades"
              + (f"; the seasonal portfolio typically trades {lag:.0f} calendar days after a report, after the "
                 "effect has mostly gone." if lag else ".")
              + " These five numbers are themselves noisy, so read the pattern, not each value.", "",
              entry_delay_table(rd / "entry_delay.csv"), "",
              "#### All measures", "", stats_table(s, cols, H), "", noise_note(s, cols, H), ""]
    if not meta.get("has_factors"):
        parts += ["_Factor rows are empty: run `earnsig factors` to download the Fama-French factor data._", ""]
    parts += [f"![Return after the report, by tone group]({rel}/figures/car_by_quintile.png)", "",
              "#### Which topic matters most?", "",
              "Each report also got separate scores for what management said about future guidance, profit "
              "margins and customer demand.", "",
              topic_table(s, [prim, *cfg["signals"].get("topics", []), base]), "",
              f"![Information coefficient by score]({rel}/figures/ic_by_signal.png)", ""]
    cons = rd / "consistency.json"
    if cons.exists():
        parts += ["#### Does the model give the same answer twice?", "",
                  consistency_table(json.loads(cons.read_text())), ""]
    parts += [END]
    return "\n".join(parts)


def update_readme(cfg: dict, readme=None) -> None:
    readme = readme or ROOT / "README.md"
    text = readme.read_text()
    outside = re.sub(re.escape(START) + r".*?" + re.escape(END), "", text, flags=re.S)
    section = build_section(cfg, include_chart="figures/cumulative_long_short.png" not in outside)
    if START in text and END in text:
        text = re.sub(re.escape(START) + r".*?" + re.escape(END), lambda m: section, text, flags=re.S)
    else:
        text += "\n" + section + "\n"
    readme.write_text(text)


# ---------------------------------------------------------------- second-model comparison
CMP_START, CMP_END = "<!-- CLEAN_TEST:START -->", "<!-- CLEAN_TEST:END -->"


def _model_agreement(main_cfg: dict, other_cfg: dict) -> dict:
    from scipy import stats

    from .config import load_events

    a, b = main_cfg["paths"].llm_scores, other_cfg["paths"].llm_scores
    if not (a.exists() and b.exists()):
        return {}
    ids = set(load_events(main_cfg)["event_id"])
    m = pd.read_csv(a).merge(pd.read_csv(b), on="event_id", suffixes=("_a", "_b"))
    m = m[m["event_id"].isin(ids)]
    if len(m) < 10:
        return {}
    return {"n": len(m), "rank_corr": stats.spearmanr(m["llm_a"], m["llm_b"]).statistic,
            "same_guidance": (m["llm_guidance_a"] == m["llm_guidance_b"]).mean(),
            "within_one": ((m["llm_guidance_a"] - m["llm_guidance_b"]).abs() <= 1).mean()}


def _period_table(main_cfg: dict, other_cfg: dict, cutoffs: tuple[str, str], names: tuple[str, str], H: int) -> str:
    """Information coefficient in the periods each model could or could not have read about."""
    from scipy import stats

    target = f"car_1_{H}"
    pa = main_cfg["paths"].results / "events_scored.csv"
    pb = other_cfg["paths"].results / "events_scored.csv"
    if not (pa.exists() and pb.exists()):
        return ""
    a = pd.read_csv(pa, parse_dates=["t0"])
    b = pd.read_csv(pb, usecols=["event_id", "llm"]).rename(columns={"llm": "llm_other"})
    e = a.merge(b, on="event_id").dropna(subset=[target])
    cut_main, cut_other = pd.Timestamp(cutoffs[0]), pd.Timestamp(cutoffs[1])
    early, late = sorted([cut_main, cut_other])
    short = [n.split(" (")[0] for n in names]
    who_early = short[1] if cut_other == early else short[0]
    periods = [(f"Up to the end of {early:%B %Y} (both models may have read about these)", e[e["t0"] <= early])]
    if late > early:
        periods.append((f"{early + pd.Timedelta(days=1):%B %Y} to {late:%B %Y} (new to {who_early} only)",
                        e[(e["t0"] > early) & (e["t0"] <= late)]))
    periods.append((f"After {late:%B %Y} (new to both models)", e[e["t0"] > late]))
    head = (f"| Period | Reports | {names[0]} | {names[1]} | Finance word list |\n"
            "|---|---:|---:|---:|---:|")
    rows = []
    for label, d in periods:
        if len(d) < 10:
            rows.append(f"| {label} | {len(d)} | too few reports | too few reports | too few reports |")
            continue
        cells = []
        for col in ("llm", "llm_other", "lm"):
            r = stats.spearmanr(d[col], d[target], nan_policy="omit")
            cells.append(f"{r.statistic:+.3f} (p {r.pvalue:.2f})")
        rows.append(f"| {label} | {len(d)} | " + " | ".join(cells) + " |")
    return head + "\n" + "\n".join(rows)


def build_comparison(main_cfg: dict, other_cfg: dict) -> str:
    sa = pd.read_csv(main_cfg["paths"].results / "summary.csv", index_col=0)
    sb = pd.read_csv(other_cfg["paths"].results / "summary.csv", index_col=0)
    ma = json.loads((main_cfg["paths"].results / "meta.json").read_text())
    mb = json.loads((other_cfg["paths"].results / "meta.json").read_text())
    prim, base = main_cfg["signals"]["primary"], main_cfg["signals"]["baseline"]
    A, B, L = sa.loc[prim], sb.loc[prim], sa.loc[base]
    H = ma["hold_days"]

    def after(r):
        n = int(r["post_cutoff_n"])
        return f"{_num(r['post_cutoff_pooled_ic'], 3, True)} ({n} reports)" if n > 0 else "none: it may have read about every report"

    rows = [
        ("Training data ends", ma["training_cutoff"], mb["training_cutoff"], "not applicable"),
        ("Reports scored", f"{int(A['n_events']):,}", f"{int(B['n_events']):,}", f"{int(L['n_events']):,}"),
        ("Reports published after the model's training data ends",
         f"{int(A['post_cutoff_n'])}", f"{int(B['post_cutoff_n'])}", "not applicable"),
        (f"Information coefficient, all reports (rank correlation with the {H}-day return above the market)",
         _num(A["mean_ic"], 3, True), _num(B["mean_ic"], 3, True), _num(L["mean_ic"], 3, True)),
        (f"t-statistic across seasons (5% significance needs about {_num(A.get('ic_t_crit'))})",
         _num(A["ic_t"]), _num(B["ic_t"]), _num(L["ic_t"])),
        ("p-value", _num(A.get("ic_p"), 2), _num(B.get("ic_p"), 2), _num(L.get("ic_p"), 2)),
        ("Hit rate (top and bottom fifth that moved the predicted way)",
         _pct(A["hit_rate"]), _pct(B["hit_rate"]), _pct(L["hit_rate"])),
        (f"Sharpe ratio, each report held {H} trading days, after costs",
         _num(A.get("event_net_sharpe")), _num(B.get("event_net_sharpe")), _num(L.get("event_net_sharpe"))),
        ("Sharpe ratio, rebalanced each season, after costs",
         _num(A.get("seasonal_net_sharpe")), _num(B.get("seasonal_net_sharpe")), _num(L.get("seasonal_net_sharpe"))),
        ("**Information coefficient using only reports the model could not have read about**",
         f"**{after(A)}**", f"**{after(B)}**", "not applicable"),
    ]
    name_a, name_b = ma["model"], mb["model"]
    table = [f"| Measure | {name_a} | {name_b} | Finance word list |", "|---|---:|---:|---:|"]
    table += [f"| {r[0]} | {r[1]} | {r[2]} | {r[3]} |" for r in rows]
    parts = [CMP_START, "", "## Clean test: an older model that cannot have seen what happened", "",
             f"The main results use {name_a}, which learned from text written after every report in the sample. "
             f"Here the same reports are scored by {name_b}, whose training data ends on {mb['training_cutoff']}, so "
             "for reports published after that date it cannot have read how the stock moved. If its score still "
             "predicts returns on those reports, the effect is more likely to be real reading skill; if it does not, "
             "memory is the likelier explanation for the main result. One caution: the second model is much "
             "smaller, so a weaker result can also mean it simply reads less well.", "", *table, "",
             "**The same comparison split at each model's training cutoff** (rank correlation of each score "
             f"with the {H}-day return above the market, pooled across reports, with its p-value):", "",
             _period_table(main_cfg, other_cfg, (ma["training_cutoff"], mb["training_cutoff"]), (name_a, name_b), H), ""]
    agree = _model_agreement(main_cfg, other_cfg)
    if agree:
        verdict = (f"Their scores agree closely, so the second model reads the reports much like {name_a}; its "
                   "weak result on unseen reports then points to memory." if agree["rank_corr"] >= 0.7 else
                   f"That is well below the 0.7 or so that would show the two models read the reports alike, so the "
                   f"second model may simply be reading worse, and this test **cannot tell memory apart from weaker "
                   f"reading**. A larger model with an equally early cutoff would settle it.")
        parts += [f"**Do the two models read the reports alike?** On the same {agree['n']} reports they gave the "
                  f"identical guidance score {_pct(agree['same_guidance'], 0)} of the time and were within one step "
                  f"{_pct(agree['within_one'], 0)} of the time; the rank correlation of their scores is "
                  f"{_num(agree['rank_corr'])}. {verdict}", ""]
    parts += [f"Full tables and charts for the second model: [`{other_cfg['paths'].results.relative_to(ROOT).as_posix()}/`]"
              f"({other_cfg['paths'].results.relative_to(ROOT).as_posix()}/).", "", CMP_END]
    return "\n".join(parts)


def _cmp_markers(other_cfg: dict) -> tuple[str, str]:
    name = other_cfg["paths"].results.name
    if name == "llama":
        return CMP_START, CMP_END
    tag = re.sub(r"[^A-Za-z0-9]+", "_", name).upper()
    return f"<!-- CLEAN_TEST_{tag}:START -->", f"<!-- CLEAN_TEST_{tag}:END -->"


def update_readme_comparison(main_cfg: dict, other_cfg: dict, readme=None) -> None:
    readme = readme or ROOT / "README.md"
    text = readme.read_text()
    CMP_START, CMP_END = _cmp_markers(other_cfg)
    section = build_comparison(main_cfg, other_cfg)
    section = section.replace(globals()["CMP_START"], CMP_START).replace(globals()["CMP_END"], CMP_END)
    if CMP_START in text and CMP_END in text:
        text = re.sub(re.escape(CMP_START) + r".*?" + re.escape(CMP_END), lambda m: section, text, flags=re.S)
    elif "## How the study works" in text:
        text = text.replace("## How the study works", section + "\n\n---\n\n## How the study works", 1)
    else:
        text += "\n" + section + "\n"
    readme.write_text(text)
