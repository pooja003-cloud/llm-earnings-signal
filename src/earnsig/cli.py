"""Command-line entry point: ``earnsig <step>`` (or ``python -m earnsig <step>``)."""
from __future__ import annotations

import argparse
import json
import logging
import sys

import pandas as pd

from .config import load_config, load_events, load_universe

log = logging.getLogger("earnsig")


def _events(cfg, limit=None):
    ev = load_events(cfg)
    return ev.head(limit) if limit else ev


def cmd_collect(cfg, a):
    from .collect import collect_local, collect_sec

    if a.source == "local":
        if not a.manifest:
            sys.exit("--manifest is required with --source local")
        collect_local(cfg, a.manifest)
    else:
        collect_sec(cfg, load_universe(cfg), a.tickers)


def cmd_prices(cfg, a):
    from .market_data import download_prices

    download_prices(cfg, load_universe(cfg))


def cmd_factors(cfg, a):
    from .market_data import download_factors

    download_factors(cfg)


def cmd_score(cfg, a):
    from .llm_score import build_llm_signals, make_scorer, score_events

    ev = _events(cfg, a.limit)
    if a.dry_run:
        chars = sum(min(len((cfg["paths"].data / p).read_text()), cfg["llm"]["max_chars"]) for p in ev["path"])
        print(f"{len(ev)} documents, ~{chars / 4 / 1e6:.2f}M input tokens (+~1k prompt tokens each).")
        prov = cfg["llm"]["provider"]
        if prov == "claude_code":
            print("Claude Code: this counts against your Pro usage limits. Scoring stops when you hit a limit; "
                  "re-run later to resume. Try `earnsig score --limit 20` first.")
        elif prov == "anthropic":
            print(f"Check current API pricing for {cfg['llm']['model']} before running.")
        return
    raw = score_events(cfg, ev, make_scorer(cfg), rep=1, universe=load_universe(cfg))
    sig = build_llm_signals(raw, ev)
    sig.to_csv(cfg["paths"].llm_scores, index=False)
    log.info("Wrote %d LLM scores to %s", len(sig), cfg["paths"].llm_scores)


def cmd_consistency(cfg, a):
    from .llm_score import consistency, make_scorer

    res = consistency(cfg, _events(cfg), make_scorer(cfg), universe=load_universe(cfg), n=a.n)
    (cfg["paths"].results / "consistency.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


def cmd_baseline(cfg, a):
    from .lm_baseline import score_lm

    score_lm(cfg, _events(cfg))


def cmd_backtest(cfg, a):
    from .analysis import run_backtest

    out = run_backtest(cfg)
    cols = ["n_events", "mean_ic", "ic_t", "hit_rate", "seasonal_net_sharpe", "seasonal_net_max_drawdown",
            "event_net_sharpe", "ff_alpha_ann", "ff_alpha_t"]
    with pd.option_context("display.width", 160, "display.float_format", "{:.3f}".format):
        print(out["summary"][[c for c in cols if c in out["summary"]]])


def cmd_report(cfg, a):
    from .report import update_readme

    update_readme(cfg)
    log.info("README.md results section updated")


def cmd_compare(cfg, a):
    from .report import update_readme_comparison

    main = load_config(a.main)
    if main["paths"].results == cfg["paths"].results:
        sys.exit("Run compare with the second model's config, e.g. "
                 "`earnsig --config config/llama.yaml compare`.")
    update_readme_comparison(main, cfg)
    log.info("README.md comparison section updated")


def cmd_demo(cfg, a):
    from .synthetic import generate

    cfg = load_config(a.config, data_dir="data/demo", results_dir="results/demo")
    cfg["universe_file"] = str(cfg["paths"].data / "universe.csv")
    cfg["analysis_start"] = None  # the demo always uses its full synthetic period
    cfg["llm"].update(provider="mock", model="mock (synthetic demo)", training_cutoff="2024-01-01")
    log.info("Generating synthetic data in %s", cfg["paths"].data)
    if cfg["paths"].llm_cache.exists():
        cfg["paths"].llm_cache.unlink()
    generate(cfg)
    for step in (cmd_score, cmd_consistency, cmd_baseline, cmd_backtest):
        step(cfg, argparse.Namespace(limit=None, dry_run=False, n=None))
    if a.readme:
        cmd_report(cfg, a)


def cmd_all(cfg, a):
    for step in (cmd_prices, cmd_factors, cmd_collect, cmd_score, cmd_consistency, cmd_baseline, cmd_backtest, cmd_report):
        log.info("== %s", step.__name__.removeprefix("cmd_"))
        step(cfg, argparse.Namespace(source="sec", manifest=None, tickers=None, limit=None, dry_run=False, n=None))


def main(argv=None):
    p = argparse.ArgumentParser(prog="earnsig", description="LLM earnings tone -> investment signal")
    p.add_argument("--config", default=None, help="path to config.yaml")
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("collect", help="download 8-K earnings releases from SEC EDGAR (or register local transcripts)")
    c.add_argument("--source", choices=["sec", "local"], default="sec")
    c.add_argument("--manifest", help="CSV with ticker,published_at,path (for --source local)")
    c.add_argument("--tickers", nargs="*", help="subset of tickers")
    sub.add_parser("prices", help="download adjusted prices with yfinance")
    sub.add_parser("factors", help="download Fama-French 5 factors + momentum")
    s = sub.add_parser("score", help="score documents with the LLM")
    s.add_argument("--limit", type=int, help="score only the first N documents (for testing)")
    s.add_argument("--dry-run", action="store_true", help="estimate token count only")
    k = sub.add_parser("consistency", help="score a sample twice and measure agreement")
    k.add_argument("--n", type=int, default=None)
    sub.add_parser("baseline", help="Loughran-McDonald dictionary scores")
    sub.add_parser("backtest", help="event study, portfolios, statistics, figures")
    sub.add_parser("report", help="write results into README.md")
    k2 = sub.add_parser("compare", help="add a README section comparing this config's model with the main one")
    k2.add_argument("--main", default=None, help="config of the main results (default config/config.yaml)")
    d = sub.add_parser("demo", help="run everything offline on synthetic data")
    d.add_argument("--readme", action="store_true", help="also write the demo results into README.md")
    sub.add_parser("all", help="run every real step in order")
    a = p.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if a.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s", datefmt="%H:%M:%S")
    for noisy in ("httpx", "anthropic", "urllib3", "yfinance", "matplotlib"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    cfg = load_config(a.config)
    globals()[f"cmd_{a.cmd}"](cfg, a)


if __name__ == "__main__":
    main()
