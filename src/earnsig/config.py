"""Configuration loading and project paths."""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]

try:  # optional: load .env if python-dotenv is installed
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
except ImportError:  # pragma: no cover
    pass

_ENV = re.compile(r"\$\{(\w+)\}")


def _expand(obj):
    if isinstance(obj, str):
        return _ENV.sub(lambda m: os.environ.get(m.group(1), ""), obj)
    if isinstance(obj, dict):
        return {k: _expand(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_expand(v) for v in obj]
    return obj


@dataclass
class Paths:
    data: Path
    results: Path
    scores_name: str = "scores_llm.csv"

    @property
    def docs(self) -> Path:
        return self.data / "raw" / "docs"

    @property
    def events(self) -> Path:
        return self.data / "events.csv"

    @property
    def prices(self) -> Path:
        return self.data / "prices.parquet"

    @property
    def factors(self) -> Path:
        return self.data / "factors.csv"

    @property
    def llm_cache(self) -> Path:
        return self.data / "llm_cache.jsonl"

    @property
    def llm_scores(self) -> Path:
        return self.data / self.scores_name

    @property
    def lm_scores(self) -> Path:
        return self.data / "scores_lm.csv"

    @property
    def lm_dictionary(self) -> Path:
        return self.data / "external" / "lm_master_dictionary.csv"

    def ensure(self) -> "Paths":
        for p in (self.data, self.results, self.docs, self.results / "figures"):
            p.mkdir(parents=True, exist_ok=True)
        return self


def load_config(path: str | Path | None = None, data_dir: str | None = None,
                results_dir: str | None = None) -> dict:
    path = Path(path) if path else ROOT / "config" / "config.yaml"
    cfg = _expand(_read_with_base(path))
    if data_dir:
        cfg["data_dir"] = data_dir
    if results_dir:
        cfg["results_dir"] = results_dir
    cfg["paths"] = Paths(_abs(cfg["data_dir"]), _abs(cfg["results_dir"]),
                         cfg["llm"].get("scores_file", "scores_llm.csv")).ensure()
    cfg["config_path"] = str(path)
    return cfg


def _read_with_base(path: Path) -> dict:
    """Read a YAML config. A ``base:`` key names another config file (relative to this one)
    whose settings are loaded first; this file's settings then override them."""
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    base = cfg.pop("base", None)
    if base:
        return _merge(_read_with_base((Path(path).parent / base).resolve()), cfg)
    return cfg


def _merge(base: dict, over: dict) -> dict:
    out = dict(base)
    for k, v in over.items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def _abs(p: str) -> Path:
    p = Path(p)
    return p if p.is_absolute() else ROOT / p


def load_universe(cfg: dict) -> pd.DataFrame:
    return pd.read_csv(_abs(cfg["universe_file"]))


def load_events(cfg: dict) -> pd.DataFrame:
    """Collected documents inside the analysis window (``analysis_start`` .. ``end_date``).

    Collection can cover a longer period; scoring, the baseline and the backtest
    all use only this window, so every signal is compared on the same documents.
    """
    ev = pd.read_csv(cfg["paths"].events, parse_dates=["published_at"])
    start = pd.Timestamp(cfg.get("analysis_start") or cfg["start_date"])
    end = pd.Timestamp(cfg["end_date"]) + pd.Timedelta(days=1)
    return ev[(ev["published_at"] >= start) & (ev["published_at"] < end)].reset_index(drop=True)
