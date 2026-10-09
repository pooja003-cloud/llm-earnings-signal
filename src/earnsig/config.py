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
        return self.data / "scores_llm.csv"

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
    with open(path) as f:
        cfg = _expand(yaml.safe_load(f))
    if data_dir:
        cfg["data_dir"] = data_dir
    if results_dir:
        cfg["results_dir"] = results_dir
    cfg["paths"] = Paths(_abs(cfg["data_dir"]), _abs(cfg["results_dir"])).ensure()
    return cfg


def _abs(p: str) -> Path:
    p = Path(p)
    return p if p.is_absolute() else ROOT / p


def load_universe(cfg: dict) -> pd.DataFrame:
    return pd.read_csv(_abs(cfg["universe_file"]))
