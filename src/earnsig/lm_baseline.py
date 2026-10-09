"""Loughran-McDonald (2011) dictionary baseline.

Net tone = (positive - negative) / (positive + negative), counting words from the
LM finance-specific word lists. Uses the official Master Dictionary CSV if you
place it at ``data/external/lm_master_dictionary.csv`` (download from
https://sraf.nd.edu/loughranmcdonald-master-dictionary/), otherwise the copy
bundled with the ``pysentiment2`` package.
"""
from __future__ import annotations

import logging
import re
from functools import lru_cache
from pathlib import Path

import pandas as pd

from .collect import load_text

log = logging.getLogger(__name__)
_WORD = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")
# LM's negation rule: a positive word preceded (within 3 words) by one of these counts as negative
_NEGATORS = {"NO", "NOT", "NONE", "NEITHER", "NEVER", "NOBODY"}


@lru_cache(maxsize=2)
def load_word_lists(path: str | None = None) -> tuple[frozenset, frozenset]:
    if path and Path(path).exists():
        src = path
    else:
        import pysentiment2

        src = Path(pysentiment2.__file__).parent / "static" / "LM.csv"
        log.info("Using LM dictionary bundled with pysentiment2 (%s)", src)
    d = pd.read_csv(src, keep_default_na=False)
    words = d["Word"].astype(str).str.upper()
    return frozenset(words[d["Positive"] != 0]), frozenset(words[d["Negative"] != 0])


def lm_tone(text: str, pos: frozenset, neg: frozenset) -> dict:
    toks = [t.upper() for t in _WORD.findall(text)]
    p = n = 0
    for i, t in enumerate(toks):
        if t in pos:
            if _NEGATORS.intersection(toks[max(0, i - 3):i]):
                n += 1
            else:
                p += 1
        elif t in neg:
            n += 1
    return {
        "lm": (p - n) / (p + n) if p + n else 0.0,
        "lm_pos": p,
        "lm_neg": n,
        "lm_words": len(toks),
    }


def score_lm(cfg: dict, events: pd.DataFrame) -> pd.DataFrame:
    pos, neg = load_word_lists(str(cfg["paths"].lm_dictionary))
    rows = [{"event_id": r["event_id"], **lm_tone(load_text(cfg, r["path"]), pos, neg)}
            for _, r in events.iterrows()]
    out = pd.DataFrame(rows)
    out.to_csv(cfg["paths"].lm_scores, index=False)
    return out
