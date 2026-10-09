"""README results section: the training-cutoff warning."""
import json

import pandas as pd

from earnsig import report
from earnsig.config import Paths

COLS = ["n_events", "mean_ic", "ic_t", "ic_pos_share", "seasons", "hit_rate", "top_q_car", "bottom_q_car",
        "post_cutoff_n", "post_cutoff_pooled_ic"]


def _cfg(tmp_path, monkeypatch, post_n, provider="claude_code"):
    monkeypatch.setattr(report, "ROOT", tmp_path)
    paths = Paths(tmp_path / "data", tmp_path / "results").ensure()
    row = dict(zip(COLS, [525, 0.08, 1.9, 0.8, 10, 0.55, 0.004, -0.005, post_n, 0.1 if post_n else float("nan")]))
    pd.DataFrame([{**row, "signal": "llm"}, {**row, "signal": "lm"}]).set_index("signal").to_csv(paths.results / "summary.csv")
    (paths.results / "meta.json").write_text(json.dumps({
        "n_events": 525, "n_tickers": 50, "first_event": "2023-07-07", "last_event": "2025-12-23",
        "provider": provider, "model": "Claude Code (haiku)", "benchmark": "market", "hold_days": 20,
        "cost_bps": 10, "borrow_bps": 50, "training_cutoff": "2026-06-30", "has_factors": True}))
    return {"paths": paths, "signals": {"primary": "llm", "baseline": "lm", "topics": []}}


def test_caution_when_every_document_predates_cutoff(tmp_path, monkeypatch):
    text = report.build_section(_cfg(tmp_path, monkeypatch, post_n=0))
    assert "[!CAUTION]" in text and "2026-06-30" in text
    assert "none: all documents predate the cutoff" in text


def test_note_when_some_documents_are_after_cutoff(tmp_path, monkeypatch):
    text = report.build_section(_cfg(tmp_path, monkeypatch, post_n=179))
    assert "[!CAUTION]" not in text and "346 of 525" in text and "(179)" in text


def test_demo_gets_neither(tmp_path, monkeypatch):
    text = report.build_section(_cfg(tmp_path, monkeypatch, post_n=0, provider="mock"))
    assert "[!CAUTION]" not in text and "synthetic demo" in text
