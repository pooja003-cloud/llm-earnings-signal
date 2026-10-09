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
    assert "none: every report is older than the cutoff" in text


def test_note_when_some_documents_are_after_cutoff(tmp_path, monkeypatch):
    text = report.build_section(_cfg(tmp_path, monkeypatch, post_n=179))
    assert "[!CAUTION]" not in text and "346 of 525" in text and "(179)" in text


def test_demo_gets_neither(tmp_path, monkeypatch):
    text = report.build_section(_cfg(tmp_path, monkeypatch, post_n=0, provider="mock"))
    assert "[!CAUTION]" not in text and "synthetic demo" in text


def test_config_base_inheritance(tmp_path):
    from earnsig.config import load_config

    (tmp_path / "main.yaml").write_text(
        "start_date: '2021-01-01'\nend_date: '2025-12-31'\ndata_dir: d\nresults_dir: r\n"
        "llm: {provider: claude_code, workers: 2, training_cutoff: '2026-06-30'}\n")
    (tmp_path / "second.yaml").write_text(
        "base: main.yaml\nresults_dir: r2\nllm: {provider: ollama, scores_file: s2.csv}\n")
    cfg = load_config(tmp_path / "second.yaml", data_dir=str(tmp_path / "d"))
    assert cfg["llm"] == {"provider": "ollama", "workers": 2, "training_cutoff": "2026-06-30", "scores_file": "s2.csv"}
    assert cfg["results_dir"] == "r2" and cfg["paths"].llm_scores.name == "s2.csv"


def test_comparison_section(tmp_path, monkeypatch):
    main = _cfg(tmp_path, monkeypatch, post_n=0)
    other_res = tmp_path / "results" / "llama"
    other = {"paths": Paths(tmp_path / "data", other_res, "scores_llama.csv").ensure(), "signals": main["signals"]}
    row = dict(zip(COLS, [525, 0.05, 1.2, 0.6, 10, 0.53, 0.002, -0.003, 420, 0.06]))
    pd.DataFrame([{**row, "signal": "llm"}, {**row, "signal": "lm"}]).set_index("signal").to_csv(other_res / "summary.csv")
    meta = json.loads((tmp_path / "results" / "meta.json").read_text())
    (other_res / "meta.json").write_text(json.dumps({**meta, "model": "Ollama llama3.2:3b", "training_cutoff": "2023-12-31"}))
    readme = tmp_path / "README.md"
    readme.write_text("# T\n\n## How the study works\n\ntext\n")
    report.update_readme_comparison(main, other, readme)
    text = readme.read_text()
    assert text.index("CLEAN_TEST:START") < text.index("## How the study works")
    assert "Ollama llama3.2:3b" in text and "**+0.060 (420 reports)**" in text and "**none: it may have read about every report**" in text
    report.update_readme_comparison(main, other, readme)  # running twice replaces, not duplicates
    assert readme.read_text().count("CLEAN_TEST:START") == 1


def test_main_test_verdict_and_delay_table(tmp_path):
    s = pd.DataFrame({"mean_ic": [0.082], "ic_p": [0.079], "seasonal_net_sharpe": [0.21]}, index=["llm"])
    v = report.main_test_verdict(s, "llm")
    assert "**not** statistically significant" in v and "p = 0.08" in v and "suggestive" in v
    s.loc["llm", "ic_p"] = 0.01
    assert "statistically significant at the usual 5% level," in report.main_test_verdict(s, "llm")
    p = tmp_path / "entry_delay.csv"
    pd.DataFrame({"delay_days": [0, 5], "sharpe_net": [1.41, 0.45]}).to_csv(p, index=False)
    t = report.entry_delay_table(p)
    assert "| 0 | 5 |" in t and "| 1.41 | 0.45 |" in t


def test_notes_and_second_comparison_section(tmp_path, monkeypatch):
    s = pd.DataFrame({"event_median_names_long": [2.0], "event_median_names_short": [3.0],
                      "event_share_one_leg": [0.23], "event_net_sharpe_se": [0.92], "event_net_sharpe": [1.41],
                      "ff_alpha_t": [0.2], "ffe_alpha_t": [-2.32]}, index=["lm"])
    assert "2 stocks long and 3 short" in report.thin_portfolio_note(s, "lm", 20)
    assert "t = -2.32" in report.noise_note(s, ["lm"], 20)
    s.loc["lm", "ffe_alpha_t"] = 1.0
    assert report.noise_note(s, ["lm"], 20) == ""
    other = {"paths": Paths(tmp_path / "data", tmp_path / "results" / "llama8b")}
    assert report._cmp_markers(other)[0] == "<!-- CLEAN_TEST_LLAMA8B:START -->"
