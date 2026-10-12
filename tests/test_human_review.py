"""Hand-scored reading check: sample files give nothing away, and both models are compared with the human."""
import numpy as np
import pandas as pd
import pytest

from earnsig import human_review
from earnsig.config import Paths


def _cfg(tmp_path, results="res"):
    paths = Paths(tmp_path / "data", tmp_path / results).ensure()
    return {"paths": paths, "start_date": "2021-01-01", "end_date": "2026-08-31", "analysis_start": "2023-07-01",
            "universe_file": str(tmp_path / "universe.csv"),
            "llm": {"max_chars": 1000, "anonymize": True, "provider": "mock"}}


def _setup(tmp_path, n=40):
    cfg = _cfg(tmp_path)
    pd.DataFrame({"ticker": ["ACME"], "name": ["Acme Corp"]}).to_csv(tmp_path / "universe.csv", index=False)
    rows = []
    for i in range(n):
        p = cfg["paths"].docs / "ACME" / f"{i}.txt"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(f"Acme Corp (ACME) raised guidance on March 3, 2025. Report {i}.")
        rows.append({"event_id": f"ACME_{i}", "ticker": "ACME", "published_at": "2024-05-01 07:00",
                     "path": str(p.relative_to(cfg["paths"].data))})
    pd.DataFrame(rows).to_csv(cfg["paths"].events, index=False)
    return cfg


def test_sample_hides_identity_and_does_not_overwrite_scores(tmp_path):
    cfg = _setup(tmp_path)
    labels = tmp_path / "labels" / "human_labels.csv"
    human_review.make_sample(cfg, n=30, labels_path=labels)
    review = cfg["paths"].data / "human_review"
    text = (review / "01.txt").read_text()
    assert "Acme" not in text and "ACME" not in text and "2025" not in text
    sheet = pd.read_csv(labels)
    assert len(sheet) == 30 and list(sheet.columns) == ["number", "guidance_tone", "notes"]
    assert "event_id" not in sheet.columns  # the sheet itself names no company
    assert "+2" in (review / "RUBRIC.txt").read_text()
    sheet.loc[0, "guidance_tone"] = 1
    sheet.to_csv(labels, index=False)
    with pytest.raises(SystemExit):
        human_review.make_sample(cfg, n=30, labels_path=labels)


def test_evaluate_finds_the_model_that_reads_like_the_human(tmp_path):
    rng = np.random.default_rng(3)
    main, other = _cfg(tmp_path, "res"), _cfg(tmp_path, "res_other")
    n = 30
    human = rng.integers(-2, 3, n)
    ids = [f"e{i}" for i in range(n)]
    key = main["paths"].data / "human_review" / "key.csv"
    key.parent.mkdir(parents=True)
    pd.DataFrame({"number": range(1, n + 1), "event_id": ids}).to_csv(key, index=False)
    labels = tmp_path / "labels.csv"
    pd.DataFrame({"number": range(1, n + 1), "guidance_tone": human, "notes": ""}).to_csv(labels, index=False)
    pd.DataFrame({"event_id": ids, "llm_guidance": human}).to_csv(main["paths"].results / "events_scored.csv", index=False)
    pd.DataFrame({"event_id": ids, "llm_guidance": rng.integers(-2, 3, n)}).to_csv(
        other["paths"].results / "events_scored.csv", index=False)
    res = human_review.evaluate(main, other, labels_path=labels, n_boot=300)
    assert res["main"]["exact"] == 1.0 and res["main"]["kappa"] == pytest.approx(1.0)
    assert res["kappa_diff_ci"][0] > 0
    text = human_review.section(res, ("Claude", "Llama"))
    assert "Claude agrees with the human scores clearly more than Llama" in text
    assert text.startswith(human_review.START) and text.endswith(human_review.END)


def test_readme_section_is_inserted_before_next_steps(tmp_path, monkeypatch):
    readme = tmp_path / "README.md"
    readme.write_text("intro\n\n## Next steps\n\nmore\n")
    monkeypatch.setattr(human_review, "evaluate", lambda a, b: {
        "main": {"n": 30, "exact": .6, "within_one": .9, "spearman": .7, "kappa": .7},
        "other": {"n": 30, "exact": .5, "within_one": .9, "spearman": .6, "kappa": .65},
        "models": {"n": 30, "exact": .4, "within_one": .8, "spearman": .5, "kappa": .5},
        "kappa_diff": .05, "kappa_diff_ci": (-.1, .2)})
    cfg = {"llm": {"provider": "mock", "model": "mock"}}
    human_review.update_readme(cfg, cfg, readme=readme)
    t = readme.read_text()
    assert t.index("Reading check") < t.index("## Next steps")
    human_review.update_readme(cfg, cfg, readme=readme)
    assert readme.read_text().count("Reading check") == 1
