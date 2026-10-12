"""Human check of reading quality: you score a sample of reports, then both models are compared with you.

The clean test cannot tell "the older model has no memory of what happened" apart from
"the older model reads worse". Your own scores settle the second part: if Llama disagrees
with you much more than Claude does, weaker reading is the likelier explanation.

Steps:
  earnsig review-sample          # writes the reports to read and an empty score sheet
  (fill in labels/human_labels.csv: guidance_tone from -2 to +2 for each report)
  earnsig review-eval            # compares Claude and Llama with your scores, updates README.md
"""
from __future__ import annotations

import logging
import re

import numpy as np
import pandas as pd
from scipy import stats

from .collect import load_text
from .config import ROOT, load_config, load_events, load_universe
from .llm_score import SYSTEM_PROMPT, anonymize, quadratic_kappa

log = logging.getLogger(__name__)

START, END = "<!-- HUMAN_EVAL:START -->", "<!-- HUMAN_EVAL:END -->"
LEVELS = (-2, -1, 0, 1, 2)


def _paths(cfg):
    review = cfg["paths"].data / "human_review"
    return review, review / "key.csv", ROOT / "labels" / "human_labels.csv"


def make_sample(cfg: dict, n: int = 30, seed: int = 11, labels_path=None) -> pd.DataFrame:
    """Pick n random reports and save them exactly as the models saw them (names and dates hidden).

    Files are numbered 01, 02, ... so the file name gives nothing away. The key that links
    numbers to reports stays in the data folder, which is not uploaded.
    """
    review, key_path, default_labels = _paths(cfg)
    labels_path = labels_path or default_labels
    if labels_path.exists() and pd.read_csv(labels_path)["guidance_tone"].notna().any():
        raise SystemExit(f"{labels_path} already has scores in it; not overwriting. Delete it to start again.")
    ev = load_events(cfg)
    uni = load_universe(cfg)
    names = dict(zip(uni["ticker"], uni["name"]))
    sample = ev.sample(min(n, len(ev)), random_state=seed).reset_index(drop=True)
    review.mkdir(parents=True, exist_ok=True)
    llm = cfg["llm"]
    for i, row in sample.iterrows():
        text = load_text(cfg, row["path"])[: llm["max_chars"]]
        if llm.get("anonymize", True):
            text = anonymize(text, row["ticker"], names.get(row["ticker"]))
        (review / f"{i + 1:02d}.txt").write_text(text)
    rubric = SYSTEM_PROMPT.split("Scale for every score:")[1].split("Fields:")[0]
    (review / "RUBRIC.txt").write_text(
        "Score each report's FUTURE GUIDANCE (forward-looking outlook for revenue, earnings or other targets)\n"
        "on this scale, the same one the models used:\n" + rubric +
        "\nRead only the report. Do not look up the company or its share price.\n")
    pd.DataFrame({"number": range(1, len(sample) + 1), "event_id": sample["event_id"]}).to_csv(key_path, index=False)
    labels_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"number": range(1, len(sample) + 1), "guidance_tone": "", "notes": ""}).to_csv(labels_path, index=False)
    log.info("Wrote %d reports to %s and an empty score sheet to %s", len(sample), review, labels_path)
    return sample


def _agreement(a: np.ndarray, b: np.ndarray) -> dict:
    return {"n": len(a), "exact": float((a == b).mean()), "within_one": float((abs(a - b) <= 1).mean()),
            "spearman": float(stats.spearmanr(a, b).statistic) if len(set(a)) > 1 and len(set(b)) > 1 else np.nan,
            "kappa": quadratic_kappa(a, b, LEVELS)}


def evaluate(main_cfg: dict, other_cfg: dict, labels_path=None, n_boot: int = 2000, seed: int = 0) -> dict:
    _, key_path, default_labels = _paths(main_cfg)
    labels = pd.read_csv(labels_path or default_labels)
    labels = labels[pd.to_numeric(labels["guidance_tone"], errors="coerce").notna()].copy()
    labels["human"] = labels["guidance_tone"].astype(int)
    if not labels["human"].isin(LEVELS).all():
        raise SystemExit("guidance_tone must be one of -2, -1, 0, 1, 2")
    df = labels.merge(pd.read_csv(key_path), on="number")
    a = pd.read_csv(main_cfg["paths"].results / "events_scored.csv", usecols=["event_id", "llm_guidance"])
    b = pd.read_csv(other_cfg["paths"].results / "events_scored.csv", usecols=["event_id", "llm_guidance"])
    df = df.merge(a.rename(columns={"llm_guidance": "main"}), on="event_id")
    df = df.merge(b.rename(columns={"llm_guidance": "other"}), on="event_id").dropna(subset=["main", "other"])
    h, m, o = df["human"].to_numpy(), df["main"].astype(int).to_numpy(), df["other"].astype(int).to_numpy()
    res = {"main": _agreement(h, m), "other": _agreement(h, o), "models": _agreement(m, o)}
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n_boot):
        i = rng.integers(0, len(h), len(h))
        d = quadratic_kappa(h[i], m[i], LEVELS) - quadratic_kappa(h[i], o[i], LEVELS)
        if not np.isnan(d):
            diffs.append(d)
    res["kappa_diff"] = res["main"]["kappa"] - res["other"]["kappa"]
    res["kappa_diff_ci"] = (float(np.percentile(diffs, 5)), float(np.percentile(diffs, 95))) if diffs else (np.nan, np.nan)
    return res


def verdict(res: dict, names: tuple[str, str]) -> str:
    lo, hi = res["kappa_diff_ci"]
    a, b = names
    if lo > 0:
        return (f"**{a} agrees with the human scores clearly more than {b} does**, so weaker reading is a likely "
                f"part of why {b} found nothing in the clean test. The clean test therefore says little about memory.")
    if hi < 0:
        return (f"**{b} agrees with the human scores more than {a} does**, so weaker reading does not explain {b}'s "
                "null result. Memory becomes the likelier explanation for the main result.")
    if abs(res["kappa_diff"]) < 0.1:
        return (f"**Both models agree with the human scores about equally well.** Weaker reading is then an "
                f"unlikely explanation for {b}'s null result, which makes memory the likelier explanation for the "
                f"main result. With {res['main']['n']} reports this is suggestive, not proof.")
    lean = a if res["kappa_diff"] > 0 else b
    return (f"**{lean} agrees somewhat more with the human scores, but the difference is within the margin of error** "
            f"for {res['main']['n']} reports, so this check cannot settle whether {b} reads worse.")


def section(res: dict, names: tuple[str, str]) -> str:
    a, b = names
    lo, hi = res["kappa_diff_ci"]

    def row(label, r):
        return (f"| {label} | {r['n']} | {r['exact']:.0%} | {r['within_one']:.0%} | "
                f"{r['spearman']:.2f} | {r['kappa']:.2f} |")

    return "\n".join([
        START, "",
        "## Reading check: both models compared with a human",
        "",
        f"The clean test cannot tell whether {b} found nothing because it has no memory of what happened or "
        "because it reads less well. To check reading quality directly, the author scored "
        f"{res['main']['n']} randomly chosen reports by hand on the same −2 to +2 guidance scale, seeing the "
        "same text the models saw (names and dates hidden), without looking at the models' scores or the "
        "stock moves.",
        "",
        "| Comparison | Reports | Same score | Within one step | Rank correlation | "
        "Agreement beyond chance (weighted kappa, 1 = perfect) |",
        "|---|---:|---:|---:|---:|---:|",
        row(f"{a} versus the human", res["main"]),
        row(f"{b} versus the human", res["other"]),
        row(f"{a} versus {b}", res["models"]),
        "",
        f"Difference in agreement with the human ({a} minus {b}, weighted kappa): {res['kappa_diff']:+.2f}, "
        f"90% bootstrap interval {lo:+.2f} to {hi:+.2f}.",
        "",
        verdict(res, names),
        "",
        END,
    ])


def update_readme(main_cfg: dict, other_cfg: dict, readme=None) -> str:
    from .llm_score import scorer_label

    names = (scorer_label(main_cfg).split(" (")[0], scorer_label(other_cfg).split(" (")[0])
    block = section(evaluate(main_cfg, other_cfg), names)
    readme = readme or ROOT / "README.md"
    text = readme.read_text()
    if START in text:
        text = re.sub(re.escape(START) + r".*?" + re.escape(END), lambda _: block, text, flags=re.S)
    elif "## Next steps" in text:
        text = text.replace("## Next steps", block + "\n\n---\n\n## Next steps", 1)
    else:
        text += "\n\n" + block + "\n"
    readme.write_text(text)
    return block


def default_other(cfg_path: str | None = None) -> dict:
    return load_config(cfg_path or str(ROOT / "config" / "llama.yaml"))
