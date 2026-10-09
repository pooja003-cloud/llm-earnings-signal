"""Score earnings documents with an LLM using a fixed rubric and JSON-schema output.

Every response is cached in ``data/llm_cache.jsonl`` keyed by
(event, model, prompt version, repetition), so re-runs cost nothing and the
exact model output behind every number in the README is on disk.
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
from scipy import stats

from .collect import load_text

log = logging.getLogger(__name__)

SCORE = {"type": "integer", "enum": [-2, -1, 0, 1, 2]}
TOPICS = ("guidance", "margins", "demand")

SCHEMA = {
    "type": "object",
    "properties": {
        "guidance_tone": SCORE,
        "guidance_mentioned": {"type": "boolean"},
        "margins_tone": SCORE,
        "margins_mentioned": {"type": "boolean"},
        "demand_tone": SCORE,
        "demand_mentioned": {"type": "boolean"},
        "overall_tone": SCORE,
        "reason": {"type": "string", "description": "One line, max 25 words, on the guidance score."},
    },
    "required": ["guidance_tone", "guidance_mentioned", "margins_tone", "margins_mentioned",
                 "demand_tone", "demand_mentioned", "overall_tone", "reason"],
    "additionalProperties": False,
}

SYSTEM_PROMPT = """You are an equity research analyst. You read one company earnings document
(a press release or call transcript) and rate management's tone on a fixed scale.
Rate only what the document says. Ignore anything you may know about the company,
its later results or its stock price.

Scale for every score:
 +2 = clearly raised / strongly positive (e.g. guidance raised above prior range)
 +1 = mildly positive (e.g. guidance reaffirmed at the high end, upbeat language)
  0 = neutral, mixed, or simply reaffirmed
 -1 = mildly negative (e.g. cautious language, guidance at the low end, headwinds flagged)
 -2 = clearly cut / strongly negative (e.g. guidance lowered or withdrawn)

Fields:
- guidance_tone: forward-looking outlook for revenue, earnings or other targets.
- margins_tone: gross/operating margin trend and cost commentary.
- demand_tone: customer demand, orders, bookings, volumes, pipeline.
- overall_tone: the document as a whole.
- *_mentioned: false if the document says nothing about that topic (then score 0).
- reason: one line justifying guidance_tone, citing the key phrase or fact."""

USER_TEMPLATE = "Rate this earnings document.\n\n<document>\n{text}\n</document>"


# ---------------------------------------------------------------- anonymization
_MONTHS = r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|June?|July?|Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\.?"


def anonymize(text: str, ticker: str, name: str | None) -> str:
    """Mask identity and calendar clues that help a model recall what happened next.

    This lowers (but cannot remove) memorization: distinctive products or
    numbers can still identify a company.
    """
    if name:
        core = re.sub(r"\b(Inc|Corp|Corporation|Company|Co|plc|Holdings)\.?$", "", name).strip()
        for n in {name, core}:
            if len(n) > 2:
                text = re.sub(re.escape(n) + r"(?:,? (?:Inc|Corp|Corporation|Company|Co|plc)\.?)?",
                              "the Company", text, flags=re.I)
    text = re.sub(rf"(?<![A-Za-z]){re.escape(ticker)}(?![A-Za-z])", "TICKER", text)
    text = re.sub(rf"\b{_MONTHS}\s+\d{{1,2}},?\s+(?:19|20)\d{{2}}\b", "[DATE]", text)
    text = re.sub(r"\b(?:19|20)\d{2}\b", "[YEAR]", text)
    text = re.sub(r"\b(?:fiscal|FY)\s?'?\d{2}\b", "fiscal [YEAR]", text, flags=re.I)
    return text


# ---------------------------------------------------------------- scorers
def validate(d: dict) -> dict:
    out = {}
    for k in ("guidance", "margins", "demand", "overall"):
        v = int(d[f"{k}_tone"])
        if v not in (-2, -1, 0, 1, 2):
            raise ValueError(f"{k}_tone out of range: {v}")
        out[f"{k}_tone"] = v
    for k in TOPICS:
        out[f"{k}_mentioned"] = bool(d[f"{k}_mentioned"])
    out["reason"] = str(d.get("reason", ""))[:300]
    return out


class AnthropicScorer:
    def __init__(self, model: str, max_tokens: int = 600):
        import anthropic

        self.client = anthropic.Anthropic(max_retries=6)
        self.model = model
        self.max_tokens = max_tokens

    def score(self, text: str, row: pd.Series | None = None) -> dict:
        resp = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": USER_TEMPLATE.format(text=text)}],
            output_config={"format": {"type": "json_schema", "schema": SCHEMA}},
        )
        if resp.stop_reason in ("refusal", "max_tokens"):
            raise RuntimeError(f"stop_reason={resp.stop_reason}")
        raw = next(b.text for b in resp.content if b.type == "text")
        out = validate(json.loads(raw))
        out["input_tokens"] = resp.usage.input_tokens
        out["output_tokens"] = resp.usage.output_tokens
        return out


class UsageLimitReached(RuntimeError):
    """The subscription's usage limit is exhausted; stop and resume later from the cache."""


class ClaudeCodeScorer:
    """Score through the Claude Code CLI (``claude -p``) using a Claude Pro/Max login.

    No API key needed: run ``claude`` once and sign in with your subscription.
    Usage counts against the same limits as the Claude apps, so a full run may
    take several limit windows. When the limit is hit, scoring stops cleanly;
    re-run ``earnsig score`` later and it resumes from the cache.
    """

    def __init__(self, model: str = "haiku", system_prompt_path=None, timeout: int = 300):
        import shutil

        self.exe = shutil.which("claude")
        if self.exe is None:
            raise RuntimeError("`claude` not found. Install Claude Code (https://code.claude.com) and run "
                               "`claude` once to sign in with your Pro account.")
        self.cli_model = model
        self.model = f"claude-code:{model}"
        self.timeout = timeout
        self.prompt_file = system_prompt_path
        self.prompt_file.parent.mkdir(parents=True, exist_ok=True)
        self.prompt_file.write_text(SYSTEM_PROMPT)

    def command(self) -> list[str]:
        return [self.exe, "-p", "--output-format", "json", "--json-schema", json.dumps(SCHEMA),
                "--model", self.cli_model, "--system-prompt-file", str(self.prompt_file),
                "--tools", "", "--strict-mcp-config", "--disable-slash-commands",
                "--no-session-persistence", "--max-turns", "3"]

    def score(self, text: str, row: pd.Series | None = None) -> dict:
        import subprocess

        proc = subprocess.run(self.command(), input=USER_TEMPLATE.format(text=text),
                              capture_output=True, text=True, timeout=self.timeout)
        try:
            out = json.loads(proc.stdout)
        except json.JSONDecodeError:
            msg = (proc.stdout + proc.stderr).strip()[:300]
            if "limit" in msg.lower():
                raise UsageLimitReached(msg)
            raise RuntimeError(f"claude exited {proc.returncode}: {msg}")
        if out.get("is_error") or not out.get("structured_output"):
            msg = str(out.get("result", ""))[:300]
            if "limit" in msg.lower() or out.get("subtype") == "error_rate_limit":
                raise UsageLimitReached(msg)
            raise RuntimeError(f"claude returned no structured output: {msg}")
        res = validate(out["structured_output"])
        usage = out.get("usage") or {}
        res["input_tokens"] = sum(usage.get(k) or 0 for k in
                                  ("input_tokens", "cache_read_input_tokens", "cache_creation_input_tokens"))
        res["output_tokens"] = usage.get("output_tokens")
        return res


class OllamaScorer:
    """Score with a free local model through Ollama (https://ollama.com).

    Ollama constrains the output to the JSON schema. A model with an old,
    published training cutoff (e.g. Llama 3.1: December 2023) gives a cleaner
    out-of-sample test: it cannot have read about 2024-2025 stock moves.
    """

    def __init__(self, model: str = "llama3.1:8b", url: str = "http://localhost:11434",
                 num_ctx: int = 16384, timeout: int = 600):
        self.ollama_model = model
        self.model = f"ollama:{model}"
        self.url = url.rstrip("/")
        self.num_ctx = num_ctx
        self.timeout = timeout

    def score(self, text: str, row: pd.Series | None = None) -> dict:
        import requests

        r = requests.post(f"{self.url}/api/chat", timeout=self.timeout, json={
            "model": self.ollama_model,
            "messages": [{"role": "system", "content": SYSTEM_PROMPT},
                         {"role": "user", "content": USER_TEMPLATE.format(text=text)}],
            "format": SCHEMA,
            "stream": False,
            "options": {"temperature": 0, "num_ctx": self.num_ctx},
        })
        r.raise_for_status()
        body = r.json()
        res = validate(json.loads(body["message"]["content"]))
        res["input_tokens"] = body.get("prompt_eval_count")
        res["output_tokens"] = body.get("eval_count")
        return res


class MockScorer:
    """Offline stand-in used only by the synthetic demo.

    It reads the hidden ``latent_tone`` that the demo generator planted and adds
    noise, imitating a reasonably (not perfectly) accurate reader. Its output
    says nothing about how a real LLM performs.
    """

    model = "mock"

    def __init__(self, noise: float = 0.8, seed: int = 0):
        self.noise = noise
        self.seed = seed

    def score(self, text: str, row: pd.Series | None = None) -> dict:
        # the main noise is fixed per document; a little extra varies by repetition,
        # imitating run-to-run variation of a real model
        base = np.random.default_rng(int(hashlib.sha1(f"{row['event_id']}|{self.seed}".encode()).hexdigest()[:8], 16))
        rep = np.random.default_rng(int(hashlib.sha1(f"{row['event_id']}|{row.get('_rep', 1)}".encode()).hexdigest()[:8], 16))

        class _R:  # mixes the two noise sources
            def normal(self, mu, sd):
                return base.normal(mu, sd * 0.9) + rep.normal(0, sd * 0.45)
        rng = _R()
        z = float(row["latent_tone"])
        clip = lambda x: int(np.clip(np.round(x), -2, 2))  # noqa: E731
        return {
            "guidance_tone": clip(1.6 * z + rng.normal(0, self.noise)),
            "margins_tone": clip(0.8 * z + rng.normal(0, 1.0)),
            "demand_tone": clip(1.2 * z + rng.normal(0, 0.9)),
            "overall_tone": clip(1.3 * z + rng.normal(0, 0.8)),
            "guidance_mentioned": True, "margins_mentioned": True, "demand_mentioned": True,
            "reason": "mock score (synthetic demo)",
        }


def scorer_label(cfg: dict) -> str:
    llm = cfg["llm"]
    return {"claude_code": f"Claude Code ({llm.get('claude_code_model', 'haiku')})",
            "ollama": f"Ollama {llm.get('ollama_model', '')}",
            }.get(llm["provider"], llm["model"])


def make_scorer(cfg: dict):
    llm = cfg["llm"]
    if llm["provider"] == "mock":
        return MockScorer()
    if llm["provider"] == "claude_code":
        return ClaudeCodeScorer(llm.get("claude_code_model", "haiku"),
                                cfg["paths"].data / "prompts" / "system_prompt.txt")
    if llm["provider"] == "ollama":
        return OllamaScorer(llm.get("ollama_model", "llama3.1:8b"), llm.get("ollama_url", "http://localhost:11434"),
                            llm.get("ollama_num_ctx", 16384))
    if llm["provider"] == "anthropic":
        return AnthropicScorer(llm["model"], llm.get("max_tokens", 600))
    raise ValueError(f"unknown provider {llm['provider']} (use claude_code, ollama, anthropic or mock)")


# ---------------------------------------------------------------- caching + batch scoring
def _key(event_id: str, model: str, prompt_version: str, rep: int) -> str:
    return hashlib.sha1(f"{event_id}|{model}|{prompt_version}|{rep}".encode()).hexdigest()


def read_cache(path) -> dict[str, dict]:
    cache = {}
    if path.exists():
        with open(path) as f:
            for line in f:
                if line.strip():
                    rec = json.loads(line)
                    cache[rec["key"]] = rec
    return cache


def score_events(cfg: dict, events: pd.DataFrame, scorer, rep: int = 1,
                 universe: pd.DataFrame | None = None) -> pd.DataFrame:
    llm = cfg["llm"]
    cache_path = cfg["paths"].llm_cache
    cache = read_cache(cache_path)
    names = dict(zip(universe["ticker"], universe["name"])) if universe is not None else {}
    lock = threading.Lock()
    todo = [r for _, r in events.iterrows()
            if _key(r["event_id"], scorer.model, llm["prompt_version"], rep) not in cache]
    log.info("LLM scoring rep %d: %d cached, %d to score", rep, len(events) - len(todo), len(todo))

    stop = threading.Event()

    def work(row):
        if stop.is_set():
            return
        text = load_text(cfg, row["path"])[: llm["max_chars"]]
        if llm.get("anonymize", True):
            text = anonymize(text, row["ticker"], names.get(row["ticker"]))
        res = scorer.score(text, pd.concat([row, pd.Series({"_rep": rep})]))
        rec = {"key": _key(row["event_id"], scorer.model, llm["prompt_version"], rep),
               "event_id": row["event_id"], "model": scorer.model,
               "prompt_version": llm["prompt_version"], "rep": rep, **res}
        with lock:
            with open(cache_path, "a") as f:
                f.write(json.dumps(rec) + "\n")
            cache[rec["key"]] = rec

    failures = 0
    with ThreadPoolExecutor(max_workers=llm.get("workers", 4)) as pool:
        futures = {pool.submit(work, r): r["event_id"] for r in todo}
        for i, fut in enumerate(as_completed(futures), 1):
            try:
                fut.result()
            except UsageLimitReached as e:
                if not stop.is_set():
                    log.warning("Usage limit reached (%s). Stopping; run the same command again after "
                                "your limit resets and it will continue where it left off.", e)
                stop.set()
            except Exception as e:  # keep going; report at the end
                failures += 1
                log.warning("score failed for %s: %s", futures[fut], e)
            if i % 50 == 0:
                log.info("  %d / %d", i, len(todo))
    if failures:
        log.warning("%d documents failed to score (re-run to retry)", failures)
    if stop.is_set():
        log.warning("Scored %d of %d documents so far.", sum(
            _key(e, scorer.model, llm["prompt_version"], rep) in cache for e in events["event_id"]), len(events))

    recs = [cache[k] for k in (_key(e, scorer.model, llm["prompt_version"], rep) for e in events["event_id"]) if k in cache]
    return pd.DataFrame(recs)


def build_llm_signals(scores: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    """Turn raw rubric scores into signal columns.

    llm           guidance tone, with the other scores as a small tie-breaker
                  (integer scores tie a lot, and quintiles need an ordering)
    llm_<topic>   each topic on its own (stretch goal)
    llm_delta     change in guidance tone vs. the same company's previous document
    """
    s = scores.merge(events[["event_id", "ticker", "published_at"]], on="event_id")
    s = s.sort_values("published_at")
    tie = (s["overall_tone"] + s["margins_tone"] + s["demand_tone"]) / 3.0
    out = pd.DataFrame({
        "event_id": s["event_id"],
        "llm": s["guidance_tone"] + 0.1 * tie,
        "llm_guidance": s["guidance_tone"].astype(float),
        "llm_margins": s["margins_tone"].where(s["margins_mentioned"], 0).astype(float),
        "llm_demand": s["demand_tone"].where(s["demand_mentioned"], 0).astype(float),
        "llm_overall": s["overall_tone"].astype(float),
        "llm_reason": s["reason"],
    })
    out["llm_delta"] = (s["guidance_tone"] - s.groupby("ticker")["guidance_tone"].shift(1)).astype(float)
    return out


def consistency(cfg: dict, events: pd.DataFrame, scorer, universe=None, n: int | None = None) -> dict:
    """Score a random sample a second time and measure agreement with the first pass."""
    n = n or cfg["llm"]["consistency_sample"]
    sample = events.sample(min(n, len(events)), random_state=7)
    a = score_events(cfg, sample, scorer, rep=1, universe=universe).set_index("event_id")
    b = score_events(cfg, sample, scorer, rep=2, universe=universe).set_index("event_id")
    idx = a.index.intersection(b.index)
    res = {"n": len(idx)}
    for col in ("guidance_tone", "overall_tone", "margins_tone", "demand_tone"):
        x, y = a.loc[idx, col].astype(int), b.loc[idx, col].astype(int)
        res[col] = {
            "exact_agreement": float((x == y).mean()),
            "within_one": float(((x - y).abs() <= 1).mean()),
            "spearman": float(stats.spearmanr(x, y).statistic) if x.nunique() > 1 and y.nunique() > 1 else float("nan"),
            "weighted_kappa": quadratic_kappa(x.values, y.values),
        }
    return res


def quadratic_kappa(a, b, levels=(-2, -1, 0, 1, 2)) -> float:
    """Cohen's kappa with quadratic weights (standard for ordinal ratings)."""
    k = len(levels)
    pos = {v: i for i, v in enumerate(levels)}
    O = np.zeros((k, k))
    for x, y in zip(a, b):
        O[pos[x], pos[y]] += 1
    W = np.array([[(i - j) ** 2 for j in range(k)] for i in range(k)]) / (k - 1) ** 2
    E = np.outer(O.sum(1), O.sum(0)) / O.sum()
    denom = (W * E).sum()
    return float(1 - (W * O).sum() / denom) if denom > 0 else float("nan")


__all__ = ["SCHEMA", "SYSTEM_PROMPT", "anonymize", "validate", "make_scorer", "score_events",
           "build_llm_signals", "consistency", "quadratic_kappa", "MockScorer", "AnthropicScorer"]
