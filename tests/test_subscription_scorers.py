"""Claude Code (Pro login) and Ollama backends, with the CLI and HTTP server stubbed."""
import json
import subprocess
from types import SimpleNamespace

import pandas as pd
import pytest

from earnsig import llm_score
from earnsig.config import Paths
from earnsig.llm_score import SCHEMA, ClaudeCodeScorer, OllamaScorer, UsageLimitReached, score_events

GOOD = {"guidance_tone": -2, "guidance_mentioned": True, "margins_tone": 0, "margins_mentioned": True,
        "demand_tone": -1, "demand_mentioned": True, "overall_tone": -1, "reason": "Outlook withdrawn"}


@pytest.fixture
def cc(tmp_path, monkeypatch):
    monkeypatch.setattr("shutil.which", lambda name: "/usr/local/bin/claude")
    return ClaudeCodeScorer("haiku", tmp_path / "prompt.txt")


def _fake_run(stdout, returncode=0, record=None):
    def run(cmd, input=None, **kw):
        if record is not None:
            record.update(cmd=cmd, input=input)
        return SimpleNamespace(stdout=stdout, stderr="", returncode=returncode)
    return run


def test_claude_code_command_and_parse(cc, monkeypatch):
    rec = {}
    payload = {"type": "result", "is_error": False, "structured_output": GOOD,
               "usage": {"input_tokens": 900, "output_tokens": 50}}
    monkeypatch.setattr(subprocess, "run", _fake_run(json.dumps(payload), record=rec))
    out = cc.score("We are withdrawing guidance.")
    cmd = rec["cmd"]
    assert cmd[:2] == ["/usr/local/bin/claude", "-p"]
    assert json.loads(cmd[cmd.index("--json-schema") + 1]) == SCHEMA
    assert cmd[cmd.index("--tools") + 1] == ""  # no tools: the model only reads and rates
    assert cmd[cmd.index("--model") + 1] == "haiku"
    assert "--bare" not in cmd  # bare mode would ignore the subscription login
    assert "<document>" in rec["input"]
    assert cc.prompt_file.read_text() == llm_score.SYSTEM_PROMPT
    assert out["guidance_tone"] == -2 and out["input_tokens"] == 900
    assert cc.model == "claude-code:haiku"


def test_claude_code_usage_limit_is_detected(cc, monkeypatch):
    payload = {"type": "result", "is_error": True, "result": "Claude usage limit reached. Resets at 5pm."}
    monkeypatch.setattr(subprocess, "run", _fake_run(json.dumps(payload), returncode=1))
    with pytest.raises(UsageLimitReached):
        cc.score("x")


def test_claude_code_other_errors_are_plain_failures(cc, monkeypatch):
    monkeypatch.setattr(subprocess, "run", _fake_run(json.dumps({"is_error": True, "result": "Not logged in"})))
    with pytest.raises(RuntimeError) as e:
        cc.score("x")
    assert not isinstance(e.value, UsageLimitReached)


def test_missing_cli_gives_install_hint(tmp_path, monkeypatch):
    monkeypatch.setattr("shutil.which", lambda name: None)
    with pytest.raises(RuntimeError, match="Install Claude Code"):
        ClaudeCodeScorer("haiku", tmp_path / "p.txt")


def _ollama_tags(models):
    return lambda url, timeout=None: SimpleNamespace(json=lambda: {"models": [{"name": m} for m in models]})


def test_ollama_request_and_parse(monkeypatch):
    rec = {}
    monkeypatch.setattr("requests.get", _ollama_tags(["llama3.1:8b"]))

    def post(url, json=None, timeout=None):
        rec.update(url=url, body=json)
        return SimpleNamespace(raise_for_status=lambda: None,
                               json=lambda: {"message": {"content": __import__("json").dumps(GOOD)},
                                             "prompt_eval_count": 800, "eval_count": 40})
    monkeypatch.setattr("requests.post", post)
    s = OllamaScorer("llama3.1:8b")
    out = s.score("text")
    assert rec["url"].endswith("/api/chat")
    assert rec["body"]["format"] == SCHEMA and rec["body"]["options"]["temperature"] == 0
    assert out["demand_tone"] == -1 and s.model == "ollama:llama3.1:8b"


def test_usage_limit_stops_and_resumes(tmp_path):
    """Hitting the limit stops scoring cleanly; a second run continues from the cache."""
    paths = Paths(tmp_path, tmp_path / "res").ensure()
    rows = []
    for i in range(6):
        (tmp_path / f"d{i}.txt").write_text(f"document {i}")
        rows.append({"event_id": f"e{i}", "ticker": "AAA", "path": f"d{i}.txt"})
    events = pd.DataFrame(rows)
    cfg = {"paths": paths, "llm": {"prompt_version": "v1", "max_chars": 1000, "anonymize": False, "workers": 1}}

    class Limited:
        model = "fake"

        def __init__(self, budget):
            self.budget = budget

        def score(self, text, row=None):
            if self.budget == 0:
                raise UsageLimitReached("usage limit reached")
            self.budget -= 1
            return dict(GOOD)

    first = score_events(cfg, events, Limited(budget=4))
    assert len(first) == 4
    second = score_events(cfg, events, Limited(budget=10))
    assert len(second) == 6


def test_ctrl_c_stops_without_draining_queue(tmp_path, monkeypatch):
    paths = Paths(tmp_path, tmp_path / "res").ensure()
    rows = []
    for i in range(50):
        (tmp_path / f"d{i}.txt").write_text("doc")
        rows.append({"event_id": f"e{i}", "ticker": "AAA", "path": f"d{i}.txt"})
    cfg = {"paths": paths, "llm": {"prompt_version": "v1", "max_chars": 100, "anonymize": False, "workers": 1}}
    calls = []

    class Slow:
        model = "fake"

        def score(self, text, row=None):
            calls.append(1)
            return dict(GOOD)

    real = llm_score.as_completed

    def interrupted(futures):
        it = real(futures)
        yield next(it)
        raise KeyboardInterrupt

    monkeypatch.setattr(llm_score, "as_completed", interrupted)
    with pytest.raises(SystemExit):
        score_events(cfg, pd.DataFrame(rows), Slow())
    assert len(calls) < 50  # the rest of the queue was cancelled


def test_half_written_cache_line_is_ignored(tmp_path):
    p = tmp_path / "cache.jsonl"
    p.write_text(json.dumps({"key": "k1", "x": 1}) + "\n" + '{"key": "k2", "x"')
    assert list(llm_score.read_cache(p)) == ["k1"]


def test_ollama_not_running_or_model_missing(monkeypatch):
    def down(url, timeout=None):
        raise ConnectionError("refused")
    monkeypatch.setattr("requests.get", down)
    with pytest.raises(RuntimeError, match="Ollama is not running"):
        OllamaScorer("llama3.2:3b")
    monkeypatch.setattr("requests.get", _ollama_tags(["llama3.1:8b"]))
    with pytest.raises(RuntimeError, match="ollama pull llama3.2:3b"):
        OllamaScorer("llama3.2:3b")
    monkeypatch.setattr("requests.get", _ollama_tags(["llama3.2:3b"]))
    assert OllamaScorer("llama3.2:3b").model == "ollama:llama3.2:3b"
