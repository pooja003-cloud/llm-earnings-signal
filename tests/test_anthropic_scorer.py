"""The real scorer, with the API client stubbed out (no network, no key)."""
import json
from types import SimpleNamespace

import pytest

from earnsig.llm_score import SCHEMA, AnthropicScorer

GOOD = {"guidance_tone": 1, "guidance_mentioned": True, "margins_tone": -1, "margins_mentioned": True,
        "demand_tone": 0, "demand_mentioned": False, "overall_tone": 1, "reason": "Reaffirmed at high end"}


class FakeMessages:
    def __init__(self, payload, stop="end_turn"):
        self.payload, self.stop, self.kwargs = payload, stop, None

    def create(self, **kwargs):
        self.kwargs = kwargs
        return SimpleNamespace(stop_reason=self.stop,
                               content=[SimpleNamespace(type="text", text=json.dumps(self.payload))],
                               usage=SimpleNamespace(input_tokens=1200, output_tokens=60))


def _scorer(fake):
    s = AnthropicScorer.__new__(AnthropicScorer)
    s.client, s.model, s.max_tokens = SimpleNamespace(messages=fake), "claude-haiku-5-5", 600
    return s


def test_request_uses_json_schema_and_parses():
    fake = FakeMessages(GOOD)
    out = _scorer(fake).score("Revenue rose. We reaffirm guidance.")
    assert fake.kwargs["output_config"] == {"format": {"type": "json_schema", "schema": SCHEMA}}
    assert "<document>" in fake.kwargs["messages"][0]["content"]
    assert out["guidance_tone"] == 1 and out["demand_mentioned"] is False and out["input_tokens"] == 1200


def test_refusal_and_truncation_raise():
    for stop in ("refusal", "max_tokens"):
        with pytest.raises(RuntimeError):
            _scorer(FakeMessages(GOOD, stop)).score("x")
