import numpy as np
import pandas as pd
import pytest

from earnsig.collect import find_press_release, parse_acceptance
from earnsig.lm_baseline import lm_tone
from earnsig.llm_score import SCHEMA, anonymize, build_llm_signals, quadratic_kappa, validate
from earnsig.market_data import parse_french_csv

POS = frozenset({"STRONG", "IMPROVED", "RECORD"})
NEG = frozenset({"DECLINE", "LOSS", "WEAK"})


def test_lm_net_tone_and_negation():
    r = lm_tone("Record quarter with strong growth. Weak demand caused a decline.", POS, NEG)
    assert (r["lm_pos"], r["lm_neg"]) == (2, 2) and r["lm"] == 0
    r = lm_tone("Results were not strong.", POS, NEG)
    assert (r["lm_pos"], r["lm_neg"]) == (0, 1)
    assert lm_tone("Nothing here.", POS, NEG)["lm"] == 0


def test_anonymize_masks_identity_and_dates():
    t = "Apple Inc. (NASDAQ: AAPL) reported results on January 30, 2025 for fiscal 2025. Apple expects..."
    a = anonymize(t, "AAPL", "Apple")
    for leak in ("Apple", "AAPL", "2025", "January 30"):
        assert leak not in a
    assert "the Company" in a


def test_anonymize_short_ticker_does_not_eat_words():
    a = anonymize("AT&T (T) said T-Mobile and The team. Total revenue", "T", "AT&T")
    assert "Total" in a and "team" in a


def test_validate_rejects_out_of_range():
    ok = {"guidance_tone": 2, "margins_tone": 0, "demand_tone": -1, "overall_tone": 1,
          "guidance_mentioned": True, "margins_mentioned": False, "demand_mentioned": True, "reason": "raised"}
    assert validate(ok)["guidance_tone"] == 2
    with pytest.raises(ValueError):
        validate({**ok, "guidance_tone": 3})


def test_schema_is_strict():
    assert SCHEMA["additionalProperties"] is False
    assert set(SCHEMA["required"]) == set(SCHEMA["properties"])


def test_quadratic_kappa():
    a = np.array([-2, -1, 0, 1, 2, 0, 1])
    assert quadratic_kappa(a, a) == pytest.approx(1.0)
    assert quadratic_kappa(a, -a) < 0


def test_llm_signals_delta_uses_previous_document():
    events = pd.DataFrame({"event_id": ["a1", "a2", "b1"], "ticker": ["A", "A", "B"],
                           "published_at": pd.to_datetime(["2024-01-01", "2024-04-01", "2024-01-05"])})
    base = {"margins_tone": 0, "demand_tone": 0, "overall_tone": 0, "margins_mentioned": True,
            "demand_mentioned": True, "reason": ""}
    scores = pd.DataFrame([{"event_id": "a2", "guidance_tone": 2, **base},
                           {"event_id": "a1", "guidance_tone": -1, **base},
                           {"event_id": "b1", "guidance_tone": 1, **base}])
    s = build_llm_signals(scores, events).set_index("event_id")
    assert s.loc["a2", "llm_delta"] == 3 and np.isnan(s.loc["a1", "llm_delta"])


def test_find_press_release_prefers_ex99_1():
    html = """<table><tr><td>1</td><td><a href="/Archives/x/form8k.htm">form8k.htm</a></td><td>8-K</td></tr>
    <tr><td>2</td><td><a href="/Archives/x/ex99-2.htm">ex99-2.htm</a></td><td>EX-99.2</td></tr>
    <tr><td>3</td><td><a href="/Archives/x/ex991.htm">ex991.htm</a></td><td>EX-99.1</td></tr></table>"""
    assert find_press_release(html) == "/Archives/x/ex991.htm"
    assert find_press_release("<table></table>") is None


def test_acceptance_time_read_as_wall_clock():
    assert parse_acceptance("2024-01-25T16:05:12.000Z") == pd.Timestamp("2024-01-25 16:05:12")


def test_parse_french_csv():
    raw = ("This file was created by CMPT_ME_BEME_RETS_DAILY\n\n"
           ",Mkt-RF,SMB,HML,RMW,CMA,RF\n20240102,  -0.71,  0.12, 1.02, 0.3, 0.2, 0.021\n"
           "20240103,  -0.50, -0.20, 0.10, 0.1, 0.0, 0.021\n\nCopyright 2024 Kenneth R. French\n")
    df = parse_french_csv(raw)
    assert list(df.columns) == ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "RF"]
    assert df.loc["2024-01-02", "Mkt-RF"] == pytest.approx(-0.0071)
