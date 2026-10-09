"""SEC collector with EDGAR stubbed: pinned CIKs for companies that reorganized."""
import pandas as pd

from earnsig import collect
from earnsig.config import Paths

INDEX = """<table><tr><td>1</td><td><a href="/Archives/edgar/data/34088/1/x-8k.htm">x</a></td><td>8-K</td></tr>
<tr><td>2</td><td><a href="/Archives/edgar/data/34088/1/ex991.htm">ex</a></td><td>EX-99.1</td></tr></table>"""


class FakeSec:
    def __init__(self, *a, **k):
        self.urls = []

    def get(self, url):
        self.urls.append(url)
        text = INDEX if url.endswith("-index.htm") else "<p>Exxon raised its outlook.</p>"
        return type("R", (), {"text": text})()


def _filings(client, cik):
    if cik == 999:  # new parent company: no earnings history
        return pd.DataFrame({"form": ["8-K"], "items": ["1.01,2.01"], "filingDate": ["2026-07-01"],
                             "accessionNumber": ["0000999-26-000001"], "acceptanceDateTime": ["2026-07-01T08:00:00.000Z"]})
    return pd.DataFrame({"form": ["8-K", "8-K", "10-Q"], "items": ["2.02,7.01", "5.02", ""],
                         "filingDate": ["2024-11-01", "2024-12-01", "2024-11-01"],
                         "accessionNumber": ["0000034088-24-000066", "0000034088-24-000070", "0000034088-24-000067"],
                         "acceptanceDateTime": ["2024-11-01T06:30:00.000Z"] * 3})


def _setup(tmp_path, monkeypatch):
    monkeypatch.setattr(collect, "SecClient", FakeSec)
    monkeypatch.setattr(collect, "ticker_to_cik", lambda c: {"XOM": 999})
    monkeypatch.setattr(collect, "_filings_frame", _filings)
    return {"paths": Paths(tmp_path, tmp_path / "res").ensure(), "sec": {"user_agent": "T t@x.com"},
            "start_date": "2021-01-01", "end_date": "2025-12-31"}


def test_pinned_cik_overrides_ticker_map(tmp_path, monkeypatch):
    cfg = _setup(tmp_path, monkeypatch)
    uni = pd.DataFrame({"ticker": ["XOM"], "name": ["Exxon Mobil"], "cik": [34088]})
    ev = collect.collect_sec(cfg, uni)
    assert len(ev) == 1
    assert ev["event_id"].iloc[0] == "XOM_0000034088-24-000066"
    assert ev["published_at"].iloc[0] == pd.Timestamp("2024-11-01 06:30:00")
    assert "raised its outlook" in (tmp_path / ev["path"].iloc[0]).read_text()


def test_without_pin_finds_nothing_and_warns(tmp_path, monkeypatch, caplog):
    cfg = _setup(tmp_path, monkeypatch)
    uni = pd.DataFrame({"ticker": ["XOM"], "name": ["Exxon Mobil"], "cik": [pd.NA]})
    ev = collect.collect_sec(cfg, uni)
    assert len(ev) == 0
    assert "add its old CIK" in caplog.text
