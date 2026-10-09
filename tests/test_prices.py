"""Price download: small batches, retries, both yfinance column layouts (yfinance stubbed)."""
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from earnsig.config import Paths
from earnsig.market_data import download_prices


class FakeYF:
    """Fails each ticker in `flaky` on its first request, raises for one whole batch once."""

    def __init__(self, flaky=(), boom_once=False):
        self.flaky, self.seen, self.calls, self.boom_once = set(flaky), set(), [], boom_once
        self.cache = None

    def set_tz_cache_location(self, path):
        self.cache = path

    def download(self, tickers, **kw):
        self.calls.append((list(tickers), kw))
        if self.boom_once:
            self.boom_once = False
            raise RuntimeError("curl: (27)")
        idx = pd.bdate_range("2021-01-01", periods=5)
        cols = {}
        for t in tickers:
            if t in self.flaky and t not in self.seen:
                self.seen.add(t)
                cols[("Close", t)] = np.nan
            else:
                cols[("Close", t)] = np.arange(5.0) + 100
        df = pd.DataFrame(cols, index=idx)
        df.columns = pd.MultiIndex.from_tuples(df.columns)
        return df


def _cfg(tmp_path):
    return {"paths": Paths(tmp_path, tmp_path / "res").ensure(), "start_date": "2021-01-01",
            "end_date": "2021-12-31", "event": {"market_ticker": "SPY"}}


def test_batches_retries_and_saves(tmp_path, monkeypatch):
    fake = FakeYF(flaky={"MS", "XLE"}, boom_once=True)
    monkeypatch.setitem(sys.modules, "yfinance", fake)
    uni = pd.DataFrame({"ticker": ["AAPL", "MS", "OXY", "BRK.B"], "sector_etf": ["XLK", "XLF", "XLE", "XLF"]})
    close = download_prices(_cfg(tmp_path), uni, batch_size=3, pause=0)
    assert set(close.columns) == {"AAPL", "MS", "OXY", "BRK.B", "XLK", "XLF", "XLE", "SPY"}
    assert all(len(c[0]) <= 3 and c[1]["threads"] is False for c in fake.calls)
    assert "BRK-B" in sum((c[0] for c in fake.calls), [])  # Yahoo spelling for class shares
    assert fake.cache.endswith("yfinance_cache")
    assert pd.read_parquet(tmp_path / "prices.parquet").shape == close.shape


def test_flat_columns_single_ticker(tmp_path, monkeypatch):
    def download(tickers, **kw):
        return pd.DataFrame({"Close": [1.0, 2.0]}, index=pd.bdate_range("2021-01-01", periods=2))
    monkeypatch.setitem(sys.modules, "yfinance", SimpleNamespace(download=download, set_tz_cache_location=lambda p: None))
    uni = pd.DataFrame({"ticker": [], "sector_etf": []})
    close = download_prices(_cfg(tmp_path), uni, pause=0)
    assert list(close.columns) == ["SPY"]


def test_nothing_downloaded_stops_clearly(tmp_path, monkeypatch):
    def download(tickers, **kw):
        raise RuntimeError("offline")
    monkeypatch.setitem(sys.modules, "yfinance", SimpleNamespace(download=download, set_tz_cache_location=lambda p: None))
    with pytest.raises(SystemExit, match="No prices downloaded"):
        download_prices(_cfg(tmp_path), pd.DataFrame({"ticker": ["AAPL"], "sector_etf": ["XLK"]}), retries=1, pause=0)
