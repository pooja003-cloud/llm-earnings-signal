"""Prices (yfinance) and Fama-French factors (Ken French data library)."""
from __future__ import annotations

import io
import logging
import zipfile

import pandas as pd
import requests

log = logging.getLogger(__name__)

FF5_URL = ("https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
           "F-F_Research_Data_5_Factors_2x3_daily_CSV.zip")
MOM_URL = ("https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
           "F-F_Momentum_Factor_daily_CSV.zip")


def download_prices(cfg: dict, universe: pd.DataFrame) -> pd.DataFrame:
    """Daily split- and dividend-adjusted closes for stocks, SPY and sector ETFs."""
    import yfinance as yf

    tickers = sorted(set(universe["ticker"]) | set(universe["sector_etf"]) | {cfg["event"]["market_ticker"]})
    start = pd.Timestamp(cfg["start_date"]) - pd.Timedelta(days=30)
    end = pd.Timestamp(cfg["end_date"]) + pd.Timedelta(days=200)  # room for holding windows
    yf_tickers = [t.replace(".", "-") for t in tickers]
    raw = yf.download(yf_tickers, start=start, end=end, auto_adjust=True, progress=False, threads=True)
    close = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw[["Close"]]
    close.columns = [c.replace("-", ".") for c in close.columns]
    close.index = pd.to_datetime(close.index).tz_localize(None).normalize()
    close = close.sort_index()
    missing = [t for t in tickers if t not in close or close[t].notna().sum() == 0]
    if missing:
        log.warning("No price data for: %s", missing)
    close.to_parquet(cfg["paths"].prices)
    log.info("prices: %d days x %d tickers", *close.shape)
    return close


def _read_french_zip(url: str) -> pd.DataFrame:
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(r.content)) as z:
        raw = z.read(z.namelist()[0]).decode("latin-1")
    return parse_french_csv(raw)


def parse_french_csv(raw: str) -> pd.DataFrame:
    """Parse a Ken French daily CSV (header block, table, copyright footer)."""
    lines = raw.splitlines()
    start = next(i for i, l in enumerate(lines) if l.strip().startswith(",") and any(c.isalpha() for c in l))
    body = [lines[start]]
    for l in lines[start + 1:]:
        if not l.strip() or not l.strip()[0].isdigit():
            break
        body.append(l)
    df = pd.read_csv(io.StringIO("\n".join(body)))
    df = df.rename(columns={df.columns[0]: "date"})
    df.columns = [c.strip() for c in df.columns]
    df["date"] = pd.to_datetime(df["date"].astype(str).str.strip(), format="%Y%m%d")
    df = df.set_index("date").astype(float) / 100.0  # percent -> decimal
    return df


def download_factors(cfg: dict) -> pd.DataFrame:
    ff5 = _read_french_zip(FF5_URL)
    mom = _read_french_zip(MOM_URL)
    mom.columns = ["Mom"]
    f = ff5.join(mom, how="inner")
    f.to_csv(cfg["paths"].factors)
    log.info("factors: %d days, %s .. %s", len(f), f.index.min().date(), f.index.max().date())
    return f


def load_prices(cfg: dict) -> pd.DataFrame:
    return pd.read_parquet(cfg["paths"].prices)


def load_factors(cfg: dict) -> pd.DataFrame | None:
    p = cfg["paths"].factors
    return pd.read_csv(p, index_col=0, parse_dates=True) if p.exists() else None
