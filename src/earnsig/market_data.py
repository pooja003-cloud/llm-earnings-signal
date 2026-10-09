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


def _closes_from_download(raw: pd.DataFrame, batch: list[str]) -> dict[str, pd.Series]:
    """Pull one Close series per ticker out of a yf.download result (either column layout)."""
    if raw is None or raw.empty:
        return {}
    if isinstance(raw.columns, pd.MultiIndex):
        close = raw["Close"]
    else:  # single ticker, flat columns
        close = raw[["Close"]].rename(columns={"Close": batch[0]})
    out = {}
    for t in batch:
        if t in close and close[t].notna().sum() > 0:
            out[t] = close[t]
    return out


def download_prices(cfg: dict, universe: pd.DataFrame, batch_size: int = 10, retries: int = 3,
                    pause: float = 1.0) -> pd.DataFrame:
    """Daily split- and dividend-adjusted closes for stocks, SPY and sector ETFs.

    Downloads in small sequential batches and retries failures: one big
    multi-threaded request makes yfinance's cache and connections fail.
    """
    import time

    import yfinance as yf

    try:  # keep yfinance's cache inside the project instead of a shared system folder
        cache = cfg["paths"].data / "yfinance_cache"
        cache.mkdir(parents=True, exist_ok=True)
        yf.set_tz_cache_location(str(cache))
    except Exception as e:  # pragma: no cover - older/newer yfinance without this function
        log.debug("could not set yfinance cache location: %s", e)

    tickers = sorted(set(universe["ticker"]) | set(universe["sector_etf"]) | {cfg["event"]["market_ticker"]})
    start = pd.Timestamp(cfg["start_date"]) - pd.Timedelta(days=30)
    end = pd.Timestamp(cfg["end_date"]) + pd.Timedelta(days=200)  # room for holding windows
    series: dict[str, pd.Series] = {}
    todo = list(tickers)
    for attempt in range(retries + 1):
        if attempt:
            log.info("Retrying %d tickers (attempt %d of %d)", len(todo), attempt, retries)
            time.sleep(pause * 5 * attempt)
        for i in range(0, len(todo), batch_size):
            batch = todo[i:i + batch_size]
            yf_batch = [t.replace(".", "-") for t in batch]
            try:
                raw = yf.download(yf_batch, start=start, end=end, auto_adjust=True, progress=False, threads=False)
            except Exception as e:
                log.warning("batch %s failed: %s", batch, e)
                continue
            got = _closes_from_download(raw, yf_batch)
            series.update({t.replace("-", "."): s for t, s in got.items()})
            log.info("prices: %d / %d tickers", len(series), len(tickers))
            time.sleep(pause)
        todo = [t for t in tickers if t not in series]
        if not todo:
            break
    if not series:
        raise SystemExit("No prices downloaded. Check your internet connection and run `earnsig prices` again.")
    close = pd.DataFrame(series)
    close.index = pd.to_datetime(close.index).tz_localize(None).normalize()
    close = close.sort_index()
    if todo:
        log.warning("Still no price data for %s. Run `earnsig prices` again later; events for these "
                    "tickers are skipped until then.", todo)
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
