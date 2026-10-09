"""Collect earnings documents and their publication timestamps.

Two sources:

* ``sec`` (default): 8-K filings with Item 2.02 ("Results of Operations and
  Financial Condition") from SEC EDGAR. The earnings press release is the
  EX-99.1 exhibit. The event timestamp is EDGAR's acceptance time, which is
  the moment the filing became public on EDGAR.
* ``local``: your own transcripts (e.g. licensed earnings-call transcripts),
  given as a CSV with columns ``ticker, published_at, path`` (path to a .txt).

Output: ``data/events.csv`` with one row per document and the text saved under
``data/raw/docs/<TICKER>/<id>.txt``.
"""
from __future__ import annotations

import logging
import re
import time
from pathlib import Path

import pandas as pd
import requests

log = logging.getLogger(__name__)

SEC_TICKERS = "https://www.sec.gov/files/company_tickers.json"
SEC_SUBMISSIONS = "https://data.sec.gov/submissions/CIK{cik:010d}.json"
SEC_ARCHIVE = "https://www.sec.gov/Archives/edgar/data/{cik}/{acc_nodash}/"

EVENT_COLUMNS = ["event_id", "ticker", "published_at", "source", "form", "url", "path"]


class SecClient:
    """Tiny polite EDGAR client: identifies itself and stays under the rate limit."""

    def __init__(self, user_agent: str, rps: float = 8):
        if not user_agent or "@" not in user_agent:
            raise ValueError(
                "SEC requires a User-Agent with your name and email. "
                "Set SEC_USER_AGENT='Jane Doe jane@example.com' in .env"
            )
        self.s = requests.Session()
        self.s.headers.update({"User-Agent": user_agent, "Accept-Encoding": "gzip, deflate"})
        self.min_interval = 1.0 / rps
        self._last = 0.0

    def get(self, url: str, retries: int = 4) -> requests.Response:
        for attempt in range(retries):
            wait = self.min_interval - (time.time() - self._last)
            if wait > 0:
                time.sleep(wait)
            self._last = time.time()
            r = self.s.get(url, timeout=30)
            if r.status_code == 200:
                return r
            if r.status_code in (429, 503):
                time.sleep(2 ** attempt)
                continue
            r.raise_for_status()
        r.raise_for_status()
        return r


def ticker_to_cik(client: SecClient) -> dict[str, int]:
    data = client.get(SEC_TICKERS).json()
    return {v["ticker"].upper(): int(v["cik_str"]) for v in data.values()}


def _filings_frame(client: SecClient, cik: int) -> pd.DataFrame:
    """All filings for a company, including older pages beyond 'recent'."""
    sub = client.get(SEC_SUBMISSIONS.format(cik=cik)).json()
    frames = [pd.DataFrame(sub["filings"]["recent"])]
    for extra in sub["filings"].get("files", []):
        url = "https://data.sec.gov/submissions/" + extra["name"]
        frames.append(pd.DataFrame(client.get(url).json()))
    return pd.concat(frames, ignore_index=True)


def parse_acceptance(ts: str) -> pd.Timestamp:
    """EDGAR acceptance time as a naive US/Eastern wall-clock timestamp.

    The submissions API writes e.g. ``2024-01-25T16:05:12.000Z``. We read the
    clock time as Eastern. If the value were really UTC this reads it 4-5 hours
    *later* than reality, which can only delay the trade - never create
    look-ahead. That is the safe direction to be wrong in.
    """
    return pd.Timestamp(ts.replace("Z", "")).tz_localize(None).floor("s")


def find_press_release(index_html: str) -> str | None:
    """Return the relative href of the EX-99.1 exhibit from a filing index page."""
    rows = re.findall(
        r"<tr[^>]*>(.*?)</tr>", index_html, flags=re.S | re.I
    )
    candidates = []
    for row in rows:
        href = re.search(r'href="([^"]+\.(?:htm|html|txt))"', row, re.I)
        cells = [re.sub(r"<[^>]+>", "", c).strip() for c in re.findall(r"<td[^>]*>(.*?)</td>", row, re.S | re.I)]
        if not href or not cells:
            continue
        doc_type = next((c for c in cells if c.upper().startswith("EX-99")), None)
        if doc_type:
            candidates.append((doc_type.upper(), href.group(1)))
    if not candidates:
        return None
    # prefer EX-99.1, then any EX-99.x
    candidates.sort(key=lambda c: (c[0] not in ("EX-99.1", "EX-99"), c[0]))
    return candidates[0][1]


def html_to_text(html: str) -> str:
    try:
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(html, "html.parser")
        for tag in soup(["script", "style"]):
            tag.decompose()
        text = soup.get_text(" ")
    except ImportError:  # pragma: no cover
        text = re.sub(r"<[^>]+>", " ", html)
    text = re.sub(r"&nbsp;|&#160;|\xa0", " ", text)
    text = re.sub(r"[ \t]+", " ", text)
    return re.sub(r"\s*\n\s*", "\n", text).strip()


def collect_sec(cfg: dict, universe: pd.DataFrame, tickers: list[str] | None = None) -> pd.DataFrame:
    paths = cfg["paths"]
    client = SecClient(cfg["sec"]["user_agent"], cfg["sec"].get("requests_per_second", 8))
    cik_map = ticker_to_cik(client)
    # optional `cik` column in universe.csv pins a company number, e.g. when a company
    # reorganized under a new parent and the ticker now maps to a CIK with no history
    pinned = {}
    if "cik" in universe.columns:
        pinned = {t: int(c) for t, c in zip(universe["ticker"], universe["cik"]) if pd.notna(c) and str(c).strip()}
    start, end = pd.Timestamp(cfg["start_date"]), pd.Timestamp(cfg["end_date"])
    existing = _read_events(paths.events)
    done = set(existing["event_id"]) if len(existing) else set()
    rows = []
    for ticker in tickers or universe["ticker"].tolist():
        cik = pinned.get(ticker) or cik_map.get(ticker.replace(".", "-")) or cik_map.get(ticker)
        if cik is None:
            log.warning("No CIK for %s, skipping", ticker)
            continue
        f = _filings_frame(client, cik)
        f = f[(f["form"] == "8-K") & f["items"].fillna("").str.contains(r"\b2\.02\b")]
        f = f[(pd.to_datetime(f["filingDate"]) >= start) & (pd.to_datetime(f["filingDate"]) <= end)]
        log.info("%s: %d earnings 8-Ks", ticker, len(f))
        if len(f) == 0:
            log.warning("%s: no earnings 8-Ks under CIK %d. If the company reorganized or changed its "
                        "SEC registrant, add its old CIK in a `cik` column of universe.csv.", ticker, cik)
        for _, fil in f.iterrows():
            acc = fil["accessionNumber"]
            event_id = f"{ticker}_{acc}"
            if event_id in done:
                continue
            base = SEC_ARCHIVE.format(cik=cik, acc_nodash=acc.replace("-", ""))
            try:
                index_html = client.get(base + f"{acc}-index.htm").text
                href = find_press_release(index_html)
                if href is None:
                    log.warning("%s %s: no EX-99 exhibit", ticker, acc)
                    continue
                url = href if href.startswith("http") else "https://www.sec.gov" + href if href.startswith("/") else base + href
                text = html_to_text(client.get(url).text)
            except requests.HTTPError as e:
                log.warning("%s %s: %s", ticker, acc, e)
                continue
            out = paths.docs / ticker / f"{acc}.txt"
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(text)
            rows.append({
                "event_id": event_id,
                "ticker": ticker,
                "published_at": parse_acceptance(fil["acceptanceDateTime"]),
                "source": "sec_8k_ex99",
                "form": "8-K",
                "url": url,
                "path": str(out.relative_to(paths.data)),
            })
    return _write_events(paths.events, existing, rows)


def collect_local(cfg: dict, manifest: str | Path) -> pd.DataFrame:
    """Register user-supplied transcripts. Manifest columns: ticker, published_at, path.

    ``published_at`` must be the time the call/transcript became public, in
    US/Eastern. For calls, use the call start time (not the transcript upload).
    """
    paths = cfg["paths"]
    m = pd.read_csv(manifest)
    missing = {"ticker", "published_at", "path"} - set(m.columns)
    if missing:
        raise ValueError(f"manifest missing columns: {missing}")
    rows = []
    for _, r in m.iterrows():
        src = Path(r["path"])
        if not src.is_absolute():
            src = Path(manifest).parent / src
        ts = pd.Timestamp(r["published_at"])
        out = paths.docs / r["ticker"] / f"call_{ts:%Y%m%d%H%M}.txt"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(src.read_text())
        rows.append({
            "event_id": f"{r['ticker']}_call_{ts:%Y%m%d%H%M}",
            "ticker": r["ticker"],
            "published_at": ts,
            "source": "local_transcript",
            "form": "transcript",
            "url": "",
            "path": str(out.relative_to(paths.data)),
        })
    return _write_events(paths.events, _read_events(paths.events), rows)


def _read_events(path: Path) -> pd.DataFrame:
    if path.exists():
        return pd.read_csv(path, parse_dates=["published_at"])
    return pd.DataFrame(columns=EVENT_COLUMNS)


def _write_events(path: Path, existing: pd.DataFrame, rows: list[dict]) -> pd.DataFrame:
    new = pd.DataFrame(rows, columns=EVENT_COLUMNS)
    ev = pd.concat([existing, new], ignore_index=True) if len(existing) else new
    ev = ev.drop_duplicates("event_id").sort_values(["published_at", "ticker"])
    ev.to_csv(path, index=False)
    log.info("events.csv: %d documents", len(ev))
    return ev


def load_text(cfg: dict, rel_path: str) -> str:
    return (cfg["paths"].data / rel_path).read_text()
