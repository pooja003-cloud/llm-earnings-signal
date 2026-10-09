"""Synthetic dataset for an offline end-to-end run (no API keys, no network).

Everything here is fake. A hidden ``latent_tone`` per document drives (a) which
sentences appear in the document, (b) a large day-0 price reaction that the
strategy cannot trade, and (c) a small drift over days +1..+20 that it can.
The point is to exercise every step of the pipeline and to check that the
backtest recovers a planted effect of known size - not to say anything about
real markets.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

SECTORS = ["TECH", "FIN", "HLTH", "ENGY", "CONS", "INDU"]

POS = ["We are raising our full-year outlook on strong momentum across the business.",
       "Demand remained robust and our backlog reached a record level.",
       "Gross margin expanded as productivity gains and favorable pricing improved profitability.",
       "We are pleased with the excellent execution and see attractive opportunities ahead.",
       "Orders strengthened throughout the quarter and customer enthusiasm is high."]
NEU = ["We are reaffirming our guidance for the fiscal year.",
       "Results were in line with our expectations.",
       "Operating expenses were consistent with the prior year period.",
       "We continue to execute on our strategic priorities."]
NEG = ["We are lowering our outlook to reflect weaker demand and difficult conditions.",
       "Margins declined due to cost inflation and unfavorable mix.",
       "Order volumes softened and we experienced delays in several regions.",
       "We recorded impairment charges and restructuring losses during the quarter.",
       "Results were adversely affected by challenging market headwinds."]
BOILER = ["This release contains forward-looking statements that involve risks and uncertainties.",
          "Actual results may differ materially due to litigation, regulatory and other risks.",
          "Revenue was ${rev:.1f} billion, compared with ${prev:.1f} billion a year ago.",
          "Diluted earnings per share were ${eps:.2f}."]


def make_document(tone: float, rng: np.random.Generator) -> str:
    sents = []
    for _ in range(12):
        u = rng.random()
        p_pos = 1 / (1 + np.exp(-1.2 * tone)) * 0.7
        p_neg = 1 / (1 + np.exp(1.2 * tone)) * 0.7
        pool = POS if u < p_pos else NEG if u < p_pos + p_neg else NEU
        sents.append(pool[rng.integers(len(pool))])
    rev = rng.uniform(1, 40)
    extra = [b.format(rev=rev, prev=rev / (1 + rng.normal(0.05, 0.08)), eps=rng.uniform(0.2, 5)) for b in BOILER]
    rng.shuffle(sents)
    return "\n".join(["Quarterly results press release."] + sents[:6] + extra[2:] + sents[6:] + extra[:2])


def generate(cfg: dict, n_stocks: int = 80, seed: int = 42,
             drift_per_sd: float = 0.006, day0_per_sd: float = 0.03) -> None:
    rng = np.random.default_rng(seed)
    paths = cfg["paths"]
    days = pd.bdate_range("2020-12-01", "2026-06-30")
    T = len(days)

    # factors (daily, decimal)
    fac = pd.DataFrame({
        "Mkt-RF": rng.normal(0.0004, 0.011, T), "SMB": rng.normal(0, 0.005, T),
        "HML": rng.normal(0, 0.006, T), "RMW": rng.normal(0, 0.004, T),
        "CMA": rng.normal(0, 0.004, T), "RF": 0.00008, "Mom": rng.normal(0, 0.007, T),
    }, index=days)
    fac.index.name = "date"
    fac.to_csv(paths.factors)

    tickers = [f"SYN{i:02d}" for i in range(n_stocks)]
    sector = {t: SECTORS[i % len(SECTORS)] for i, t in enumerate(tickers)}
    uni = pd.DataFrame({"ticker": tickers, "name": [f"Synthetic Co {i}" for i in range(n_stocks)],
                        "sector": [sector[t] for t in tickers], "sector_etf": [f"ETF_{sector[t]}" for t in tickers]})
    uni.to_csv(paths.data / "universe.csv", index=False)

    sec_shock = {s: rng.normal(0, 0.006, T) for s in SECTORS}
    rets = {}
    for t in tickers:
        beta = rng.uniform(0.7, 1.4)
        loads = rng.normal(0, 0.3, 4)
        rets[t] = (fac["RF"].values + beta * fac["Mkt-RF"].values + sec_shock[sector[t]]
                   + loads @ fac[["SMB", "HML", "RMW", "Mom"]].values.T + rng.normal(0, 0.016, T))
    rets = pd.DataFrame(rets, index=days)

    # earnings events: 4 per year per stock, timestamps before the open or after the close
    windows = [(1, 20, 2, 25), (4, 18, 5, 20), (7, 18, 8, 20), (10, 18, 11, 20)]
    events, prev_tone = [], {t: 0.0 for t in tickers}
    for year in range(2021, 2026):
        for (m1, d1, m2, d2) in windows:
            lo, hi = pd.Timestamp(year, m1, d1), pd.Timestamp(year, m2, d2)
            for t in tickers:
                day = lo + pd.Timedelta(days=int(rng.integers(0, (hi - lo).days)))
                if day.weekday() >= 5:
                    day += pd.Timedelta(days=7 - day.weekday())
                after_close = rng.random() < 0.55
                ts = day + (pd.Timedelta(hours=16, minutes=5) if after_close else pd.Timedelta(hours=7))
                tone = 0.3 * prev_tone[t] + rng.normal(0, 1)
                prev_tone[t] = tone
                eid = f"{t}_{ts:%Y%m%d}"
                text = make_document(tone, rng)
                rel = f"raw/docs/{t}/{eid}.txt"
                (paths.data / rel).parent.mkdir(parents=True, exist_ok=True)
                (paths.data / rel).write_text(text)
                events.append({"event_id": eid, "ticker": t, "published_at": ts, "source": "synthetic",
                               "form": "synthetic", "url": "", "path": rel, "latent_tone": tone})
                # plant price effects relative to the first tradable close
                p0 = days.searchsorted(day + pd.Timedelta(days=int(after_close)))
                if p0 + 21 < T:
                    rets.iloc[p0, rets.columns.get_loc(t)] += day0_per_sd * tone
                    rets.iloc[p0 + 1:p0 + 21, rets.columns.get_loc(t)] += drift_per_sd * tone / 20
    ev = pd.DataFrame(events).sort_values("published_at")
    ev.to_csv(paths.events, index=False)

    mkt = fac["RF"] + fac["Mkt-RF"]
    rets["SPY"] = mkt
    for s in SECTORS:
        rets[f"ETF_{s}"] = mkt + pd.Series(sec_shock[s], index=days)
    prices = 100 * (1 + rets).cumprod()
    prices.to_parquet(paths.prices)
