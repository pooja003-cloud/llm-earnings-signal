import numpy as np
import pandas as pd
import pytest

from earnsig.metrics import factor_regression, hit_rate, ic_by_season, perf_stats


def test_perf_stats_drawdown_and_sharpe():
    r = pd.Series([0.1, -0.5, 0.2])
    s = perf_stats(r)
    assert s["max_drawdown"] == pytest.approx(-0.5)
    assert s["total_return"] == pytest.approx(1.1 * 0.5 * 1.2 - 1)
    assert s["sharpe"] == pytest.approx(r.mean() / r.std() * np.sqrt(252))


def test_ic_and_hit_rate_perfect_signal():
    ev = pd.DataFrame({"season": ["s1"] * 20 + ["s2"] * 20, "sig": list(range(20)) * 2,
                       "car": [x - 9.5 for x in range(20)] * 2})
    ics = ic_by_season(ev, "sig", "car")
    assert (ics == 1).all()
    assert hit_rate(ev, "sig", "car")["hit_rate"] == 1.0


def test_factor_regression_recovers_betas():
    rng = np.random.default_rng(0)
    idx = pd.bdate_range("2020-01-01", periods=1000)
    f = pd.DataFrame(rng.normal(0, 0.01, (1000, 6)), index=idx, columns=["Mkt-RF", "SMB", "HML", "RMW", "CMA", "Mom"])
    y = 0.0004 + 0.5 * f["Mkt-RF"] - 0.3 * f["Mom"] + rng.normal(0, 0.001, 1000)
    out = factor_regression(y, f)
    assert out["beta_Mkt-RF"] == pytest.approx(0.5, abs=0.02)
    assert out["beta_Mom"] == pytest.approx(-0.3, abs=0.02)
    assert out["alpha_ann"] == pytest.approx(0.0004 * 252, abs=0.02)
