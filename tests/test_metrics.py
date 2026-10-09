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


def test_ic_p_value_uses_t_distribution_with_few_seasons():
    from earnsig.metrics import ic_summary

    ics = pd.Series([0.20, -0.05, 0.15, 0.10, 0.02, 0.12, -0.03, 0.18, 0.05, 0.08])
    out = ic_summary(ics)
    assert out["ic_t_crit"] == pytest.approx(2.262, abs=1e-3)  # 9 degrees of freedom
    from scipy import stats
    assert out["ic_p"] == pytest.approx(2 * stats.t.sf(out["ic_t"], 9))


def test_sharpe_standard_error():
    r = pd.Series(np.random.default_rng(1).normal(0.001, 0.01, 504))
    s = perf_stats(r)
    assert s["sharpe_se"] == pytest.approx(np.sqrt((1 + 0.5 * s["sharpe"] ** 2) / 2), rel=1e-6)


def test_event_time_exposure_counts_names_and_one_leg_days():
    from earnsig.metrics import event_time_exposure

    events = pd.DataFrame({"t0_pos": [0, 0, 2], "leg": [1, 1, -1]})
    idx = pd.RangeIndex(8)
    rets = pd.DataFrame({"long": [np.nan] + [0.01] * 5 + [np.nan] * 2,
                         "short": [np.nan] * 3 + [0.0] * 5}, index=idx)
    out = event_time_exposure(events, rets, 8, H=5)
    assert out["event_positions_long"] == 2 and out["event_positions_short"] == 1
    assert out["event_median_names_long"] == 2.0
    assert out["event_share_one_leg"] == pytest.approx(4 / 8)
