import numpy as np
import pandas as pd
import pytest
from conftest import dates

from bse_quant import metrics


def series(values):
    return pd.Series(values, index=dates(len(values)), dtype=float)


def test_total_return_and_cagr():
    r = series([0.001] * 252)
    assert metrics.total_return(r) == pytest.approx(1.001**252 - 1)
    assert metrics.cagr(r) == pytest.approx(1.001**252 - 1)


def test_annual_volatility():
    r = series([0.01, -0.01] * 100)
    assert metrics.annual_volatility(r) == pytest.approx(r.std() * np.sqrt(252))


def test_sharpe_ratio():
    r = series([0.02, -0.01] * 100)
    expected = r.mean() / r.std() * np.sqrt(252)
    assert metrics.sharpe_ratio(r) == pytest.approx(expected)


def test_sharpe_is_zero_when_no_risk():
    assert metrics.sharpe_ratio(series([0.0] * 10)) == 0.0


def test_sortino_only_penalises_downside():
    up_heavy = series([0.03, -0.01] * 50)
    assert metrics.sortino_ratio(up_heavy) > metrics.sharpe_ratio(up_heavy)


def test_max_drawdown():
    # equity: 1.0 -> 1.1 -> 0.88 -> 0.968
    r = series([0.1, -0.2, 0.1])
    assert metrics.max_drawdown(r) == pytest.approx(-0.2)


def test_max_drawdown_counts_a_loss_on_day_one():
    assert metrics.max_drawdown(series([-0.1, 0.05])) == pytest.approx(-0.1)


def test_hit_rate_ignores_flat_days():
    r = series([0.01, -0.01, 0.0, 0.02])
    assert metrics.hit_rate(r) == pytest.approx(2 / 3)


def test_summary_has_all_fields():
    out = metrics.summary(series([0.01, -0.005] * 100))
    for key in ["CAGR", "Volatility", "Sharpe", "Sortino", "Max drawdown", "Calmar", "Hit rate", "Time in market"]:
        assert key in out


def test_summary_of_empty_series_is_safe():
    out = metrics.summary(series([]))
    assert out["Sharpe"] == 0.0
