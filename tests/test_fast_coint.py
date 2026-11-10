"""The batched (JAX) cointegration test must agree with statsmodels."""

import warnings

import numpy as np
import pytest
from conftest import random_market_data
from statsmodels.tsa.stattools import coint

from bse_quant import fast_coint
from bse_quant.strategies import pairs_cointegration
from test_strategies import cointegrated_pair


def statsmodels_pvalues(ys, xs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return np.array([coint(y, x)[1] for y, x in zip(ys, xs)])


def random_windows(n_windows, length, seed):
    rng = np.random.default_rng(seed)
    xs = np.cumsum(rng.normal(0, 0.01, (n_windows, length)), axis=1)
    # mix of cointegrated (small noise) and unrelated (random walk) partners
    noise_scale = rng.uniform(0.001, 0.02, (n_windows, 1))
    related = xs * rng.uniform(0.5, 1.5, (n_windows, 1)) + rng.normal(0, 1, (n_windows, length)) * noise_scale
    unrelated = np.cumsum(rng.normal(0, 0.01, (n_windows, length)), axis=1)
    pick = rng.random((n_windows, 1)) < 0.5
    ys = np.where(pick, related, unrelated) + 4.6
    return ys, xs + 4.6


@pytest.mark.parametrize("length", [252, 120])
def test_jax_matches_statsmodels(length):
    ys, xs = random_windows(40, length, seed=length)
    expected = statsmodels_pvalues(ys, xs)
    got = fast_coint.coint_pvalues(ys, xs, backend="jax")
    np.testing.assert_allclose(got, expected, rtol=1e-6, atol=1e-10)


def test_statsmodels_backend_matches_statsmodels():
    ys, xs = random_windows(5, 252, seed=1)
    np.testing.assert_allclose(fast_coint.coint_pvalues(ys, xs, backend="statsmodels"), statsmodels_pvalues(ys, xs))


def test_perfectly_collinear_pair_counts_as_cointegrated():
    xs = np.cumsum(np.random.default_rng(3).normal(0, 0.01, (1, 252)), axis=1)
    ys = 2 * xs + 1
    assert fast_coint.coint_pvalues(ys, xs, backend="jax")[0] == pytest.approx(0.0)


def test_unknown_backend_rejected():
    with pytest.raises(ValueError):
        fast_coint.coint_pvalues(np.zeros((1, 50)), np.zeros((1, 50)), backend="abacus")


@pytest.mark.parametrize("make_data", [cointegrated_pair, random_market_data])
def test_pairs_strategy_trades_identically_on_both_backends(monkeypatch, make_data):
    data = make_data()
    monkeypatch.setattr(fast_coint, "DEFAULT_BACKEND", "statsmodels")
    slow = pairs_cointegration(data, formation_days=200, refit_days=20).positions
    monkeypatch.setattr(fast_coint, "DEFAULT_BACKEND", "jax")
    fast = pairs_cointegration(data, formation_days=200, refit_days=20).positions
    np.testing.assert_allclose(fast.to_numpy(), slow.to_numpy(), atol=1e-9)
