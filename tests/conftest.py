"""Shared test helpers: build small, fully controlled fake markets."""

import numpy as np
import pandas as pd
import pytest

from bse_quant.data import MarketData


def build_market_data(close, open_=None, high=None, low=None, volume=None):
    """Make a MarketData from a close-price DataFrame.

    Anything not given is filled in sensibly: open = previous close,
    high/low = 0.5% above/below the open-close range, volume = 1,000,000.
    """
    close = pd.DataFrame(close).astype(float)
    if open_ is None:
        open_ = close.shift(1).fillna(close)
    open_ = pd.DataFrame(open_, index=close.index, columns=close.columns).astype(float)
    if high is None:
        high = np.maximum(open_, close) * 1.005
    if low is None:
        low = np.minimum(open_, close) * 0.995
    if volume is None:
        volume = pd.DataFrame(1_000_000.0, index=close.index, columns=close.columns)
    return MarketData(
        open=open_,
        high=pd.DataFrame(high, index=close.index, columns=close.columns).astype(float),
        low=pd.DataFrame(low, index=close.index, columns=close.columns).astype(float),
        close=close,
        volume=pd.DataFrame(volume, index=close.index, columns=close.columns).astype(float),
    )


def dates(n_days):
    return pd.bdate_range("2020-01-01", periods=n_days)


def prices_from_returns(returns, start=100.0):
    """Turn a DataFrame (or dict) of daily returns into prices."""
    returns = pd.DataFrame(returns)
    return start * (1 + returns).cumprod()


def random_market_data(n_days=400, n_assets=4, seed=0):
    """A random-walk market with realistic-looking OHLCV bars."""
    rng = np.random.default_rng(seed)
    idx = dates(n_days)
    cols = [chr(ord("A") + i) for i in range(n_assets)]
    overnight = rng.normal(0.0002, 0.006, (n_days, n_assets))
    intraday = rng.normal(0.0002, 0.012, (n_days, n_assets))
    log_open = np.zeros((n_days, n_assets))
    log_close = np.zeros((n_days, n_assets))
    prev_close = np.log(100.0)
    for t in range(n_days):
        log_open[t] = prev_close + overnight[t]
        log_close[t] = log_open[t] + intraday[t]
        prev_close = log_close[t]
    if n_assets >= 2:
        # make the last asset track the one before it, so pair strategies have a real pair
        gap = np.zeros(n_days)
        for t in range(1, n_days):
            gap[t] = 0.8 * gap[t - 1] + rng.normal(0, 0.01)
        log_close[:, -1] = log_close[:, -2] + gap
        log_open[:, -1] = log_close[:, -1] - intraday[:, -1]
    open_ = pd.DataFrame(np.exp(log_open), idx, cols)
    close = pd.DataFrame(np.exp(log_close), idx, cols)
    wiggle = np.abs(rng.normal(0, 0.006, (n_days, n_assets)))
    high = np.maximum(open_, close) * (1 + wiggle)
    low = np.minimum(open_, close) * (1 - wiggle)
    volume = pd.DataFrame(rng.lognormal(14, 0.3, (n_days, n_assets)), idx, cols)
    return build_market_data(close, open_, high, low, volume)


@pytest.fixture
def market():
    return random_market_data()
