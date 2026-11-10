"""The backtester: turns strategy positions into daily profit and loss.

Conventions
- Every asset gets an equal slice of the account. A position of +1 in one
  asset means that slice is fully invested; the strategy's daily return is the
  average over all assets.
- Trading costs (`cost_bps`, in 1/100ths of a percent) are charged on every
  unit bought or sold.
- Strategies are blended by their returns, using relative weights.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from bse_quant.config import Config
from bse_quant.data import MarketData
from bse_quant.strategies.base import Signal

TRADING_DAYS = 252


def leg_returns(data: MarketData, leg: str) -> pd.DataFrame:
    """Each asset's return over the part of the day a position is held."""
    if leg == "close_to_close":
        r = data.close / data.close.shift(1) - 1
    elif leg == "close_to_open":
        r = data.open / data.close.shift(1) - 1
    elif leg == "open_to_close":
        r = data.close / data.open - 1
    else:
        raise ValueError(f"Unknown leg {leg!r}")
    return r.fillna(0.0)


def strategy_returns(signal: Signal, data: MarketData, cost_bps: float, allow_short: bool = True) -> pd.Series:
    """Daily returns of one strategy, after trading costs."""
    pos = signal.positions.reindex(index=data.dates, columns=data.assets).fillna(0.0)
    if not allow_short:
        pos = pos.clip(lower=0)
    if signal.leg == "open_to_close":
        # bought at today's open, sold at today's close
        held = pos
        traded = 2 * pos.abs()
    elif signal.leg == "close_to_open":
        # bought at yesterday's close, sold at today's open
        held = pos.shift(1).fillna(0.0)
        traded = 2 * held.abs()
    else:
        # decided at yesterday's close, held through today
        held = pos.shift(1).fillna(0.0)
        traded = (pos - held).abs()
    pnl = held * leg_returns(data, signal.leg) - traded * cost_bps / 10_000
    return pnl.sum(axis=1) / len(data.assets)


def volatility_scale(returns, target_vol, lookback_days, max_leverage, strength=1.0):
    """Scale returns up in calm times and down in wild times to aim at `target_vol`.

    Uses only volatility measured up to *yesterday*. `strength` blends between
    no scaling (0) and full scaling (1).
    """
    vol = returns.rolling(lookback_days, min_periods=max(5, lookback_days // 3)).std() * np.sqrt(TRADING_DAYS)
    with np.errstate(divide="ignore"):
        scale = (target_vol / vol).clip(upper=max_leverage)
    scale = scale.shift(1).fillna(1.0)
    scale = 1 + strength * (scale - 1)
    return returns * scale


def combine_strategies(returns: pd.DataFrame, weights: dict) -> pd.Series:
    """Weighted average of strategy returns (weights are relative: 3 and 1 means 75% / 25%)."""
    total = sum(weights[name] for name in returns.columns)
    return sum(returns[name] * weights[name] / total for name in returns.columns)


@dataclass
class BacktestResult:
    strategy_returns: pd.DataFrame  # one column per strategy, inside the test window
    portfolio_returns: pd.Series
    positions: dict  # strategy name -> positions table
    weights: dict
    config: Config
    _full_strategy_returns: pd.DataFrame
    _window: pd.Index

    def portfolio_without(self, name: str) -> pd.Series:
        """What the portfolio would have done with one strategy switched off."""
        rest = self._full_strategy_returns.drop(columns=[name])
        if rest.empty:
            return pd.Series(0.0, index=self._window)
        return _portfolio(rest, self.weights, self.config).loc[self._window]


def _portfolio(full_returns: pd.DataFrame, weights: dict, config: Config) -> pd.Series:
    combined = combine_strategies(full_returns, weights)
    vt = config.volatility_targeting
    if vt.enabled and vt.weight > 0:
        combined = volatility_scale(combined, vt.target_vol, vt.lookback_days, vt.max_leverage, vt.weight)
    return combined


def run_backtest(data: MarketData, config: Config, registry=None) -> BacktestResult:
    if registry is None:
        from bse_quant.strategies import STRATEGIES as registry

    bt = config.backtest
    active = {name: s for name, s in config.strategies.items() if s.enabled and s.weight > 0}
    if not active:
        raise ValueError("No strategies are switched on. In config.toml set enabled = true and weight > 0 for one.")

    full_returns = {}
    positions = {}
    for name, settings in active.items():
        signal = registry[name](data, **settings.params)
        r = strategy_returns(signal, data, bt.cost_bps, bt.allow_short)
        if bt.normalize_strategy_risk:
            r = volatility_scale(r, bt.strategy_risk_target, bt.strategy_risk_lookback_days, bt.strategy_max_leverage)
        full_returns[name] = r
        positions[name] = signal.positions
    full = pd.DataFrame(full_returns)

    in_window = data.dates >= pd.Timestamp(bt.start_date)
    if not bt.include_holdout:
        in_window &= data.dates < pd.Timestamp(bt.holdout_start)
    window = data.dates[in_window]
    if len(window) == 0:
        raise ValueError("The date range is empty. Check start_date / holdout_start in config.toml.")

    weights = {name: s.weight for name, s in active.items()}
    return BacktestResult(
        strategy_returns=full.loc[window],
        portfolio_returns=_portfolio(full, weights, config).loc[window],
        positions={name: p.loc[window] for name, p in positions.items()},
        weights=weights,
        config=config,
        _full_strategy_returns=full,
        _window=window,
    )
