"""The competition's scoring rules, written out exactly."""

import numpy as np
import pandas as pd

from bse_quant import metrics
from bse_quant.data import MarketData

COST_BPS = 5.0  # charged on entry AND on exit
MAX_GROSS = 1.0
TOLERANCE = 1e-9


def intraday_returns(data: MarketData) -> pd.DataFrame:
    """Open-to-close return of every asset on every day."""
    return data.close / data.open - 1


def check_weights(weights) -> None:
    """Raise ValueError unless the weights are finite and sum(|w|) <= 1."""
    values = np.asarray(list(weights.values()) if isinstance(weights, dict) else weights, dtype=float)
    if not np.isfinite(values).all():
        raise ValueError(f"All weights must be finite numbers, got {weights}")
    gross = np.abs(values).sum()
    if gross > MAX_GROSS + TOLERANCE:
        raise ValueError(f"Total gross exposure {gross:.6f} is above the {MAX_GROSS:.0%} limit")


def portfolio_returns(weights: pd.DataFrame, data: MarketData) -> pd.Series:
    """Daily portfolio return after costs.

    `weights` row t = the allocation TRADED on day t (bought at t's open, sold at t's close).
    """
    w = weights.reindex(index=data.dates, columns=data.assets).fillna(0.0)
    gross = w.abs().sum(axis=1)
    if (gross > MAX_GROSS + TOLERANCE).any():
        worst = gross.idxmax()
        raise ValueError(f"Total gross exposure {gross.max():.6f} on {worst.date()} is above the {MAX_GROSS:.0%} limit")
    r = intraday_returns(data).fillna(0.0)
    pnl = (w * r).sum(axis=1)
    entry_cost = w.abs().sum(axis=1)
    exit_cost = (w.abs() * (1 + r)).sum(axis=1)
    return pnl - (entry_cost + exit_cost) * COST_BPS / 10_000


def day_return(weights, moves) -> float:
    """One day's return after costs, for arrays of weights and open-to-close moves."""
    w, r = np.asarray(weights, dtype=float), np.asarray(moves, dtype=float)
    cost = (np.abs(w) + np.abs(w) * (1 + r)).sum() * COST_BPS / 10_000
    return float(w @ r - cost)


def summary(returns: pd.Series, active: pd.Series | None = None) -> dict:
    """The three judged metrics plus a few extras."""
    if active is None:
        active = returns != 0
    return {
        "Sharpe": metrics.sharpe_ratio(returns),
        "CAGR": metrics.cagr(returns),
        "Max drawdown": metrics.max_drawdown(returns),
        "Total return": metrics.total_return(returns),
        "Volatility": metrics.annual_volatility(returns),
        "Active days": int(active.sum()),
        "Days": len(returns),
    }
