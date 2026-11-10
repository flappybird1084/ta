"""A stand-in for the competition evaluator: feeds a Trader one day at a time."""

from dataclasses import dataclass

import pandas as pd

from bse_quant.competition import rules
from bse_quant.data import MarketData

FIELDS = ["open", "high", "low", "close", "volume"]


@dataclass
class SimulationResult:
    weights: pd.DataFrame  # row t = weights traded on day t
    returns: pd.Series


def _rows(data, sl):
    return MarketData(*(getattr(data, f).iloc[sl] for f in FIELDS))


def simulate(make_trader, data: MarketData, start) -> SimulationResult:
    """Trade every day from `start` on. `make_trader(history)` builds the trader from the days before."""
    first = int(data.dates.searchsorted(pd.Timestamp(start)))
    if first < 1:
        raise ValueError("Need at least one day of history before `start`")
    trader = make_trader(_rows(data, slice(0, first - 1)))
    weights = pd.DataFrame(0.0, index=data.dates[first:], columns=data.assets)
    equity = peak = 1.0
    previous = {a: 0.0 for a in data.assets}
    for t in range(first - 1, len(data.dates) - 1):
        bars = {a: {f: getattr(data, f).iat[t, j] for f in FIELDS} | {"date": data.dates[t]} for j, a in enumerate(data.assets)}
        state = {"equity": equity, "peak_equity": peak, "previous_weights": previous}
        w = trader.on_day_close(bars, state)
        rules.check_weights(w)
        weights.iloc[t + 1 - first] = [w[a] for a in data.assets]
        moves = (data.close.iloc[t + 1] / data.open.iloc[t + 1] - 1).to_numpy()
        equity *= 1 + rules.day_return(weights.iloc[t + 1 - first].to_numpy(), moves)
        peak, previous = max(peak, equity), w
    returns = rules.portfolio_returns(weights, data).loc[weights.index]
    return SimulationResult(weights=weights, returns=returns)
