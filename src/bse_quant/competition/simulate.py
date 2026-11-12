"""A stand-in for the competition evaluator: feeds a Trader one day at a time.

The returns are booked from the trader's orders (`last_orders`: what to buy at
each open and how many sessions to hold it), or, for a trader without them,
from the returned weights as one-session trades.
"""

from dataclasses import dataclass

import pandas as pd

from bse_quant.competition import rules
from bse_quant.competition.book import MAX_HOLD_DAYS, Book, book_returns
from bse_quant.data import MarketData

FIELDS = ["open", "high", "low", "close", "volume"]


@dataclass
class SimulationResult:
    weights: pd.DataFrame  # row t = weights the trader said it would hold on day t
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
    holds = range(1, MAX_HOLD_DAYS + 1)
    orders = {h: weights.copy() for h in holds}
    book, gap = Book(len(data.assets), holds), (data.open / data.close.shift(1) - 1).fillna(0.0)
    peak = 1.0
    previous = {a: 0.0 for a in data.assets}
    for t in range(first - 1, len(data.dates) - 1):
        bars = {a: {f: getattr(data, f).iat[t, j] for f in FIELDS} | {"date": data.dates[t]} for j, a in enumerate(data.assets)}
        state = {"equity": book.equity, "peak_equity": peak, "previous_weights": previous}
        w = trader.on_day_close(bars, state)
        rules.check_weights(w)
        row = t + 1 - first
        weights.iloc[row] = [w[a] for a in data.assets]
        todays = getattr(trader, "last_orders", None) or {1: w}
        for h, o in todays.items():
            orders[h].iloc[row] = [o.get(a, 0.0) for a in data.assets]
        moves = (data.close.iloc[t + 1] / data.open.iloc[t + 1] - 1).to_numpy()
        book.session(gap.iloc[t + 1].to_numpy(), moves, {h: orders[h].iloc[row].to_numpy() for h in holds})
        peak, previous = max(peak, book.equity), w
    used = {h: o for h, o in orders.items() if o.abs().to_numpy().sum() > 0} or {1: orders[1]}
    returns = book_returns(used, data).returns.loc[weights.index]
    return SimulationResult(weights=weights, returns=returns)
