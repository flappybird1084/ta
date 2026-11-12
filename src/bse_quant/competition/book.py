"""Positions held for several days (the organisers allow holds of up to a month).

Every position is a "lot": bought at some day's open, held for a fixed number
of sessions, and sold at the close of its last session. So the program always
knows exactly how long it holds each position and what it is worth.

    res = book_returns({1: intraday_weights, 21: trend_entries}, data)

`{hold_days: entries}`: entries row t = the weights (fractions of equity at
t's open) bought at t's open and held for `hold_days` sessions. A one-session
hold is exactly the daily open-to-close trade of rules.portfolio_returns.

Costs: 5 bps of entry notional at the open and 5 bps of exit notional at the
close, nothing while a lot is just held. A new lot only gets capital that is
free: if the open lots plus the new ones would exceed 100% of equity, the new
ones are scaled down.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from bse_quant.competition import rules
from bse_quant.competition.settings import MAX_HOLD_DAYS  # noqa: F401  (about a month of trading days)
from bse_quant.data import MarketData


@dataclass
class BookResult:
    returns: pd.Series  # daily return after costs
    gross: pd.Series  # sum(|positions|) / equity just after the open's trades


class Book:
    """The open lots and the equity of an account, one session at a time.

    Used by both the research backtest (book_returns) and the live Trader, so
    they keep exactly the same books.
    """

    def __init__(self, n_assets: int, holds):
        self.equity = 1.0
        # lots[h][k] = signed exposure (in units of starting equity) of the h-session lots with k+1 sessions left
        self.lots = {h: np.zeros((h, n_assets)) for h in holds}

    def add_assets(self, k: int):
        self.lots = {h: np.hstack([v, np.zeros((h, k))]) for h, v in self.lots.items()}

    def exposure(self) -> np.ndarray:
        return sum(v.sum(axis=0) for v in self.lots.values())

    def session(self, gap, intraday, entries: dict) -> tuple:
        """One session: the overnight gap, the open's new lots, the open-to-close move, the close's exits.

        `entries[h]` = weights (fractions of equity at the open) to buy now and hold h sessions.
        Returns (the day's return, gross exposure / equity just after the open).
        """
        cost = rules.COST_BPS / 10_000
        start = self.equity
        for v in self.lots.values():
            self.equity += (v * gap).sum()  # overnight move of the lots carried in
            v *= 1 + gap
        held = sum(np.abs(v).sum() for v in self.lots.values())
        new = {h: np.asarray(w, dtype=float) * self.equity for h, w in entries.items()}
        want = sum(np.abs(v).sum() for v in new.values())
        free = max(self.equity - held, 0.0)
        scale = min(1.0, free / want) if want > 0 else 0.0
        for h, v in new.items():
            self.lots[h][h - 1] += v * scale
        gross = (held + want * scale) / self.equity if self.equity > 0 else 0.0  # vs. equity at the open
        self.equity -= want * scale * cost
        for h, v in self.lots.items():
            self.equity += (v * intraday).sum()
            v *= 1 + intraday
            self.equity -= np.abs(v[0]).sum() * cost  # lots whose last session is today are sold at the close
            self.lots[h] = np.vstack([v[1:], np.zeros((1, v.shape[1]))])
        return self.equity / start - 1, gross


def check_holds(holds):
    if any(h < 1 or h > MAX_HOLD_DAYS for h in holds):
        raise ValueError(f"Hold periods must be 1..{MAX_HOLD_DAYS} sessions (positions close within a month)")


def book_returns(entries: dict, data: MarketData) -> BookResult:
    holds = sorted(entries)
    check_holds(holds)
    tables = {h: entries[h].reindex(index=data.dates, columns=data.assets).fillna(0.0).to_numpy() for h in holds}
    total = sum(np.abs(t) for t in tables.values()).sum(axis=1)
    if (total > rules.MAX_GROSS + rules.TOLERANCE).any():
        raise ValueError(f"New entries on one day add up to {total.max():.4f} gross, above the 100% limit")
    opens, closes = data.open.to_numpy(), data.close.to_numpy()
    gap = np.vstack([np.zeros(opens.shape[1]), np.nan_to_num(opens[1:] / closes[:-1] - 1)])
    intraday = np.nan_to_num(closes / opens - 1)
    book = Book(len(data.assets), holds)
    returns, gross = np.zeros(len(data.dates)), np.zeros(len(data.dates))
    for t in range(len(data.dates)):
        returns[t], gross[t] = book.session(gap[t], intraday[t], {h: tables[h][t] for h in holds})
    return BookResult(pd.Series(returns, data.dates), pd.Series(gross, data.dates))


def trend_weights(data: MarketData, lookback: int, hold_days: int, long_only: bool = True) -> pd.DataFrame:
    """Row t = the trend cohort decided at t's close (to buy at the next open). See trend_entries."""
    close = data.close
    if lookback == 0:
        signal = pd.DataFrame(1.0, close.index, close.columns).where(close.notna())
    else:
        signal = np.sign(close / close.shift(lookback) - 1)
    if long_only:
        signal = signal.clip(lower=0)
    signal = signal.fillna(0.0)
    gross = signal.abs().sum(axis=1).replace(0, np.nan)
    return signal.div(gross, axis=0).fillna(0.0) / hold_days


def trend_entries(data: MarketData, lookback: int, hold_days: int, long_only: bool = True) -> pd.DataFrame:
    """Daily cohorts of a simple trend rule, decided at the previous close.

    Signal: the sign of the last `lookback` days' close-to-close return
    (lookback 0 = always long, i.e. buy and hold). The account is split into
    `hold_days` cohorts: each day one cohort (1 / hold_days of the account)
    is invested equally across the signalled assets and held `hold_days` sessions.
    """
    return trend_weights(data, lookback, hold_days, long_only).shift(1).fillna(0.0)
