"""The program the competition evaluator runs, one day at a time.

    trader = Trader.from_files("submission/model.json", "competition.toml", history=...)
    weights = trader.on_day_close(daily_data, portfolio_state)

It keeps the recent price history, recomputes the features after every close
and returns tomorrow's weights. It also keeps its own books (book.Book): every
position, how many sessions it has left, and the account's equity, so it
always knows its real PnL. `last_orders` = {hold_days: {asset: weight}} are
the new positions to buy at the next open and how long to hold each. It is checked against the research backtest
(tests/test_competition.py) so both trade exactly the same way.
"""

import json

import numpy as np
import pandas as pd

from bse_quant.competition import rules
from bse_quant.competition.book import Book, trend_weights
from bse_quant.competition.features import compute_features, intraday_vol
from bse_quant.competition.model import Model, models_from_dict
from bse_quant.competition.portfolio import RiskManager, forecast_to_weights
from bse_quant.data import MarketData, parse_price

FIELDS = ["open", "high", "low", "close", "volume"]
ALIASES = {"close/last": "close", "adj close": "close", "o": "open", "h": "high", "l": "low", "c": "close", "v": "volume"}
MAX_HISTORY = 600  # enough for the slowest feature (1-year momentum + volatility scaling)


def _field(key) -> str:
    k = str(key).strip().lower()
    return ALIASES.get(k, k)


def parse_daily_data(daily_data):
    """(date, table with one row per asset and columns open/high/low/close/volume).

    Accepts {asset: {field: value}} or a DataFrame with one row per asset.
    Field names are case-insensitive; prices like "$28.43" are fine.
    """
    if isinstance(daily_data, pd.DataFrame):
        rows = {asset: row.to_dict() for asset, row in daily_data.iterrows()}
    else:
        rows = dict(daily_data)
    date, table = None, {}
    for asset, values in rows.items():
        values = {_field(k): v for k, v in dict(values).items() if not (np.isscalar(v) and pd.isna(v))}
        if date is None and values.get("date") is not None:
            date = pd.Timestamp(values["date"])
        table[asset] = {f: parse_price(values[f]) for f in FIELDS}
    return date, pd.DataFrame(table).T[FIELDS].astype(float)


class Trader:
    def __init__(self, model, settings, history: MarketData | None = None):
        """`model`: one Model, or {sleeve name: Model} when competition.toml has [sleeves]."""
        self.sleeves = settings.sleeve_list()
        self.models = model if isinstance(model, dict) else {name: model for name, _, _ in self.sleeves}
        missing = [name for name, _, _ in self.sleeves if name not in self.models]
        if missing:
            raise ValueError(f"No fitted model for sleeve(s) {missing}; re-run `uv run competition-fit`")
        self.risk = RiskManager(settings.risk.drawdown_limit, settings.risk.drawdown_scale)
        self.trend = settings.trend
        self.history = None if history is None else {f: getattr(history, f).iloc[-MAX_HISTORY:].copy() for f in FIELDS}
        self.assets = [] if self.history is None else list(self.history["close"].columns)
        self.holds = sorted({1, self.trend.hold_days} if self.trend.weight else {1})
        self.book = Book(len(self.assets), self.holds)
        self.last_orders = None  # {hold_days: {asset: weight}} bought at the open of the session about to be reported
        self.peak = 1.0

    @property
    def equity(self) -> float:
        return self.book.equity

    def positions(self) -> list:
        """Every open position: asset, weight (of equity), its hold period and the sessions it has left."""
        out = []
        for h, lots in self.book.lots.items():
            for k, row in enumerate(lots):
                for asset, x in zip(self.assets, row):
                    if x:
                        out.append({"asset": asset, "weight": float(x / self.equity), "hold_days": h, "sessions_left": k + 1})
        return out

    @classmethod
    def from_files(cls, model_path, settings_path, history: MarketData | None = None):
        from bse_quant.competition.settings import load_settings

        model = models_from_dict(json.loads(open(model_path).read()))
        return cls(model, load_settings(settings_path), history)

    def on_day_close(self, daily_data, portfolio_state=None) -> dict:
        """Take the day that just closed; return weights for the next session.

        Equity and peak are tracked here from our own trades (they match the
        evaluator's), so `portfolio_state` is accepted but not needed.
        """
        date, bars = parse_daily_data(daily_data)
        self._book_session(bars)
        self._append(date, bars)
        new_assets = [a for a in self.history["close"].columns if a not in self.assets]
        if new_assets:
            self.assets += new_assets
            self.book.add_assets(len(new_assets))
        orders = self._decide()
        orders[1] = orders[1] * self.risk.scale(self.equity, self.peak)
        self.last_orders = {h: {a: float(w.get(a, 0.0)) for a in self.assets} for h, w in orders.items()}
        # what we expect to hold next session: the lots carried over plus the new ones
        carried = pd.Series(self.book.exposure() / self.equity if self.equity > 0 else 0.0, self.assets)
        total = carried + sum(orders.values()).reindex(self.assets).fillna(0.0)
        gross = total.abs().sum()
        if gross > rules.MAX_GROSS:  # price drift of the carried lots; the book scales new buys to fit the same way
            total *= rules.MAX_GROSS / gross
        out = {asset: float(total.get(asset, 0.0)) for asset in bars.index}
        rules.check_weights(out)
        return out

    def _book_session(self, bars):
        """Book the session that just closed: overnight gap, the open's buys, the day's move, the close's sales."""
        if self.last_orders is None:
            return
        prev_close = self.history["close"].iloc[-1].reindex(self.assets)
        b = bars.reindex(self.assets)
        gap = (b["open"] / prev_close - 1).fillna(0.0).to_numpy()
        intraday = (b["close"] / b["open"] - 1).fillna(0.0).to_numpy()
        entries = {h: np.array([w.get(a, 0.0) for a in self.assets]) for h, w in self.last_orders.items()}
        self.book.session(gap, intraday, entries)
        self.peak = max(self.peak, self.equity)

    def _append(self, date, bars):
        if self.history is None:
            self.history = {f: pd.DataFrame(columns=bars.index, dtype=float) for f in FIELDS}
        last = self.history["close"].index
        if date is None:
            date = (last[-1] + pd.offsets.BDay(1)) if len(last) else pd.Timestamp("2000-01-03")
        new_assets = [a for a in bars.index if a not in self.history["close"].columns]
        for f in FIELDS:
            df = self.history[f]
            if new_assets:  # an asset we haven't seen before: its history starts today
                df = df.reindex(columns=[*df.columns, *new_assets])
            row = bars[f].reindex(df.columns)
            if date in df.index:  # a day we already have (e.g. the history is replayed): update it in place
                df.loc[date] = row
                self.history[f] = df
            else:
                self.history[f] = pd.concat([df, row.to_frame(date).T]).iloc[-MAX_HISTORY:]

    def _decide(self) -> dict:
        """{hold_days: weights} to buy at the next open."""
        data = MarketData(**self.history)
        if len(data.dates) < 2:
            return {h: pd.Series(0.0, index=data.assets) for h in self.holds}
        names = sorted({n for m in self.models.values() for n in m.names})
        feats = {k: v.iloc[[-1]] for k, v in compute_features(data, names).items()}
        vol = intraday_vol(data).iloc[[-1]]
        total = 0.0
        for name, weight, settings in self.sleeves:
            forecast = self.models[name].predict(feats, vol)
            s = settings.sizing
            w = forecast_to_weights(
                forecast, vol, s.hurdle_bps, s.risk_per_trade, s.max_asset_weight, s.long_only, s.floor_weight, s.floor_weekday
            )
            total = total + weight * w.iloc[0]
        orders = {1: total.replace([np.inf, -np.inf], 0.0).fillna(0.0)}
        t = self.trend
        if t.weight:
            trend = trend_weights(data, t.lookback, t.hold_days, t.long_only).iloc[-1] * t.weight
            orders[t.hold_days] = orders.get(t.hold_days, 0.0) + trend
        return orders
