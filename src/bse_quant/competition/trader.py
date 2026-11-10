"""The program the competition evaluator runs, one day at a time.

    trader = Trader.from_files("submission/model.json", "competition.toml", history=...)
    weights = trader.on_day_close(daily_data, portfolio_state)

It keeps the recent price history, recomputes the features after every close
and returns tomorrow's weights. It is checked against the research backtest
(tests/test_competition.py) so both trade exactly the same way.
"""

import json

import numpy as np
import pandas as pd

from bse_quant.competition import rules
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
        self.history = None if history is None else {f: getattr(history, f).iloc[-MAX_HISTORY:].copy() for f in FIELDS}
        self.last_weights = None  # what we hold during the session that is about to be reported
        self.equity = self.peak = 1.0

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
        weights = self._decide() * self.risk.scale(self.equity, self.peak)
        self.last_weights = weights
        out = {asset: float(weights.get(asset, 0.0)) for asset in bars.index}
        rules.check_weights(out)
        return out

    def _book_session(self, bars):
        if self.last_weights is None:
            return
        w = self.last_weights.reindex(bars.index).fillna(0.0)
        moves = (bars["close"] / bars["open"] - 1).fillna(0.0)
        self.equity *= 1 + rules.day_return(w.to_numpy(), moves.to_numpy())
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

    def _decide(self) -> pd.Series:
        data = MarketData(**self.history)
        if len(data.dates) < 2:
            return pd.Series(0.0, index=data.assets)
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
        return total.replace([np.inf, -np.inf], 0.0).fillna(0.0)
