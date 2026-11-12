"""Research backtest under the exact competition rules."""

from dataclasses import dataclass

import pandas as pd

from bse_quant.competition import rules
from bse_quant.competition.book import book_returns, trend_entries
from bse_quant.competition.features import compute_features, intraday_vol
from bse_quant.competition.model import Model, walk_forward_forecast
from bse_quant.competition.portfolio import RiskManager, forecast_to_weights


@dataclass
class BacktestResult:
    weights: pd.DataFrame  # row t = one-session weights traded on day t (all dates; zero outside the window)
    returns: pd.Series  # daily returns inside the window
    forecast: pd.DataFrame | dict  # row t = forecast for day t+1 (one table per sleeve if there are several)
    summary: dict
    trend: pd.DataFrame | None = None  # row t = trend lots bought at t's open (held settings.trend.hold_days sessions)


def window_mask(data, settings) -> pd.Series:
    d = settings.data
    keep = data.dates >= pd.Timestamp(d.start_date)
    if not d.include_holdout:
        keep &= data.dates < pd.Timestamp(d.holdout_start)
    if not keep.any():
        raise ValueError("The test window is empty; check start_date / holdout_start")
    return pd.Series(keep, index=data.dates)


def forecasts(data, settings, model: Model | None = None) -> pd.DataFrame:
    """Walk-forward forecasts, or a fixed model's forecasts if one is given."""
    m = settings.model
    if model is None:
        return walk_forward_forecast(
            data, settings.feature_names, ridge=m.ridge, refit_days=m.refit_days, min_train_days=m.min_train_days,
            train_start=settings.data.train_start, fit_intercept=m.fit_intercept,
        )  # fmt: skip
    return model.predict(compute_features(data, model.names), intraday_vol(data))


def decision_weights(forecast, data, settings) -> pd.DataFrame:
    """Row t = weights chosen at t's close, before any drawdown brake."""
    s = settings.sizing
    return forecast_to_weights(
        forecast, intraday_vol(data), s.hurdle_bps, s.risk_per_trade, s.max_asset_weight, s.long_only,
        s.floor_weight, s.floor_weekday,
    )  # fmt: skip


def apply_drawdown_brake(traded: pd.DataFrame, data, risk: RiskManager) -> pd.DataFrame:
    """Day by day, shrink weights while the account is in a drawdown (equity starts at 1)."""
    if not risk.drawdown_limit:
        return traded
    moves = rules.intraday_returns(data).reindex(index=traded.index, columns=traded.columns).fillna(0.0).to_numpy()
    w = traded.to_numpy().copy()
    equity = peak = 1.0
    for i in range(len(w)):
        w[i] *= risk.scale(equity, peak)
        equity *= 1 + rules.day_return(w[i], moves[i])
        peak = max(peak, equity)
    return pd.DataFrame(w, index=traded.index, columns=traded.columns)


def blended_decisions(data, settings, model=None, forecast=None):
    """(weights chosen at each close summed over the sleeves, forecasts).

    `model`: None (walk-forward), a Model (one sleeve) or {sleeve name: Model}.
    `forecast`: ready-made forecasts for a single sleeve (used by the settings search).
    """
    sleeves = settings.sleeve_list()
    if forecast is not None:
        if len(sleeves) > 1:
            raise ValueError("Ready-made forecasts only work with a single sleeve")
        return decision_weights(forecast, data, sleeves[0][2]), forecast
    total, all_forecasts = 0.0, {}
    for name, weight, s in sleeves:
        m = model.get(name) if isinstance(model, dict) else model
        f = forecasts(data, s, m)
        all_forecasts[name] = f
        total = total + weight * decision_weights(f, data, s)
    if len(sleeves) == 1:
        return total, all_forecasts[sleeves[0][0]]
    return total, all_forecasts


def run_backtest(data, settings, model=None, forecast: pd.DataFrame | None = None) -> BacktestResult:
    """Backtest under the competition rules.

    `model`: None = walk-forward (re-fit as it goes); a Model, or {sleeve: Model}, = fixed models.
    Pass `forecast` to reuse forecasts already computed (single sleeve only).
    """
    window = window_mask(data, settings)
    decided, forecast = blended_decisions(data, settings, model, forecast)
    traded = decided.shift(1).fillna(0.0)
    traded = traded.where(window, 0.0)
    risk = RiskManager(settings.risk.drawdown_limit, settings.risk.drawdown_scale)
    in_window = traded.loc[window]
    traded.loc[window] = apply_drawdown_brake(in_window, data, risk)
    active = traded.loc[window].abs().sum(axis=1) > 0
    t = settings.trend
    if not t.weight:
        returns = rules.portfolio_returns(traded, data).loc[window]
        return BacktestResult(weights=traded, returns=returns, forecast=forecast, summary=rules.summary(returns, active))
    trend = trend_entries(data, t.lookback, t.hold_days, t.long_only).where(window, 0.0) * t.weight
    entries = {1: traded}
    entries[t.hold_days] = entries.get(t.hold_days, 0.0) + trend
    book = book_returns(entries, data)
    returns = book.returns.loc[window]
    active |= book.gross.loc[window] > 0
    return BacktestResult(traded, returns, forecast, rules.summary(returns, active), trend)
