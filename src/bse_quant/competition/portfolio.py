"""From forecasts to weights, and risk control."""

from dataclasses import dataclass

import numpy as np
import pandas as pd


def forecast_to_weights(
    forecast, vol, hurdle_bps, risk_per_trade, max_asset_weight, long_only=False, floor_weight=0.0, floor_weekday=-1
) -> pd.DataFrame:
    """Trade only where the forecast beats the hurdle; size each trade for equal risk.

    - An asset is traded when |forecast| > hurdle_bps (it must at least pay the ~10 bps of costs).
    - Its weight is risk_per_trade / volatility (so a wild asset gets a smaller weight),
      capped at max_asset_weight, with the forecast's sign.
    - If the total is above 100%, everything is scaled down to fit.
    - Activity floor: on a day with no trade, put `floor_weight` (e.g. 0.01 = 1%) in
      the asset with the strongest forecast, so the account always trades a little
      (the competition requires >= 30 trading days). `floor_weekday` = 0..4 limits
      this to days decided on that weekday (4 = Friday); -1 = every day.
    """
    f = forecast.fillna(0.0)
    if long_only:
        f = f.clip(lower=0)
    active = f.abs() > hurdle_bps / 10_000
    with np.errstate(divide="ignore", invalid="ignore"):
        size = (risk_per_trade / vol).clip(upper=max_asset_weight)
    w = (np.sign(f) * size).where(active, 0.0).fillna(0.0)
    gross = w.abs().sum(axis=1)
    w = w.div(gross.clip(lower=1.0), axis=0)
    if floor_weight > 0:
        w = _add_floor(w, f, floor_weight, floor_weekday)
    return w


def _add_floor(w, f, floor_weight, floor_weekday):
    quiet = (w.abs().sum(axis=1) == 0) & (f.abs().max(axis=1) > 0)
    if floor_weekday >= 0:
        quiet &= w.index.day_of_week == floor_weekday
    if not quiet.any():
        return w
    strongest = f.loc[quiet].abs().idxmax(axis=1)
    w = w.copy()
    for day, asset in strongest.items():
        w.loc[day, asset] = np.sign(f.loc[day, asset]) * floor_weight
    return w


@dataclass
class RiskManager:
    """Cut exposure while the account is in a deep drawdown."""

    drawdown_limit: float | None = None  # e.g. 0.10 = act once 10% below the peak; None or 0 = off
    drawdown_scale: float = 0.5  # multiply all weights by this while in the drawdown

    def scale(self, equity: float, peak: float) -> float:
        if not self.drawdown_limit:
            return 1.0
        peak = max(peak, 1.0)
        return self.drawdown_scale if equity / peak - 1 < -self.drawdown_limit else 1.0
