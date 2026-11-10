"""#5 Volatility-scaled trend.

Average the trend direction over several look-backs (a score from -1 to +1),
then size each asset so wild assets get smaller positions than calm ones.
"""

import numpy as np

from bse_quant.indicators import TRADING_DAYS, daily_vol
from bse_quant.strategies.base import Signal


def volatility_scaled_trend(
    data, lookbacks=(21, 63, 126, 252), target_asset_vol=0.20, vol_lookback_days=60, max_leverage=2.0
):
    c = data.close
    trend = sum(np.sign(c / c.shift(n) - 1) for n in lookbacks) / len(lookbacks)
    annual_vol = daily_vol(c, vol_lookback_days) * np.sqrt(TRADING_DAYS)
    size = (target_asset_vol / annual_vol).clip(upper=max_leverage)
    return Signal(trend * size)
