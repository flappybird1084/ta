"""#7 Short-term reversal.

After an unusually big move over the last few days, bet on it snapping back.
`threshold_z` = how unusual (in standard deviations) the move must be.
With `trend_filter_days` > 0, only buy dips in uptrends and sell rallies in
downtrends (0 turns the filter off).
"""

import numpy as np

from bse_quant.indicators import daily_vol, hold, sma
from bse_quant.strategies.base import Signal


def short_term_reversal(
    data, lookback_days=3, threshold_z=1.5, vol_lookback_days=60, hold_days=3, trend_filter_days=200
):
    c = data.close
    move = c / c.shift(lookback_days) - 1
    normal_move = daily_vol(c, vol_lookback_days).replace(0, np.nan) * np.sqrt(lookback_days)
    z = move / normal_move
    signal = (-np.sign(z)).where(z.abs() > threshold_z, 0.0)
    if trend_filter_days > 0:
        uptrend = c > sma(c, trend_filter_days)
        downtrend = c < sma(c, trend_filter_days)
        signal = signal.where(((signal > 0) & uptrend) | ((signal < 0) & downtrend), 0.0)
    return Signal(hold(signal.fillna(0.0), hold_days))
