"""#1 Time-series momentum.

Idea: assets that went up over the last year tend to keep going up.
Long if the price is higher than `lookback_days` ago, short if lower.
"""

import numpy as np

from bse_quant.strategies.base import Signal


def time_series_momentum(data, lookback_days=252):
    past_return = data.close / data.close.shift(lookback_days) - 1
    return Signal(np.sign(past_return))
