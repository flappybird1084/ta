"""#2 Moving-average crossover.

Long when the fast average is above the slow one (an uptrend), short when below.
"""

import numpy as np

from bse_quant.indicators import sma
from bse_quant.strategies.base import Signal


def moving_average_crossover(data, fast_days=50, slow_days=200):
    if fast_days >= slow_days:
        raise ValueError("fast_days must be smaller than slow_days")
    return Signal(np.sign(sma(data.close, fast_days) - sma(data.close, slow_days)))
