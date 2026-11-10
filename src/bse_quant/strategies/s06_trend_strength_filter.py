"""#6 Trend with a strength filter.

Only follow the trend when the market is actually trending. The efficiency
ratio measures this: 1 = moved in a straight line, near 0 = went nowhere.
"""

import numpy as np

from bse_quant.indicators import efficiency_ratio
from bse_quant.strategies.base import Signal


def trend_strength_filter(data, lookback_days=63, min_efficiency=0.3):
    c = data.close
    direction = np.sign(c - c.shift(lookback_days))
    strong = efficiency_ratio(c, lookback_days) > min_efficiency
    return Signal(direction.where(strong, 0.0))
