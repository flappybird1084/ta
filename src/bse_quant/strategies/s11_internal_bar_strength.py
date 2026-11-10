"""#11 Internal Bar Strength.

A close near the day's low tends to bounce the next day; a close near the high
tends to slip. IBS = (close - low) / (high - low).
"""

from bse_quant.indicators import internal_bar_strength as ibs
from bse_quant.strategies.base import Signal


def internal_bar_strength(data, low_threshold=0.2, high_threshold=0.8):
    value = ibs(data.high, data.low, data.close)
    pos = (value < low_threshold).astype(float) - (value > high_threshold).astype(float)
    return Signal(pos)
