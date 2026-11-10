"""#20 Range-compression breakout (NR7).

A day with the narrowest high-low range of the last `nr_days` often comes
before a big move. Within the next `setup_window` days, go long if the price
closes above that narrow day's high, short if below its low. Hold `hold_days`.
"""

from bse_quant.indicators import hold
from bse_quant.strategies.base import Signal


def range_compression_breakout(data, nr_days=7, setup_window=5, hold_days=10):
    day_range = data.high - data.low
    narrow = day_range <= day_range.rolling(nr_days).min()
    setup_high = data.high.where(narrow)
    setup_low = data.low.where(narrow)
    if setup_window > 1:
        setup_high = setup_high.ffill(limit=setup_window - 1)
        setup_low = setup_low.ffill(limit=setup_window - 1)
    # compare today's close with a narrow day from *before* today
    setup_high, setup_low = setup_high.shift(1), setup_low.shift(1)
    c = data.close
    signal = (c > setup_high).astype(float) - (c < setup_low).astype(float)
    return Signal(hold(signal, hold_days))
