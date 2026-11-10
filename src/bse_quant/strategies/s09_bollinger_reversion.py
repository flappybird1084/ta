"""#9 Bollinger band / z-score reversion.

When the price stretches `entry_z` standard deviations away from its moving
average, bet on it coming back; close once it's within `exit_z`.
"""

from bse_quant.indicators import mean_reversion_bands, rolling_zscore
from bse_quant.strategies.base import Signal


def bollinger_reversion(data, lookback_days=20, entry_z=2.0, exit_z=0.5):
    z = rolling_zscore(data.close, lookback_days)
    return Signal(mean_reversion_bands(z, entry_z, exit_z))
