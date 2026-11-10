"""#3 Donchian channel breakout (the "turtle" rule).

Buy when the close beats the highest high of the last `entry_days`;
get out when it drops below the lowest low of the last `exit_days`.
Shorts are the mirror image.
"""

from bse_quant.indicators import positions_from_signals
from bse_quant.strategies.base import Signal


def donchian_breakout(data, entry_days=55, exit_days=20):
    c = data.close
    # .shift(1) so today's bar is compared with the channel from *before* today
    entry_high = data.high.rolling(entry_days).max().shift(1)
    entry_low = data.low.rolling(entry_days).min().shift(1)
    exit_high = data.high.rolling(exit_days).max().shift(1)
    exit_low = data.low.rolling(exit_days).min().shift(1)
    return Signal(
        positions_from_signals(
            long_entry=c > entry_high,
            long_exit=c < exit_low,
            short_entry=c < entry_low,
            short_exit=c > exit_high,
        )
    )
