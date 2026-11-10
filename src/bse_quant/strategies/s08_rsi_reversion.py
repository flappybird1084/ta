"""#8 RSI(2) mean reversion with a trend filter.

A very short RSI flags sharp pull-backs. Buy when RSI drops below `oversold`
while the price is above its `trend_days` average; sell when RSI recovers past
`exit_level`. Shorts are the mirror image.
"""

from bse_quant.indicators import positions_from_signals, rsi, sma
from bse_quant.strategies.base import Signal


def rsi_reversion(data, rsi_days=2, oversold=10.0, overbought=90.0, exit_level=50.0, trend_days=200):
    c = data.close
    r = rsi(c, rsi_days)
    trend = sma(c, trend_days)
    return Signal(
        positions_from_signals(
            long_entry=(r < oversold) & (c > trend),
            long_exit=r > exit_level,
            short_entry=(r > overbought) & (c < trend),
            short_exit=r < exit_level,
        )
    )
