"""#19 On-balance volume confirmation.

Only trust a trend when volume agrees with it. Long when both the price and
on-balance volume are above their `ma_days` averages; short when both are below.
"""

from bse_quant.indicators import on_balance_volume, sma
from bse_quant.strategies.base import Signal


def obv_confirmation(data, ma_days=50):
    c = data.close
    obv = on_balance_volume(c, data.volume)
    up = (c > sma(c, ma_days)) & (obv > sma(obv, ma_days))
    down = (c < sma(c, ma_days)) & (obv < sma(obv, ma_days))
    return Signal(up.astype(float) - down.astype(float))
