"""#16 Volatility regime switch.

Measure volatility from each day's open/high/low/close (Garman-Klass). When
it's below its median of the last `regime_days` (a calm market), follow the
`trend_days` trend; when above (a wild market), bet against the
`reversion_days` move instead.
"""

import numpy as np

from bse_quant.indicators import garman_klass_vol
from bse_quant.strategies.base import Signal


def volatility_regime_switch(data, vol_days=20, regime_days=252, trend_days=63, reversion_days=5):
    c = data.close
    vol = garman_klass_vol(data.open, data.high, data.low, c, vol_days)
    typical = vol.rolling(regime_days).median()
    calm = vol < typical
    trend = np.sign(c / c.shift(trend_days) - 1)
    reversion = -np.sign(c / c.shift(reversion_days) - 1)
    pos = trend.where(calm, reversion).where(typical.notna(), 0.0)
    return Signal(pos)
