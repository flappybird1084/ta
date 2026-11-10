"""#10 Gap fade (intraday).

If an asset opens far above yesterday's close, short it at the open and buy it
back at the close (and the reverse for big gaps down). "Far" means more than
`threshold_z` times the asset's normal daily move.
"""

import numpy as np

from bse_quant.indicators import daily_vol
from bse_quant.strategies.base import Signal


def gap_fade(data, threshold_z=1.0, vol_lookback_days=60):
    gap = data.open / data.close.shift(1) - 1
    normal_move = daily_vol(data.close, vol_lookback_days).replace(0, np.nan).shift(1)  # known before today's open
    z = gap / normal_move
    pos = (-np.sign(z)).where(z.abs() > threshold_z, 0.0)
    return Signal(pos, leg="open_to_close")
