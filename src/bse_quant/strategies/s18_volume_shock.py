"""#18 Volume shocks.

A day with unusually heavy trading (volume more than `volume_z` standard
deviations above the last `lookback_days`) often marks news. In
"continuation" mode, follow that day's direction; in "reversal" mode, bet
against it (exhaustion). Hold for `hold_days`.
"""

import numpy as np

from bse_quant.indicators import hold
from bse_quant.strategies.base import Signal


def volume_shock(data, lookback_days=20, volume_z=2.0, mode="continuation", hold_days=5):
    if mode not in ("continuation", "reversal"):
        raise ValueError('mode must be "continuation" or "reversal"')
    log_volume = np.log(data.volume.where(data.volume > 0))
    baseline = log_volume.shift(1).rolling(lookback_days)
    z = (log_volume - baseline.mean()) / baseline.std()
    direction = np.sign(data.close.pct_change())
    flip = 1.0 if mode == "continuation" else -1.0
    signal = (direction * flip).where(z > volume_z, 0.0).fillna(0.0)
    return Signal(hold(signal, hold_days))
