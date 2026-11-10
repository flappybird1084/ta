"""#15 Lead-lag.

Sometimes one asset's move today predicts another's move tomorrow. For each
asset, find the other asset whose yesterday-return has been most correlated
with its today-return over the last `window_days`. If that correlation is
stronger than `min_correlation`, copy (or invert, if negative) the leader's
move from today.
"""

import numpy as np
import pandas as pd

from bse_quant.strategies.base import Signal


def lead_lag(data, window_days=126, min_correlation=0.1):
    r = data.close.pct_change()
    pos = pd.DataFrame(0.0, index=data.dates, columns=data.assets)
    for target in data.assets:
        best_strength = pd.Series(0.0, index=data.dates)
        prediction = pd.Series(0.0, index=data.dates)
        for leader in data.assets:
            if leader == target:
                continue
            corr = r[leader].shift(1).rolling(window_days).corr(r[target])
            stronger = (corr.abs() > best_strength).fillna(False)
            best_strength = best_strength.where(~stronger, corr.abs())
            prediction = prediction.where(~stronger, np.sign(corr) * np.sign(r[leader]))
        pos[target] = prediction.where(best_strength > min_correlation, 0.0).fillna(0.0)
    return Signal(pos)
