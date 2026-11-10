"""#4 Cross-sectional momentum.

Every `rebalance_days`, rank the assets by their return over the last
`lookback_days` (ignoring the most recent `skip_days`, which tend to reverse).
Buy the `top_n` best, short the `bottom_n` worst, hold until the next rebalance.
"""

import numpy as np
import pandas as pd

from bse_quant.strategies.base import Signal


def cross_sectional_momentum(data, lookback_days=126, skip_days=21, rebalance_days=21, top_n=2, bottom_n=2):
    if top_n + bottom_n > len(data.assets):
        raise ValueError(f"top_n + bottom_n can't exceed the number of assets ({len(data.assets)})")
    score = data.close.shift(skip_days) / data.close.shift(lookback_days) - 1
    rank_best = score.rank(axis=1, ascending=False, method="first")
    rank_worst = score.rank(axis=1, ascending=True, method="first")
    target = (rank_best <= top_n).astype(float) - (rank_worst <= bottom_n).astype(float)
    target = target.where(score.notna().all(axis=1), 0.0, axis=0)

    is_rebalance_day = np.arange(len(target)) % rebalance_days == 0
    pos = target.where(pd.Series(is_rebalance_day, index=target.index), axis=0).ffill()
    return Signal(pos.fillna(0.0))
