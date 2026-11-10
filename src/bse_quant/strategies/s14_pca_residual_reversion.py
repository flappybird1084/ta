"""#14 PCA residual reversion.

Most of each day's move is "the market" moving everything together. Principal
component analysis over the last `window_days` finds that common part
(`n_components` of them). What's left over for each asset is its own
idiosyncratic move. If that left-over has piled up over the last
`residual_days`, bet on it reversing.
"""

import numpy as np
import pandas as pd

from bse_quant.indicators import mean_reversion_bands
from bse_quant.strategies.base import Signal


def pca_residual_reversion(data, window_days=60, n_components=1, residual_days=5, entry_z=1.5, exit_z=0.5):
    if not 0 < n_components < len(data.assets):
        raise ValueError("n_components must be at least 1 and less than the number of assets")
    returns = data.close.pct_change().to_numpy()
    n = len(returns)
    z = np.full(returns.shape, np.nan)
    for t in range(window_days, n):
        window = returns[t - window_days + 1 : t + 1]
        if not np.isfinite(window).all():
            continue
        centered = window - window.mean(axis=0)
        _, vectors = np.linalg.eigh(np.cov(centered, rowvar=False))
        common = vectors[:, ::-1][:, :n_components]  # biggest components first
        residuals = centered - centered @ common @ common.T
        piled_up = residuals[-residual_days:].sum(axis=0)
        typical = residuals.std(axis=0, ddof=1) * np.sqrt(residual_days)
        with np.errstate(divide="ignore", invalid="ignore"):
            z[t] = np.where(typical > 0, piled_up / typical, np.nan)
    z_frame = pd.DataFrame(z, index=data.dates, columns=data.assets)
    return Signal(mean_reversion_bands(z_frame, entry_z, exit_z))
