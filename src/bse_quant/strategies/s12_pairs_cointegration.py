"""#12 Pairs trading with a cointegration test.

For every pair of assets, look at the last `formation_days` of (log) prices.
If the pair passes an Engle-Granger cointegration test (p-value below
`max_pvalue`), fit y = beta * x + alpha and trade the gap ("spread") back
towards normal: short the spread when it's `entry_z` standard deviations too
wide, long when too narrow, close within `exit_z`. Re-checked every
`refit_days`.

(All the cointegration tests are run in one batch, on the GPU when available;
see fast_coint.py.)
"""

import numpy as np
import pandas as pd

from bse_quant import fast_coint
from bse_quant.indicators import all_pairs, mean_reversion_bands, pair_positions_to_assets
from bse_quant.strategies.base import Signal


def pairs_cointegration(data, formation_days=252, refit_days=21, max_pvalue=0.05, entry_z=2.0, exit_z=0.5):
    log_prices = np.log(data.close)
    n = len(log_prices)
    pairs = all_pairs(data.assets)
    refit_days_list = np.arange(formation_days - 1, n, refit_days)  # day index of each re-fit

    # Every (pair, re-fit) window, tested in one batch
    ys, xs = [], []
    for y_name, x_name in pairs:
        y, x = log_prices[y_name].to_numpy(), log_prices[x_name].to_numpy()
        for t in refit_days_list:
            ys.append(y[t - formation_days + 1 : t + 1])
            xs.append(x[t - formation_days + 1 : t + 1])
    ys = np.array(ys).reshape(len(pairs) * len(refit_days_list), formation_days)
    xs = np.array(xs).reshape(ys.shape)
    usable = np.isfinite(ys).all(axis=1) & np.isfinite(xs).all(axis=1)
    pvalues = np.ones(len(ys))
    pvalues[usable] = fast_coint.coint_pvalues(ys[usable], xs[usable])

    # Fit y = beta * x + alpha in each window (ordinary least squares)
    xc = xs - xs.mean(axis=1, keepdims=True)
    yc = ys - ys.mean(axis=1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        beta = (xc * yc).sum(axis=1) / (xc * xc).sum(axis=1)
    alpha = ys.mean(axis=1) - beta * xs.mean(axis=1)
    spread_std = (yc - beta[:, None] * xc).std(axis=1)
    passed = usable & (pvalues <= max_pvalue)

    # Each day uses the most recent re-fit at or before it
    latest_fit = np.searchsorted(refit_days_list, np.arange(n), side="right") - 1
    has_fit = latest_fit >= 0
    pair_pos, betas = {}, {}
    for i, (y_name, x_name) in enumerate(pairs):
        z = np.full(n, np.nan)
        pair_beta = np.full(n, np.nan)
        rows = i * len(refit_days_list) + latest_fit[has_fit]
        ok = passed[rows]
        days = np.flatnonzero(has_fit)[ok]
        rows = rows[ok]
        y, x = log_prices[y_name].to_numpy(), log_prices[x_name].to_numpy()
        z[days] = (y[days] - beta[rows] * x[days] - alpha[rows]) / spread_std[rows]
        pair_beta[days] = beta[rows]
        z_frame = pd.DataFrame({"z": z}, index=data.dates)
        pair_pos[(y_name, x_name)] = mean_reversion_bands(z_frame, entry_z, exit_z)["z"]
        betas[(y_name, x_name)] = pd.Series(pair_beta, index=data.dates)
    return Signal(pair_positions_to_assets(pair_pos, betas, data.assets, data.dates))
