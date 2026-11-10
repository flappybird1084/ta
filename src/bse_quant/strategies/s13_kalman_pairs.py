"""#13 Pairs trading with a Kalman-filter hedge ratio.

Like #12, but the relationship y = beta * x + alpha is re-estimated every day
by a Kalman filter, so it can drift slowly over time. Today's surprise
(actual y minus predicted y), divided by its expected size, is the z-score.
`delta` = how fast beta/alpha may drift; `observation_noise` = how noisy the
relationship is day to day. The defaults give z-scores of roughly unit size.
"""

import numpy as np
import pandas as pd

from bse_quant.indicators import all_pairs, mean_reversion_bands, pair_positions_to_assets
from bse_quant.strategies.base import Signal


def kalman_pairs(data, delta=1e-5, observation_noise=1e-4, entry_z=2.0, exit_z=0.5, warmup_days=60):
    pairs = all_pairs(data.assets)
    log_prices = np.log(data.close)
    # one column per pair; all pairs are filtered side by side
    y = log_prices[[a for a, _ in pairs]].to_numpy()
    x = log_prices[[b for _, b in pairs]].to_numpy()
    n, k = y.shape
    drift = delta / (1 - delta)
    z = np.full((n, k), np.nan)
    beta = np.full((n, k), np.nan)
    state = np.zeros((k, 2))  # [beta, alpha] for each pair
    cov = np.zeros((k, 2, 2))
    seen = np.zeros(k, dtype=int)
    for t in range(n):
        ok = np.isfinite(y[t]) & np.isfinite(x[t])
        obs = np.stack([x[t], np.ones(k)], axis=1)
        prior_cov = cov + drift * np.eye(2)
        surprise = y[t] - (obs * state).sum(axis=1)
        surprise_var = np.einsum("ki,kij,kj->k", obs, prior_cov, obs) + observation_noise
        gain = np.einsum("kij,kj->ki", prior_cov, obs) / surprise_var[:, None]
        new_state = state + gain * surprise[:, None]
        new_cov = prior_cov - np.einsum("ki,kj,kjl->kil", gain, obs, prior_cov)
        state = np.where(ok[:, None], new_state, state)
        cov = np.where(ok[:, None, None], new_cov, cov)
        seen += ok
        ready = ok & (seen > warmup_days)
        z[t, ready] = surprise[ready] / np.sqrt(surprise_var[ready])
        beta[t, ready] = state[ready, 0]
    pair_pos, betas = {}, {}
    for i, pair in enumerate(pairs):
        pair_pos[pair] = mean_reversion_bands(pd.DataFrame({"z": z[:, i]}, index=data.dates), entry_z, exit_z)["z"]
        betas[pair] = pd.Series(beta[:, i], index=data.dates)
    return Signal(pair_positions_to_assets(pair_pos, betas, data.assets, data.dates))
