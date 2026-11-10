"""Engle-Granger cointegration p-values for many windows at once.

Computes exactly what `statsmodels.tsa.stattools.coint(y, x)` does with its
defaults, but for a whole batch of windows in one go on the GPU (via JAX):

1. regress y on [x, 1] and take the residuals
2. Augmented Dickey-Fuller test on the residuals (no trend), picking the
   number of lags (0..maxlag) with the lowest AIC on a common sample
3. refit with that lag on the full sample; the t-stat on the lagged level
   is turned into a p-value with MacKinnon's tables

Set the environment variable BSE_QUANT_COINT_BACKEND=statsmodels to use the
plain (slow) statsmodels loop instead.
"""

import os
import warnings
from functools import lru_cache

import numpy as np
from statsmodels.tsa.adfvalues import mackinnonp

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")  # several processes share the GPU

DEFAULT_BACKEND = os.environ.get("BSE_QUANT_COINT_BACKEND", "auto")
COLLINEAR_R2 = 1 - 100 * np.sqrt(np.finfo(float).eps)  # statsmodels' "perfectly colinear" cut-off


def coint_pvalues(ys, xs, backend=None) -> np.ndarray:
    """p-values for each row: is ys[i] cointegrated with xs[i]?"""
    backend = backend or DEFAULT_BACKEND
    if backend == "auto":
        backend = "jax" if _jax_available() else "statsmodels"
    ys, xs = np.asarray(ys, float), np.asarray(xs, float)
    if len(ys) == 0:
        return np.zeros(0)
    if backend == "statsmodels":
        return _statsmodels_pvalues(ys, xs)
    if backend == "jax":
        stats = np.asarray(_jax_kernel(ys.shape[1])(ys, xs))
        return np.array([mackinnonp(s, regression="c", N=2) for s in stats])
    raise ValueError(f"Unknown backend {backend!r}; use 'auto', 'jax' or 'statsmodels'")


def _statsmodels_pvalues(ys, xs):
    from statsmodels.tsa.stattools import coint

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return np.array([coint(y, x)[1] for y, x in zip(ys, xs)])


@lru_cache(maxsize=1)
def _jax_available():
    try:
        import jax  # noqa: F401

        return True
    except ImportError:
        return False


def adf_maxlag(n):
    """statsmodels' default ADF lag limit (Schwert 1989) for a series of length n."""
    return min(n // 2 - 1, int(np.ceil(12.0 * (n / 100.0) ** 0.25)))


@lru_cache(maxsize=None)
def _jax_kernel(n):
    """A compiled function: (ys, xs) of shape (batch, n) -> ADF statistics of shape (batch,)."""
    import jax
    import jax.numpy as jnp

    jax.config.update("jax_enable_x64", True)
    max_lag = adf_maxlag(n)
    n_cols = max_lag + 1  # lagged level + up to max_lag lagged differences
    rows = jnp.arange(n - 1)
    cols = jnp.arange(n_cols)

    def masked_ols(X, target, first_row, used_cols):
        """OLS on rows >= first_row and the first `used_cols` columns, with fixed array shapes."""
        row_on = (rows >= first_row).astype(X.dtype)
        col_on = (cols < used_cols).astype(X.dtype)
        Xm = X * row_on[:, None] * col_on[None, :]
        ym = target * row_on
        xtx = Xm.T @ Xm + jnp.diag(1 - col_on)  # unused columns get a harmless 1 on the diagonal
        coef = jnp.linalg.solve(xtx, Xm.T @ ym)
        resid = ym - Xm @ coef
        return coef, resid @ resid, row_on.sum(), xtx

    def one_window(y, x):
        # 1. cointegrating regression y = a + b x
        xc, yc = x - x.mean(), y - y.mean()
        b = (xc @ yc) / (xc @ xc)
        e = yc - b * xc
        r_squared = 1 - (e @ e) / (yc @ yc)

        # 2. ADF design: row t (0..n-2) explains diff[t] by level e[t] and diff[t-1..t-max_lag]
        diff = jnp.diff(e)
        lagged = [jnp.concatenate([jnp.zeros(j), diff[: n - 1 - j]]) for j in range(1, n_cols)]
        X = jnp.stack([e[:-1]] + lagged, axis=1)

        def aic(k):  # k columns, on the common sample (rows >= max_lag)
            _, ssr, m, _ = masked_ols(X, diff, max_lag, k)
            llf = -m / 2 * (jnp.log(2 * jnp.pi) + jnp.log(ssr / m) + 1)
            return -2 * llf + 2 * k

        aics = jax.vmap(aic)(jnp.arange(1, n_cols + 1))
        best_lag = jnp.argmin(aics)  # first (smallest) lag wins ties, as in statsmodels

        # 3. refit with the chosen lag on all usable rows
        coef, ssr, m, xtx = masked_ols(X, diff, best_lag, best_lag + 1)
        sigma2 = ssr / (m - (best_lag + 1))
        inv00 = jnp.linalg.solve(xtx, jnp.eye(n_cols)[0])[0]
        t_stat = coef[0] / jnp.sqrt(sigma2 * inv00)
        return jnp.where(r_squared >= COLLINEAR_R2, -jnp.inf, t_stat)

    return jax.jit(jax.vmap(one_window))
