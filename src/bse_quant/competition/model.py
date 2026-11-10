"""The forecast: a ridge regression of tomorrow's open-to-close move on today's features.

One model is shared by all assets. The target is tomorrow's open-to-close
return divided by the asset's recent intraday volatility, so a forecast of
+0.1 means "a tenth of a typical day's move, upwards". Multiplying by the
volatility turns it back into a return.

A training row is a (day s, asset) pair: features at s's close, target = the
move on day s+1. A model "fit at the close of day t" may only use rows with
s <= t-1, because day t's move is the last one we have seen.
"""

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from bse_quant.competition.features import VOL_DAYS, compute_features, intraday_vol
from bse_quant.data import MarketData


@dataclass
class Model:
    names: list
    coefs: dict  # per feature, applied to feature / scale
    scales: dict  # typical size of each feature in the training data
    intercept: float = 0.0

    def predict(self, features: dict, vol: pd.DataFrame) -> pd.DataFrame:
        """Forecast of the next session's open-to-close return (0.002 = +0.2%)."""
        z = sum(features[n] / self.scales[n] * self.coefs[n] for n in self.names) + self.intercept
        return z * vol

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Model":
        return cls(**d)


def _target(data: MarketData, vol_days: int = VOL_DAYS) -> pd.DataFrame:
    """Row s: the move on day s+1, in units of the intraday volatility known at s."""
    oc = data.close / data.open - 1
    return oc.shift(-1) / intraday_vol(data, vol_days)


def _stack(features: dict, target: pd.DataFrame, names, rows: np.ndarray):
    """Training matrix from the given row positions, with incomplete rows dropped."""
    X = np.stack([features[n].to_numpy()[rows].ravel() for n in names], axis=1)
    y = target.to_numpy()[rows].ravel()
    ok = np.isfinite(X).all(axis=1) & np.isfinite(y)
    return X[ok], y[ok]


def _solve(X, y, names, ridge, fit_intercept) -> Model:
    scales = X.std(axis=0)
    scales[scales == 0] = 1.0
    Z = X / scales
    intercept = 0.0
    if fit_intercept:
        intercept = float(y.mean())
        y = y - intercept
    n = len(y)
    beta = np.linalg.solve(Z.T @ Z / n + ridge * np.eye(len(names)), Z.T @ y / n)
    return Model(
        names=list(names),
        coefs={k: float(b) for k, b in zip(names, beta)},
        scales={k: float(s) for k, s in zip(names, scales)},
        intercept=intercept,
    )


def fit_model(data, names, end, ridge=0.05, start=None, fit_intercept=False, vol_days=VOL_DAYS, features=None) -> Model:
    """Fit on days in [start, end), each paired with the following day's move (which is <= end)."""
    features = compute_features(data, names) if features is None else features
    target = _target(data, vol_days)
    end_pos = data.dates.searchsorted(pd.Timestamp(end), side="right") - 1  # the close we're standing at
    first = 0 if start is None else data.dates.searchsorted(pd.Timestamp(start))
    X, y = _stack(features, target, names, np.arange(first, max(first, end_pos)))
    if len(y) == 0:
        raise ValueError("No complete training rows; need more history")
    return _solve(X, y, names, ridge, fit_intercept)


def walk_forward_forecast(
    data, names, ridge=0.05, refit_days=252, min_train_days=252, train_start=None, fit_intercept=False, vol_days=VOL_DAYS
) -> pd.DataFrame:
    """Forecasts made the honest way: re-fit every `refit_days` on everything seen so far.

    Row t is the forecast for day t+1, made at t's close by a model fit at or before t.
    """
    features = compute_features(data, names)
    target = _target(data, vol_days)
    vol = intraday_vol(data, vol_days)
    first = 0 if train_start is None else int(data.dates.searchsorted(pd.Timestamp(train_start)))
    out = pd.DataFrame(np.nan, index=data.dates, columns=data.assets)
    n = len(data.dates)
    for fit_at in range(first + min_train_days, n, refit_days):
        X, y = _stack(features, target, names, np.arange(first, fit_at))
        if len(y) < len(names) + 1:
            continue
        model = _solve(X, y, names, ridge, fit_intercept)
        until = min(n, fit_at + refit_days)
        block = {k: v.iloc[fit_at:until] for k, v in features.items()}
        out.iloc[fit_at:until] = model.predict(block, vol.iloc[fit_at:until]).to_numpy()
    return out


def models_to_dict(models) -> dict:
    """What model.json holds: one model, or {"sleeves": {name: model}}."""
    if isinstance(models, Model):
        return models.to_dict()
    return {"sleeves": {name: m.to_dict() for name, m in models.items()}}


def models_from_dict(d: dict):
    if "sleeves" in d:
        return {name: Model.from_dict(m) for name, m in d["sleeves"].items()}
    return Model.from_dict(d)
