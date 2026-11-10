"""Features: numbers known at a day's close that might predict the NEXT day's
open-to-close move.

Every feature is a table (rows = dates, columns = assets) and uses only data up
to and including that row's close. Most are scaled by the asset's recent
volatility so that one model can be shared across all assets.
"""

import numpy as np
import pandas as pd

from bse_quant.data import MarketData
from bse_quant.indicators import rsi

VOL_DAYS = 60


def _vol(returns: pd.DataFrame) -> pd.DataFrame:
    return returns.rolling(VOL_DAYS, min_periods=20).std()


def _daily_vol(data):
    return _vol(data.close.pct_change())


def _broadcast(series: pd.Series, columns) -> pd.DataFrame:
    return pd.DataFrame({c: series for c in columns})


def _relative(df: pd.DataFrame) -> pd.DataFrame:
    """Each asset compared with the average of all assets that day."""
    return df.sub(df.mean(axis=1), axis=0)


# --- reversal: big moves tend to partly undo themselves the next day --------


def return_1d(data):
    """Today's close-to-close return, in units of daily volatility."""
    r = data.close.pct_change()
    return r / _vol(r)


def return_2d(data):
    """Last two days' return, in volatility units."""
    return data.close.pct_change(2) / (_daily_vol(data) * np.sqrt(2))


def intraday_1d(data):
    """Today's open-to-close move, in volatility units."""
    return (data.close / data.open - 1) / _daily_vol(data)


def overnight_1d(data):
    """Today's opening gap (open vs. yesterday's close), in volatility units."""
    return (data.open / data.close.shift(1) - 1) / _daily_vol(data)


def rsi_2(data):
    """2-day RSI, centred on 0 (-50 = very oversold, +50 = very overbought)."""
    return rsi(data.close, 2) - 50


def rsi_5(data):
    """5-day RSI, centred on 0."""
    return rsi(data.close, 5) - 50


def bar_position(data):
    """Where the close sits in today's high-low range (-0.5 = at the low, +0.5 = at the high)."""
    span = (data.high - data.low).where(lambda s: s > 0)
    return ((data.close - data.low) / span - 0.5).fillna(0.0)


def relative_return_1d(data):
    """Today's return vs. the other assets, in volatility units."""
    return _relative(return_1d(data))


def market_return_1d(data):
    """The average asset's return today (same value for every asset), in volatility units."""
    m = data.close.pct_change().mean(axis=1)
    return _broadcast(m / m.rolling(VOL_DAYS, min_periods=20).std(), data.assets)


def market_intraday_1d(data):
    """The average asset's open-to-close move today, in volatility units."""
    m = (data.close / data.open - 1).mean(axis=1)
    return _broadcast(m / m.rolling(VOL_DAYS, min_periods=20).std(), data.assets)


# --- momentum / drift: slower effects ---------------------------------------


def momentum_60(data):
    """Return over ~3 months, in volatility units."""
    return data.close.pct_change(60) / (_daily_vol(data) * np.sqrt(60))


def momentum_250(data):
    """Return over ~1 year, in volatility units."""
    return data.close.pct_change(250) / (_daily_vol(data) * np.sqrt(250))


def relative_momentum_60(data):
    """3-month return vs. the other assets."""
    return _relative(momentum_60(data))


def intraday_drift_60(data):
    """Average open-to-close move over ~3 months (does this asset tend to rise during the day?)."""
    oc = data.close / data.open - 1
    return oc.rolling(60, min_periods=20).mean() / _daily_vol(data)


def distance_from_sma200(data):
    """How far the price is above its 200-day average, in volatility units."""
    sma = data.close.rolling(200, min_periods=100).mean()
    return (data.close / sma - 1) / (_daily_vol(data) * np.sqrt(20))


# --- activity -----------------------------------------------------------------


def volume_z(data):
    """Today's volume vs. the last 20 days (z-score of log volume)."""
    lv = np.log(data.volume.where(data.volume > 0))
    return (lv - lv.rolling(20, min_periods=10).mean()) / lv.rolling(20, min_periods=10).std()


def range_z(data):
    """Today's high-low range vs. its 20-day average (0 = normal, 1 = twice as wide)."""
    rng = (data.high - data.low) / data.close
    return rng / rng.rolling(20, min_periods=10).mean() - 1


def volatility_ratio(data):
    """Last week's volatility vs. the last 3 months (above 0 = getting wilder)."""
    r = data.close.pct_change()
    return r.rolling(5).std() / _vol(r) - 1


FEATURES = {
    f.__name__: f
    for f in [
        return_1d, return_2d, intraday_1d, overnight_1d, rsi_2, rsi_5, bar_position,
        relative_return_1d, market_return_1d, market_intraday_1d,
        momentum_60, momentum_250, relative_momentum_60, intraday_drift_60, distance_from_sma200,
        volume_z, range_z, volatility_ratio,
    ]
}  # fmt: skip


def compute_features(data: MarketData, names) -> dict:
    return {name: FEATURES[name](data).replace([np.inf, -np.inf], np.nan) for name in names}


def intraday_vol(data: MarketData, days: int = VOL_DAYS) -> pd.DataFrame:
    """Recent volatility of open-to-close moves: the risk of holding an asset for one session."""
    return (data.close / data.open - 1).rolling(days, min_periods=20).std()
