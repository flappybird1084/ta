"""Small building blocks shared by the strategies.

Every function takes and returns tables (rows = dates, columns = assets) and
only ever looks backwards in time.
"""

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def sma(df: pd.DataFrame, days: int) -> pd.DataFrame:
    """Simple moving average."""
    return df.rolling(days).mean()


def rolling_zscore(df: pd.DataFrame, days: int) -> pd.DataFrame:
    """How many standard deviations today's value is from its recent average."""
    mean = df.rolling(days).mean()
    std = df.rolling(days).std()
    return (df - mean) / std.replace(0, np.nan)


def daily_vol(close: pd.DataFrame, days: int) -> pd.DataFrame:
    """Standard deviation of daily returns over the last `days` days."""
    return close.pct_change().rolling(days).std()


def rsi(close: pd.DataFrame, days: int) -> pd.DataFrame:
    """Wilder's Relative Strength Index, 0-100."""
    change = close.diff()
    gain = change.clip(lower=0).ewm(alpha=1 / days, adjust=False, min_periods=days).mean()
    loss = (-change).clip(lower=0).ewm(alpha=1 / days, adjust=False, min_periods=days).mean()
    out = 100 - 100 / (1 + gain / loss)
    out = out.where(loss > 0, 100.0)  # no losses at all -> maximally strong
    return out.where(gain.notna())


def efficiency_ratio(close: pd.DataFrame, days: int) -> pd.DataFrame:
    """Kaufman efficiency ratio: net move / total distance travelled (1 = straight line)."""
    net = (close - close.shift(days)).abs()
    path = close.diff().abs().rolling(days).sum()
    return net / path.replace(0, np.nan)


def on_balance_volume(close: pd.DataFrame, volume: pd.DataFrame) -> pd.DataFrame:
    """Running total of volume, added on up days and subtracted on down days."""
    direction = np.sign(close.diff()).fillna(0)
    return (direction * volume).cumsum()


def garman_klass_vol(open_, high, low, close, days: int) -> pd.DataFrame:
    """Annualised volatility estimated from each day's open/high/low/close."""
    hl = np.log(high / low) ** 2
    co = np.log(close / open_) ** 2
    daily_variance = (0.5 * hl - (2 * np.log(2) - 1) * co).clip(lower=0)
    return np.sqrt(daily_variance.rolling(days).mean() * TRADING_DAYS)


def internal_bar_strength(high, low, close) -> pd.DataFrame:
    """Where the close sits in the day's range: 0 = at the low, 1 = at the high."""
    day_range = high - low
    return ((close - low) / day_range.replace(0, np.nan)).fillna(0.5)


def positions_from_signals(long_entry, long_exit, short_entry, short_exit) -> pd.DataFrame:
    """Turn entry/exit signals (True/False tables) into positions of +1, 0 or -1.

    Each day: exits are checked first, then entries, so a long can flip
    straight to short on the same day.
    """
    out = np.zeros(long_entry.shape)
    le, lx, se, sx = (s.fillna(False).to_numpy(bool) for s in (long_entry, long_exit, short_entry, short_exit))
    state = np.zeros(long_entry.shape[1])
    for t in range(len(out)):
        state = np.where((state > 0) & lx[t], 0, state)
        state = np.where((state < 0) & sx[t], 0, state)
        state = np.where(le[t] & ~se[t], 1, state)
        state = np.where(se[t] & ~le[t], -1, state)
        out[t] = state
    return pd.DataFrame(out, index=long_entry.index, columns=long_entry.columns)


def mean_reversion_bands(z: pd.DataFrame, entry: float, exit: float) -> pd.DataFrame:
    """Bet on a z-score returning to zero.

    Go short when z rises above +entry, long when it falls below -entry, and
    close the trade once z is back inside +/-exit. Missing z means flat.
    """
    missing = z.isna()
    return positions_from_signals(
        long_entry=z < -entry,
        long_exit=(z > -exit) | missing,
        short_entry=z > entry,
        short_exit=(z < exit) | missing,
    )


def hold(signal: pd.DataFrame, days: int) -> pd.DataFrame:
    """Keep each non-zero signal in place for `days` days (the newest signal wins)."""
    marked = signal.where(signal != 0)
    return marked.ffill(limit=days - 1).fillna(0) if days > 1 else signal.fillna(0)


def pair_positions_to_assets(pair_positions: dict, betas: dict, assets: list, index) -> pd.DataFrame:
    """Turn spread trades into per-asset positions.

    A pair (y, x) with position +1 means: buy 1 unit of y, sell `beta` units of x.
    Each asset can be in at most (n_assets - 1) pairs, so totals are divided by
    that to keep each asset's position roughly within -1..+1.
    """
    out = pd.DataFrame(0.0, index=index, columns=assets)
    for (y, x), pos in pair_positions.items():
        pos = pos.fillna(0)
        out[y] += pos
        out[x] -= pos * betas[(y, x)].fillna(0)
    return out / max(len(assets) - 1, 1)


def all_pairs(assets: list) -> list[tuple[str, str]]:
    return [(a, b) for i, a in enumerate(assets) for b in assets[i + 1 :]]
