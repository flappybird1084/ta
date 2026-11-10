"""#22 Calendar effects.

Be long every asset only on historically strong days:
- turn_of_month: the last `days_before` and first `days_after` trading days of each month
- pre_holiday: the last trading day before a market holiday
- weekdays: any days listed, e.g. ["Mon", "Fri"]
A position held overnight from day t earns day t+1's return, so we buy at the
close *before* each chosen day. (The trading calendar is known in advance.)
"""

import numpy as np
import pandas as pd

from bse_quant.strategies.base import Signal

WEEKDAYS = ["Mon", "Tue", "Wed", "Thu", "Fri"]


def calendar_effects(data, turn_of_month=True, days_before=1, days_after=3, pre_holiday=True, weekdays=()):
    for day in weekdays:
        if day not in WEEKDAYS:
            raise ValueError(f"weekdays must be from {WEEKDAYS}, got {day!r}")
    idx = data.dates
    as_series = pd.Series(idx, index=idx)
    chosen = pd.Series(False, index=idx)

    if turn_of_month:
        month = idx.to_period("M")
        from_start = as_series.groupby(month).cumcount()
        from_end = as_series.groupby(month).cumcount(ascending=False)
        unfinished_month = month == month[-1]  # we can't know the last day of the final month yet
        chosen |= (from_start < days_after) | ((from_end < days_before) & ~unfinished_month)

    if pre_holiday:
        next_day = as_series.shift(-1)
        has_next = next_day.notna().to_numpy()
        gap = np.zeros(len(idx), dtype=int)
        gap[has_next] = np.busday_count(
            idx[has_next].values.astype("datetime64[D]"), next_day[has_next].values.astype("datetime64[D]")
        )
        chosen |= gap > 1  # more than one business day until the next session

    if weekdays:
        chosen |= pd.Series(np.isin(idx.day_of_week, [WEEKDAYS.index(d) for d in weekdays]), index=idx)

    buy_today = chosen.shift(-1, fill_value=False).to_numpy()
    pos = pd.DataFrame(np.repeat(buy_today[:, None], len(data.assets), axis=1).astype(float), idx, data.assets)
    return Signal(pos)
