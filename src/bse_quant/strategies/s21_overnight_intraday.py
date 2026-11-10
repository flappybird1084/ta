"""#21 Overnight vs intraday.

A day's return splits into the overnight part (close -> next open) and the
intraday part (open -> close), which often behave differently. Hold every
asset only during one `session` ("overnight" or "intraday"); `direction` is
1 for long, -1 for short. Note it trades twice a day, so costs add up fast.
"""

import pandas as pd

from bse_quant.strategies.base import Signal

SESSIONS = {"overnight": "close_to_open", "intraday": "open_to_close"}


def overnight_intraday(data, session="overnight", direction=1):
    if session not in SESSIONS:
        raise ValueError('session must be "overnight" or "intraday"')
    if direction not in (1, -1):
        raise ValueError("direction must be 1 (long) or -1 (short)")
    pos = pd.DataFrame(float(direction), index=data.dates, columns=data.assets)
    return Signal(pos, leg=SESSIONS[session])
