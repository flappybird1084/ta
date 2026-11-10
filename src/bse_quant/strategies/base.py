from dataclasses import dataclass

import pandas as pd

# When a position is held:
#   close_to_close - decided at today's close, held until tomorrow's close (the usual case)
#   close_to_open  - decided at today's close, sold at tomorrow's open (overnight only)
#   open_to_close  - decided at today's open, sold at today's close (intraday only)
LEGS = ("close_to_close", "close_to_open", "open_to_close")


@dataclass
class Signal:
    """What a strategy wants to hold.

    positions: rows = dates, columns = assets. +1 means "fully long this
    asset's slice of the account", -1 fully short, 0 flat. Values in between
    (or a bit beyond) are allowed.
    """

    positions: pd.DataFrame
    leg: str = "close_to_close"

    def __post_init__(self):
        if self.leg not in LEGS:
            raise ValueError(f"leg must be one of {LEGS}, got {self.leg!r}")
        self.positions = self.positions.astype(float).fillna(0.0)
