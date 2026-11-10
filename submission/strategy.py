"""BSE Quant Track submission: the program the evaluator calls.

    import strategy
    strategy.initialize(history="path/to/development/csvs")   # optional, see below
    weights = strategy.on_day_close(daily_data, portfolio_state)

- `daily_data`: the day that just closed, one entry per asset, e.g.
  {"Asset_A": {"Date": "2023-10-09", "Open": ..., "High": ..., "Low": ..., "Close": ..., "Volume": ...}, ...}
  (a DataFrame with one row per asset also works; field names are case-insensitive).
- Returns {"Asset_A": 0.25, "Asset_B": -0.10, ...}: signed fractions of equity
  for the NEXT session, with sum(|w|) <= 1.

`initialize(history=...)` loads the public development history (a folder of
Asset_*.csv files, or a MarketData) so the indicators are warm on day one. If
it's never called, the first call to on_day_close loads ../data automatically
when that folder exists; otherwise the strategy warms up from the days it is
fed (it returns zero weights until it has enough history).

The model coefficients (model.json) and settings (settings.toml) were fixed
before the hidden evaluation, using only the development data; see README.md.
"""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
try:
    import bse_quant  # noqa: F401
except ImportError:  # running from a checkout without installing the package
    sys.path.insert(0, str(HERE.parent / "src"))

from bse_quant.competition.trader import Trader  # noqa: E402
from bse_quant.data import MarketData, load_market_data  # noqa: E402

DEFAULT_HISTORY = HERE.parent / "data"
_trader = None


def _load_history(history):
    if history is None or isinstance(history, MarketData):
        return history
    data = load_market_data(history)
    rename = {a: a if a.startswith("Asset_") else f"Asset_{a}" for a in data.assets}
    return MarketData(*(getattr(data, f).rename(columns=rename) for f in ("open", "high", "low", "close", "volume")))


def initialize(history=None, model_path=HERE / "model.json", settings_path=HERE / "settings.toml"):
    """Set up the trader. `history`: folder of Asset_*.csv files, a MarketData, or None."""
    global _trader
    _trader = Trader.from_files(model_path, settings_path, history=_load_history(history))
    return _trader


def on_day_close(daily_data, portfolio_state=None) -> dict:
    """The day that just closed in, tomorrow's weights out."""
    if _trader is None:
        initialize(history=DEFAULT_HISTORY if DEFAULT_HISTORY.exists() else None)
    return _trader.on_day_close(daily_data, portfolio_state)
