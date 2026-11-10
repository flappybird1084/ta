"""A stand-in for the other teams, used to estimate our rank-based score.

We can't see the other submissions, so we assume a field of plausible ones:
the 20 classic strategies from src/bse_quant/strategies (momentum, moving
averages, RSI, pairs, ...) translated to the competition rules (decided at the
close, traded the next session, sum(|w|) <= 1), plus simple "always long"
entries. Each is a table of weights TRADED on each day.
"""

import pandas as pd

from bse_quant.data import MarketData
from bse_quant.strategies import STRATEGIES


def student_field(data: MarketData) -> dict:
    n = len(data.assets)
    field = {
        "always_long_all": pd.DataFrame(1.0 / n, index=data.dates, columns=data.assets),
        "always_long_half": pd.DataFrame(0.5 / n, index=data.dates, columns=data.assets),
    }
    for name, generate in STRATEGIES.items():
        signal = generate(data)
        if signal.leg != "close_to_close":  # these use the same day's open/close: not allowed
            continue
        pos = signal.positions.reindex(index=data.dates, columns=data.assets).fillna(0.0).clip(-1, 1) / n
        field[f"naive_{name}"] = pos.shift(1).fillna(0.0)
    return field


# A few "competent" teams: versions of our own kind of model with other settings.
COMPETENT = [
    {"features": ["return_1d", "intraday_1d", "overnight_1d", "rsi_2", "bar_position", "market_intraday_1d"], "ridge": 1.0, "hurdle_bps": 20.0, "long_only": False},
    {"features": ["return_1d", "intraday_1d", "overnight_1d"], "ridge": 0.05, "hurdle_bps": 20.0, "long_only": False},
    {"features": ["return_1d"], "ridge": 0.05, "hurdle_bps": 10.0, "long_only": False},
    {"features": ["market_intraday_1d"], "ridge": 1.0, "hurdle_bps": 15.0, "long_only": True},
]  # fmt: skip


def competitor_fields(data: MarketData, settings, seed=0) -> dict:
    """Daily returns of four assumed fields of other teams (all on the same days).

    - naive: the classic strategies + always-long entries
    - naive+competent: plus four teams running models like ours (at ~1% risk per trade)
    - cautious: naive teams that cut their exposure to 30%, + the competent ones
    - strong: naive+competent plus three teams with a Sharpe of about 1 (10% volatility)
    """
    import copy

    import numpy as np

    from bse_quant.competition import rules
    from bse_quant.competition.backtest import run_backtest

    naive = pd.DataFrame({k: rules.portfolio_returns(w, data) for k, w in student_field(data).items()})
    competent = {}
    for i, c in enumerate(COMPETENT):
        s = copy.deepcopy(settings)
        s.sleeves = {}
        s.data.include_holdout = True
        s.data.start_date = str(data.dates[0].date())
        s.features = {name: name in c["features"] for name in s.features}
        s.model.ridge, s.sizing.hurdle_bps, s.sizing.long_only = c["ridge"], c["hurdle_bps"], c["long_only"]
        s.sizing.risk_per_trade, s.sizing.floor_weight = 0.01, 0.0
        competent[f"competent_{i}"] = run_backtest(data, s).returns.reindex(data.dates).fillna(0.0)
    competent = pd.DataFrame(competent)
    rng = np.random.default_rng(seed)
    strong = pd.DataFrame(rng.normal(0.10 / 252, 0.10 / np.sqrt(252), (len(data.dates), 3)), data.dates,
                          [f"sharpe1_{i}" for i in range(3)])  # fmt: skip
    return {
        "naive": naive,
        "naive+competent": pd.concat([naive, competent], axis=1),
        "cautious": pd.concat([naive * 0.3, competent], axis=1),
        "strong": pd.concat([naive, competent, strong], axis=1),
    }
