import numpy as np
import pandas as pd
import pytest
from conftest import build_market_data, dates, random_market_data

from bse_quant.config import BacktestSettings, Config, StrategySettings, VolTargetSettings
from bse_quant.engine import (
    combine_strategies,
    leg_returns,
    run_backtest,
    strategy_returns,
    volatility_scale,
)
from bse_quant.strategies.base import Signal


@pytest.fixture
def tiny():
    idx = dates(3)
    close = pd.DataFrame({"A": [100.0, 110.0, 99.0], "B": [50.0, 50.0, 50.0]}, idx)
    open_ = pd.DataFrame({"A": [100.0, 105.0, 100.0], "B": [50.0, 50.0, 50.0]}, idx)
    return build_market_data(close, open_)


def positions(tiny, a_values):
    return pd.DataFrame({"A": a_values, "B": [0.0] * len(a_values)}, tiny.close.index)


def test_leg_returns(tiny):
    cc = leg_returns(tiny, "close_to_close")["A"]
    co = leg_returns(tiny, "close_to_open")["A"]
    oc = leg_returns(tiny, "open_to_close")["A"]
    assert list(cc.round(6)) == [0.0, 0.1, -0.1]
    assert list(co.round(6)) == [0.0, 0.05, round(100 / 110 - 1, 6)]
    assert list(oc.round(6)) == [0.0, round(110 / 105 - 1, 6), -0.01]


def test_leg_returns_rejects_unknown_leg(tiny):
    with pytest.raises(ValueError):
        leg_returns(tiny, "sideways")


def test_close_to_close_uses_yesterdays_position(tiny):
    sig = Signal(positions(tiny, [1.0, 1.0, 1.0]))
    r = strategy_returns(sig, tiny, cost_bps=0)
    # each asset is an equal slice of capital (1/2 here)
    assert list(r.round(6)) == [0.0, 0.05, -0.05]


def test_close_to_close_charges_costs_on_trades(tiny):
    sig = Signal(positions(tiny, [1.0, 1.0, -1.0]))
    r = strategy_returns(sig, tiny, cost_bps=10)
    cost = 10 / 10_000
    assert r.iloc[0] == pytest.approx(-cost * 1 / 2)
    assert r.iloc[1] == pytest.approx(0.05)
    assert r.iloc[2] == pytest.approx(-0.05 - cost * 2 / 2)


def test_open_to_close_uses_same_day_position_and_round_trip_costs(tiny):
    sig = Signal(positions(tiny, [0.0, 1.0, 0.0]), leg="open_to_close")
    r = strategy_returns(sig, tiny, cost_bps=10)
    cost = 10 / 10_000
    assert r.iloc[0] == pytest.approx(0.0)
    assert r.iloc[1] == pytest.approx((110 / 105 - 1) / 2 - 2 * cost / 2)
    assert r.iloc[2] == pytest.approx(0.0)


def test_close_to_open_holds_overnight(tiny):
    sig = Signal(positions(tiny, [1.0, 0.0, 0.0]), leg="close_to_open")
    r = strategy_returns(sig, tiny, cost_bps=10)
    cost = 10 / 10_000
    assert r.iloc[0] == pytest.approx(0.0)
    assert r.iloc[1] == pytest.approx(0.05 / 2 - 2 * cost / 2)
    assert r.iloc[2] == pytest.approx(0.0)


def test_long_only_removes_shorts(tiny):
    sig = Signal(positions(tiny, [-1.0, -1.0, -1.0]))
    r = strategy_returns(sig, tiny, cost_bps=0, allow_short=False)
    assert (r == 0).all()


def test_missing_positions_count_as_flat(tiny):
    sig = Signal(positions(tiny, [np.nan, np.nan, 1.0]))
    r = strategy_returns(sig, tiny, cost_bps=0)
    assert (r == 0).all()


def test_volatility_scale_hits_target_roughly():
    rng = np.random.default_rng(1)
    r = pd.Series(rng.normal(0, 0.02, 3000), index=dates(3000))
    scaled = volatility_scale(r, target_vol=0.10, lookback_days=63, max_leverage=10)
    realized = scaled.iloc[200:].std() * np.sqrt(252)
    assert realized == pytest.approx(0.10, rel=0.15)


def test_volatility_scale_respects_max_leverage():
    r = pd.Series([0.0001, -0.0001] * 200, index=dates(400))
    scaled = volatility_scale(r, target_vol=0.5, lookback_days=20, max_leverage=2)
    assert (scaled.abs() <= r.abs() * 2 + 1e-12).all()


def test_volatility_scale_strength_zero_is_unchanged():
    r = pd.Series(np.random.default_rng(2).normal(0, 0.01, 300), index=dates(300))
    out = volatility_scale(r, target_vol=0.1, lookback_days=20, max_leverage=5, strength=0)
    pd.testing.assert_series_equal(out, r)


def test_volatility_scale_does_not_peek_at_the_future():
    rng = np.random.default_rng(3)
    r = pd.Series(rng.normal(0, 0.01, 300), index=dates(300))
    changed = r.copy()
    changed.iloc[200:] *= 5
    a = volatility_scale(r, 0.1, 20, 5)
    b = volatility_scale(changed, 0.1, 20, 5)
    pd.testing.assert_series_equal(a.iloc[:200], b.iloc[:200])


def test_combine_strategies_uses_relative_weights():
    idx = dates(3)
    rets = pd.DataFrame({"x": [0.01, 0.02, 0.03], "y": [0.03, 0.02, 0.01]}, idx)
    out = combine_strategies(rets, {"x": 3.0, "y": 1.0})
    assert list(out.round(6)) == [0.015, 0.02, 0.025]


def always(value, leg="close_to_close"):
    def generate(data):
        return Signal(pd.DataFrame(value, index=data.close.index, columns=data.assets), leg)

    return generate


def make_config(strategies, **backtest_overrides):
    settings = dict(
        start_date="2020-03-01",
        holdout_start="2021-01-01",
        include_holdout=False,
        cost_bps=0.0,
        normalize_strategy_risk=False,
    )
    settings.update(backtest_overrides)
    return Config(
        backtest=BacktestSettings(**settings),
        strategies=strategies,
        volatility_targeting=VolTargetSettings(enabled=False),
    )


def test_run_backtest_blends_enabled_strategies():
    data = random_market_data(n_days=300)
    registry = {"long": always(1.0), "short": always(-1.0), "off": always(1.0)}
    config = make_config(
        {
            "long": StrategySettings(enabled=True, weight=3.0),
            "short": StrategySettings(enabled=True, weight=1.0),
            "off": StrategySettings(enabled=False, weight=5.0),
        }
    )
    result = run_backtest(data, config, registry=registry)

    assert list(result.strategy_returns.columns) == ["long", "short"]
    expected = 0.75 * result.strategy_returns["long"] + 0.25 * result.strategy_returns["short"]
    pd.testing.assert_series_equal(result.portfolio_returns, expected, check_names=False)


def test_run_backtest_respects_dates_and_holdout():
    data = random_market_data(n_days=300)  # 2020-01-01 .. 2021-02-23
    config = make_config({"long": StrategySettings()})
    result = run_backtest(data, config, registry={"long": always(1.0)})
    assert result.portfolio_returns.index.min() >= pd.Timestamp("2020-03-01")
    assert result.portfolio_returns.index.max() < pd.Timestamp("2021-01-01")

    config = make_config({"long": StrategySettings()}, include_holdout=True)
    result = run_backtest(data, config, registry={"long": always(1.0)})
    assert result.portfolio_returns.index.max() == data.dates[-1]


def test_run_backtest_needs_at_least_one_strategy():
    data = random_market_data(n_days=100)
    config = make_config({"long": StrategySettings(enabled=False)})
    with pytest.raises(ValueError, match="No strategies"):
        run_backtest(data, config, registry={"long": always(1.0)})


def test_run_backtest_passes_params_to_strategy():
    seen = {}

    def generate(data, lookback_days=10):
        seen["lookback_days"] = lookback_days
        return always(1.0)(data)

    data = random_market_data(n_days=100)
    config = make_config({"s": StrategySettings(params={"lookback_days": 42})}, start_date="2020-01-01")
    run_backtest(data, config, registry={"s": generate})
    assert seen["lookback_days"] == 42


def test_run_backtest_leave_one_out_report():
    data = random_market_data(n_days=300)
    registry = {"long": always(1.0), "short": always(-1.0)}
    config = make_config({"long": StrategySettings(), "short": StrategySettings()})
    result = run_backtest(data, config, registry=registry)
    without_short = result.portfolio_without("short")
    pd.testing.assert_series_equal(
        without_short, result.strategy_returns["long"], check_names=False
    )
