import numpy as np
import pandas as pd
import pytest
from conftest import build_market_data, dates, random_market_data

from bse_quant.competition import rules
from bse_quant.competition.book import MAX_HOLD_DAYS, book_returns, trend_entries

BP = rules.COST_BPS / 10_000


def one_asset(opens, closes):
    idx = dates(len(closes))
    return build_market_data(pd.DataFrame({"A": closes}, idx), open_=pd.DataFrame({"A": opens}, idx))


def test_one_day_holds_match_the_daily_rules():
    data = random_market_data(n_days=200, n_assets=3, seed=1)
    rng = np.random.default_rng(1)
    w = pd.DataFrame(rng.uniform(-0.3, 0.3, (200, 3)), data.dates, data.assets)
    expected = rules.portfolio_returns(w, data)
    got = book_returns({1: w}, data).returns
    np.testing.assert_allclose(got.to_numpy(), expected.to_numpy(), atol=1e-12)


def test_a_three_day_hold_earns_the_whole_move_and_pays_costs_once():
    # enter at day 0's open (100), hold through day 2's close (121)
    data = one_asset(opens=[100.0, 112.0, 115.0, 120.0], closes=[110.0, 111.0, 121.0, 120.0])
    entries = pd.DataFrame({"A": [1.0, 0.0, 0.0, 0.0]}, data.dates)
    res = book_returns({3: entries}, data)
    equity = (1 + res.returns).prod()
    # costs come out of cash (as in the daily rules): entry on 1.0, exit on the 1.21 the position is worth
    assert equity == pytest.approx(1.21 - BP - 1.21 * BP)
    assert res.returns.iloc[3] == 0.0  # sold at day 2's close: nothing held on day 3


def test_costs_are_only_charged_on_entry_and_exit():
    data = one_asset(opens=[100.0] * 6, closes=[100.0] * 6)
    entries = pd.DataFrame({"A": [0.5, 0, 0, 0, 0, 0]}, data.dates)
    r = book_returns({5: entries}, data).returns
    assert r.iloc[0] == pytest.approx(-0.5 * BP)  # entry
    assert r.iloc[1:4].abs().max() == 0.0  # just holding: free
    assert r.iloc[4] == pytest.approx(-0.5 * BP / (1 - 0.5 * BP))  # exit, relative to the day's equity


def test_overnight_gaps_count_while_held_but_not_before_entry():
    data = one_asset(opens=[100.0, 120.0, 130.0], closes=[100.0, 120.0, 130.0])
    late = pd.DataFrame({"A": [0.0, 1.0, 0.0]}, data.dates)  # bought at day 1's open, after the first gap
    held = pd.DataFrame({"A": [1.0, 0.0, 0.0]}, data.dates)  # held through day 1's opening gap
    assert book_returns({2: late}, data).returns.iloc[1] == pytest.approx(-BP)
    assert book_returns({2: held}, data).returns.iloc[1] == pytest.approx((0.2 - 1.2 * BP) / (1 - BP))


def test_shorts_profit_when_the_price_falls():
    data = one_asset(opens=[100.0, 95.0], closes=[95.0, 90.0])
    entries = pd.DataFrame({"A": [-1.0, 0.0]}, data.dates)
    equity = (1 + book_returns({2: entries}, data).returns).prod()
    assert equity == pytest.approx(1 + 0.10 - BP - 0.9 * BP)


def test_new_positions_only_use_capital_that_is_free():
    data = random_market_data(n_days=120, n_assets=3, seed=2)
    greedy = pd.DataFrame(1 / 3, data.dates, data.assets)  # tries to buy a full account every day
    res = book_returns({5: greedy}, data)
    assert (res.gross <= 1 + 2.02 * BP).all()  # fees (entry + exit) can push it over by a hair
    assert res.gross.iloc[20:].min() > 0.9  # stays (nearly) fully invested
    assert (res.returns > -1).all()


def test_holds_must_end_within_a_month():
    data = random_market_data(n_days=50, n_assets=2, seed=3)
    w = pd.DataFrame(0.1, data.dates, data.assets)
    with pytest.raises(ValueError, match="month"):
        book_returns({MAX_HOLD_DAYS + 1: w}, data)
    with pytest.raises(ValueError, match="gross"):
        book_returns({2: w * 10}, data)


def test_trend_entries_use_only_past_data_and_split_capital_into_cohorts():
    data = random_market_data(n_days=300, n_assets=4, seed=4)
    e = trend_entries(data, lookback=20, hold_days=5, long_only=True)
    assert (e >= 0).all().all()
    assert (e.abs().sum(axis=1) <= 1 / 5 + 1e-9).all()  # one fifth of the account per daily cohort
    # changing day t's prices must not change the entry made at day t's open
    t = 200
    close = data.close.copy()
    close.iloc[t:] *= 1.5
    changed = build_market_data(close, open_=data.open, high=data.high * 1.5, low=data.low, volume=data.volume)
    pd.testing.assert_series_equal(trend_entries(changed, 20, 5, True).iloc[t], e.iloc[t])


def test_buy_and_hold_is_a_trend_entry_with_no_lookback():
    data = random_market_data(n_days=60, n_assets=3, seed=5)
    e = trend_entries(data, lookback=0, hold_days=10, long_only=True)
    assert e.iloc[5:].sum(axis=1).round(9).eq(0.1).all()


# ---------------------------------------------------------------- the trend sleeve in the full system

from bse_quant.competition.backtest import run_backtest  # noqa: E402
from bse_quant.competition.model import fit_model  # noqa: E402
from bse_quant.competition.settings import load_settings  # noqa: E402
from bse_quant.competition.simulate import simulate  # noqa: E402
from bse_quant.competition.trader import Trader  # noqa: E402
from bse_quant.config import ConfigError  # noqa: E402

TREND_TOML = """
[sleeves.market]
weight = 0.4
features = ["market_intraday_1d"]
hurdle_bps = 2.0
floor_weight = 0.01

[sleeves.reversal]
weight = 0.4
features = ["return_1d", "rsi_2"]
hurdle_bps = 3.0
risk_per_trade = 0.01

[trend]
weight = 0.2
lookback = 20
hold_days = 5
long_only = true
"""


def trend_settings(tmp_path, text=TREND_TOML):
    p = tmp_path / "c.toml"
    p.write_text(text)
    s = load_settings(p)
    s.data.start_date = "2020-09-01"
    s.data.holdout_start = "2022-06-01"
    s.model.refit_days = 63
    s.model.min_train_days = 120
    return s


def models_for(s, data):
    return {name: fit_model(data, one.feature_names, end=data.dates[250], ridge=one.model.ridge) for name, _, one in s.sleeve_list()}


def test_trend_settings_load_and_share_the_account(tmp_path):
    s = trend_settings(tmp_path)
    assert (s.trend.weight, s.trend.lookback, s.trend.hold_days, s.trend.long_only) == (0.2, 20, 5, True)
    with pytest.raises(ConfigError, match="add up"):
        trend_settings(tmp_path, TREND_TOML.replace("weight = 0.2", "weight = 0.3"))
    with pytest.raises(ConfigError, match="month"):
        trend_settings(tmp_path, TREND_TOML.replace("hold_days = 5", "hold_days = 40"))
    with pytest.raises(ConfigError, match="lookback"):
        trend_settings(tmp_path, TREND_TOML.replace("lookback = 20", "lookbak = 20"))


def test_backtest_with_trend_is_the_book_of_both_parts(tmp_path):
    data = random_market_data(n_days=700, n_assets=5, seed=11)
    s = trend_settings(tmp_path)
    models = models_for(s, data)
    res = run_backtest(data, s, model=models)
    window = res.returns.index
    live = pd.Series(data.dates.isin(window), data.dates)
    trend = trend_entries(data, 20, 5, True).where(live, 0.0) * 0.2
    expected = book_returns({1: res.weights, 5: trend}, data).returns.loc[window]
    pd.testing.assert_series_equal(res.returns, expected, check_exact=False, atol=1e-12, check_freq=False)
    assert res.summary["Active days"] == len(window) or res.summary["Active days"] > 0.9 * len(window)


def test_streaming_trader_matches_backtest_with_trend(tmp_path):
    data = random_market_data(n_days=700, n_assets=5, seed=11)
    s = trend_settings(tmp_path)
    models = models_for(s, data)
    vectorised = run_backtest(data, s, model=models)
    window = vectorised.returns.index
    traders = []

    def make(history):
        traders.append(Trader(model=models, settings=s, history=history))
        return traders[-1]

    streamed = simulate(make, data, start=window[0])
    pd.testing.assert_series_equal(streamed.returns.loc[window], vectorised.returns, check_exact=False, atol=1e-12, check_freq=False)
    trader = traders[0]
    # the trader tracks its own positions and PnL: its equity matches the evaluator's book
    # (up to the last day it was told about; the final session's close is never reported to it)
    assert trader.equity == pytest.approx((1 + streamed.returns.iloc[:-1]).prod(), rel=1e-9)
    orders = trader.last_orders
    assert set(orders) == {1, 5}  # one-session trades and 5-session trend holds
    assert sum(abs(v) for v in orders[5].values()) <= 0.2 / 5 + 1e-9
    assert all(p["sessions_left"] >= 1 and p["hold_days"] == 5 for p in trader.positions())
