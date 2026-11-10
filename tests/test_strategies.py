"""Tests for every trading strategy.

The first block runs against *all* strategies (shape, sanity, no peeking at
the future). The rest check that each idea does what it says on the tin,
using small hand-built markets where the right answer is obvious.
"""

import numpy as np
import pandas as pd
import pytest
from conftest import build_market_data, dates, prices_from_returns, random_market_data

from bse_quant.data import MarketData
from bse_quant.strategies import STRATEGIES
from bse_quant.strategies import (
    bollinger_reversion,
    calendar_effects,
    cross_sectional_momentum,
    donchian_breakout,
    gap_fade,
    internal_bar_strength,
    kalman_pairs,
    lead_lag,
    moving_average_crossover,
    obv_confirmation,
    overnight_intraday,
    pairs_cointegration,
    pca_residual_reversion,
    range_compression_breakout,
    rsi_reversion,
    short_term_reversal,
    time_series_momentum,
    trend_strength_filter,
    volatility_regime_switch,
    volatility_scaled_trend,
    volume_shock,
)
from bse_quant.strategies.base import LEGS

ALL = sorted(STRATEGIES)


# --------------------------------------------------------------------------
# Rules every strategy must follow
# --------------------------------------------------------------------------


def test_registry_has_all_twenty_one_signal_strategies():
    # ideas 1-22, minus #17 (volatility targeting), which is a portfolio overlay
    assert len(STRATEGIES) == 21


@pytest.mark.parametrize("name", ALL)
def test_output_shape_and_values(name, market):
    signal = STRATEGIES[name](market)
    pos = signal.positions
    assert signal.leg in LEGS
    assert pos.index.equals(market.close.index)
    assert list(pos.columns) == market.assets
    assert np.isfinite(pos.to_numpy()).all(), "positions must be numbers (no NaN/inf)"
    assert (pos.abs() <= 5).all().all(), "positions should be sized sensibly"


@pytest.mark.parametrize("name", ALL)
def test_some_trading_happens(name, market):
    assert STRATEGIES[name](market).positions.abs().to_numpy().sum() > 0


@pytest.mark.parametrize("name", ALL)
def test_no_lookahead(name, market):
    """Scrambling the future must not change any decision made before it.

    A close_to_close / close_to_open position on day T is decided at the
    close of day T, so it may use all of day T's bar. An open_to_close
    position on day T is decided at the open, so it may only use day T's open.
    """
    cut = 300
    leg = STRATEGIES[name](market).leg
    rng = np.random.default_rng(99)

    def scramble(df, first_row):
        out = df.copy()
        n = len(out) - first_row
        out.iloc[first_row:] = out.iloc[first_row:].to_numpy() * rng.uniform(0.7, 1.3, (n, out.shape[1]))
        return out

    first_changed = cut if leg == "open_to_close" else cut + 1
    changed = MarketData(
        open=scramble(market.open, cut + 1),
        high=scramble(market.high, first_changed),
        low=scramble(market.low, first_changed),
        close=scramble(market.close, first_changed),
        volume=scramble(market.volume, first_changed),
    )
    before = STRATEGIES[name](market).positions.iloc[: cut + 1]
    after = STRATEGIES[name](changed).positions.iloc[: cut + 1]
    pd.testing.assert_frame_equal(before, after)


# --------------------------------------------------------------------------
# Hand-built markets for behaviour checks
# --------------------------------------------------------------------------


def trend_market(n=300):
    idx = dates(n)
    close = pd.DataFrame(
        {"up": np.linspace(100, 200, n), "down": np.linspace(200, 100, n)}, index=idx
    )
    return build_market_data(close)


def zigzag_returns(n, size=0.005):
    """+size, -size, +size, ... a calm, non-trending market."""
    return np.array([size if i % 2 == 0 else -size for i in range(n)])


def one_asset(returns, **bars):
    idx = dates(len(returns))
    close = prices_from_returns(pd.DataFrame({"A": returns}, index=idx))
    return build_market_data(close, **bars)


# 1 ------------------------------------------------------------------------
def test_time_series_momentum_follows_the_trend():
    pos = time_series_momentum(trend_market(), lookback_days=20).positions
    assert pos["up"].iloc[25] == 1
    assert pos["down"].iloc[25] == -1
    assert pos.iloc[:20].abs().to_numpy().sum() == 0  # not enough history yet


# 2 ------------------------------------------------------------------------
def test_moving_average_crossover():
    pos = moving_average_crossover(trend_market(), fast_days=5, slow_days=20).positions
    assert pos["up"].iloc[30] == 1
    assert pos["down"].iloc[30] == -1


def test_moving_average_crossover_rejects_fast_slower_than_slow():
    with pytest.raises(ValueError):
        moving_average_crossover(trend_market(), fast_days=50, slow_days=20)


# 3 ------------------------------------------------------------------------
def test_donchian_breakout_enters_and_reverses():
    close = [100.0] * 30 + [110.0] * 10 + [90.0] * 5
    data = build_market_data(pd.DataFrame({"A": close}, index=dates(len(close))))
    pos = donchian_breakout(data, entry_days=10, exit_days=5).positions["A"]
    assert pos.iloc[29] == 0
    assert pos.iloc[30] == 1  # broke above the 10-day high
    assert pos.iloc[39] == 1
    assert pos.iloc[40] == -1  # crashed below the 10-day low


# 4 ------------------------------------------------------------------------
def test_cross_sectional_momentum_longs_winners_shorts_losers():
    n = 120
    idx = dates(n)
    drifts = {"A": 0.003, "B": 0.001, "C": -0.001, "D": -0.003}
    close = pd.DataFrame({k: 100 * np.exp(d * np.arange(n)) for k, d in drifts.items()}, idx)
    data = build_market_data(close)
    pos = cross_sectional_momentum(
        data, lookback_days=20, skip_days=0, rebalance_days=5, top_n=1, bottom_n=1
    ).positions
    assert list(pos.iloc[-1]) == [1, 0, 0, -1]


def test_cross_sectional_momentum_only_trades_on_rebalance_days(market):
    pos = cross_sectional_momentum(market, rebalance_days=5).positions
    changed_rows = np.flatnonzero(pos.diff().abs().sum(axis=1).to_numpy() > 0)
    assert len(changed_rows) > 0
    assert all(i % 5 == 0 for i in changed_rows)


def test_cross_sectional_momentum_rejects_too_many_picks(market):
    with pytest.raises(ValueError):
        cross_sectional_momentum(market, top_n=3, bottom_n=2)  # only 4 assets


# 5 ------------------------------------------------------------------------
def test_volatility_scaled_trend_sizes_down_wild_assets():
    rng = np.random.default_rng(5)
    n = 400
    returns = pd.DataFrame(
        {
            "calm": 0.002 + 0.005 * rng.standard_normal(n),
            "wild": 0.004 + 0.03 * rng.standard_normal(n),
        },
        index=dates(n),
    )
    data = build_market_data(prices_from_returns(returns))
    pos = volatility_scaled_trend(data, target_asset_vol=0.2, max_leverage=2.0).positions
    late = pos.iloc[-100:]
    assert late["calm"].mean() > late["wild"].mean() > 0
    assert pos.abs().max().max() <= 2.0


# 6 ------------------------------------------------------------------------
def test_trend_strength_filter_trades_trends_not_chop():
    n = 100
    close = pd.DataFrame(
        {"trend": np.linspace(100, 150, n), "chop": [100.0, 101.0] * (n // 2)}, index=dates(n)
    )
    pos = trend_strength_filter(build_market_data(close), lookback_days=20, min_efficiency=0.3).positions
    assert pos["trend"].iloc[-1] == 1
    assert pos["chop"].iloc[20:].abs().sum() == 0


# 7 ------------------------------------------------------------------------
def test_short_term_reversal_buys_a_crash_and_holds():
    r = zigzag_returns(80)
    r[60] = -0.10
    pos = short_term_reversal(
        one_asset(r), lookback_days=1, threshold_z=2.0, vol_lookback_days=20, hold_days=3, trend_filter_days=0
    ).positions["A"]
    assert pos.iloc[59] == 0
    assert list(pos.iloc[60:63]) == [1, 1, 1]
    assert pos.iloc[63] == 0


def test_short_term_reversal_trend_filter_blocks_shorts_in_uptrend():
    r = np.full(300, 0.002)
    r[250] = 0.10  # a spike up in a strong uptrend: no short allowed
    pos = short_term_reversal(
        one_asset(r), lookback_days=1, threshold_z=2.0, vol_lookback_days=20, hold_days=1, trend_filter_days=100
    ).positions["A"]
    assert pos.iloc[250] == 0


# 8 ------------------------------------------------------------------------
def test_rsi_reversion_buys_a_dip_in_an_uptrend():
    r = np.full(62, 0.01)
    r[60] = r[61] = -0.05
    pos = rsi_reversion(one_asset(r), rsi_days=2, oversold=10, overbought=90, exit_level=50, trend_days=50).positions[
        "A"
    ]
    assert pos.iloc[59] == 0
    assert pos.iloc[61] == 1


# 9 ------------------------------------------------------------------------
def test_bollinger_reversion_fades_a_spike():
    r = zigzag_returns(60)
    r[50] = 0.10
    pos = bollinger_reversion(one_asset(r), lookback_days=20, entry_z=2.0, exit_z=0.5).positions["A"]
    assert pos.iloc[49] == 0
    assert pos.iloc[50] == -1


# 10 -----------------------------------------------------------------------
def test_gap_fade_shorts_a_gap_up_for_the_day():
    r = zigzag_returns(80)
    idx = dates(80)
    close = prices_from_returns(pd.DataFrame({"A": r}, index=idx))
    open_ = close.shift(1).fillna(close)
    open_.iloc[60, 0] = close.iloc[59, 0] * 1.05
    sig = gap_fade(build_market_data(close, open_), threshold_z=1.0, vol_lookback_days=20)
    assert sig.leg == "open_to_close"
    assert sig.positions["A"].iloc[60] == -1
    assert sig.positions["A"].iloc[21:].drop(idx[60]).abs().sum() == 0


# 11 -----------------------------------------------------------------------
def test_internal_bar_strength():
    idx = dates(3)
    close = pd.DataFrame({"A": [9.0, 10.0, 11.0]}, idx)
    high = pd.DataFrame({"A": [10.0, 12.0, 11.0]}, idx)
    low = pd.DataFrame({"A": [9.0, 8.0, 9.0]}, idx)
    data = build_market_data(close, high=high, low=low)
    pos = internal_bar_strength(data, low_threshold=0.2, high_threshold=0.8).positions["A"]
    assert list(pos) == [1, 0, -1]


# 12 & 13 ------------------------------------------------------------------
def cointegrated_pair(n=400, shock_day=350, shock=0.15, seed=12):
    rng = np.random.default_rng(seed)
    x = np.cumsum(rng.normal(0, 0.01, n))
    noise = np.zeros(n)
    for t in range(1, n):
        noise[t] = 0.5 * noise[t - 1] + rng.normal(0, 0.005)
    y = x + noise
    y[shock_day:] += shock  # y suddenly gets expensive relative to x
    close = pd.DataFrame({"Y": 100 * np.exp(y), "X": 100 * np.exp(x)}, index=dates(n))
    return build_market_data(close)


def test_pairs_cointegration_shorts_the_expensive_leg():
    pos = pairs_cointegration(
        cointegrated_pair(), formation_days=200, refit_days=20, max_pvalue=0.05, entry_z=2.0, exit_z=0.5
    ).positions
    assert pos["Y"].iloc[352] < 0
    assert pos["X"].iloc[352] > 0


def test_kalman_pairs_shorts_the_expensive_leg():
    pos = kalman_pairs(cointegrated_pair(), entry_z=2.0, exit_z=0.5).positions
    assert pos["Y"].iloc[350] < 0
    assert pos["X"].iloc[350] > 0


# 14 -----------------------------------------------------------------------
def test_pca_residual_reversion_fades_an_idiosyncratic_jump():
    rng = np.random.default_rng(14)
    n = 320
    factor = rng.normal(0, 0.01, n)
    returns = pd.DataFrame(
        {k: factor + rng.normal(0, 0.004, n) for k in "ABCDE"}, index=dates(n)
    )
    returns.iloc[300, 0] += 0.08
    data = build_market_data(prices_from_returns(returns))
    pos = pca_residual_reversion(data, window_days=60, n_components=1, residual_days=5, entry_z=1.5).positions
    assert pos["A"].iloc[300] < 0


# 15 -----------------------------------------------------------------------
def test_lead_lag_copies_the_leader():
    rng = np.random.default_rng(15)
    n = 400
    leader = rng.normal(0, 0.01, n)
    follower = np.r_[0.0, 0.8 * leader[:-1]] + rng.normal(0, 0.003, n)
    other = rng.normal(0, 0.01, n)
    returns = pd.DataFrame({"L": leader, "F": follower, "O": other}, index=dates(n))
    data = build_market_data(prices_from_returns(returns))
    pos = lead_lag(data, window_days=60, min_correlation=0.1).positions
    late = slice(100, None)
    agreement = (pos["F"].iloc[late] == np.sign(returns["L"].iloc[late])).mean()
    assert agreement > 0.95


# 16 -----------------------------------------------------------------------
def test_volatility_regime_switch_trends_when_calm_reverts_when_wild():
    n = 400
    sign = np.where(np.arange(n) % 2 == 0, 1.0, -1.0)
    calming = 0.004 + np.linspace(0.004, 0.001, n) * sign  # vol shrinking, strong uptrend
    heating = np.linspace(0.005, 0.04, n) * sign  # vol growing
    returns = pd.DataFrame({"calm": calming, "wild": heating}, index=dates(n))
    data = build_market_data(prices_from_returns(returns))
    pos = volatility_regime_switch(data, vol_days=20, regime_days=150, trend_days=63, reversion_days=5).positions

    assert (pos["calm"].iloc[-50:] == 1).all()
    five_day = np.sign(data.close["wild"] / data.close["wild"].shift(5) - 1)
    assert (pos["wild"].iloc[-50:] == -five_day.iloc[-50:]).all()


# 18 -----------------------------------------------------------------------
def volume_spike_market():
    n = 70
    r = zigzag_returns(n, 0.002)
    r[50] = 0.02
    volume = pd.DataFrame({"A": [1e6 if i % 2 else 1.1e6 for i in range(n)]}, index=dates(n))
    volume.iloc[50, 0] = 5e6
    return one_asset(r, volume=volume)


def test_volume_shock_continuation():
    pos = volume_shock(volume_spike_market(), lookback_days=20, volume_z=2.0, mode="continuation", hold_days=5)
    pos = pos.positions["A"]
    assert pos.iloc[49] == 0
    assert list(pos.iloc[50:55]) == [1] * 5
    assert pos.iloc[55] == 0


def test_volume_shock_reversal_mode_flips():
    pos = volume_shock(volume_spike_market(), lookback_days=20, volume_z=2.0, mode="reversal", hold_days=5)
    assert pos.positions["A"].iloc[50] == -1


def test_volume_shock_rejects_unknown_mode():
    with pytest.raises(ValueError):
        volume_shock(volume_spike_market(), mode="sideways")


# 19 -----------------------------------------------------------------------
def test_obv_confirmation():
    pos = obv_confirmation(trend_market(100), ma_days=20).positions
    assert pos["up"].iloc[-1] == 1
    assert pos["down"].iloc[-1] == -1


# 20 -----------------------------------------------------------------------
def test_range_compression_breakout():
    n = 45
    idx = dates(n)
    close = pd.DataFrame({"A": [100.0] * n}, idx)
    high = pd.DataFrame({"A": [102.0] * n}, idx)
    low = pd.DataFrame({"A": [98.0] * n}, idx)
    high.iloc[30, 0], low.iloc[30, 0] = 100.5, 99.5  # a very narrow day
    close.iloc[31, 0] = 101.0  # ...then a close above its high
    data = build_market_data(close, high=high, low=low)
    pos = range_compression_breakout(data, nr_days=7, setup_window=3, hold_days=5).positions["A"]
    assert pos.iloc[30] == 0
    assert list(pos.iloc[31:36]) == [1] * 5
    assert pos.iloc[36] == 0


# 21 -----------------------------------------------------------------------
def test_overnight_intraday_sessions(market):
    night = overnight_intraday(market, session="overnight", direction=1)
    day = overnight_intraday(market, session="intraday", direction=-1)
    assert night.leg == "close_to_open"
    assert day.leg == "open_to_close"
    assert (night.positions == 1).all().all()
    assert (day.positions == -1).all().all()


def test_overnight_intraday_rejects_unknown_session(market):
    with pytest.raises(ValueError):
        overnight_intraday(market, session="lunch")


# 22 -----------------------------------------------------------------------
def calendar_market(index):
    close = pd.DataFrame({"A": 100.0}, index=index)
    return build_market_data(close)


def test_calendar_turn_of_month():
    idx = dates(300)
    pos = calendar_effects(
        calendar_market(idx), turn_of_month=True, days_before=1, days_after=3, pre_holiday=False, weekdays=()
    ).positions["A"]

    s = pd.Series(idx, index=idx)
    month = idx.to_period("M")
    rank_from_start = s.groupby(month).cumcount()
    rank_from_end = s.groupby(month).cumcount(ascending=False)
    last_month = month == month[-1]
    is_tom = (rank_from_start < 3) | ((rank_from_end < 1) & ~last_month)
    # a position held overnight on day t earns day t+1's return
    expected = is_tom.shift(-1, fill_value=False).astype(float)
    assert list(pos) == list(expected)


def test_calendar_weekdays():
    idx = dates(50)
    pos = calendar_effects(
        calendar_market(idx), turn_of_month=False, pre_holiday=False, weekdays=("Mon",)
    ).positions["A"]
    fridays = [d.weekday() == 4 for d in idx]
    fridays[-1] = False  # no next day to earn
    assert list(pos == 1) == fridays


def test_calendar_pre_holiday():
    idx = pd.bdate_range("2020-06-22", "2020-07-17").drop(pd.Timestamp("2020-07-03"))
    pos = calendar_effects(calendar_market(idx), turn_of_month=False, pre_holiday=True, weekdays=()).positions["A"]
    # 2020-07-02 is the last day before the holiday; hold it from the close of 07-01
    assert pos.loc["2020-07-01"] == 1
    assert pos.drop(pd.Timestamp("2020-07-01")).abs().sum() == 0


def test_calendar_rejects_bad_weekday():
    with pytest.raises(ValueError):
        calendar_effects(calendar_market(dates(10)), weekdays=("Funday",))


def reference_kalman_pairs(data, delta=1e-5, observation_noise=1e-3, entry_z=2.0, exit_z=0.5, warmup_days=60):
    """The original one-pair-at-a-time Kalman loop, kept to check the fast version against."""
    from bse_quant.indicators import all_pairs, mean_reversion_bands, pair_positions_to_assets

    log_prices = np.log(data.close)
    n = len(log_prices)
    drift = delta / (1 - delta) * np.eye(2)
    pair_pos, betas = {}, {}
    for y_name, x_name in all_pairs(data.assets):
        y, x = log_prices[y_name].to_numpy(), log_prices[x_name].to_numpy()
        z, beta = np.full(n, np.nan), np.full(n, np.nan)
        state, cov = np.zeros(2), np.zeros((2, 2))
        for t in range(n):
            obs = np.array([x[t], 1.0])
            prior_cov = cov + drift
            surprise = y[t] - obs @ state
            surprise_var = obs @ prior_cov @ obs + observation_noise
            gain = prior_cov @ obs / surprise_var
            state = state + gain * surprise
            cov = prior_cov - np.outer(gain, obs) @ prior_cov
            if t >= warmup_days:
                z[t], beta[t] = surprise / np.sqrt(surprise_var), state[0]
        pair_pos[(y_name, x_name)] = mean_reversion_bands(pd.DataFrame({"z": z}, data.dates), entry_z, exit_z)["z"]
        betas[(y_name, x_name)] = pd.Series(beta, index=data.dates)
    return pair_positions_to_assets(pair_pos, betas, data.assets, data.dates)


@pytest.mark.parametrize("settings", [{}, {"delta": 1e-4, "observation_noise": 1e-3, "entry_z": 1.0}])
def test_kalman_pairs_matches_reference_loop(market, settings):
    params = {"delta": 1e-5, "observation_noise": 1e-4, **settings}
    fast = kalman_pairs(market, **params).positions
    slow = reference_kalman_pairs(market, **params)
    np.testing.assert_allclose(fast.to_numpy(), slow.to_numpy(), atol=1e-9)
