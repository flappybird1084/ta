import numpy as np
import pandas as pd
import pytest

from bse_quant import indicators as ind


def frame(values, name="A"):
    return pd.DataFrame({name: values}, index=pd.bdate_range("2020-01-01", periods=len(values)))


def test_sma():
    out = ind.sma(frame([1, 2, 3, 4]), 2)
    assert np.isnan(out["A"].iloc[0])
    assert list(out["A"].iloc[1:]) == [1.5, 2.5, 3.5]


def test_rsi_extremes():
    up = ind.rsi(frame(np.arange(1, 30, dtype=float)), 2)
    down = ind.rsi(frame(np.arange(30, 1, -1, dtype=float)), 2)
    assert up["A"].iloc[-1] == pytest.approx(100)
    assert down["A"].iloc[-1] == pytest.approx(0)


def test_efficiency_ratio_straight_line_is_one_and_chop_is_near_zero():
    straight = ind.efficiency_ratio(frame(np.arange(50, dtype=float)), 10)
    chop = ind.efficiency_ratio(frame([100.0, 101.0] * 25), 10)
    assert straight["A"].iloc[-1] == pytest.approx(1.0)
    assert chop["A"].iloc[-1] < 0.15


def test_on_balance_volume():
    close = frame([10, 11, 10, 10, 12])
    volume = frame([100, 200, 50, 70, 30])
    obv = ind.on_balance_volume(close, volume)
    assert list(obv["A"]) == [0, 200, 150, 150, 180]


def test_garman_klass_vol_is_zero_for_flat_prices():
    flat = frame([10.0] * 30)
    vol = ind.garman_klass_vol(flat, flat, flat, flat, 10)
    assert vol["A"].iloc[-1] == pytest.approx(0.0)


def test_garman_klass_vol_grows_with_range():
    close = frame([10.0] * 30)
    narrow = ind.garman_klass_vol(close, close * 1.01, close * 0.99, close, 10)
    wide = ind.garman_klass_vol(close, close * 1.05, close * 0.95, close, 10)
    assert wide["A"].iloc[-1] > narrow["A"].iloc[-1] > 0


def test_internal_bar_strength():
    ibs = ind.internal_bar_strength(
        high=frame([10.0, 10.0, 10.0]), low=frame([8.0, 8.0, 10.0]), close=frame([8.0, 10.0, 10.0])
    )
    assert list(ibs["A"]) == [0.0, 1.0, 0.5]  # a zero-range bar counts as the middle


def test_positions_from_signals_state_machine():
    f = lambda xs: frame([bool(x) for x in xs])
    pos = ind.positions_from_signals(
        long_entry=f([0, 1, 0, 0, 0, 0, 0, 0]),
        long_exit=f([0, 0, 0, 1, 0, 0, 0, 0]),
        short_entry=f([0, 0, 0, 0, 0, 1, 0, 0]),
        short_exit=f([0, 0, 0, 0, 0, 0, 0, 1]),
    )
    assert list(pos["A"]) == [0, 1, 1, 0, 0, -1, -1, 0]


def test_positions_from_signals_can_flip_directly():
    f = lambda xs: frame([bool(x) for x in xs])
    pos = ind.positions_from_signals(
        long_entry=f([1, 0, 0]),
        long_exit=f([0, 1, 0]),
        short_entry=f([0, 1, 0]),
        short_exit=f([0, 0, 0]),
    )
    assert list(pos["A"]) == [1, -1, -1]


def test_mean_reversion_bands():
    z = frame([0.0, 2.5, 1.0, 0.2, -2.5, -1.0, -0.4, np.nan])
    pos = ind.mean_reversion_bands(z, entry=2.0, exit=0.5)
    assert list(pos["A"]) == [0, -1, -1, 0, 1, 1, 0, 0]


def test_hold_extends_signals():
    sig = frame([0, 1, 0, 0, 0, -1, 0, 0, 0])
    held = ind.hold(sig, 3)
    assert list(held["A"]) == [0, 1, 1, 1, 0, -1, -1, -1, 0]


def test_hold_newest_signal_wins():
    sig = frame([1, 0, -1, 0, 0])
    assert list(ind.hold(sig, 3)["A"]) == [1, 1, -1, -1, -1]


def test_hold_one_day_is_identity():
    sig = frame([0, 1, -1, 0])
    assert list(ind.hold(sig, 1)["A"]) == [0, 1, -1, 0]


def test_pair_positions_to_assets():
    idx = pd.bdate_range("2020-01-01", periods=2)
    pair_pos = {("A", "B"): pd.Series([1.0, -1.0], idx)}
    betas = {("A", "B"): pd.Series([2.0, 0.5], idx)}
    out = ind.pair_positions_to_assets(pair_pos, betas, ["A", "B", "C"], idx)
    # one pair per asset slot is the max (3 assets -> 2 pairs each), so divide by 2
    assert list(out["A"]) == [0.5, -0.5]
    assert list(out["B"]) == [-1.0, 0.25]
    assert list(out["C"]) == [0.0, 0.0]
