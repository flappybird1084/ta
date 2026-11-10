import numpy as np
import pandas as pd
import pytest
from conftest import random_market_data

from bse_quant.config import BacktestSettings, Config, StrategySettings, VolTargetSettings
from bse_quant.permutation import (
    familywise_pvalues,
    load_permutations,
    permute_bars,
    pvalues,
    run_permutation,
    run_worker,
)

START, END = "2020-06-01", "2021-03-01"


@pytest.fixture
def data():
    return random_market_data(n_days=400)  # 2020-01-01 .. 2021-07-13


def test_drops_holdout_and_keeps_warmup(data):
    p = permute_bars(data, START, END, seed=1)
    assert p.dates.max() < pd.Timestamp(END)
    assert p.dates.equals(data.dates[data.dates < pd.Timestamp(END)])
    warmup = data.dates < pd.Timestamp(START)
    for field in ["open", "high", "low", "close", "volume"]:
        pd.testing.assert_frame_equal(getattr(p, field)[warmup[: len(p.dates)]], getattr(data, field)[warmup])


def test_bars_stay_valid(data):
    p = permute_bars(data, START, END, seed=2)
    assert (p.high >= np.maximum(p.open, p.close) - 1e-9).all().all()
    assert (p.low <= np.minimum(p.open, p.close) + 1e-9).all().all()
    assert (p.close > 0).all().all()


def bar_pieces(d):
    """Per-day log gap, and intraday high/low/close (relative to the open), inside the shuffled segment."""
    gap = np.log(d.open / d.close.shift(1))
    hi, lo, cl = (np.log(f / d.open) for f in (d.high, d.low, d.close))
    seg = (d.dates >= pd.Timestamp(START)) & (d.dates < pd.Timestamp(END))
    return gap[seg], hi[seg], lo[seg], cl[seg], d.volume[seg]


def test_same_pieces_just_reordered(data):
    p = permute_bars(data, START, END, seed=3)
    orig, new = bar_pieces(data), bar_pieces(p)
    for a, b in zip(orig, new):
        np.testing.assert_allclose(np.sort(a.to_numpy(), axis=0), np.sort(b.to_numpy(), axis=0), rtol=1e-9)
        assert not np.allclose(a.to_numpy(), b.to_numpy())  # but in a different order


def test_all_assets_shuffled_together(data):
    """Same shuffle for every asset, so how assets move together on a day is kept."""
    p = permute_bars(data, START, END, seed=4)
    _, _, _, orig_close, _ = bar_pieces(data)
    _, _, _, new_close, _ = bar_pieces(p)
    as_rows = lambda df: sorted(map(tuple, np.round(df.to_numpy(), 12)))
    assert as_rows(orig_close) == as_rows(new_close)


def test_end_price_unchanged(data):
    p = permute_bars(data, START, END, seed=5)
    np.testing.assert_allclose(p.close.iloc[-1], data.close[data.dates < pd.Timestamp(END)].iloc[-1], rtol=1e-9)


def test_seed_controls_the_shuffle(data):
    a = permute_bars(data, START, END, seed=6).close
    b = permute_bars(data, START, END, seed=6).close
    c = permute_bars(data, START, END, seed=7).close
    pd.testing.assert_frame_equal(a, b)
    assert not np.allclose(a, c)


def test_pvalues():
    real = pd.Series({"x": 1.0, "y": 0.0})
    perms = pd.DataFrame({"x": [0.5, 1.5, 0.2, 0.1], "y": [0.5, 1.5, 0.2, 0.1]})
    p = pvalues(real, perms)
    assert p["x"] == pytest.approx((1 + 1) / (4 + 1))
    assert p["y"] == pytest.approx((1 + 4) / (4 + 1))


def test_familywise_pvalues_use_best_strategy_in_each_shuffle():
    real = pd.Series({"x": 1.0, "y": 0.4})
    perms = pd.DataFrame({"x": [0.5, 0.1, 0.2], "y": [0.3, 1.2, 0.1]})
    # best of each shuffle: 0.5, 1.2, 0.2
    p = familywise_pvalues(real, perms)
    assert p["x"] == pytest.approx((1 + 1) / (3 + 1))
    assert p["y"] == pytest.approx((1 + 2) / (3 + 1))


def tiny_config():
    return Config(
        backtest=BacktestSettings(start_date=START, holdout_start=END),
        strategies={
            "time_series_momentum": StrategySettings(params={"lookback_days": 20}),
            "internal_bar_strength": StrategySettings(weight=2.0),
            "gap_fade": StrategySettings(enabled=False),
        },
        volatility_targeting=VolTargetSettings(),
    )


def test_run_permutation_scores_every_enabled_strategy(data):
    scores = run_permutation(data, tiny_config(), seed=None)
    assert set(scores) == {"time_series_momentum", "internal_bar_strength", "PORTFOLIO"}
    assert all(np.isfinite(v) for v in scores.values())


def test_unshuffled_score_matches_the_normal_backtest(data):
    from bse_quant import metrics
    from bse_quant.engine import run_backtest

    result = run_backtest(data, tiny_config())
    scores = run_permutation(data, tiny_config(), seed=None)
    assert scores["PORTFOLIO"] == pytest.approx(metrics.sharpe_ratio(result.portfolio_returns))


def test_worker_writes_and_resumes(tmp_path, data):
    config = tiny_config()
    run_worker(data, config, tmp_path, worker_id=3, max_permutations=2)
    run_worker(data, config, tmp_path, worker_id=3, max_permutations=3)
    real, perms = load_permutations(tmp_path)

    assert len(perms) == 3
    assert perms["seed"].tolist() == [3_000_000, 3_000_001, 3_000_002]
    assert set(real.index) == {"time_series_momentum", "internal_bar_strength", "PORTFOLIO"}


def test_worker_stops_at_deadline(tmp_path, data):
    import time

    run_worker(data, tiny_config(), tmp_path, worker_id=0, deadline=time.time() - 1)
    _, perms = load_permutations(tmp_path)
    assert len(perms) == 0


def test_permutation_table():
    from bse_quant.permutation import permutation_table

    real = pd.Series({"x": 1.0, "y": 0.4, "PORTFOLIO": 0.8})
    perms = pd.DataFrame({"seed": [1, 2, 3], "x": [0.5, 0.1, 0.2], "y": [0.3, 1.2, 0.1], "PORTFOLIO": [0.9, 0.1, 0.0]})
    table = permutation_table(real, perms)
    assert list(table.index) == ["x", "y", "PORTFOLIO"]
    assert table.loc["x", "Real Sharpe"] == 1.0
    assert table.loc["x", "p-value"] == pytest.approx(0.25)
    assert table.loc["PORTFOLIO", "p-value"] == pytest.approx(0.5)
    assert np.isnan(table.loc["PORTFOLIO", "Family-wise p"])
    assert table.loc["y", "Shuffled mean"] == pytest.approx(np.mean([0.3, 1.2, 0.1]))


def test_report_command(tmp_path, data, capsys):
    from bse_quant.permutation_cli import report_main

    run_worker(data, tiny_config(), tmp_path, worker_id=0, max_permutations=3, log=None)
    report_main(["--out", str(tmp_path)])
    printed = capsys.readouterr().out
    assert "3 shuffles" in printed
    assert "time_series_momentum" in printed
    assert (tmp_path / "pvalues.csv").exists()
    assert (tmp_path / "distributions.png").exists()
