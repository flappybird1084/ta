import numpy as np
import pandas as pd
import pytest
from conftest import random_market_data

from bse_quant import metrics
from bse_quant.competition import rules
from bse_quant.competition.competitors import student_field
from bse_quant.competition.scoring import (
    POINTS,
    block_bootstrap,
    competition_points,
    metrics_matrix,
    simulate_scores,
)


def test_points_are_linear_in_rank():
    table = pd.DataFrame(
        {"Sharpe": [2.0, 1.0, 0.0], "CAGR": [0.1, 0.3, 0.2], "Max drawdown": [-0.30, -0.10, -0.20]},
        index=["x", "y", "z"],
    )
    pts = competition_points(table)
    assert pts.loc["x", "Sharpe"] == 50 and pts.loc["y", "Sharpe"] == 25 and pts.loc["z", "Sharpe"] == 0
    assert pts.loc["y", "CAGR"] == 30 and pts.loc["z", "CAGR"] == 15 and pts.loc["x", "CAGR"] == 0
    # smaller drawdown is better
    assert pts.loc["y", "Max drawdown"] == 20 and pts.loc["x", "Max drawdown"] == 0
    assert pts.loc["y", "Total"] == 25 + 30 + 20
    assert POINTS == {"Sharpe": 50, "CAGR": 30, "Max drawdown": 20}


def test_ties_share_their_average_rank():
    table = pd.DataFrame({"Sharpe": [1.0, 1.0, 0.0], "CAGR": [0.0, 0.0, 0.0], "Max drawdown": [0.0, 0.0, 0.0]})
    pts = competition_points(table)
    assert pts["Sharpe"].tolist() == [pytest.approx(37.5), pytest.approx(37.5), 0.0]
    assert pts["CAGR"].tolist() == [pytest.approx(15.0)] * 3  # everyone tied: average rank


def test_metrics_matrix_matches_the_metrics_module():
    rng = np.random.default_rng(0)
    r = rng.normal(0.0005, 0.01, (300, 3))
    m = metrics_matrix(r)
    for j in range(3):
        s = pd.Series(r[:, j])
        assert m["Sharpe"][j] == pytest.approx(metrics.sharpe_ratio(s))
        assert m["CAGR"][j] == pytest.approx(metrics.cagr(s))
        assert m["Max drawdown"][j] == pytest.approx(metrics.max_drawdown(s))
    assert m["Active days"].tolist() == [300, 300, 300]


def test_block_bootstrap_keeps_runs_of_days_together():
    rng = np.random.default_rng(1)
    idx = block_bootstrap(n_days=100, length=50, block=10, rng=rng)
    assert len(idx) == 50 and idx.min() >= 0 and idx.max() < 100
    steps = np.diff(idx)
    assert (steps == 1).mean() >= 0.8  # mostly consecutive


def test_simulate_scores_ranks_a_clear_winner_first():
    rng = np.random.default_rng(2)
    days = pd.bdate_range("2020-01-01", periods=800)
    field = pd.DataFrame(rng.normal(-0.0005, 0.01, (800, 8)), days, [f"team{i}" for i in range(8)])
    ours = pd.DataFrame({"great": rng.normal(0.002, 0.005, 800), "awful": rng.normal(-0.003, 0.02, 800)}, days)
    res = simulate_scores(ours, field, n_trials=40, length=500, block=20, seed=0)
    assert res.loc["great", "Score"] > 90
    assert res.loc["awful", "Score"] < 10
    assert set(res.columns) >= {"Score", "Score p10", "Sharpe pts", "CAGR pts", "Max drawdown pts", "Win rate"}


def test_ineligible_candidate_scores_zero():
    days = pd.bdate_range("2020-01-01", periods=400)
    rng = np.random.default_rng(3)
    field = pd.DataFrame(rng.normal(0, 0.01, (400, 5)), days)
    rare = np.zeros(400)
    rare[::40] = 0.01  # only ~10 trading days per trial
    res = simulate_scores(pd.DataFrame({"rare": rare}, days), field, n_trials=10, length=300, block=10, seed=0)
    assert res.loc["rare", "Score"] == 0


def test_student_field_is_valid():
    data = random_market_data(n_days=400, n_assets=4, seed=4)
    field = student_field(data)
    assert len(field) >= 15
    for name, w in field.items():
        assert w.shape == data.close.shape, name
        assert (w.abs().sum(axis=1) <= 1 + 1e-9).all(), name
        rules.portfolio_returns(w, data)  # passes the rules check


def test_stricter_eligibility_for_our_candidates():
    days = pd.bdate_range("2020-01-01", periods=400)
    rng = np.random.default_rng(5)
    field = pd.DataFrame(rng.normal(0, 0.01, (400, 5)), days)
    some = np.zeros(400)
    some[::5] = rng.normal(0.002, 0.003, 80)  # ~60 active days per 300-day trial
    ours = pd.DataFrame({"some": some}, days)
    loose = simulate_scores(ours, field, n_trials=10, length=300, block=10, seed=0)
    strict = simulate_scores(ours, field, n_trials=10, length=300, block=10, seed=0, our_min_active=200)
    assert loose.loc["some", "Score"] > 0 and strict.loc["some", "Score"] == 0
