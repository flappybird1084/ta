import numpy as np
import pandas as pd
import pytest
from conftest import random_market_data

from bse_quant.competition.backtest import run_backtest
from bse_quant.competition.search import (
    apply_overrides,
    config_name,
    expand_grid,
    load_search,
    run_grid,
    run_search_worker,
    search_table,
)
from bse_quant.competition.settings import CompetitionSettings


@pytest.fixture
def data():
    return random_market_data(n_days=700, n_assets=5, seed=5)


def base():
    s = CompetitionSettings()
    s.data.train_start = "2020-02-03"
    s.data.start_date = "2020-09-01"
    s.data.holdout_start = "2022-06-01"
    s.model.refit_days = 63
    s.model.min_train_days = 120
    return s


GRID = {
    "features": {"rev": ["return_1d"], "rev2": ["return_1d", "rsi_2"]},
    "sizing.hurdle_bps": [2.0, 10.0],
    "sizing.long_only": [False, True],
}


def test_expand_grid_makes_every_combination():
    configs = expand_grid(GRID)
    assert len(configs) == 2 * 2 * 2
    assert {c["features"] for c in configs} == {"rev", "rev2"}
    assert len({config_name(c) for c in configs}) == 8


def test_apply_overrides_leaves_the_base_alone():
    b = base()
    s = apply_overrides(b, {"features": "rev2", "sizing.hurdle_bps": 7.0}, GRID["features"])
    assert s.feature_names == ["return_1d", "rsi_2"]
    assert s.sizing.hurdle_bps == 7.0
    assert b.sizing.hurdle_bps == 20.0


def test_run_grid_matches_individual_backtests(data):
    configs = expand_grid(GRID)
    table = run_grid(data, base(), configs, GRID["features"])
    assert list(table.index) == [config_name(c) for c in configs]
    for c in configs[:3]:
        single = run_backtest(data, apply_overrides(base(), c, GRID["features"]))
        assert table.loc[config_name(c), "Sharpe"] == pytest.approx(single.summary["Sharpe"])


def test_search_worker_and_table(tmp_path, data):
    run_search_worker(data, base(), GRID, tmp_path, worker_id=1, max_permutations=2, log=None)
    run_search_worker(data, base(), GRID, tmp_path, worker_id=1, max_permutations=3, log=None)
    perms = pd.read_csv(tmp_path / "worker_01.csv")
    assert perms["seed"].tolist() == [1_000_000, 1_000_001, 1_000_002]
    real = pd.read_csv(tmp_path / "real.csv", index_col=0)
    table = search_table(real, perms)
    assert set(table.columns) >= {"Sharpe", "p-value", "Family-wise p"}
    assert len(table) == 8
    assert table["Score"].max() > -99

    assert (table["Family-wise p"] >= table["p-value"] - 1e-12).all()


def test_rarely_traded_settings_are_not_eligible(data):
    configs = expand_grid(GRID)
    table = run_grid(data, base(), configs, GRID["features"], min_active_days=10**6)
    assert (table["Score"] == -99).all()


def test_load_search_reads_a_grid_file(tmp_path):
    p = tmp_path / "grid.toml"
    p.write_text('[search]\nmin_active_days = 5\n[feature_sets]\na = ["return_1d"]\n[grid]\n"sizing.hurdle_bps" = [10.0, 20.0]\n')
    grid = load_search(p)
    assert grid["_options"]["min_active_days"] == 5
    assert len(expand_grid(grid)) == 2
    assert grid["features"] == {"a": ["return_1d"]}
    assert grid["sizing.hurdle_bps"] == [10.0, 20.0]
