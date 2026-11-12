"""The submission entry point behaves like the evaluator expects."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from conftest import random_market_data

from bse_quant.competition import rules
from bse_quant.competition.model import fit_model
from bse_quant.competition.settings import CompetitionSettings
from bse_quant.competition.simulate import simulate
from bse_quant.competition.trader import Trader

ROOT = Path(__file__).parent.parent
FIELDS = ["open", "high", "low", "close", "volume"]


def load_strategy_module():
    spec = importlib.util.spec_from_file_location("strategy", ROOT / "submission" / "strategy.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def bars_for(data, t, prefix="Asset_"):
    return {
        f"{prefix}{a}": {"Date": str(data.dates[t].date()), **{f.capitalize(): getattr(data, f).iat[t, j] for f in FIELDS}}
        for j, a in enumerate(data.assets)
    }


@pytest.fixture
def setup(tmp_path):
    data = random_market_data(n_days=500, n_assets=4, seed=21)
    settings = CompetitionSettings()
    settings.sizing.hurdle_bps = 2.0
    model = fit_model(data, settings.feature_names, end=data.dates[250], ridge=settings.model.ridge)
    (tmp_path / "model.json").write_text(json.dumps(model.to_dict()))
    (tmp_path / "settings.toml").write_text("[sizing]\nhurdle_bps = 2.0\n")
    return data, settings, model, tmp_path


def test_strategy_matches_the_simulated_trader(setup):
    data, settings, model, folder = setup
    strategy = load_strategy_module()
    start = 300
    history = data.__class__(*(getattr(data, f).iloc[: start - 1].rename(columns=lambda a: f"Asset_{a}") for f in FIELDS))
    strategy.initialize(history=history, model_path=folder / "model.json", settings_path=folder / "settings.toml")
    got = []
    for t in range(start - 1, len(data.dates) - 1):
        w = strategy.on_day_close(bars_for(data, t), {"equity": 1.0})
        assert set(w) == {f"Asset_{a}" for a in data.assets}
        rules.check_weights(w)
        got.append([w[f"Asset_{a}"] for a in data.assets])
    expected = simulate(lambda h: Trader(model=model, settings=settings, history=h), data, start=data.dates[start])
    np.testing.assert_allclose(np.array(got), expected.weights.to_numpy(), atol=1e-12)
    assert np.abs(np.array(got)).sum() > 0


def test_strategy_warms_up_from_the_days_it_is_given(setup):
    """No history at all: it should return zeros at first, then start trading."""
    data, _, _, folder = setup
    strategy = load_strategy_module()
    strategy.initialize(history=None, model_path=folder / "model.json", settings_path=folder / "settings.toml")
    weights = [strategy.on_day_close(bars_for(data, t)) for t in range(len(data.dates))]
    assert all(v == 0 for v in weights[0].values())
    assert any(v != 0 for w in weights for v in w.values())


def test_a_new_asset_is_handled(setup):
    data, _, _, folder = setup
    strategy = load_strategy_module()
    strategy.initialize(history=None, model_path=folder / "model.json", settings_path=folder / "settings.toml")
    for t in range(len(data.dates)):
        bars = bars_for(data, t)
        if t >= 100:
            bars["Asset_Z"] = dict(bars["Asset_A"])  # an asset we've never seen before appears
        w = strategy.on_day_close(bars)
        rules.check_weights(w)
        assert set(w) == set(bars)


def test_history_can_be_a_csv_folder(setup, tmp_path):
    data, _, _, folder = setup
    csv_dir = tmp_path / "csv"
    csv_dir.mkdir()
    for a in data.assets:
        pd.DataFrame({
            "Date": data.dates.strftime("%m/%d/%Y"), "Close/Last": [f"${v:.4f}" for v in data.close[a]],
            "Volume": data.volume[a].astype(int), "Open": data.open[a], "High": data.high[a], "Low": data.low[a],
        }).iloc[::-1].to_csv(csv_dir / f"Asset_{a}.csv", index=False)  # fmt: skip
    strategy = load_strategy_module()
    strategy.initialize(history=csv_dir, model_path=folder / "model.json", settings_path=folder / "settings.toml")
    w = strategy.on_day_close(bars_for(data, len(data.dates) - 1))
    assert set(w) == {f"Asset_{a}" for a in data.assets}


def test_replaying_a_known_day_does_not_duplicate_it(setup):
    data, _, _, folder = setup
    strategy = load_strategy_module()
    history = data.__class__(*(getattr(data, f).rename(columns=lambda a: f"Asset_{a}") for f in FIELDS))
    trader = strategy.initialize(history=history, model_path=folder / "model.json", settings_path=folder / "settings.toml")
    before = trader.history["close"].copy()
    strategy.on_day_close(bars_for(data, len(data.dates) - 1))
    pd.testing.assert_frame_equal(trader.history["close"], before, check_freq=False)


def test_submission_reports_orders_positions_and_equity():
    import importlib
    import sys
    from pathlib import Path

    from bse_quant.data import load_market_data

    root = Path(__file__).resolve().parent.parent
    if not (root / "data").exists():
        pytest.skip("needs the public data folder")
    sys.path.insert(0, str(root / "submission"))
    strategy = importlib.import_module("strategy")
    data = load_market_data(root / "data")
    strategy.initialize(history=root / "data")
    for t in range(len(data.dates) - 10, len(data.dates)):
        bars = {f"Asset_{a}".replace("Asset_Asset_", "Asset_"): {f: getattr(data, f).iat[t, j] for f in ("open", "high", "low", "close", "volume")} | {"date": data.dates[t]} for j, a in enumerate(data.assets)}
        w = strategy.on_day_close(bars)
    assert sum(abs(v) for v in w.values()) <= 1 + 1e-9
    assert set(strategy.last_orders()) == {1, 5}  # one-session trades + 5-session trend holds
    assert all(p["hold_days"] in (1, 5) and 1 <= p["sessions_left"] <= 5 for p in strategy.positions())
    assert isinstance(strategy.equity(), float)
