"""Tests for the competition system (bse_quant.competition).

Competition rules: weights are decided after a day's close, traded at the next
day's open and closed at that day's close; the sum of |weights| is at most 1;
each trade pays 5 bps on entry and 5 bps on exit.
"""

import numpy as np
import pandas as pd
import pytest
from conftest import build_market_data, dates, random_market_data

from bse_quant.competition import features as feat
from bse_quant.competition import rules
from bse_quant.competition.backtest import run_backtest
from bse_quant.competition.model import Model, fit_model, walk_forward_forecast
from bse_quant.competition.portfolio import RiskManager, forecast_to_weights
from bse_quant.competition.settings import CompetitionSettings, load_settings
from bse_quant.competition.simulate import simulate
from bse_quant.competition.trader import Trader, parse_daily_data
from bse_quant.config import ConfigError
from bse_quant.data import MarketData

FIELDS = ["open", "high", "low", "close", "volume"]


def scramble_after(data: MarketData, cut: int, seed=0) -> MarketData:
    """Same data up to row `cut`, random junk afterwards."""
    rng = np.random.default_rng(seed)
    out = {}
    for f in FIELDS:
        df = getattr(data, f).copy()
        tail = df.iloc[cut + 1 :]
        df.iloc[cut + 1 :] = tail.to_numpy()[rng.permutation(len(tail))] * rng.uniform(0.8, 1.2, tail.shape)
        out[f] = df
    out["high"] = np.maximum(out["high"], np.maximum(out["open"], out["close"]))
    out["low"] = np.minimum(out["low"], np.minimum(out["open"], out["close"]))
    return MarketData(**out)


@pytest.fixture
def data():
    return random_market_data(n_days=700, n_assets=5, seed=11)


# ---------------------------------------------------------------- rules


def two_asset_day():
    idx = dates(2)
    close = pd.DataFrame({"A": [100.0, 103.0], "B": [50.0, 49.0]}, idx)
    open_ = pd.DataFrame({"A": [100.0, 100.0], "B": [50.0, 50.0]}, idx)
    return build_market_data(close, open_)


def test_portfolio_return_matches_hand_calculation():
    data = two_asset_day()
    weights = pd.DataFrame({"A": [0.0, 0.5], "B": [0.0, -0.5]}, index=data.dates)  # traded on day 2
    r = rules.portfolio_returns(weights, data)
    ra, rb = 0.03, -0.02
    pnl = 0.5 * ra - 0.5 * rb
    cost = 0.0005 * (0.5 * 1 + 0.5 * (1 + ra)) + 0.0005 * (0.5 * 1 + 0.5 * (1 + rb))
    assert r.iloc[0] == 0.0
    assert r.iloc[1] == pytest.approx(pnl - cost)


def test_check_weights():
    rules.check_weights({"A": 0.6, "B": -0.4})
    with pytest.raises(ValueError, match="gross"):
        rules.check_weights({"A": 0.7, "B": -0.4})
    with pytest.raises(ValueError, match="finite"):
        rules.check_weights({"A": np.nan})


def test_portfolio_returns_rejects_too_much_exposure():
    data = two_asset_day()
    weights = pd.DataFrame({"A": [0.0, 0.8], "B": [0.0, 0.8]}, index=data.dates)
    with pytest.raises(ValueError, match="gross"):
        rules.portfolio_returns(weights, data)


def test_summary_has_the_judging_metrics():
    r = pd.Series([0.01, -0.005, 0.0, 0.002], index=dates(4))
    s = rules.summary(r, active=pd.Series([True, True, False, True], index=r.index))
    assert set(s) >= {"Sharpe", "CAGR", "Max drawdown", "Total return", "Active days"}
    assert s["Active days"] == 3


# ---------------------------------------------------------------- features


def test_every_feature_has_the_right_shape(data):
    for name, f in feat.compute_features(data, list(feat.FEATURES)).items():
        assert f.shape == data.close.shape, name
        assert list(f.columns) == data.assets, name


@pytest.mark.parametrize("name", list(feat.FEATURES))
def test_features_do_not_look_ahead(data, name):
    cut = 450
    real = feat.FEATURES[name](data)
    junk = feat.FEATURES[name](scramble_after(data, cut))
    pd.testing.assert_frame_equal(real.iloc[: cut + 1], junk.iloc[: cut + 1], check_exact=False, rtol=1e-9)


def test_reversal_feature_is_yesterdays_move_scaled_by_vol(data):
    f = feat.FEATURES["return_1d"](data)
    r = data.close.pct_change()
    expected = r / r.rolling(60, min_periods=20).std()
    pd.testing.assert_frame_equal(f, expected)


# ---------------------------------------------------------------- model


def planted_signal_market(n_days=900, beta=0.4, seed=3):
    """Tomorrow's intraday move = beta * (today's 'return_1d' feature) * vol + noise."""
    rng = np.random.default_rng(seed)
    base = random_market_data(n_days=n_days, n_assets=4, seed=seed)
    close, open_ = base.close.copy(), base.open.copy()
    sigma = 0.01
    for t in range(1, n_days):
        prev = close.iloc[t - 1] / close.iloc[t - 2] - 1 if t >= 2 else 0 * close.iloc[0]
        intraday = beta * prev + rng.normal(0, sigma, close.shape[1])
        open_.iloc[t] = close.iloc[t - 1] * (1 + rng.normal(0, 0.003, close.shape[1]))
        close.iloc[t] = open_.iloc[t] * (1 + intraday)
    return build_market_data(close, open_, volume=base.volume)


def test_fit_finds_a_planted_signal():
    data = planted_signal_market()
    model = fit_model(data, ["return_1d", "volume_z"], end=data.dates[-1], ridge=1.0)
    assert model.coefs["return_1d"] > 0.1
    assert abs(model.coefs["volume_z"]) < abs(model.coefs["return_1d"]) / 3


def test_model_roundtrips_through_a_dict():
    m = Model(names=["a", "b"], coefs={"a": 0.1, "b": -0.2}, scales={"a": 2.0, "b": 0.5}, intercept=0.01)
    assert Model.from_dict(m.to_dict()) == m


def test_walk_forward_forecast_does_not_look_ahead(data):
    names = ["return_1d", "rsi_2", "momentum_60"]
    cut = 500
    kw = dict(names=names, ridge=1.0, refit_days=63, min_train_days=120)
    real = walk_forward_forecast(data, **kw)
    junk = walk_forward_forecast(scramble_after(data, cut), **kw)
    assert real.iloc[: cut + 1].notna().any().any()
    pd.testing.assert_frame_equal(real.iloc[: cut + 1], junk.iloc[: cut + 1], check_exact=False, rtol=1e-9)


def test_walk_forward_is_blank_before_enough_history(data):
    f = walk_forward_forecast(data, names=["return_1d"], ridge=1.0, refit_days=63, min_train_days=200)
    assert f.iloc[:200].isna().all().all()


# ---------------------------------------------------------------- portfolio


def test_small_forecasts_are_not_traded():
    idx = dates(3)
    forecast = pd.DataFrame({"A": [0.0005, 0.003, -0.003], "B": [0.0, -0.0001, 0.0025]}, idx)
    vol = pd.DataFrame(0.01, idx, ["A", "B"])
    w = forecast_to_weights(forecast, vol, hurdle_bps=20, risk_per_trade=0.005, max_asset_weight=0.5)
    assert (w.iloc[0] == 0).all()
    assert w.loc[idx[1], "A"] > 0 and w.loc[idx[1], "B"] == 0
    assert w.loc[idx[2], "A"] < 0 and w.loc[idx[2], "B"] > 0


def test_weights_respect_caps():
    rng = np.random.default_rng(0)
    idx = dates(200)
    forecast = pd.DataFrame(rng.normal(0, 0.004, (200, 7)), idx, list("ABCDEFG"))
    vol = pd.DataFrame(rng.uniform(0.002, 0.03, (200, 7)), idx, forecast.columns)
    w = forecast_to_weights(forecast, vol, hurdle_bps=5, risk_per_trade=0.02, max_asset_weight=0.4)
    assert (w.abs().sum(axis=1) <= 1 + 1e-12).all()
    assert (w.abs() <= 0.4 + 1e-12).all().all()
    assert (np.sign(w) * np.sign(forecast) >= 0).all().all()


def test_long_only_never_shorts():
    idx = dates(2)
    forecast = pd.DataFrame({"A": [0.003, -0.003]}, idx)
    w = forecast_to_weights(forecast, pd.DataFrame(0.01, idx, ["A"]), 10, 0.005, 0.5, long_only=True)
    assert w.loc[idx[0], "A"] > 0 and w.loc[idx[1], "A"] == 0


def test_risk_manager_cuts_exposure_in_a_drawdown():
    rm = RiskManager(drawdown_limit=0.10, drawdown_scale=0.5)
    assert rm.scale(equity=1.0, peak=1.05) == 1.0
    assert rm.scale(equity=0.90, peak=1.05) == 0.5
    assert RiskManager(drawdown_limit=None).scale(equity=0.5, peak=1.0) == 1.0


# ---------------------------------------------------------------- settings


def write(tmp_path, text):
    p = tmp_path / "c.toml"
    p.write_text(text)
    return p


def test_settings_defaults_and_overrides(tmp_path):
    s = load_settings(write(tmp_path, "[sizing]\nhurdle_bps = 25\n[features]\nreturn_1d = false\n"))
    assert s.sizing.hurdle_bps == 25
    assert "return_1d" not in s.feature_names
    assert s.sizing.max_asset_weight == CompetitionSettings().sizing.max_asset_weight


def test_settings_typos_are_explained(tmp_path):
    with pytest.raises(ConfigError, match="hurdle_bps"):
        load_settings(write(tmp_path, "[sizing]\nhurdle_bp = 25\n"))
    with pytest.raises(ConfigError, match="return_1d"):
        load_settings(write(tmp_path, "[features]\nreturn_1 = true\n"))


def test_project_competition_toml_loads():
    from pathlib import Path

    s = load_settings(Path(__file__).parent.parent / "competition.toml")
    assert s.feature_names


# ---------------------------------------------------------------- backtest + trader


def small_settings(**sizing):
    s = CompetitionSettings()
    s.data.start_date = "2020-09-01"
    s.data.holdout_start = "2022-06-01"
    s.model.refit_days = 63
    s.model.min_train_days = 120
    for k, v in sizing.items():
        setattr(s.sizing, k, v)
    return s


def test_backtest_respects_the_rules_and_window(data):
    res = run_backtest(data, small_settings(hurdle_bps=2))
    assert res.returns.index.min() >= pd.Timestamp("2020-09-01")
    assert res.returns.index.max() < pd.Timestamp("2022-06-01")
    assert (res.weights.abs().sum(axis=1) <= 1 + 1e-9).all()
    assert res.summary["Active days"] > 0
    pd.testing.assert_series_equal(res.returns, rules.portfolio_returns(res.weights, data).loc[res.returns.index])


def test_parse_daily_data_accepts_common_shapes():
    rows = {
        "Asset_A": {"Date": "2024-01-02", "Open": 1.0, "High": 2.0, "Low": 0.5, "Close": 1.5, "Volume": 10},
        "Asset_B": {"date": "2024-01-02", "open": 3.0, "high": 4.0, "low": 2.5, "close/last": "$3.5", "volume": 20},
    }
    date, bars = parse_daily_data(rows)
    assert date == pd.Timestamp("2024-01-02")
    assert bars.loc["Asset_B", "close"] == 3.5
    frame = pd.DataFrame(rows).T
    date2, bars2 = parse_daily_data(frame)
    assert date2 == date and bars2.loc["Asset_A", "high"] == 2.0


def test_trader_output_is_valid(data):
    model = fit_model(data, ["return_1d", "rsi_2"], end=data.dates[300], ridge=1.0)
    trader = Trader(model=model, settings=small_settings(hurdle_bps=1))
    for t in range(len(data.dates)):
        bars = {a: {f: getattr(data, f).iloc[t][a] for f in FIELDS} | {"date": data.dates[t]} for a in data.assets}
        w = trader.on_day_close(bars, {"equity": 1.0, "peak_equity": 1.0})
        assert set(w) == set(data.assets)
        assert all(np.isfinite(v) for v in w.values())
        rules.check_weights(w)
    assert any(v != 0 for v in w.values())


def test_streaming_trader_matches_the_vectorised_backtest(data):
    """The program the evaluator runs must trade exactly like our research backtest."""
    settings = small_settings(hurdle_bps=3)
    model = fit_model(data, settings.feature_names, end=data.dates[250], ridge=settings.model.ridge)
    vectorised = run_backtest(data, settings, model=model)
    start = vectorised.returns.index[0]
    streamed = simulate(lambda history: Trader(model=model, settings=settings, history=history), data, start=start)
    window = vectorised.returns.index
    assert vectorised.summary["Active days"] > 10
    pd.testing.assert_frame_equal(
        streamed.weights.loc[window], vectorised.weights.loc[window], check_exact=False, atol=1e-9, check_freq=False
    )
    pd.testing.assert_series_equal(
        streamed.returns.loc[window], vectorised.returns, check_exact=False, atol=1e-12, check_freq=False
    )


def test_drawdown_brake_is_used_in_both_backtest_and_trader(data):
    settings = small_settings(hurdle_bps=1, risk_per_trade=0.05, max_asset_weight=0.5)
    settings.risk.drawdown_limit = 0.02
    settings.risk.drawdown_scale = 0.25
    model = fit_model(data, settings.feature_names, end=data.dates[250], ridge=settings.model.ridge)
    vectorised = run_backtest(data, settings, model=model)
    streamed = simulate(lambda h: Trader(model=model, settings=settings, history=h), data, start=vectorised.returns.index[0])
    pd.testing.assert_series_equal(
        streamed.returns.loc[vectorised.returns.index], vectorised.returns, check_exact=False, atol=1e-12, check_freq=False
    )


# ---------------------------------------------------------------- activity floor


def floor_inputs():
    idx = pd.bdate_range("2024-01-01", periods=5)  # Mon..Fri
    forecast = pd.DataFrame({"A": [0.0003, -0.0001, 0.003, 0.0, -0.0002], "B": [-0.0005, 0.0002, 0.0, 0.0, 0.0001]}, idx)
    return forecast, pd.DataFrame(0.01, idx, ["A", "B"])


def test_floor_trades_the_strongest_forecast_on_quiet_days():
    forecast, vol = floor_inputs()
    w = forecast_to_weights(forecast, vol, 20, 0.005, 0.5, floor_weight=0.01)
    d = forecast.index
    assert w.loc[d[0]].to_dict() == {"A": 0.0, "B": -0.01}  # strongest is B, short
    assert w.loc[d[1]].to_dict() == {"A": 0.0, "B": 0.01}
    assert w.loc[d[2], "A"] == pytest.approx(0.5)  # a real trade: no floor added
    assert w.loc[d[2], "B"] == 0.0
    assert (w.loc[d[3]] == 0).all()  # no forecast at all: nothing to trade
    assert w.loc[d[4]].to_dict() == {"A": -0.01, "B": 0.0}


def test_floor_can_be_limited_to_one_weekday():
    forecast, vol = floor_inputs()
    w = forecast_to_weights(forecast, vol, 20, 0.005, 0.5, floor_weight=0.01, floor_weekday=4)  # decided on Fridays
    assert (w.iloc[:2] == 0).all().all()
    assert w.iloc[4].abs().sum() == pytest.approx(0.01)


def test_floor_respects_long_only():
    forecast, vol = floor_inputs()
    w = forecast_to_weights(forecast, vol, 20, 0.005, 0.5, long_only=True, floor_weight=0.01)
    assert (w >= 0).all().all()
    assert w.iloc[0].to_dict() == {"A": 0.01, "B": 0.0}


def test_streaming_matches_backtest_with_the_floor(data):
    settings = small_settings(hurdle_bps=40, floor_weight=0.01, floor_weekday=-1)
    model = fit_model(data, settings.feature_names, end=data.dates[250], ridge=settings.model.ridge)
    vectorised = run_backtest(data, settings, model=model)
    assert vectorised.summary["Active days"] >= len(vectorised.returns) - 30
    streamed = simulate(lambda h: Trader(model=model, settings=settings, history=h), data, start=vectorised.returns.index[0])
    pd.testing.assert_series_equal(
        streamed.returns.loc[vectorised.returns.index], vectorised.returns, check_exact=False, atol=1e-12, check_freq=False
    )


# ---------------------------------------------------------------- blended sleeves

SLEEVES_TOML = """
[sleeves.market]
weight = 0.5
features = ["market_intraday_1d"]
ridge = 0.25
hurdle_bps = 2.0
floor_weight = 0.01

[sleeves.reversal]
weight = 0.5
features = ["return_1d", "rsi_2"]
hurdle_bps = 3.0
risk_per_trade = 0.01
"""


def sleeve_settings(tmp_path):
    s = load_settings(write(tmp_path, SLEEVES_TOML))
    s.data.start_date = "2020-09-01"
    s.data.holdout_start = "2022-06-01"
    s.model.refit_days = 63
    s.model.min_train_days = 120
    return s


def test_sleeves_load_with_their_own_settings(tmp_path):
    s = sleeve_settings(tmp_path)
    sleeves = s.sleeve_list()
    assert [(name, weight) for name, weight, _ in sleeves] == [("market", 0.5), ("reversal", 0.5)]
    market = sleeves[0][2]
    assert market.feature_names == ["market_intraday_1d"]
    assert market.model.ridge == 0.25 and market.sizing.hurdle_bps == 2.0 and market.sizing.floor_weight == 0.01
    assert market.model.refit_days == 63  # shared settings come from the main sections
    assert sleeves[1][2].sizing.risk_per_trade == 0.01


def test_no_sleeves_means_one_sleeve_of_everything():
    s = CompetitionSettings()
    assert [(n, w) for n, w, _ in s.sleeve_list()] == [("main", 1.0)]


def test_sleeve_mistakes_are_explained(tmp_path):
    with pytest.raises(ConfigError, match="hurdle_bps"):
        load_settings(write(tmp_path, "[sleeves.a]\nweight = 0.5\nhurdle_bp = 3.0\n"))
    with pytest.raises(ConfigError, match="add up"):
        load_settings(write(tmp_path, "[sleeves.a]\nweight = 0.7\n[sleeves.b]\nweight = 0.6\n"))
    with pytest.raises(ConfigError, match="return_1d"):
        load_settings(write(tmp_path, '[sleeves.a]\nweight = 0.5\nfeatures = ["return_1"]\n'))


def test_blend_is_the_weighted_sum_of_its_sleeves(tmp_path, data):
    s = sleeve_settings(tmp_path)
    blended = run_backtest(data, s)
    expected = sum(w * run_backtest(data, one).weights for _, w, one in s.sleeve_list())
    pd.testing.assert_frame_equal(blended.weights, expected, check_exact=False, atol=1e-12)
    assert (blended.weights.abs().sum(axis=1) <= 1 + 1e-9).all()


def test_streaming_matches_backtest_with_sleeves(tmp_path, data):
    s = sleeve_settings(tmp_path)
    models = {
        name: fit_model(data, one.feature_names, end=data.dates[250], ridge=one.model.ridge)
        for name, _, one in s.sleeve_list()
    }
    vectorised = run_backtest(data, s, model=models)
    assert vectorised.summary["Active days"] > 10
    streamed = simulate(lambda h: Trader(model=models, settings=s, history=h), data, start=vectorised.returns.index[0])
    pd.testing.assert_series_equal(
        streamed.returns.loc[vectorised.returns.index], vectorised.returns, check_exact=False, atol=1e-12, check_freq=False
    )


def test_fit_command_saves_every_sleeve(tmp_path, data, monkeypatch):
    import json

    from bse_quant.competition import cli

    cfg = write(tmp_path, SLEEVES_TOML)
    monkeypatch.setattr(cli, "load_market_data", lambda folder: data)
    out = tmp_path / "model.json"
    cli.fit_main(["--config", str(cfg), "--out", str(out), "--end", str(data.dates[-1].date())])
    saved = json.loads(out.read_text())
    assert set(saved["sleeves"]) == {"market", "reversal"}
    trader = Trader.from_files(out, cfg)
    assert set(trader.models) == {"market", "reversal"}
