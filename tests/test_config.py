from pathlib import Path

import pytest

from bse_quant.config import ConfigError, load_config
from bse_quant.strategies import STRATEGIES

PROJECT_CONFIG = Path(__file__).resolve().parent.parent / "config.toml"


def dummy(data, lookback_days=10, allow_flip=True, name="x", windows=(1, 2)):
    return None


REGISTRY = {"dummy": dummy}


def write(tmp_path, text):
    path = tmp_path / "config.toml"
    path.write_text(text)
    return path


def test_loads_settings_and_strategy_params(tmp_path):
    path = write(
        tmp_path,
        """
[backtest]
cost_bps = 2.5
allow_short = false

[strategies.dummy]
enabled = true
weight = 2.0
lookback_days = 30

[strategies.volatility_targeting]
enabled = true
weight = 0.5
target_vol = 0.12
""",
    )
    cfg = load_config(path, registry=REGISTRY)
    assert cfg.backtest.cost_bps == 2.5
    assert cfg.backtest.allow_short is False
    assert cfg.strategies["dummy"].enabled is True
    assert cfg.strategies["dummy"].weight == 2.0
    assert cfg.strategies["dummy"].params == {"lookback_days": 30}
    assert cfg.volatility_targeting.enabled is True
    assert cfg.volatility_targeting.weight == 0.5
    assert cfg.volatility_targeting.target_vol == 0.12


def test_missing_sections_use_defaults(tmp_path):
    cfg = load_config(write(tmp_path, ""), registry=REGISTRY)
    assert cfg.backtest.cost_bps == 5.0
    assert cfg.strategies == {}
    assert cfg.volatility_targeting.enabled is False


def test_unknown_strategy_is_a_helpful_error(tmp_path):
    path = write(tmp_path, "[strategies.dumy]\nenabled = true\n")
    with pytest.raises(ConfigError, match="dumy.*dummy"):
        load_config(path, registry=REGISTRY)


def test_unknown_parameter_lists_valid_ones(tmp_path):
    path = write(tmp_path, "[strategies.dummy]\nlookback = 5\n")
    with pytest.raises(ConfigError, match="lookback_days"):
        load_config(path, registry=REGISTRY)


def test_wrong_parameter_type(tmp_path):
    path = write(tmp_path, '[strategies.dummy]\nlookback_days = "ten"\n')
    with pytest.raises(ConfigError, match="lookback_days"):
        load_config(path, registry=REGISTRY)


def test_int_is_accepted_where_float_expected(tmp_path):
    def f(data, threshold=1.5):
        return None

    path = write(tmp_path, "[strategies.f]\nthreshold = 2\n")
    cfg = load_config(path, registry={"f": f})
    assert cfg.strategies["f"].params["threshold"] == 2


def test_negative_weight_rejected(tmp_path):
    path = write(tmp_path, "[strategies.dummy]\nweight = -1\n")
    with pytest.raises(ConfigError, match="weight"):
        load_config(path, registry=REGISTRY)


def test_unknown_backtest_setting_rejected(tmp_path):
    path = write(tmp_path, "[backtest]\ncost = 3\n")
    with pytest.raises(ConfigError, match="cost_bps"):
        load_config(path, registry=REGISTRY)


def test_volatility_targeting_weight_must_be_between_zero_and_one(tmp_path):
    path = write(tmp_path, "[strategies.volatility_targeting]\nweight = 1.5\n")
    with pytest.raises(ConfigError, match="between 0 and 1"):
        load_config(path, registry=REGISTRY)


def test_project_config_lists_every_strategy():
    cfg = load_config(PROJECT_CONFIG)
    assert set(cfg.strategies) == set(STRATEGIES)


def test_project_config_values_match_code_defaults():
    """config.toml should start out showing each strategy's real defaults."""
    import inspect

    cfg = load_config(PROJECT_CONFIG)
    for name, settings in cfg.strategies.items():
        sig = inspect.signature(STRATEGIES[name])
        for param, value in settings.params.items():
            default = sig.parameters[param].default
            if isinstance(default, tuple):
                default = list(default)
            assert value == default, f"{name}.{param}"
