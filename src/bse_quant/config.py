"""Reading config.toml and checking it for mistakes."""

import inspect
import tomllib
from dataclasses import dataclass, field, fields
from difflib import get_close_matches
from pathlib import Path


class ConfigError(ValueError):
    """Something in config.toml doesn't make sense. The message says what."""


@dataclass
class BacktestSettings:
    data_folder: str = "data"
    results_folder: str = "results"
    start_date: str = "2017-11-01"
    holdout_start: str = "2022-01-01"
    include_holdout: bool = False
    cost_bps: float = 5.0
    allow_short: bool = True
    normalize_strategy_risk: bool = True
    strategy_risk_target: float = 0.10
    strategy_risk_lookback_days: int = 63
    strategy_max_leverage: float = 3.0


@dataclass
class StrategySettings:
    enabled: bool = True
    weight: float = 1.0
    params: dict = field(default_factory=dict)


@dataclass
class VolTargetSettings:
    enabled: bool = False
    weight: float = 1.0  # 0 = no effect, 1 = full volatility targeting
    target_vol: float = 0.10
    lookback_days: int = 63
    max_leverage: float = 2.0


@dataclass
class Config:
    backtest: BacktestSettings
    strategies: dict
    volatility_targeting: VolTargetSettings


def _suggest(name, options):
    close = get_close_matches(name, options, n=1)
    hint = f" Did you mean {close[0]!r}?" if close else ""
    return f"{hint} Valid options: {', '.join(sorted(options))}"


def _type_ok(value, default):
    if isinstance(default, bool):
        return isinstance(value, bool)
    if isinstance(default, float):
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if isinstance(default, int):
        return isinstance(value, int) and not isinstance(value, bool)
    if isinstance(default, (list, tuple)):
        return isinstance(value, (list, tuple))
    if isinstance(default, str):
        return isinstance(value, str)
    return True


def _check_types(section, values, defaults):
    for key, value in values.items():
        if not _type_ok(value, defaults[key]):
            raise ConfigError(
                f"[{section}] {key} = {value!r} has the wrong type; expected something like {defaults[key]!r}"
            )


def _parse_backtest(raw: dict) -> BacktestSettings:
    defaults = {f.name: f.default for f in fields(BacktestSettings)}
    for key in raw:
        if key not in defaults:
            raise ConfigError(f"[backtest] has no setting {key!r}." + _suggest(key, defaults))
    _check_types("backtest", raw, defaults)
    return BacktestSettings(**raw)


def _split_common(section: str, raw: dict):
    raw = dict(raw)
    enabled = raw.pop("enabled", True)
    weight = raw.pop("weight", 1.0)
    if not isinstance(enabled, bool):
        raise ConfigError(f"[{section}] enabled must be true or false")
    if not isinstance(weight, (int, float)) or isinstance(weight, bool) or weight < 0:
        raise ConfigError(f"[{section}] weight must be a number >= 0, got {weight!r}")
    return enabled, float(weight), raw


def _parse_vol_target(raw: dict) -> VolTargetSettings:
    section = "strategies.volatility_targeting"
    enabled, weight, rest = _split_common(section, raw)
    if weight > 1:
        raise ConfigError(f"[{section}] weight must be between 0 and 1 (it's how strongly to apply it)")
    defaults = {f.name: f.default for f in fields(VolTargetSettings)}
    for key in rest:
        if key not in defaults or key in ("enabled", "weight"):
            raise ConfigError(f"[{section}] has no setting {key!r}." + _suggest(key, list(defaults)[2:]))
    _check_types(section, rest, defaults)
    return VolTargetSettings(enabled=enabled, weight=weight, **rest)


def strategy_parameters(generate) -> dict:
    """The tunable settings of a strategy function and their defaults."""
    params = inspect.signature(generate).parameters
    return {name: p.default for name, p in params.items() if name != "data"}


def load_config(path, registry=None) -> Config:
    if registry is None:
        from bse_quant.strategies import STRATEGIES as registry

    with open(Path(path), "rb") as f:
        try:
            raw = tomllib.load(f)
        except tomllib.TOMLDecodeError as e:
            raise ConfigError(f"{path} is not valid TOML: {e}") from e

    for key in raw:
        if key not in ("backtest", "strategies"):
            raise ConfigError(f"Unknown section [{key}]. Use [backtest] or [strategies.<name>].")

    backtest = _parse_backtest(raw.get("backtest", {}))
    strategies = {}
    vol_target = VolTargetSettings()
    for name, section in raw.get("strategies", {}).items():
        if name == "volatility_targeting":
            vol_target = _parse_vol_target(section)
            continue
        if name not in registry:
            raise ConfigError(
                f"Unknown strategy {name!r}." + _suggest(name, list(registry) + ["volatility_targeting"])
            )
        enabled, weight, params = _split_common(f"strategies.{name}", section)
        defaults = strategy_parameters(registry[name])
        for key in params:
            if key not in defaults:
                raise ConfigError(f"[strategies.{name}] has no setting {key!r}." + _suggest(key, defaults))
        _check_types(f"strategies.{name}", params, defaults)
        strategies[name] = StrategySettings(enabled=enabled, weight=weight, params=params)

    return Config(backtest=backtest, strategies=strategies, volatility_targeting=vol_target)
