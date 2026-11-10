"""Reading competition.toml."""

import copy
import tomllib
from dataclasses import dataclass, field, fields
from pathlib import Path

from bse_quant.competition.features import FEATURES
from bse_quant.config import ConfigError, _check_types, _suggest

DEFAULT_FEATURES = [
    "return_1d", "intraday_1d", "overnight_1d", "rsi_2", "bar_position", "market_intraday_1d",
]  # fmt: skip


@dataclass
class DataSettings:
    data_folder: str = "data"
    results_folder: str = "results/competition"
    train_start: str = "2017-01-01"  # first day the model learns from (earlier = indicator warm-up)
    start_date: str = "2018-01-01"  # first day of the reported (walk-forward) backtest
    holdout_start: str = "2022-01-01"  # data from here on is kept aside for one final check
    include_holdout: bool = False


@dataclass
class ModelSettings:
    ridge: float = 0.05  # shrinks the model's coefficients; bigger = more cautious
    refit_days: int = 252  # walk-forward: re-fit the model this often
    min_train_days: int = 252  # walk-forward: days of history before the first fit
    fit_intercept: bool = False  # true = also learn the average daily drift


@dataclass
class SizingSettings:
    hurdle_bps: float = 20.0  # only trade when the forecast beats this (costs are ~10)
    risk_per_trade: float = 0.005  # each trade sized to move the account ~0.5% on a typical day
    max_asset_weight: float = 0.5  # largest |weight| in one asset
    long_only: bool = False
    floor_weight: float = 0.0  # 0 = off; e.g. 0.01 = on days with no trade, put 1% in the strongest forecast
    floor_weekday: int = -1  # -1 = every quiet day; 0..4 = only on days decided on that weekday (4 = Friday)


@dataclass
class RiskSettings:
    drawdown_limit: float = 0.0  # 0 = off; e.g. 0.10 = shrink positions once 10% below the peak
    drawdown_scale: float = 0.5  # how much to shrink them by


@dataclass
class CompetitionSettings:
    data: DataSettings = field(default_factory=DataSettings)
    model: ModelSettings = field(default_factory=ModelSettings)
    sizing: SizingSettings = field(default_factory=SizingSettings)
    risk: RiskSettings = field(default_factory=RiskSettings)
    features: dict = field(default_factory=lambda: {name: name in DEFAULT_FEATURES for name in FEATURES})
    sleeves: dict = field(default_factory=dict)  # name -> {"weight": .., "features": [..], model/sizing overrides}

    @property
    def feature_names(self) -> list:
        return [name for name, on in self.features.items() if on]

    def sleeve_list(self) -> list:
        """[(name, weight, settings)] for every sleeve; no [sleeves] = one sleeve using everything above."""
        if not self.sleeves:
            return [("main", 1.0, self)]
        out = []
        for name, overrides in self.sleeves.items():
            s = copy.deepcopy(self)
            s.sleeves = {}
            for key, value in overrides.items():
                if key == "weight":
                    continue
                if key == "features":
                    s.features = {f: f in value for f in s.features}
                else:
                    setattr(getattr(s, SLEEVE_KEYS[key]), key, value)
            out.append((name, float(overrides.get("weight", 1.0)), s))
        return out


SECTIONS = {"data": DataSettings, "model": ModelSettings, "sizing": SizingSettings, "risk": RiskSettings}
SLEEVE_KEYS = {f.name: section for section in ("model", "sizing") for f in fields(SECTIONS[section])}


def _parse_sleeves(raw: dict) -> dict:
    sleeves = {}
    for name, values in raw.items():
        where = f"[sleeves.{name}]"
        if not isinstance(values, dict):
            raise ConfigError(f"{where} must be a table of settings")
        for key, value in values.items():
            if key == "weight":
                if not isinstance(value, (int, float)) or isinstance(value, bool) or value < 0:
                    raise ConfigError(f"{where} weight must be a number >= 0")
            elif key == "features":
                if not isinstance(value, list) or not value:
                    raise ConfigError(f"{where} features must be a list of feature names")
                for f in value:
                    if f not in FEATURES:
                        raise ConfigError(f"{where} has no feature {f!r}." + _suggest(f, FEATURES))
            elif key not in SLEEVE_KEYS:
                raise ConfigError(f"{where} has no setting {key!r}." + _suggest(key, [*SLEEVE_KEYS, "weight", "features"]))
            else:
                section = SLEEVE_KEYS[key]
                _check_types(where, {key: value}, {f.name: f.default for f in fields(SECTIONS[section])})
        sleeves[name] = dict(values)
    total = sum(v.get("weight", 1.0) for v in sleeves.values())
    if total > 1 + 1e-9:
        raise ConfigError(f"[sleeves] weights add up to {total:g}; they must add up to at most 1 (100% of the account)")
    return sleeves


def load_settings(path) -> CompetitionSettings:
    with open(Path(path), "rb") as f:
        try:
            raw = tomllib.load(f)
        except tomllib.TOMLDecodeError as e:
            raise ConfigError(f"{path} is not valid TOML: {e}") from e
    settings = CompetitionSettings()
    for section, values in raw.items():
        if section == "sleeves":
            settings.sleeves = _parse_sleeves(values)
            continue
        if section == "features":
            for name, on in values.items():
                if name not in FEATURES:
                    raise ConfigError(f"[features] has no feature {name!r}." + _suggest(name, FEATURES))
                if not isinstance(on, bool):
                    raise ConfigError(f"[features] {name} must be true or false")
                settings.features[name] = on
            continue
        if section not in SECTIONS:
            raise ConfigError(f"Unknown section [{section}]." + _suggest(section, [*SECTIONS, "features", "sleeves"]))
        defaults = {f.name: f.default for f in fields(SECTIONS[section])}
        for key in values:
            if key not in defaults:
                raise ConfigError(f"[{section}] has no setting {key!r}." + _suggest(key, defaults))
        _check_types(section, values, defaults)
        for key, value in values.items():
            setattr(getattr(settings, section), key, value)
    if not settings.feature_names:
        raise ConfigError("[features] switch on at least one feature")
    return settings
