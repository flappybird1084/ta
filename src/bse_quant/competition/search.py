"""Trying many settings at once, and checking whether the best of them is just luck.

The grid lives in competition_search.toml. `run_grid` backtests every
combination (walk-forward, so each one is honest on its own). Picking the best
of many is not honest, though, so `run_search_worker` re-runs the WHOLE grid on
shuffled histories: if the best shuffled result is often as good as our best
real one, the search found nothing (that is the "Family-wise p").
"""

import copy
import itertools
import time
import tomllib
from dataclasses import astuple
from pathlib import Path

import numpy as np
import pandas as pd

from bse_quant.competition.backtest import forecasts, run_backtest
from bse_quant.data import MarketData
from bse_quant.permutation import permute_bars

FIELDS = ["open", "high", "low", "close", "volume"]
METRICS = ["Sharpe", "CAGR", "Max drawdown", "Active days"]
NOT_ELIGIBLE = -99.0  # score given to settings that trade too rarely to count


def load_search(path) -> dict:
    """{"features": {set name: [feature, ...]}, "section.setting": [values to try], ...}

    Also returns the [search] options under "_options" (e.g. min_active_days).
    """
    with open(path, "rb") as f:
        raw = tomllib.load(f)
    return {"features": raw.get("feature_sets", {}), **raw.get("grid", {}), "_options": raw.get("search", {})}


def expand_grid(grid: dict) -> list:
    keys = [k for k in grid if not k.startswith("_")]
    options = [list(grid[k]) for k in keys]
    return [dict(zip(keys, combo)) for combo in itertools.product(*options)]


def config_name(config: dict) -> str:
    return " ".join(f"{k.split('.')[-1]}={v}" for k, v in config.items())


def apply_overrides(base, config: dict, feature_sets: dict):
    s = copy.deepcopy(base)
    s.sleeves = {}  # the grid describes one model; blended sleeves are set up in competition.toml
    s.trend.weight = 0.0  # ...and no trend holds, unless the grid sets trend.weight
    for key, value in config.items():
        if key == "features":
            s.features = {name: name in feature_sets[value] for name in s.features}
            continue
        section, setting = key.split(".")
        if not hasattr(getattr(s, section), setting):
            raise ValueError(f"Unknown setting {key!r}")
        setattr(getattr(s, section), setting, value)
    return s


def run_grid(data, base, configs, feature_sets, min_active_days=0) -> pd.DataFrame:
    """One row per config: the judged metrics of its walk-forward backtest.

    "Score" is the Sharpe ratio, or NOT_ELIGIBLE if it traded on fewer than `min_active_days` days.
    """
    cache, rows = {}, {}
    for config in configs:
        s = apply_overrides(base, config, feature_sets)
        key = (tuple(s.feature_names), astuple(s.model), s.data.train_start)
        if key not in cache:
            cache[key] = forecasts(data, s)
        result = run_backtest(data, s, forecast=cache[key])
        row = {m: result.summary[m] for m in METRICS}
        row["Score"] = row["Sharpe"] if row["Active days"] >= min_active_days else NOT_ELIGIBLE
        rows[config_name(config)] = row
    return pd.DataFrame(rows).T


def _before(data, day):
    keep = data.dates < pd.Timestamp(day)
    return MarketData(*(getattr(data, f)[keep] for f in FIELDS))


def run_search_worker(data, base, grid, out_dir, worker_id, max_permutations=None, deadline=None, log=print):
    """Append one row of Sharpe ratios (one per config) per shuffled history. Resumable."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    configs = expand_grid(grid)
    feature_sets = grid["features"]
    min_active = grid.get("_options", {}).get("min_active_days", 0)
    real_path = out_dir / "real.csv"
    if not real_path.exists():
        real = run_grid(_before(data, base.data.holdout_start), base, configs, feature_sets, min_active)
        tmp = out_dir / f".real_{worker_id}.tmp"
        real.to_csv(tmp)
        tmp.replace(real_path)
    path = out_dir / f"worker_{worker_id:02d}.csv"
    done = len(pd.read_csv(path)) if path.exists() else 0
    k, started = done, time.time()
    while (deadline is None or time.time() < deadline) and (max_permutations is None or k < max_permutations):
        seed = worker_id * 1_000_000 + k
        shuffled = permute_bars(data, base.data.train_start, base.data.holdout_start, seed)
        score = run_grid(shuffled, base, configs, feature_sets, min_active)["Score"]
        pd.DataFrame([{"seed": seed, **score.to_dict()}]).to_csv(path, mode="a", header=not path.exists(), index=False)
        k += 1
        if log:
            rate = (k - done) / (time.time() - started) * 3600
            left = "" if deadline is None else f" | {(deadline - time.time()) / 3600:5.2f}h left"
            log(f"[search worker {worker_id:02d}] shuffle {k:6d} | {rate:6.0f}/h{left}")
    return k - done


def load_search_results(out_dir):
    out_dir = Path(out_dir)
    real = pd.read_csv(out_dir / "real.csv", index_col=0)
    frames = []
    for f in sorted(out_dir.glob("worker_*.csv")):
        try:
            frames.append(pd.read_csv(f, on_bad_lines="skip"))
        except pd.errors.EmptyDataError:
            continue
    perms = pd.concat(frames, ignore_index=True).dropna() if frames else pd.DataFrame(columns=["seed", *real.index])
    return real, perms


def search_table(real: pd.DataFrame, perms: pd.DataFrame) -> pd.DataFrame:
    """Real metrics per config, plus how often shuffled histories scored as well.

    Settings that trade too rarely score NOT_ELIGIBLE on real and shuffled data alike.
    """
    names = list(real.index)
    sharpe = real["Score"].astype(float)
    shuffled = perms[names].astype(float)
    n = len(shuffled)
    best = shuffled.max(axis=1)
    out = real.copy()
    out["Shuffled mean"] = shuffled.where(shuffled > NOT_ELIGIBLE).mean()
    out["p-value"] = [(1 + (shuffled[c] >= sharpe[c]).sum()) / (n + 1) for c in names]
    out["Family-wise p"] = [(1 + (best >= sharpe[c]).sum()) / (n + 1) for c in names]
    out.loc[sharpe <= NOT_ELIGIBLE, ["p-value", "Family-wise p"]] = float("nan")
    return out.sort_values("Score", ascending=False)
