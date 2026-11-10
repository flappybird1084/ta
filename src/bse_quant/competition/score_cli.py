"""uv run competition-score: estimate the competition score of many settings."""

import argparse
import copy
import time
import tomllib
from pathlib import Path

import pandas as pd

from bse_quant.competition.backtest import forecasts, run_backtest
from bse_quant.competition.competitors import competitor_fields
from bse_quant.competition.scoring import simulate_scores
from bse_quant.competition.search import apply_overrides, config_name, expand_grid
from bse_quant.competition.settings import load_settings
from bse_quant.data import load_market_data


def candidate_returns(data, settings, configs, feature_sets) -> pd.DataFrame:
    base = copy.deepcopy(settings)
    base.data.include_holdout = True
    cache, out = {}, {}
    for c in configs:
        s = apply_overrides(base, c, feature_sets)
        key = (tuple(s.feature_names), s.model.ridge, s.model.fit_intercept)
        if key not in cache:
            cache[key] = forecasts(data, s)
        out[config_name(c)] = run_backtest(data, s, forecast=cache[key]).returns
    return pd.DataFrame(out)


def current_name(settings, grid) -> str:
    """The name the current competition.toml settings would have in this grid (if they're in it)."""
    parts = []
    for key in grid:
        if key.startswith("_"):
            continue
        if key == "features":
            match = [n for n, f in grid["features"].items() if sorted(f) == sorted(settings.feature_names)]
            parts.append(f"features={match[0] if match else '?'}")
        else:
            section, name = key.split(".")
            parts.append(f"{name}={getattr(getattr(settings, section), name)}")
    return " ".join(parts)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Estimate the competition score of many settings.")
    parser.add_argument("--config", default="competition.toml")
    parser.add_argument("--grid", default="competition_score.toml")
    parser.add_argument("--out", default="results/competition_score")
    parser.add_argument("--trials", type=int, help="override [score] trials")
    args = parser.parse_args(argv)
    t0 = time.time()
    path = Path(args.config).resolve()
    settings = load_settings(path)
    data = load_market_data(path.parent / settings.data.data_folder)
    with open(args.grid, "rb") as f:
        raw = tomllib.load(f)
    opts = raw["score"]
    grid = {"features": raw["feature_sets"], **raw["grid"]}
    configs = expand_grid(grid)
    ours = candidate_returns(data, settings, configs, grid["features"])
    fields = competitor_fields(data, settings)
    print(f"{len(configs)} candidates x {len(fields)} fields x {len(opts['periods'])} periods "
          f"({time.time() - t0:.0f}s to build)", flush=True)  # fmt: skip
    trials = args.trials or opts["trials"]
    columns, detail = {}, {}
    for period, (start, end) in opts["periods"].items():
        for fname, field in fields.items():
            res = simulate_scores(
                ours.loc[start:end], field.loc[start:end], n_trials=trials, length=opts["trial_days"],
                block=opts["block_days"], seed=1, our_min_active=opts["our_min_active_days"],
            )  # fmt: skip
            columns[f"{period} | {fname}"] = res["Score"]
            detail[(period, fname)] = res
        print(f"  scored {period} ({time.time() - t0:.0f}s)", flush=True)
    table = pd.DataFrame(columns)
    for period in opts["periods"]:
        table[f"{period} avg"] = table[[c for c in table.columns if c.startswith(f"{period} |")]].mean(axis=1)
    choose = f"{opts['choose_on']} avg"
    table = table.sort_values(choose, ascending=False)
    # what the scores are made of, for the chosen period vs. the naive+competent field
    parts = detail[(opts["choose_on"], "naive+competent")]
    for col in ["Sharpe pts", "CAGR pts", "Max drawdown pts", "Median Sharpe", "Median CAGR", "Median Max drawdown", "Win rate"]:
        table[col] = parts[col]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "scores.csv")
    show = [c for c in table.columns if c.endswith(" avg")] + ["Sharpe pts", "CAGR pts", "Max drawdown pts", "Median CAGR", "Median Max drawdown"]
    pd.set_option("display.width", 250)
    print(f"\nExpected score out of 100 (average over the {len(fields)} fields), best first by '{choose}':\n")
    print(table[show].head(25).to_string(float_format=lambda v: f"{v:7.2f}"))
    now = current_name(settings, grid)
    if now in table.index:
        print(f"\nCurrent competition.toml ({now}):")
        print(table.loc[[now], show].to_string(float_format=lambda v: f"{v:7.2f}"))
    print(f"\nSaved {out / 'scores.csv'}  ({time.time() - t0:.0f}s)")
