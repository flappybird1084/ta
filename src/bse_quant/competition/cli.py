"""Command-line entry points for the competition system.

    uv run competition-backtest            # walk-forward backtest with competition.toml
    uv run competition-search              # try every setting in competition_search.toml
    uv run competition-search --worker 0 --hours 2   # ...and on shuffled histories
    uv run competition-search-report       # which settings beat luck?
    uv run competition-fit                 # fit the final model for the submission
"""

import argparse
import json
import os
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from bse_quant import metrics
from bse_quant.competition.backtest import run_backtest
from bse_quant.competition.model import fit_model, models_to_dict
from bse_quant.competition.search import (
    expand_grid,
    load_search,
    load_search_results,
    run_grid,
    run_search_worker,
    search_table,
    NOT_ELIGIBLE,
)
from bse_quant.competition.settings import load_settings
from bse_quant.config import ConfigError
from bse_quant.data import load_market_data


def _load(config):
    path = Path(config).resolve()
    try:
        settings = load_settings(path)
    except ConfigError as e:
        raise SystemExit(f"Problem in {path.name}: {e}")
    data = load_market_data(path.parent / settings.data.data_folder)
    return path, settings, data


def yearly_table(returns: pd.Series) -> pd.DataFrame:
    rows = {}
    for year, r in returns.groupby(returns.index.year):
        rows[year] = {"Sharpe": metrics.sharpe_ratio(r), "Return": metrics.total_return(r),
                      "Max drawdown": metrics.max_drawdown(r), "Active days": int((r != 0).sum())}  # fmt: skip
    return pd.DataFrame(rows).T


def _fmt(summary: dict) -> str:
    return (f"Sharpe {summary['Sharpe']:.2f} | CAGR {summary['CAGR']:.1%} | Max drawdown {summary['Max drawdown']:.1%}"
            f" | Volatility {summary['Volatility']:.1%} | traded {summary['Active days']} of {summary['Days']} days")  # fmt: skip


def backtest_main(argv=None):
    parser = argparse.ArgumentParser(description="Walk-forward backtest under the competition rules.")
    parser.add_argument("--config", default="competition.toml")
    args = parser.parse_args(argv)
    path, settings, data = _load(args.config)
    result = run_backtest(data, settings)
    r = result.returns
    print(f"Competition backtest {r.index[0].date()} to {r.index[-1].date()}  ({len(data.assets)} assets)")
    for name, weight, s in settings.sleeve_list():
        print(f"Sleeve {name} ({weight:.0%}): {', '.join(s.feature_names)}")
    if not settings.data.include_holdout:
        print(f"Holdout: data from {settings.data.holdout_start} on is kept aside (include_holdout = true to use it)")
    print("\n" + _fmt(result.summary) + "\n")
    print(yearly_table(r).to_string(formatters={"Sharpe": "{:.2f}".format, "Return": "{:.1%}".format,
                                                "Max drawdown": "{:.1%}".format, "Active days": "{:.0f}".format}))  # fmt: skip
    out = path.parent / settings.data.results_folder
    out.mkdir(parents=True, exist_ok=True)
    r.to_csv(out / "daily_returns.csv", header=["return"])
    result.weights.loc[r.index].to_csv(out / "weights.csv")
    fig, ax = plt.subplots(figsize=(11, 4))
    metrics.equity_curve(r).plot(ax=ax, color="#2b6cb0")
    ax.set_title(f"Competition system (walk-forward): {_fmt(result.summary)}", fontsize=9)
    ax.axhline(1, color="grey", linewidth=0.5)
    fig.tight_layout()
    fig.savefig(out / "equity_curve.png", dpi=110)
    plt.close(fig)
    print(f"\nSaved daily_returns.csv, weights.csv and equity_curve.png to {out}")


def search_main(argv=None):
    parser = argparse.ArgumentParser(description="Backtest every combination in the search grid.")
    parser.add_argument("--config", default="competition.toml")
    parser.add_argument("--grid", default="competition_search.toml")
    parser.add_argument("--out", default="results/competition_search")
    parser.add_argument("--worker", type=int, help="also run shuffled histories, as this worker number")
    parser.add_argument("--hours", type=float)
    parser.add_argument("--until", type=float, help="stop at this Unix time")
    parser.add_argument("--max", type=int)
    args = parser.parse_args(argv)
    _, settings, data = _load(args.config)
    grid = load_search(args.grid)
    if args.worker is None:
        configs = expand_grid(grid)
        dev = settings  # the window in competition.toml (holdout excluded unless switched on)
        min_active = grid["_options"].get("min_active_days", 0)
        table = run_grid(data, dev, configs, grid["features"], min_active).sort_values("Score", ascending=False)
        print(f"Settings trading on fewer than {min_active} days are marked not eligible (Score {NOT_ELIGIBLE:.0f}).")
        print(f"{len(configs)} combinations, walk-forward {dev.data.start_date} .. {dev.data.holdout_start}\n")
        print(table.head(30).to_string(float_format=lambda v: f"{v:8.3f}"))
        Path(args.out).mkdir(parents=True, exist_ok=True)
        table.to_csv(Path(args.out) / "grid_real.csv")
        return
    deadline = args.until or (time.time() + args.hours * 3600 if args.hours else None)
    print(f"[search worker {args.worker:02d}] pid {os.getpid()} -> {args.out}", flush=True)
    n = run_search_worker(data, settings, grid, args.out, args.worker, args.max, deadline, log=lambda m: print(m, flush=True))
    print(f"[search worker {args.worker:02d}] finished: {n} shuffles this session", flush=True)


def search_report_main(argv=None):
    parser = argparse.ArgumentParser(description="Which settings beat luck?")
    parser.add_argument("--out", default="results/competition_search")
    parser.add_argument("--top", type=int, default=25)
    args = parser.parse_args(argv)
    real, perms = load_search_results(args.out)
    table = search_table(real, perms)
    print(f"\n{len(perms)} shuffled histories x {len(real)} settings (smallest possible p: {1 / (len(perms) + 1):.2g})")
    print("p-value: how often a shuffled history did at least as well with the SAME settings.")
    print("Family-wise p: how often the BEST of all settings on a shuffled history did at least as well.\n")
    print(table.head(args.top).to_string(float_format=lambda v: f"{v:8.3f}"))
    table.to_csv(Path(args.out) / "search_table.csv")


def fit_main(argv=None):
    parser = argparse.ArgumentParser(description="Fit the final model on the development data and save it.")
    parser.add_argument("--config", default="competition.toml")
    parser.add_argument("--out", default="submission/model.json")
    parser.add_argument("--end", help="last day to learn from (default: holdout_start, or all data with include_holdout)")
    args = parser.parse_args(argv)
    _, settings, data = _load(args.config)
    end = args.end or (data.dates[-1] if settings.data.include_holdout else settings.data.holdout_start)
    models = {}
    for name, _, s in settings.sleeve_list():
        models[name] = fit_model(data, s.feature_names, end=end, ridge=s.model.ridge, start=s.data.train_start,
                                 fit_intercept=s.model.fit_intercept)  # fmt: skip
    saved = models_to_dict(models if settings.sleeves else models["main"])
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(saved, indent=2))
    print(f"Fit on {settings.data.train_start} .. {pd.Timestamp(end).date()}; saved {args.out}")
    for name, model in models.items():
        print(f"  sleeve {name}:")
        for f in model.names:
            print(f"    {f:24s} {model.coefs[f] / model.scales[f]:+.4f} per unit")
