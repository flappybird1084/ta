"""Command-line entry points for permutation testing.

    uv run permutation-test --worker 0 --hours 18     # one worker; run several side by side
    uv run permutation-report                          # p-values from everything so far
"""

import argparse
import os
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from bse_quant.config import load_config
from bse_quant.data import load_market_data
from bse_quant.permutation import load_permutations, permutation_table, run_worker

DEFAULT_OUT = "results/permutation"


def worker_main(argv=None):
    parser = argparse.ArgumentParser(description="Run shuffled-history backtests until told to stop.")
    parser.add_argument("--config", default="config.toml")
    parser.add_argument("--worker", type=int, required=True, help="a different number for each worker")
    parser.add_argument("--hours", type=float, help="stop after this many hours")
    parser.add_argument("--until", type=float, help="stop at this Unix time (lets workers share one deadline)")
    parser.add_argument("--max", type=int, help="stop after this many shuffles in total for this worker")
    parser.add_argument("--out", default=DEFAULT_OUT)
    args = parser.parse_args(argv)

    deadline = args.until if args.until else (time.time() + args.hours * 3600 if args.hours else None)
    config_path = Path(args.config).resolve()
    config = load_config(config_path)
    data = load_market_data(config_path.parent / config.backtest.data_folder)
    print(f"[worker {args.worker:02d}] pid {os.getpid()} writing to {args.out}", flush=True)
    n = run_worker(
        data, config, args.out, args.worker, args.max, deadline, log=lambda m: print(m, flush=True)
    )
    print(f"[worker {args.worker:02d}] finished: {n} shuffles this session", flush=True)


def report_main(argv=None):
    parser = argparse.ArgumentParser(description="Summarise permutation test results.")
    parser.add_argument("--out", default=DEFAULT_OUT)
    args = parser.parse_args(argv)
    out = Path(args.out)

    real, perms = load_permutations(out)
    table = permutation_table(real, perms).sort_values("p-value")
    print(f"\nPermutation test: {len(perms)} shuffles (smallest possible p-value: {1 / (len(perms) + 1):.2g})")
    print("p-value: chance a shuffled history does at least this well.")
    print("Family-wise p: same, against the BEST of all strategies on each shuffle (corrects for trying many).\n")
    print(table.to_string(float_format=lambda v: f"{v:8.4f}"))
    table.to_csv(out / "pvalues.csv")
    plot_distributions(real, perms, out / "distributions.png")
    print(f"\nSaved pvalues.csv and distributions.png to {out}")


def plot_distributions(real, perms, path):
    names = list(real.index)
    cols = 4
    rows = -(-len(names) // cols)
    fig, axes = plt.subplots(rows, cols, figsize=(16, 3 * rows), squeeze=False)
    for ax, name in zip(axes.flat, names):
        ax.hist(perms[name], bins=50, color="#8899aa")
        ax.axvline(real[name], color="crimson", linewidth=2)
        ax.set_title(name, fontsize=9)
        ax.tick_params(labelsize=7)
    for ax in list(axes.flat)[len(names):]:
        ax.axis("off")
    fig.suptitle(f"Sharpe on {len(perms)} shuffled histories (grey) vs. real (red)")
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)
