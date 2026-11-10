"""`uv run backtest` - run the backtest described in config.toml and print the results."""

import argparse
import sys
from pathlib import Path

from bse_quant.config import ConfigError, load_config
from bse_quant.data import load_market_data
from bse_quant.engine import run_backtest
from bse_quant.report import format_table, save_results, summary_table


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run the backtest described in a config file.")
    parser.add_argument("--config", default="config.toml", help="which settings file to use (default: config.toml)")
    args = parser.parse_args(argv)

    config_path = Path(args.config).resolve()
    try:
        config = load_config(config_path)
    except (ConfigError, FileNotFoundError) as e:
        print(f"Problem with {config_path.name}: {e}", file=sys.stderr)
        raise SystemExit(1)

    bt = config.backtest
    base = config_path.parent  # folders in the config are relative to the config file
    data = load_market_data(base / bt.data_folder)
    try:
        result = run_backtest(data, config)
    except ValueError as e:
        print(f"Problem: {e}", file=sys.stderr)
        raise SystemExit(1)

    window = result.portfolio_returns.index
    print(f"\nBacktest {window[0]:%Y-%m-%d} to {window[-1]:%Y-%m-%d}  ({len(window)} days, {len(data.assets)} assets)")
    print(f"Costs: {bt.cost_bps} bps per trade | shorting {'on' if bt.allow_short else 'off'}", end="")
    vt = config.volatility_targeting
    print(f" | volatility targeting {'on (' + format(vt.weight, '.0%') + ')' if vt.enabled else 'off'}")
    if not bt.include_holdout:
        print(f"Holdout: data from {bt.holdout_start} onward is kept aside (set include_holdout = true to use it)")
    print()

    table = summary_table(result)
    print(format_table(table))

    results_folder = base / bt.results_folder
    save_results(result, table, results_folder)
    print(f"\nSaved summary.csv, daily_returns.csv and equity_curves.png to {results_folder}")


if __name__ == "__main__":
    main()
