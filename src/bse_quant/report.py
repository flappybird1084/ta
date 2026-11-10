"""Turning a backtest result into a table, CSV files and a chart."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # draw to files, no window needed
import matplotlib.pyplot as plt
import pandas as pd

from bse_quant import metrics
from bse_quant.engine import BacktestResult

PERCENT_COLUMNS = ["Weight", "CAGR", "Volatility", "Max drawdown", "Hit rate", "Time in market"]


def summary_table(result: BacktestResult) -> pd.DataFrame:
    """One row per strategy (on its own) plus the blended PORTFOLIO."""
    total_weight = sum(result.weights.values())
    rows = {}
    for name in result.strategy_returns.columns:
        rows[name] = {
            "Weight": result.weights[name] / total_weight,
            **metrics.summary(result.strategy_returns[name]),
            "Portfolio Sharpe without it": metrics.sharpe_ratio(result.portfolio_without(name)),
        }
    rows["PORTFOLIO"] = {
        "Weight": 1.0,
        **metrics.summary(result.portfolio_returns),
        "Portfolio Sharpe without it": float("nan"),
    }
    return pd.DataFrame(rows).T


def format_table(table: pd.DataFrame) -> str:
    shown = table.copy().astype(object)
    for col in shown.columns:
        if col in PERCENT_COLUMNS:
            shown[col] = table[col].map(lambda v: f"{v:7.1%}")
        else:
            shown[col] = table[col].map(lambda v: "" if pd.isna(v) else f"{v:6.2f}")
    return shown.to_string()


def save_results(result: BacktestResult, table: pd.DataFrame, folder: Path) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    table.to_csv(folder / "summary.csv")
    daily = result.strategy_returns.copy()
    daily["PORTFOLIO"] = result.portfolio_returns
    daily.to_csv(folder / "daily_returns.csv")
    plot_equity(result, folder / "equity_curves.png")


def plot_equity(result: BacktestResult, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 7))
    colors = plt.get_cmap("tab20").colors
    for i, name in enumerate(result.strategy_returns.columns):
        ax.plot(
            metrics.equity_curve(result.strategy_returns[name]),
            color=colors[i % len(colors)],
            linestyle=["-", "--", ":"][i // len(colors) % 3],
            linewidth=1.0,
            alpha=0.8,
            label=name,
        )
    ax.plot(metrics.equity_curve(result.portfolio_returns), color="black", linewidth=2.5, label="PORTFOLIO")
    ax.set_yscale("log")
    ax.set_title("Growth of $1 (log scale)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
