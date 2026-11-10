import pandas as pd
from conftest import random_market_data

from bse_quant.cli import main
from bse_quant.config import BacktestSettings, Config, StrategySettings, VolTargetSettings
from bse_quant.engine import run_backtest
from bse_quant.report import summary_table


def small_result():
    config = Config(
        backtest=BacktestSettings(start_date="2020-06-01", include_holdout=True),
        strategies={
            "time_series_momentum": StrategySettings(weight=3.0, params={"lookback_days": 20}),
            "internal_bar_strength": StrategySettings(weight=1.0),
        },
        volatility_targeting=VolTargetSettings(enabled=True, weight=0.5),
    )
    return run_backtest(random_market_data(), config)


def test_summary_table_has_a_row_per_strategy_and_the_portfolio():
    table = summary_table(small_result())
    assert list(table.index) == ["time_series_momentum", "internal_bar_strength", "PORTFOLIO"]
    for column in ["Weight", "CAGR", "Volatility", "Sharpe", "Max drawdown", "Portfolio Sharpe without it"]:
        assert column in table.columns
    assert table.loc["time_series_momentum", "Weight"] == 0.75
    assert table.loc["PORTFOLIO", "Weight"] == 1.0


def write_csvs(folder):
    data = random_market_data(n_days=300, n_assets=3)
    folder.mkdir()
    for asset in data.assets:
        df = pd.DataFrame(
            {
                "Date": data.dates.strftime("%Y-%m-%d"),
                "Close/Last": data.close[asset].map("${:.2f}".format),
                "Volume": data.volume[asset].round().astype(int),
                "Open": data.open[asset].map("${:.2f}".format),
                "High": data.high[asset].map("${:.2f}".format),
                "Low": data.low[asset].map("${:.2f}".format),
            }
        )
        df.to_csv(folder / f"Asset_{asset}.csv", index=False)


def test_command_line_end_to_end(tmp_path, capsys):
    write_csvs(tmp_path / "prices")
    config = tmp_path / "my_config.toml"
    config.write_text(
        """
[backtest]
data_folder = "prices"
results_folder = "out"
start_date = "2020-06-01"
include_holdout = true

[strategies.moving_average_crossover]
fast_days = 5
slow_days = 20

[strategies.bollinger_reversion]
weight = 2
"""
    )
    main(["--config", str(config)])

    printed = capsys.readouterr().out
    assert "moving_average_crossover" in printed
    assert "PORTFOLIO" in printed
    # paths in the config are relative to the config file
    assert (tmp_path / "out" / "summary.csv").exists()
    assert (tmp_path / "out" / "daily_returns.csv").exists()
    assert (tmp_path / "out" / "equity_curves.png").exists()


def test_command_line_reports_config_mistakes_plainly(tmp_path, capsys):
    config = tmp_path / "bad.toml"
    config.write_text("[strategies.not_a_strategy]\nenabled = true\n")
    try:
        main(["--config", str(config)])
    except SystemExit as e:
        assert e.code != 0
    assert "not_a_strategy" in capsys.readouterr().err
