# bse-quant

A backtester that blends up to 22 trading ideas on six anonymous assets
(`data/Asset_A.csv` … `Asset_F.csv`, daily prices 2016-10 to 2023-10).
Each idea can be switched on or off and given a weight, all from a single file.

## Quick start

```bash
uv sync            # install everything (first time only)
uv run backtest    # run the backtest in config.toml
```

You'll get a table like this, plus files in `results/`:

```
                         Weight   CAGR Volatility  Sharpe ... Portfolio Sharpe without it
moving_average_crossover   4.8%   7.8%      10.7%    0.76 ...                        0.25
...
PORTFOLIO                100.0%   2.1%       7.3%    0.32 ...
```

- **Each strategy row** shows how that strategy did *on its own*.
- **PORTFOLIO** is the weighted blend of all switched-on strategies.
- **Portfolio Sharpe without it** shows what the portfolio's Sharpe would be with
  that one strategy switched off. If it's *higher* than PORTFOLIO's Sharpe, the
  strategy is dragging the blend down.
- `results/equity_curves.png` charts every strategy and the portfolio;
  `results/daily_returns.csv` has the day-by-day numbers.

## Experimenting

Everything you'd want to change is in **`config.toml`**:

```toml
[strategies.donchian_breakout]
enabled = true      # switch on/off
weight = 2.0        # relative share: twice as much as a weight-1 strategy
entry_days = 55     # the strategy's own settings
exit_days = 20
```

Some things to try:

1. Switch everything off except one strategy, to study it alone.
2. Double one strategy's weight and watch the PORTFOLIO row change.
3. Change `cost_bps` to see which strategies survive realistic trading costs.
4. Set `allow_short = false` to see the long-only version.
5. Keep several experiments side by side: `cp config.toml fast.toml`, edit it, then
   `uv run backtest --config fast.toml`.

Typos are caught with a hint, e.g. `[strategies.donchian_breakout] has no setting 'entry_day'. Did you mean 'entry_days'?`

### The holdout: don't peek

`include_holdout = false` keeps 2022-01-01 onward hidden. Tune on the earlier
years only. When you're done, switch it to `true` **once** to see how the
system does on data it has never seen. If you tune again after looking, the
holdout stops being an honest test.

## How it works

| File | What it does |
|---|---|
| `config.toml` | All the knobs. |
| `src/bse_quant/strategies/sNN_*.py` | One file per idea, numbered like the idea list. Each is a single function whose keyword arguments are the config settings. |
| `src/bse_quant/indicators.py` | Shared building blocks (moving averages, RSI, z-scores, entry/exit logic). |
| `src/bse_quant/engine.py` | Turns positions into daily profit and loss, applies costs, blends the strategies. |
| `src/bse_quant/metrics.py` | Sharpe, drawdown, CAGR and so on. |
| `src/bse_quant/report.py`, `cli.py` | The printed table, CSVs and chart. |
| `tests/` | Unit tests, including a "no peeking at the future" test for every strategy. |

Rules the engine follows:

- **Positions.** Each strategy returns, for every day and asset, a number like
  +1 (fully long that asset's equal slice of the account), 0 (flat) or -1 (fully short).
- **Timing.** A position decided at today's close earns tomorrow's return. Two
  exceptions: gap fade trades open→close the same day, and the overnight
  strategy holds close→next open.
- **Costs.** `cost_bps` is charged on every unit traded. Intraday and overnight
  strategies pay it twice a day.
- **Risk.** With `normalize_strategy_risk = true`, every strategy is first scaled
  to the same volatility, so `weight` means share of risk. Volatility targeting
  (#17) then scales the whole blend. Both only use past data.
- **Blending.** Strategies are blended by their returns. Costs are counted per
  strategy, so trades that cancel out *between* strategies are still charged
  (a conservative choice).

## Adding your own strategy

1. Copy a simple one, e.g. `s02_moving_average_crossover.py`, to a new file.
2. Write a function `def my_idea(data, my_setting=10): ...` that returns
   `Signal(positions)`. `data.close`, `data.open`, `data.high`, `data.low` and
   `data.volume` are tables with dates as rows and assets as columns.
3. Add it to `STRATEGIES` in `src/bse_quant/strategies/__init__.py`.
4. Add a `[strategies.my_idea]` section to `config.toml`.
5. Run `uv run pytest`. The shared tests automatically check your strategy for
   look-ahead bias.

## Tests

```bash
uv run pytest
```

## Competition system (BSE Quant Track)

The competition rules are different from the backtester above: positions are opened at a day's open and closed at a close, after 1 session or after several (up to about a month); everything held must fit in 100% of the account, and each trade pays 5 bps in and 5 bps out. The position book that tracks this is `src/bse_quant/competition/book.py`. That system lives in `src/bse_quant/competition/` and is driven by **`competition.toml`** (features on/off, hurdle, sizing, risk).

| Command | What it does |
|---|---|
| `uv run competition-backtest` | Walk-forward backtest under the competition rules |
| `uv run competition-search` | Backtests every combination in `competition_search.toml` |
| `scripts/competition_search_tmux.sh 0:3 10 3` | Re-runs the search on shuffled histories in tmux (10 workers, 3 hours) |
| `uv run competition-search-report` | p-values: do the settings beat luck? |
| `uv run competition-score` | Estimates each setting's rank-based competition score vs. assumed competitors (`competition_score.toml`) |
| `uv run competition-fit` | Fits the final model(s) into `submission/model.json` |

The program the evaluator runs is `submission/strategy.py`. The write-up, results and reproduction steps are in `submission/README.md`.
