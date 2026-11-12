# BSE Quant Track submission

## 1. Trading program

`submission/strategy.py` implements the evaluator interface:

```python
import strategy
strategy.initialize(history="data")          # optional: warm up on the public CSVs
weights = strategy.on_day_close(daily_data, portfolio_state)
# -> {"Asset_A": 0.21, "Asset_B": 0.0, ..., "Asset_G": -0.15}, sum(|w|) <= 1
strategy.last_orders()   # -> {1: {...}, 5: {...}}: new buys for the next open, by sessions held
strategy.positions()     # -> [{"asset", "weight", "hold_days", "sessions_left"}, ...]
strategy.equity()        # -> account value from our own trades (starts at 1.0)
```

- `daily_data` is the day that just closed, one entry per asset: `{"Asset_A": {"Date", "Open", "High", "Low", "Close", "Volume"}, ...}`. A DataFrame with one row per asset also works. Field names are case-insensitive, and prices like `"$28.43"` are accepted.
- The returned weights are what we will hold during the **next** session: positions carried over plus new buys. Only data up to the reported close is used.
- **Holding periods.** Every position is bought at an open and held for a fixed number of sessions, then sold at a close: **1 session** for the reversal sleeves and **5 sessions** for the trend holds. Nothing is held longer than 5 sessions, well inside the one-month limit. `last_orders()` gives the new buys and how long each is held.
- **Capital.** The program keeps its own books (`book.Book`): every open position and how many sessions it has left. A new buy only gets capital that is actually free, so open positions plus new buys never exceed 100% of equity.
- An asset the program has never seen (for example Asset_G) is handled. It gets weight 0 until there are about 20 days of its history.
- A day that is already in the history (if the evaluator replays the public data) updates that day in place instead of being appended twice.
- **PnL.** Equity is tracked from the program's own positions (overnight gaps of held positions included, costs on entry and exit), so `portfolio_state` is accepted but not required. A test checks that this equity matches an independent evaluator simulation.

Files: `model.json` holds the fitted coefficients of both sleeves, and `settings.toml` the sleeves, trend, sizing and risk settings (a copy of `competition.toml`). The code lives in `src/bse_quant/competition/`.

## 2. Strategy explanation

**Hypothesis.**
- Over a single session, large moves tend to partly reverse the next day, both for the market as a whole and for individual stocks.
- A stock's slower trend, especially its recent *intraday* drift, tends to persist.
- These effects are small next to the ~10 bps cost of each traded day, so the system trades in size only when a forecast clearly beats that cost.
- Over days to weeks, assets that have been rising tend to keep rising. Holding for several sessions pays the ~10 bps once instead of every day, so a slow trend is worth trading.

**Signals.** There are two one-session "sleeves" (42.5% of the account each) and a multi-day trend part (15%). Each sleeve is a ridge regression, pooled across assets, of the next session's open-to-close return on features known at the close. All features are scaled by volatility, so one model fits every asset.

1. **Market reversal:** one feature, the average asset's intraday move today. A big market-wide sell-off predicts a rebound the next session, and a big rally a give-back. Ridge 0.25; trades when |forecast| > 25 bps.
2. **Reversal + momentum:** nine features, covering today's return, intraday move and opening gap, 2-day RSI, where the close sits in the day's range, the market's intraday move, 3-month momentum (absolute and relative to the other assets), and the 3-month average intraday move. Ridge 0.05; trades when |forecast| > 35 bps.
3. **Trend holds (multi-day):** every day, a fifth of the trend share buys, equally weighted, the assets whose last 20 days' return is positive, and holds them for 5 sessions. There are always five overlapping cohorts, so the 15% is spread over the week. Holding costs nothing between entry and exit, and the position earns the overnight moves the one-session sleeves never see. Over 2018–2023 this part alone had a Sharpe of about 1.0, and its returns are almost uncorrelated with the sleeves (0.11), which is why it raises the blend's Sharpe.

**Allocation and risk.**
- When a sleeve trades an asset, it sizes the position for equal risk: weight = 1% ÷ the asset's recent intraday volatility, capped at 50%.
- **Activity floor:** on days a sleeve has no trade, it puts a tiny position (1% or 0.5%) in the direction of its strongest forecast. The system therefore trades every day at negligible cost, which keeps it safely above the 30-trading-day eligibility rule.
- The sleeves' weights are added together with the trend holds. The shares sum to 1, and the book scales down new buys if positions carried over have grown. So gross exposure never exceeds 100%; it averages about 18%.
- The one-session sleeves trade long and short; the trend holds are long only.

**Why small and selective: designing for a rank-based score.**
- The score ranks teams on Sharpe (50 pts), CAGR (30) and max drawdown (20). Sharpe doesn't change when positions are scaled, but drawdown does. So the cheapest way to gain points is high Sharpe with small positions.
- We estimated our expected score with `uv run competition-score`. Each candidate setting is scored in thousands of random 3-year "trials", each built from 21-day blocks of real days.
- Candidates are scored against five assumed fields of other teams:
  - the 20 classic strategies (momentum, moving averages, RSI, pairs, ...) translated to these rules, plus always-long entries;
  - that field plus four teams running models like ours;
  - a cautious version that cuts its exposure;
  - a strong version with three teams at Sharpe ≈ 1.
  - since the organisers allowed multi-day holds: the naive and competent teams plus six that buy and hold or follow trends for days (each held up to 21 sessions, costs only on entry and exit).
- Settings chosen on 2018–21 trials scored about 66–79 when re-scored on 2022–23 trials, against 53–70 for our first design.
- Multi-day teams are the main threat: buy-and-hold alone earned about 19% a year here. They take CAGR points, but their drawdowns (25–45%) give us the drawdown points. Adding the 15% trend part raised our expected score in every field and both periods: averaged over the five fields, from 93.4 to 94.8 (2018–21) and from 85.4 to 87.4 (2022–23). Against the multi-day field alone it went from 88.0 to 91.3 and from 85.3 to 88.5.
- The organiser's own naive long-only example (Sharpe 0.35, +26.6% over the development data) is within the range these fields cover.

**Assets targeted.** All of them, through shared models with no asset-specific rules, so unseen assets are treated the same way.

## 3. Development results and validation

**How development was separated from validation.**
- 2016-10 to 2016-12 is warm-up only.
- Research and the first round of settings used walk-forward results over 2018–2021: the model was re-fit every 252 trading days on the history up to that point.
- 2022-01 to 2023-10 was then used once to check the first design (Sharpe 0.20). That gap is what led to the score-based redesign.
- The redesign's scoring trials also covered 2022–23, so **2022–23 is no longer a clean out-of-sample test for the final blend**. The fairest out-of-sample evidence is the check that chose settings using 2018–21 only and scored them on 2022–23.
- The final `model.json` was fit on 2017-01 to 2023-10-06.

**Shuffled-history checks.**
- Single-model settings were re-run on thousands of histories with the days shuffled, which destroys any day-to-day pattern.
- Many settings beat their own shuffled results (p ≈ 0.002–0.05). The reversal + momentum family beat shuffles as a group (p ≈ 0.02).
- But the best of 160 settings on a shuffled history matched the real best about 12% of the time. The edge is real-looking but modest.

**Results under the competition rules** (open-to-close, 5 + 5 bps, gross ≤ 100%):

| System, period | Sharpe | CAGR | Max drawdown | Volatility | Days traded |
|---|---|---|---|---|---|
| **Final system** (sleeves + trend holds), walk-forward 2018–2021 | 1.70 | 11.7% | −6.5% | 6.6% | 1005 / 1008 |
| **Final system**, 2022-01 to 2023-10, walk-forward | 0.87 | 5.7% | −7.3% | 6.7% | 443 / 443 |
| **Final system**, 2022-01 to 2023-10, model frozen at 2021-12-31 | 0.99 | 6.7% | −7.4% | 6.8% | 443 / 443 |
| Sleeves only (previous submission), walk-forward 2018–2021 | 1.36 | 8.2% | −5.2% | 5.9% | 1005 / 1008 |
| Sleeves only, 2022-01 to 2023-10, walk-forward | 0.65 | 4.0% | −4.2% | 6.4% | 443 / 443 |
| First design (one model, 1% risk), 2018–2021 | 0.84 | 11.3% | −11.7% | 13.9% | 364 / 1008 |
| First design, 2022-01 to 2023-10 (frozen) | 0.20 | 2.0% | −13.3% | 15.9% | 209 / 443 |
| Long every asset every session, 2022-01 to 2023-10 | −0.40 | −13.5% | −37.3% | 27.0% | 443 / 443 |

Final system by year (walk-forward Sharpe): 2018 0.25, 2019 1.96, 2020 1.95, 2021 2.59, 2022 0.43, 2023 2.04. The trend holds hurt in 2018 (the sleeves alone had 1.39 that year), when the late-year sell-off came after a rising market. Starting from $10,000 in January 2018, the final system ends at about $17,160 after costs, against $14,690 for the sleeves alone.

**Assumptions.**
- Fills happen exactly at the open and close prices, with costs as specified.
- A position held for several sessions pays 5 bps when it is bought and 5 bps when it is sold, and nothing on the days in between. If held positions were charged every day, the trend holds would cost about 3.8% of the account a year and should be switched off (`[trend] weight = 0`).
- Dividends, borrow and financing are ignored, as in the rules.
- Days with missing data for any asset are dropped.

## 4. Limitations and failure conditions

- **The edge is weak.**
  - Each individual signal is close to the noise level, and the shuffled-history tests support an edge only modestly.
  - Small positions keep drawdowns low, but they also cap CAGR at a few percent a year. A field with several high-CAGR teams would cost us CAGR points.
- **The score estimates depend on assumptions.** Which settings score best depends on the assumed competitors, and the trials reuse the same 6 years of data, so the estimates are optimistic. The relative conclusions held across all four assumed fields: small, selective positions plus the activity floor beat a bigger single model.
- **The trend holds were added late.** Their settings (15%, 20 days, 5 sessions) were picked with trials that included 2022–23, so that period is not a clean test for them; they were near the top in 2018–21 too, and neighbouring settings scored about the same. They are long only, so a falling market costs them, and a sharp turn after a rally (as in 2018) catches them on the wrong side.
- **Reversal can fail.**
  - A sell-off that keeps going the next day hurts the reversal sleeves.
  - A calm, steadily trending market gives few signals; the floor keeps the system eligible, but returns would be small.
- **The data differs from the spec.** The guide mentions seven assets (A–G), but the development data has six (A–F). The program handles any set of assets.
- **The interface may change.** The evaluator interface is provisional; `parse_daily_data` accepts the common shapes.

## 5. Reproducibility

Requires Python ≥ 3.12, numpy and pandas (JAX is optional and not used here).

```bash
uv sync --no-group gpu                 # or: pip install -e .
uv run competition-backtest            # development results (walk-forward 2018-2021)
uv run competition-score               # expected-score search vs. assumed competitors (~2 min)
uv run competition-fit --end 2023-10-06   # re-creates submission/model.json from the public data
uv run pytest                          # includes a check that the program matches the backtest
```
