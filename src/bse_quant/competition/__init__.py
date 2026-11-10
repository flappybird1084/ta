"""The trading system for the BSE Quant Track competition.

Competition rules (see competition.toml for the knobs):
- Decide weights after a day's close; they are bought at the next open and
  sold at that day's close. Nothing is held overnight.
- The sum of |weights| across assets can't exceed 1 (100% of equity).
- Every trade pays 5 bps on entry and 5 bps on exit.

Pipeline: features.py (what we know at the close) -> model.py (forecast of
tomorrow's open-to-close move) -> portfolio.py (forecast -> weights, risk
control) -> backtest.py (research) / trader.py (the program the evaluator runs).
"""
