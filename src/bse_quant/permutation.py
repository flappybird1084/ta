"""Permutation testing: is a strategy's result better than luck?

We shuffle the order of the daily bars (after the warm-up, before the holdout),
which keeps every day's moves but destroys any pattern over time, then re-run
every strategy on the shuffled history. If a strategy does about as well on
shuffled data as on the real thing, its edge is probably just luck.

How bars are shuffled (Masters' method): each bar is split into its overnight
gap (open vs. previous close) and its intraday shape (high, low, close and
volume relative to the open). Gaps and intraday shapes are shuffled
separately, with the same shuffle for every asset so that assets still move
together on the same days. Each asset ends the period at the same price.

p-value = (1 + number of shuffles scoring at least as well) / (1 + number of shuffles).
"""

import time
from pathlib import Path

import numpy as np
import pandas as pd

from bse_quant import metrics
from bse_quant.data import MarketData
from bse_quant.engine import run_backtest

FIELDS = ["open", "high", "low", "close", "volume"]


def permute_bars(data: MarketData, start, end, seed) -> MarketData:
    """Shuffle the bars dated in [start, end); drop everything from `end` on."""
    keep = data.dates < pd.Timestamp(end)
    o, h, l, c, v = (getattr(data, f)[keep] for f in FIELDS)
    shuffled = np.flatnonzero(o.index >= pd.Timestamp(start))
    first = shuffled[0]
    if first == 0:
        raise ValueError("Need at least one bar before the shuffled period to start from")

    log_o, log_h, log_l, log_c = (np.log(df.to_numpy()) for df in (o, h, l, c))
    gap = log_o[first:] - log_c[first - 1 : -1]
    intraday = [log_h[first:] - log_o[first:], log_l[first:] - log_o[first:], log_c[first:] - log_o[first:]]
    volume = v.to_numpy()[first:]

    rng = np.random.default_rng(seed)
    gap = gap[rng.permutation(len(gap))]
    order = rng.permutation(len(gap))
    high_rel, low_rel, close_rel = (piece[order] for piece in intraday)
    volume = volume[order]

    closes = log_c[first - 1] + np.cumsum(gap + close_rel, axis=0)
    prev_closes = np.vstack([log_c[first - 1 : first], closes[:-1]])
    opens = prev_closes + gap

    def rebuild(original, new_tail):
        out = original.copy()
        out.iloc[first:] = new_tail
        return out

    return MarketData(
        open=rebuild(o, np.exp(opens)),
        high=rebuild(h, np.exp(opens + high_rel)),
        low=rebuild(l, np.exp(opens + low_rel)),
        close=rebuild(c, np.exp(closes)),
        volume=rebuild(v, volume),
    )


def run_permutation(data: MarketData, config, seed) -> dict:
    """Sharpe ratio of every enabled strategy and the PORTFOLIO. seed=None means the real, unshuffled data."""
    bt = config.backtest
    if seed is None:
        tested = MarketData(*(getattr(data, f)[data.dates < pd.Timestamp(bt.holdout_start)] for f in FIELDS))
    else:
        tested = permute_bars(data, bt.start_date, bt.holdout_start, seed)
    result = run_backtest(tested, config)
    scores = {name: metrics.sharpe_ratio(result.strategy_returns[name]) for name in result.strategy_returns}
    scores["PORTFOLIO"] = metrics.sharpe_ratio(result.portfolio_returns)
    return scores


def pvalues(real: pd.Series, perms: pd.DataFrame) -> pd.Series:
    """For each strategy, how often a shuffle scored at least as well as reality."""
    n = len(perms)
    return pd.Series({name: (1 + (perms[name] >= real[name]).sum()) / (n + 1) for name in real.index})


def familywise_pvalues(real: pd.Series, perms: pd.DataFrame) -> pd.Series:
    """Like pvalues, but compared with the *best* strategy in each shuffle.

    This corrects for having tried many strategies: with 21 of them, one is
    likely to look good on shuffled data purely by chance.
    """
    strategies = [name for name in real.index if name != "PORTFOLIO"]
    best = perms[strategies].max(axis=1)
    n = len(perms)
    return pd.Series({name: (1 + (best >= real[name]).sum()) / (n + 1) for name in strategies})


def worker_file(out_dir: Path, worker_id: int) -> Path:
    return Path(out_dir) / f"worker_{worker_id:02d}.csv"


def run_worker(data, config, out_dir, worker_id, max_permutations=None, deadline=None, log=print):
    """Run shuffles until `max_permutations` or the `deadline` (a time.time() value), appending to a CSV.

    Safe to stop and restart: it carries on from where it left off.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    real_path = out_dir / "real.csv"
    if not real_path.exists():
        real = pd.Series(run_permutation(data, config, seed=None), name="sharpe")
        tmp = out_dir / f".real_{worker_id}.tmp"
        real.to_csv(tmp, index_label="strategy")
        tmp.replace(real_path)

    path = worker_file(out_dir, worker_id)
    done = len(pd.read_csv(path)) if path.exists() else 0
    started = time.time()
    k = done
    while True:
        if deadline is not None and time.time() >= deadline:
            break
        if max_permutations is not None and k >= max_permutations:
            break
        seed = worker_id * 1_000_000 + k
        t0 = time.time()
        row = pd.DataFrame([{"seed": seed, **run_permutation(data, config, seed)}])
        row.to_csv(path, mode="a", header=not path.exists(), index=False)
        k += 1
        if log:
            rate = (k - done) / (time.time() - started)
            left = "" if deadline is None else f" | {(deadline - time.time()) / 3600:5.2f}h left"
            log(f"[worker {worker_id:02d}] shuffle {k:6d} | {time.time() - t0:5.2f}s | {rate * 3600:6.0f}/h{left}")
    return k - done


def load_permutations(out_dir):
    """(real Sharpe ratios, one row per shuffle from every worker)."""
    out_dir = Path(out_dir)
    real = pd.read_csv(out_dir / "real.csv", index_col="strategy")["sharpe"]
    frames = []
    for f in sorted(out_dir.glob("worker_*.csv")):
        try:
            frames.append(pd.read_csv(f, on_bad_lines="skip"))
        except pd.errors.EmptyDataError:
            continue
    perms = pd.concat(frames, ignore_index=True).dropna() if frames else pd.DataFrame(columns=["seed", *real.index])
    return real, perms


def permutation_table(real: pd.Series, perms: pd.DataFrame) -> pd.DataFrame:
    """One row per strategy (and PORTFOLIO): real Sharpe vs. the shuffled ones."""
    fw = familywise_pvalues(real, perms)
    rows = {}
    for name in real.index:
        shuffled = perms[name]
        rows[name] = {
            "Real Sharpe": real[name],
            "Shuffled mean": shuffled.mean(),
            "Shuffled 95th pct": shuffled.quantile(0.95),
            "p-value": pvalues(real[[name]], perms)[name],
            "Family-wise p": fw.get(name, np.nan),
        }
    return pd.DataFrame(rows).T
