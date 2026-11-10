"""Our own estimate of the competition score.

The judges rank every eligible team on each metric. The best team gets the
metric's full points, the worst gets zero, and the rest are spread linearly by
rank (ties share their average rank):

    points = full points * (n_teams - rank) / (n_teams - 1)

So what matters is beating the other teams, not the size of the win. That
changes which settings are best: e.g. a tiny, positive, low-drawdown strategy
can out-rank a bigger, better one on CAGR and drawdown if most teams lose.

`simulate_scores` estimates our expected score against a field of competitor
strategies over many random 3-year "trials" built from blocks of real days.
"""

import numpy as np
import pandas as pd

TRADING_DAYS = 252
POINTS = {"Sharpe": 50, "CAGR": 30, "Max drawdown": 20}  # all three: higher (less negative) is better
MIN_ACTIVE_DAYS = 30


def competition_points(table: pd.DataFrame) -> pd.DataFrame:
    """Points per team (rows) for the columns Sharpe, CAGR and Max drawdown, plus Total."""
    n = len(table)
    out = pd.DataFrame(index=table.index)
    for metric, full in POINTS.items():
        rank = table[metric].rank(ascending=False, method="average")
        out[metric] = full * (n - rank) / (n - 1) if n > 1 else float(full)
    out["Total"] = out.sum(axis=1)
    return out


def metrics_matrix(returns: np.ndarray) -> dict:
    """Judged metrics for every column of a (days x strategies) array of daily returns."""
    r = np.asarray(returns, dtype=float)
    n = r.shape[0]
    std = r.std(axis=0, ddof=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        sharpe = np.where(std > 0, r.mean(axis=0) / std * np.sqrt(TRADING_DAYS), 0.0)
    equity = np.cumprod(1 + r, axis=0)
    growth = equity[-1]
    with np.errstate(invalid="ignore"):
        cagr = np.where(growth > 0, np.abs(growth) ** (TRADING_DAYS / n) - 1, -1.0)
    peak = np.maximum(np.maximum.accumulate(equity, axis=0), 1.0)
    mdd = (equity / peak - 1).min(axis=0)
    return {
        "Sharpe": sharpe,
        "CAGR": cagr,
        "Max drawdown": mdd,
        "Active days": (r != 0).sum(axis=0),
        "Min equity": equity.min(axis=0),
    }


def block_bootstrap(n_days: int, length: int, block: int, rng) -> np.ndarray:
    """Row positions for one trial: random runs of `block` consecutive days, glued together."""
    starts = rng.integers(0, n_days - block + 1, size=-(-length // block))
    return (starts[:, None] + np.arange(block)).ravel()[:length]


def _eligible(m: dict, min_active=MIN_ACTIVE_DAYS) -> np.ndarray:
    return (m["Active days"] >= min_active) & (m["Min equity"] > 0)


def _rank_points(ours: np.ndarray, field: np.ndarray, full: float):
    """Points for each candidate (vs. the field) and for each field team (with that candidate added)."""
    k = len(field)
    n = k + 1
    better = (field[None, :] > ours[:, None]).sum(axis=1)
    ties = (field[None, :] == ours[:, None]).sum(axis=1)
    our_rank = 1 + better + 0.5 * ties
    field_rank_alone = pd.Series(field).rank(ascending=False, method="average").to_numpy()
    field_rank = field_rank_alone[None, :] + (ours[:, None] > field[None, :]) + 0.5 * (ours[:, None] == field[None, :])
    scale = full / (n - 1) if n > 1 else 0.0
    return (n - our_rank) * scale, (n - field_rank) * scale


def simulate_scores(
    ours: pd.DataFrame, field: pd.DataFrame, n_trials=2000, length=756, block=21, seed=0, our_min_active=MIN_ACTIVE_DAYS
) -> pd.DataFrame:
    """Expected competition score of each of OUR candidates (columns of `ours`), one at a time, vs. `field`.

    Both are daily returns on the same days. Each trial samples the same days for everyone.
    A candidate that isn't eligible in a trial (< 30 trading days, or equity hits zero) scores 0.
    `our_min_active` can demand more trading days from our candidates, as a safety margin.
    """
    days = ours.index.intersection(field.index)
    O, F = ours.loc[days].to_numpy(), field.loc[days].to_numpy()
    rng = np.random.default_rng(seed)
    m_c = O.shape[1]
    totals = np.zeros((n_trials, m_c))
    parts = {k: np.zeros((n_trials, m_c)) for k in POINTS}
    stats = {k: np.zeros((n_trials, m_c)) for k in POINTS}
    wins = np.zeros((n_trials, m_c))
    for t in range(n_trials):
        idx = block_bootstrap(len(days), min(length, len(days)), block, rng)
        mo, mf = metrics_matrix(O[idx]), metrics_matrix(F[idx])
        keep = _eligible(mf)
        ok = _eligible(mo, our_min_active)
        field_total = np.zeros((m_c, keep.sum()))
        for metric, full in POINTS.items():
            ours_pts, field_pts = _rank_points(mo[metric], mf[metric][keep], full)
            ours_pts = np.where(ok, ours_pts, 0.0)
            parts[metric][t] = ours_pts
            totals[t] += ours_pts
            field_total += field_pts
            stats[metric][t] = mo[metric]
        wins[t] = ok & (totals[t] >= field_total.max(axis=1, initial=-np.inf))
    out = pd.DataFrame(index=ours.columns)
    out["Score"] = totals.mean(axis=0)
    out["Score p10"] = np.percentile(totals, 10, axis=0)
    out["Score p90"] = np.percentile(totals, 90, axis=0)
    for metric in POINTS:
        out[f"{metric} pts"] = parts[metric].mean(axis=0)
    out["Win rate"] = wins.mean(axis=0)
    for metric in POINTS:
        out[f"Median {metric}"] = np.median(stats[metric], axis=0)
    return out.sort_values("Score", ascending=False)
