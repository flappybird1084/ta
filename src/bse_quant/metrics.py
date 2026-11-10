"""Performance statistics for a series of daily returns."""

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def equity_curve(returns: pd.Series) -> pd.Series:
    """Growth of $1."""
    return (1 + returns).cumprod()


def total_return(returns: pd.Series) -> float:
    return float(equity_curve(returns).iloc[-1] - 1) if len(returns) else 0.0


def cagr(returns: pd.Series) -> float:
    """Compound annual growth rate."""
    if len(returns) == 0:
        return 0.0
    growth = 1 + total_return(returns)
    return float(growth ** (TRADING_DAYS / len(returns)) - 1) if growth > 0 else -1.0


def annual_volatility(returns: pd.Series) -> float:
    return float(returns.std() * np.sqrt(TRADING_DAYS)) if len(returns) > 1 else 0.0


def sharpe_ratio(returns: pd.Series) -> float:
    """Average return per unit of risk, annualised (risk-free rate taken as 0)."""
    std = returns.std() if len(returns) > 1 else 0.0
    if not std or np.isnan(std):
        return 0.0
    return float(returns.mean() / std * np.sqrt(TRADING_DAYS))


def sortino_ratio(returns: pd.Series) -> float:
    """Like Sharpe, but only losing days count as risk."""
    downside = np.sqrt((returns.clip(upper=0) ** 2).mean()) if len(returns) else 0.0
    if not downside or np.isnan(downside):
        return 0.0
    return float(returns.mean() / downside * np.sqrt(TRADING_DAYS))


def max_drawdown(returns: pd.Series) -> float:
    """Worst peak-to-trough fall, e.g. -0.25 = -25%."""
    if len(returns) == 0:
        return 0.0
    equity = equity_curve(returns)
    peak = equity.cummax().clip(lower=1.0)  # starting capital counts as a peak
    return float((equity / peak - 1).min())


def hit_rate(returns: pd.Series) -> float:
    """Share of trading days (non-zero return) that made money."""
    active = returns[returns != 0]
    return float((active > 0).mean()) if len(active) else 0.0


def time_in_market(returns: pd.Series) -> float:
    return float((returns != 0).mean()) if len(returns) else 0.0


def summary(returns: pd.Series) -> dict:
    mdd = max_drawdown(returns)
    growth = cagr(returns)
    return {
        "CAGR": growth,
        "Volatility": annual_volatility(returns),
        "Sharpe": sharpe_ratio(returns),
        "Sortino": sortino_ratio(returns),
        "Max drawdown": mdd,
        "Calmar": growth / abs(mdd) if mdd < 0 else 0.0,
        "Hit rate": hit_rate(returns),
        "Time in market": time_in_market(returns),
    }
