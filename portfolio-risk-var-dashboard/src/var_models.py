"""VaR, Expected Shortfall, and VaR backtesting models."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm


def historical_var(returns: pd.Series, confidence: float = 0.95) -> float:
    return float(np.percentile(returns, (1 - confidence) * 100))


def historical_es(returns: pd.Series, confidence: float = 0.95) -> float:
    var = historical_var(returns, confidence)
    tail = returns[returns <= var]
    return float(tail.mean()) if len(tail) else np.nan


def parametric_var(returns: pd.Series, confidence: float = 0.95) -> float:
    mu, sigma = returns.mean(), returns.std(ddof=1)
    return float(mu + sigma * norm.ppf(1 - confidence))


def parametric_es(returns: pd.Series, confidence: float = 0.95) -> float:
    mu, sigma = returns.mean(), returns.std(ddof=1)
    alpha = 1 - confidence
    return float(mu - sigma * norm.pdf(norm.ppf(alpha)) / alpha)


def monte_carlo_var_es(
    returns: pd.Series,
    confidence: float = 0.95,
    simulations: int = 20000,
    seed: int = 42,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    simulated = rng.normal(returns.mean(), returns.std(ddof=1), simulations)
    var = np.percentile(simulated, (1 - confidence) * 100)
    es = simulated[simulated <= var].mean()
    return float(var), float(es)


def rolling_var(returns: pd.Series, window: int = 252, confidence: float = 0.95) -> pd.Series:
    return returns.rolling(window).quantile(1 - confidence)


def var_backtest(returns: pd.Series, var_series: pd.Series) -> dict[str, float]:
    joined = pd.concat([returns.rename("returns"), var_series.rename("var")], axis=1).dropna()
    breaches = joined["returns"] < joined["var"]
    return {
        "observations": int(len(joined)),
        "breaches": int(breaches.sum()),
        "breach_rate": float(breaches.mean()) if len(joined) else np.nan,
    }
