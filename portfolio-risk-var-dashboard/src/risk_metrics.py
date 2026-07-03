"""Core portfolio risk and performance metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def portfolio_returns(returns: pd.DataFrame, weights: pd.Series) -> pd.Series:
    """Calculate weighted portfolio returns."""
    aligned_weights = weights.reindex(returns.columns).fillna(0.0)
    aligned_weights = aligned_weights / aligned_weights.sum()
    return returns.dot(aligned_weights)


def annualized_return(port_returns: pd.Series) -> float:
    return float(port_returns.mean() * TRADING_DAYS)


def annualized_volatility(port_returns: pd.Series) -> float:
    return float(port_returns.std(ddof=1) * np.sqrt(TRADING_DAYS))


def sharpe_ratio(port_returns: pd.Series, risk_free_rate: float = 0.03) -> float:
    vol = annualized_volatility(port_returns)
    if vol == 0:
        return np.nan
    return (annualized_return(port_returns) - risk_free_rate) / vol


def sortino_ratio(port_returns: pd.Series, risk_free_rate: float = 0.03) -> float:
    downside = port_returns[port_returns < 0].std(ddof=1) * np.sqrt(TRADING_DAYS)
    if downside == 0 or np.isnan(downside):
        return np.nan
    return (annualized_return(port_returns) - risk_free_rate) / downside


def max_drawdown(port_returns: pd.Series) -> float:
    cumulative = (1 + port_returns).cumprod()
    drawdown = cumulative / cumulative.cummax() - 1
    return float(drawdown.min())


def cagr(port_returns: pd.Series) -> float:
    cumulative = (1 + port_returns).cumprod()
    years = len(port_returns) / TRADING_DAYS
    if years <= 0:
        return np.nan
    return float(cumulative.iloc[-1] ** (1 / years) - 1)


def beta_to_benchmark(port_returns: pd.Series, benchmark_returns: pd.Series) -> float:
    joined = pd.concat([port_returns, benchmark_returns], axis=1).dropna()
    if joined.shape[0] < 2:
        return np.nan
    cov = joined.iloc[:, 0].cov(joined.iloc[:, 1])
    var = joined.iloc[:, 1].var()
    return float(cov / var) if var != 0 else np.nan


def component_risk_contribution(returns: pd.DataFrame, weights: pd.Series) -> pd.Series:
    """Estimate each asset's percentage contribution to portfolio volatility."""
    w = weights.reindex(returns.columns).fillna(0.0).values
    cov = returns.cov().values * TRADING_DAYS
    portfolio_vol = np.sqrt(w.T @ cov @ w)
    if portfolio_vol == 0:
        return pd.Series(index=returns.columns, data=np.nan)
    marginal_contrib = cov @ w / portfolio_vol
    component_contrib = w * marginal_contrib / portfolio_vol
    return pd.Series(component_contrib, index=returns.columns).sort_values(ascending=False)
