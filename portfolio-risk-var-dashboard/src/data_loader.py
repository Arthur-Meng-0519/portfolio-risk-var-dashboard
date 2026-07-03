"""Data loading utilities for the Portfolio Risk & VaR Dashboard."""

from __future__ import annotations

import pandas as pd
import yfinance as yf


def clean_tickers(tickers: str) -> list[str]:
    """Convert a comma-separated ticker string into a clean uppercase list."""
    return [ticker.strip().upper() for ticker in tickers.split(",") if ticker.strip()]


def load_prices(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    """Download adjusted close prices from Yahoo Finance."""
    if not tickers:
        raise ValueError("Please provide at least one ticker.")

    raw = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)

    if raw.empty:
        raise ValueError("No price data returned. Check tickers and dates.")

    prices = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw[["Close"]]

    if len(tickers) == 1:
        prices = prices.rename(columns={"Close": tickers[0]})

    return prices.dropna(how="all").ffill().dropna()


def calculate_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Calculate simple daily returns from price data."""
    return prices.pct_change().dropna()
