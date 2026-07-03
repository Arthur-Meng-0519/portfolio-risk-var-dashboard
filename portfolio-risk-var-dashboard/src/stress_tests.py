"""Scenario and stress-testing helpers."""

from __future__ import annotations

import pandas as pd


def default_scenarios(tickers: list[str]) -> pd.DataFrame:
    """Create simple asset-level scenario shocks for demonstration."""
    scenarios = {
        "2008-style equity crash": -0.35,
        "COVID-style sudden selloff": -0.25,
        "High-rate risk-off shock": -0.12,
        "Tech concentration selloff": -0.20,
        "Mild recession": -0.10,
        "Climate transition shock": -0.08,
        "Liquidity squeeze": -0.15,
    }
    return pd.DataFrame({name: [shock] * len(tickers) for name, shock in scenarios.items()}, index=tickers).T


def apply_stress_tests(weights: pd.Series, scenario_table: pd.DataFrame) -> pd.Series:
    aligned = weights.reindex(scenario_table.columns).fillna(0.0)
    aligned = aligned / aligned.sum()
    return scenario_table.dot(aligned)
