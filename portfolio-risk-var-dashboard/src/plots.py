"""Plotly visualization helpers."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def cumulative_return_chart(port_returns: pd.Series):
    cumulative = (1 + port_returns).cumprod() - 1
    return px.line(cumulative, title="Portfolio Cumulative Return", labels={"value": "Cumulative Return", "index": "Date"})


def drawdown_chart(port_returns: pd.Series):
    cumulative = (1 + port_returns).cumprod()
    drawdown = cumulative / cumulative.cummax() - 1
    return px.area(drawdown, title="Portfolio Drawdown", labels={"value": "Drawdown", "index": "Date"})


def return_histogram(port_returns: pd.Series):
    return px.histogram(port_returns, nbins=60, title="Daily Return Distribution", labels={"value": "Daily Return"})


def correlation_heatmap(returns: pd.DataFrame):
    corr = returns.corr()
    fig = go.Figure(data=go.Heatmap(z=corr.values, x=corr.columns, y=corr.index, zmin=-1, zmax=1))
    fig.update_layout(title="Asset Correlation Matrix")
    return fig


def bar_chart(series: pd.Series, title: str):
    return px.bar(series, title=title, labels={"value": "Value", "index": "Category"})
