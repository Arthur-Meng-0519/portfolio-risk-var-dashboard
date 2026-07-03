from __future__ import annotations

import pandas as pd
import streamlit as st

from src.data_loader import calculate_returns, clean_tickers, load_prices
from src.plots import bar_chart, correlation_heatmap, cumulative_return_chart, drawdown_chart, return_histogram
from src.risk_metrics import (
    annualized_return,
    annualized_volatility,
    beta_to_benchmark,
    cagr,
    component_risk_contribution,
    max_drawdown,
    portfolio_returns,
    sharpe_ratio,
    sortino_ratio,
)
from src.stress_tests import apply_stress_tests, default_scenarios
from src.var_models import (
    historical_es,
    historical_var,
    monte_carlo_var_es,
    parametric_es,
    parametric_var,
    rolling_var,
    var_backtest,
)

st.set_page_config(page_title="Portfolio Risk & VaR Dashboard", layout="wide")

st.title("Portfolio Risk & VaR Dashboard")
st.caption("Market risk analytics with VaR, Expected Shortfall, stress testing, drawdowns, and risk contribution.")

with st.sidebar:
    st.header("Portfolio Inputs")
    ticker_text = st.text_input("Tickers", "AAPL,MSFT,NVDA,JPM,SPY")
    benchmark = st.text_input("Benchmark ticker", "SPY")
    start = st.date_input("Start date", pd.to_datetime("2020-01-01"))
    end = st.date_input("End date", pd.Timestamp.today())
    confidence = st.slider("VaR / ES confidence", 0.90, 0.99, 0.95, 0.01)
    risk_free_rate = st.number_input("Risk-free rate", min_value=0.0, max_value=0.20, value=0.03, step=0.005)
    simulations = st.number_input("Monte Carlo simulations", min_value=1000, max_value=100000, value=20000, step=1000)

try:
    tickers = clean_tickers(ticker_text)
    prices = load_prices(tickers, str(start), str(end))
    returns = calculate_returns(prices)

    st.sidebar.subheader("Weights")
    raw_weights = []
    for ticker in tickers:
        raw_weights.append(st.sidebar.number_input(f"{ticker}", min_value=0.0, max_value=1.0, value=1 / len(tickers), step=0.01))
    weights = pd.Series(raw_weights, index=tickers)
    weights = weights / weights.sum()

    port_ret = portfolio_returns(returns, weights)

    bench_beta = None
    if benchmark.strip().upper() not in tickers:
        benchmark_prices = load_prices([benchmark.strip().upper()], str(start), str(end))
        benchmark_returns = calculate_returns(benchmark_prices).iloc[:, 0]
    else:
        benchmark_returns = returns[benchmark.strip().upper()]
    bench_beta = beta_to_benchmark(port_ret, benchmark_returns)

    metric_cols = st.columns(6)
    metric_cols[0].metric("Annual Return", f"{annualized_return(port_ret):.2%}")
    metric_cols[1].metric("CAGR", f"{cagr(port_ret):.2%}")
    metric_cols[2].metric("Annual Volatility", f"{annualized_volatility(port_ret):.2%}")
    metric_cols[3].metric("Sharpe", f"{sharpe_ratio(port_ret, risk_free_rate):.2f}")
    metric_cols[4].metric("Sortino", f"{sortino_ratio(port_ret, risk_free_rate):.2f}")
    metric_cols[5].metric("Max Drawdown", f"{max_drawdown(port_ret):.2%}")

    st.metric(f"Beta vs {benchmark.strip().upper()}", f"{bench_beta:.2f}" if pd.notna(bench_beta) else "N/A")

    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Performance", "VaR & ES", "Backtesting", "Risk Budget", "Stress Tests"])

    with tab1:
        c1, c2 = st.columns(2)
        c1.plotly_chart(cumulative_return_chart(port_ret), use_container_width=True)
        c2.plotly_chart(drawdown_chart(port_ret), use_container_width=True)
        c3, c4 = st.columns(2)
        c3.plotly_chart(return_histogram(port_ret), use_container_width=True)
        c4.plotly_chart(correlation_heatmap(returns), use_container_width=True)

    with tab2:
        hist_var = historical_var(port_ret, confidence)
        hist_es = historical_es(port_ret, confidence)
        para_var = parametric_var(port_ret, confidence)
        para_es = parametric_es(port_ret, confidence)
        mc_var, mc_es = monte_carlo_var_es(port_ret, confidence, int(simulations))

        risk_table = pd.DataFrame(
            {
                "VaR": [hist_var, para_var, mc_var],
                "Expected Shortfall": [hist_es, para_es, mc_es],
            },
            index=["Historical", "Parametric Normal", "Monte Carlo Normal"],
        )
        st.dataframe(risk_table.style.format("{:.2%}"), use_container_width=True)
        st.plotly_chart(bar_chart(risk_table["VaR"], "VaR Model Comparison"), use_container_width=True)
        st.info("VaR estimates a loss threshold. Expected Shortfall estimates average loss after crossing that threshold, making it more focused on tail risk.")

    with tab3:
        rv = rolling_var(port_ret, window=252, confidence=confidence)
        backtest = var_backtest(port_ret, rv)
        st.write(backtest)
        bt_df = pd.concat([port_ret.rename("Portfolio Returns"), rv.rename("Rolling Historical VaR")], axis=1).dropna()
        st.line_chart(bt_df)
        st.caption("Breaches occur when actual daily return is worse than the estimated rolling VaR.")

    with tab4:
        rc = component_risk_contribution(returns, weights)
        st.dataframe(rc.rename("Risk Contribution").to_frame().style.format("{:.2%}"), use_container_width=True)
        st.plotly_chart(bar_chart(rc, "Contribution to Portfolio Volatility"), use_container_width=True)

    with tab5:
        scenarios = default_scenarios(tickers)
        st.write("Editable scenario shocks")
        edited = st.data_editor(scenarios, use_container_width=True)
        scenario_losses = apply_stress_tests(weights, edited)
        st.dataframe(scenario_losses.rename("Portfolio Shock").to_frame().style.format("{:.2%}"), use_container_width=True)
        st.plotly_chart(bar_chart(scenario_losses, "Scenario Stress Test Results"), use_container_width=True)

except Exception as exc:
    st.error(str(exc))
