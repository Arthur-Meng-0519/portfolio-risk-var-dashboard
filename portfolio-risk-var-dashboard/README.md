# Portfolio Risk & VaR Dashboard

An interactive Python dashboard for portfolio market risk analysis. The app combines classic Value at Risk with more current risk-management features such as Expected Shortfall, VaR backtesting, risk-budget contribution, stress testing, liquidity-style shocks, and climate-transition-style shocks.

## Why This Project Is Industry-Relevant in 2026

Modern market risk teams do not only report VaR. They increasingly focus on tail risk, scenario analysis, explainable risk drivers, and regulatory-style Expected Shortfall thinking. This project is designed to look more realistic than a basic school VaR calculator.

## Live Demo
**https://portfolio-risk-var-dashboard-arthurmeng0519.streamlit.app/** 

## Features

- Download live historical price data using `yfinance`
- User-defined tickers, dates, benchmark, and weights
- Portfolio return, CAGR, volatility, Sharpe ratio, Sortino ratio, beta, and max drawdown
- Historical VaR
- Parametric normal VaR
- Monte Carlo VaR
- Historical, parametric, and Monte Carlo Expected Shortfall
- Rolling VaR backtesting and breach-rate tracking
- Correlation heatmap
- Return distribution histogram
- Cumulative return chart
- Drawdown chart
- Contribution-to-risk / risk-budget analysis
- Editable stress-test table
- Example stress scenarios:
  - 2008-style equity crash
  - COVID-style sudden selloff
  - high-rate risk-off shock
  - tech concentration selloff
  - mild recession
  - climate transition shock
  - liquidity squeeze

## Tech Stack

- Python
- Streamlit
- pandas
- NumPy
- SciPy
- Plotly
- yfinance
- scikit-learn
- arch

## Project Structure

```text
portfolio-risk-var-dashboard/
├── app.py
├── requirements.txt
├── README.md
├── data/
│   └── .gitkeep
├── notebooks/
│   └── prototype.ipynb
└── src/
    ├── data_loader.py
    ├── plots.py
    ├── risk_metrics.py
    ├── stress_tests.py
    └── var_models.py
```

## How to Run Locally

1. Clone or download this repository.
2. Open a terminal in the project folder.
3. Create and activate a virtual environment.

```bash
python -m venv .venv
source .venv/bin/activate   # Mac/Linux
.venv\Scripts\activate      # Windows
```

4. Install packages.

```bash
pip install -r requirements.txt
```

5. Run the dashboard.

```bash
streamlit run app.py
```


## Future Improvement Ideas

- Add GARCH volatility forecasting
- Add factor exposure analysis such as market, size, value, momentum, and quality
- Add ETF/asset-class presets
- Add PDF report export
- Add database storage for historical portfolios
- Add VaR model validation tests such as Kupiec proportion-of-failures test
- Add multi-currency portfolio support
