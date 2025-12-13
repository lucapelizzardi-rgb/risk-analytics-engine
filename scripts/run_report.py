 Risk Analytics Engine (Python)

A modular portfolio risk analytics engine implemented in Python, designed to replicate
core functionalities used by Risk Analysts and Quantitative teams in financial institutions.

 Features

- Returns computation and aggregation
- Portfolio-level risk analysis
- Value at Risk (VaR):
  - Historical
  - Parametric
  - Monte Carlo (multivariate)
- Expected Shortfall (ES)
- Statistical backtesting (Kupiec Test)
- Stress testing (volatility and mean shocks)
- Fully tested with pytest

 Methodology

The engine estimates portfolio risk using multiple approaches:

- Historical VaR: non-parametric quantile-based estimation
- Parametric VaR: normal approximation
- Monte Carlo VaR: multivariate simulation using covariance estimation
  and Cholesky decomposition
- Backtesting: Kupiec likelihood ratio test to validate VaR accuracy
- Stress Testing: scenario analysis via volatility and mean shocks

All components are modular and independently testable.

 Project Structure

risk-analytics-engine/
│
├── src/risk_engine/
│ ├── returns.py
│ ├── portfolio.py
│ ├── var.py
│ ├── backtesting.py
│ ├── monte_carlo.py
│ └── stress.py
│
├── tests/
│ ├── test_returns.py
│ ├── test_portfolio.py
│ ├── test_var.py
│ ├── test_backtesting.py
│ ├── test_monte_carlo.py
│ └── test_stress.py
│
├── scripts/
│ └── run_report.py
│
└── README.md
