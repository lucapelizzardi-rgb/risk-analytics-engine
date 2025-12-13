import numpy as np
import pandas as pd

from risk_engine.returns import compute_returns
from risk_engine.portfolio import portfolio_returns
from risk_engine.var import historical_var, parametric_var
from risk_engine.monte_carlo import monte_carlo_var
from risk_engine.stress import apply_stress
from risk_engine.backtesting import kupiec_test

np.random.seed(42)

prices = pd.DataFrame({
    "AAPL": 100 * np.cumprod(1 + np.random.normal(0.001, 0.02, 500)),
    "MSFT": 120 * np.cumprod(1 + np.random.normal(0.0008, 0.018, 500)),
    "GOOG": 90 * np.cumprod(1 + np.random.normal(0.0012, 0.025, 500)),
})

returns = compute_returns(prices)

weights = {
    "AAPL": 0.4,
    "MSFT": 0.3,
    "GOOG": 0.3,
}

portfolio_rets = portfolio_returns(returns, weights)

alpha = 0.95

var_hist = historical_var(portfolio_rets, alpha=alpha)
var_param = parametric_var(portfolio_rets, alpha=alpha)
weights_series = pd.Series(weights).loc[returns.columns].values
var_mc = monte_carlo_var(returns, weights_series, alpha=alpha, n_sims=20000)


lr, p_value = kupiec_test(
    portfolio_rets,
    np.full(len(portfolio_rets), var_hist),
    alpha=alpha
)

stressed = apply_stress(portfolio_rets, shock_mean=-0.10)

print("RISK ANALYTICS REPORT")
print("Historical VaR:", round(var_hist, 6))
print("Parametric VaR:", round(var_param, 6))
print("Monte Carlo VaR:", round(var_mc, 6))
print("Kupiec p-value:", round(p_value, 8))
print("Stress mean:", round(stressed.mean(), 6))

