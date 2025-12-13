import numpy as np


def monte_carlo_var(returns, weights, alpha=0.95, n_sims=10000):
    """
    Monte Carlo VaR multivariato usando decomposizione di Cholesky.
    """
    returns = np.asarray(returns)
    weights = np.asarray(weights)

    if returns.ndim != 2:
        raise ValueError("returns must be a 2D array (T x N)")
    if weights.ndim != 1 or weights.shape[0] != returns.shape[1]:
        raise ValueError("weights dimension mismatch")
    if not (0 < alpha < 1):
        raise ValueError("alpha must be in (0,1)")

    mu = returns.mean(axis=0)
    cov = np.cov(returns, rowvar=False)

    chol = np.linalg.cholesky(cov)
    z = np.random.normal(size=(n_sims, returns.shape[1]))
    sims = z @ chol.T + mu

    portfolio_returns = sims @ weights
    var = -np.quantile(portfolio_returns, 1 - alpha)

    return var
