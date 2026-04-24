import numpy as np


def monte_carlo_var(returns, weights, alpha=0.95, n_sims=10000, seed=None):
    """
    Monte Carlo VaR multivariato usando decomposizione di Cholesky.
    Risultati non deterministici.
    """
    returns = np.asarray(returns)
    weights = np.asarray(weights)

    if returns.ndim != 2:
        raise ValueError("returns must be a 2D array (T x N)")
    if weights.ndim != 1 or weights.shape[0] != returns.shape[1]:
        raise ValueError("weights dimension mismatch")
    if not (0 < alpha < 1):
        raise ValueError("alpha must be in (0,1)")

    # stima media e covarianza dai rendimenti storici
    mu = returns.mean(axis=0)
    cov = np.cov(returns, rowvar=False)

    # genera n scenari correlati tramite decomposizione di Cholesky
    rng = np.random.default_rng(seed)
    chol = np.linalg.cholesky(cov)
    z = rng.standard_normal((n_sims, returns.shape[1]))
    sims = z @ chol.T + mu

    # calcola il VaR come perdita al quantile (1 - alpha)
    portfolio_returns = sims @ weights
    var = -np.quantile(portfolio_returns, 1 - alpha)

    return var
