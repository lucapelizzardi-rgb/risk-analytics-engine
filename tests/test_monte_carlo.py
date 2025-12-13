import numpy as np
import pytest

from risk_engine.monte_carlo import monte_carlo_var


def test_monte_carlo_var_positive():
    np.random.seed(0)

    returns = np.random.normal(0, 0.01, size=(1000, 2))
    weights = np.array([0.5, 0.5])

    var_95 = monte_carlo_var(returns, weights, alpha=0.95, n_sims=5000)

    assert var_95 > 0


def test_monte_carlo_invalid_alpha():
    returns = np.random.normal(0, 0.01, size=(100, 2))
    weights = np.array([0.5, 0.5])

    with pytest.raises(ValueError):
        monte_carlo_var(returns, weights, alpha=1.5)


def test_monte_carlo_weights_mismatch():
    returns = np.random.normal(0, 0.01, size=(100, 3))
    weights = np.array([0.5, 0.5])

    with pytest.raises(ValueError):
        monte_carlo_var(returns, weights)
