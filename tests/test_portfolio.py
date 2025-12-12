import numpy as np
import pandas as pd
import pytest

from risk_engine.portfolio import portfolio_returns


def test_portfolio_returns_simple_case():
    returns = pd.DataFrame(
        {
            "A": [0.01, 0.02],
            "B": [0.00, -0.01],
        }
    )
    weights = {"A": 0.6, "B": 0.4}

    port_ret = portfolio_returns(returns, weights)

    expected = 0.6 * returns["A"] + 0.4 * returns["B"]
    assert np.allclose(port_ret.values, expected.values)


def test_weights_must_sum_to_one():
    returns = pd.DataFrame({"A": [0.01], "B": [0.02]})
    weights = {"A": 0.7, "B": 0.4}

    with pytest.raises(ValueError):
        portfolio_returns(returns, weights)
