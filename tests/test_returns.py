import numpy as np
import pandas as pd
import pytest

from risk_engine.returns import compute_returns, aggregate_returns


def test_log_returns_simple_case():
    prices = pd.DataFrame(
        {
            "A": [100.0, 200.0],
        }
    )
    returns = compute_returns(prices, method="log")
    expected = np.log(2.0)

    assert np.isclose(returns.iloc[0, 0], expected)


def test_prices_must_be_positive():
    prices = pd.DataFrame(
        {
            "A": [100.0, 0.0],
        }
    )
    with pytest.raises(ValueError):
        compute_returns(prices)


def test_aggregate_returns():
    returns = pd.DataFrame(
        {
            "A": [0.01, 0.02, -0.01],
        }
    )
    agg = aggregate_returns(returns, horizon_days=2)

    expected = 0.01 + 0.02
    assert np.isclose(agg.iloc[0, 0], expected)
