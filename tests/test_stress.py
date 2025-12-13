import numpy as np
import pytest

from risk_engine.stress import apply_stress


def test_stress_increases_volatility():
    np.random.seed(0)

    returns = np.random.normal(0, 0.01, size=(1000, 2))
    stressed = apply_stress(returns, shock_vol=2.0)

    assert stressed.std() > returns.std()


def test_stress_negative_vol_error():
    returns = np.random.normal(0, 0.01, size=(100, 2))

    with pytest.raises(ValueError):
        apply_stress(returns, shock_vol=-1)
