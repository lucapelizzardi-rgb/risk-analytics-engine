import numpy as np
import pandas as pd
import pytest

from risk_engine.var import historical_var, historical_es


def test_historical_var_simple():
    returns = pd.Series([-0.02, -0.01, 0.0, 0.01, 0.02])
    var_95 = historical_var(returns, alpha=0.95)

    assert var_95 >= 0
    assert var_95 >= 0.01
    assert var_95 <= 0.02


def test_historical_es_simple():
    returns = pd.Series([-0.02, -0.01, 0.0, 0.01, 0.02])
    es_95 = historical_es(returns, alpha=0.95)

    assert es_95 >= 0
    assert es_95 >= historical_var(returns, alpha=0.95)


def test_invalid_alpha():
    returns = pd.Series([0.01, -0.01])

    with pytest.raises(ValueError):
        historical_var(returns, alpha=1.5)

from risk_engine.var import parametric_var, cornish_fisher_var


def test_parametric_var_positive():
    returns = pd.Series(np.random.normal(0, 0.01, size=10_000))
    var_99 = parametric_var(returns, alpha=0.99)

    assert var_99 > 0


def test_cornish_fisher_differs_from_gaussian():
    returns = pd.Series(np.random.normal(0, 0.01, size=10_000) ** 3)

    var_gauss = parametric_var(returns, alpha=0.99)
    var_cf = cornish_fisher_var(returns, alpha=0.99)

    assert not np.isclose(var_gauss, var_cf)
