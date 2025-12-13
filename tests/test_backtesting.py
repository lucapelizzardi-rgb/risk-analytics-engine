import numpy as np
import pytest

from risk_engine.backtesting import kupiec_test


def test_kupiec_reasonable_model():
    """
    Test di backtesting Kupiec su un modello VaR ragionevole.
    Il VaR è calibrato coerentemente con una N(0, 1%) al 95%.
    Il test NON deve rifiutare il modello.
    """
    np.random.seed(42)

    # Rendimenti simulati
    returns = np.random.normal(loc=0.0, scale=0.01, size=1000)

    # VaR teorico al 95% per una normale (≈ 1.65 * sigma)
    var_95 = np.full_like(returns, 0.0165)

    lr_stat, p_value = kupiec_test(returns, var_95, alpha=0.95)

    # Il modello non deve essere rifiutato
    assert p_value > 0.001


def test_kupiec_bad_model():
    """
    Test di backtesting Kupiec su un modello VaR sottostimato.
    Il test DEVE rifiutare il modello.
    """
    np.random.seed(123)

    returns = np.random.normal(loc=0.0, scale=0.01, size=1000)

    # VaR troppo basso → troppe violazioni
    var_95 = np.full_like(returns, 0.005)

    lr_stat, p_value = kupiec_test(returns, var_95, alpha=0.95)

    assert p_value < 0.05


def test_kupiec_invalid_inputs():
    """
    Test di robustezza: input non validi devono sollevare errore.
    """
    returns = np.array([0.01, -0.02, 0.015])
    var = np.array([0.01, 0.01])  # dimensione errata

    with pytest.raises(ValueError):
        kupiec_test(returns, var, alpha=0.95)


