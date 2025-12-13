import numpy as np
import pandas as pd


def _validate_returns(returns: pd.Series) -> None:
    if not isinstance(returns, pd.Series):
        raise TypeError("returns must be a pandas Series")

    if returns.empty:
        raise ValueError("returns series is empty")

    if returns.isnull().any():
        raise ValueError("returns contain NaN values")


def _validate_alpha(alpha: float) -> None:
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must be between 0 and 1")


def historical_var(returns: pd.Series, alpha: float = 0.99) -> float:
    """
    Historical Value at Risk (VaR).

    Parameters
    ----------
    returns : pd.Series
        Portfolio returns.
    alpha : float
        Confidence level.

    Returns
    -------
    float
        Value at Risk (positive number).
    """
    _validate_returns(returns)
    _validate_alpha(alpha)

    q = returns.quantile(1 - alpha)
    return -float(q)


def historical_es(returns: pd.Series, alpha: float = 0.99) -> float:
    """
    Historical Expected Shortfall (ES).

    Parameters
    ----------
    returns : pd.Series
        Portfolio returns.
    alpha : float
        Confidence level.

    Returns
    -------
    float
        Expected Shortfall (positive number).
    """
    _validate_returns(returns)
    _validate_alpha(alpha)

    q = returns.quantile(1 - alpha)
    tail_losses = returns[returns <= q]

    if tail_losses.empty:
        raise ValueError("not enough data in tail to compute ES")
    return -float(tail_losses.mean())

from scipy.stats import norm, skew, kurtosis

def parametric_var(returns: pd.Series, alpha: float = 0.99) -> float:
    """
    Gaussian (parametric) Value at Risk.
    """
    _validate_returns(returns)
    _validate_alpha(alpha)

    mu = returns.mean()
    sigma = returns.std(ddof=1)

    z = norm.ppf(1 - alpha)
    return -(mu + sigma * z)


def cornish_fisher_var(returns: pd.Series, alpha: float = 0.99) -> float:
    """
    Cornish-Fisher adjusted Value at Risk.
    """
    _validate_returns(returns)
    _validate_alpha(alpha)

    mu = returns.mean()
    sigma = returns.std(ddof=1)

    z = norm.ppf(1 - alpha)

    s = skew(returns)
    k = kurtosis(returns, fisher=True)

    z_cf = (
        z
        + (1 / 6) * (z**2 - 1) * s
        + (1 / 24) * (z**3 - 3 * z) * k
        - (1 / 36) * (2 * z**3 - 5 * z) * s**2
    )

    return -(mu + sigma * z_cf)
