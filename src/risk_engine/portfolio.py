import numpy as np
import pandas as pd


def validate_weights(weights, tol: float = 1e-6) -> None:
    """
    Validate portfolio weights.

    Parameters
    ----------
    weights : dict or pd.Series
        Asset weights.
    tol : float
        Tolerance for sum of weights.

    Raises
    ------
    ValueError
        If weights are invalid.
    """
    if isinstance(weights, dict):
        weights = pd.Series(weights)

    if not isinstance(weights, pd.Series):
        raise TypeError("weights must be a dict or pandas Series")

    if not np.isclose(weights.sum(), 1.0, atol=tol):
        raise ValueError("weights must sum to 1")

    if weights.isnull().any():
        raise ValueError("weights contain NaN values")


def portfolio_returns(
    asset_returns: pd.DataFrame,
    weights,
) -> pd.Series:
    """
    Compute portfolio returns as weighted sum of asset returns.

    Parameters
    ----------
    asset_returns : pd.DataFrame
        Asset return time series.
    weights : dict or pd.Series
        Portfolio weights.

    Returns
    -------
    pd.Series
        Portfolio returns.
    """
    if not isinstance(asset_returns, pd.DataFrame):
        raise TypeError("asset_returns must be a pandas DataFrame")

    if isinstance(weights, dict):
        weights = pd.Series(weights)

    validate_weights(weights)

    missing_assets = set(weights.index) - set(asset_returns.columns)
    if missing_assets:
        raise ValueError(f"Missing returns for assets: {missing_assets}")

    aligned_returns = asset_returns[weights.index]

    return aligned_returns.dot(weights)
