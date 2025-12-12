import numpy as np
import pandas as pd


def compute_returns(
    prices: pd.DataFrame,
    method: str = "log",
    dropna: bool = True,
) -> pd.DataFrame:
    """
    Compute asset returns from price data.

    Parameters
    ----------
    prices : pd.DataFrame
        Price time series (index = dates, columns = assets).
    method : str
        Return type: "log" or "simple".
    dropna : bool
        Whether to drop NaN values after computation.

    Returns
    -------
    pd.DataFrame
        Returns time series.
    """
    if not isinstance(prices, pd.DataFrame):
        raise TypeError("prices must be a pandas DataFrame")

    if (prices <= 0).any().any():
        raise ValueError("prices must be strictly positive")

    if method not in {"log", "simple"}:
        raise ValueError("method must be 'log' or 'simple'")

    if method == "log":
        returns = np.log(prices / prices.shift(1))
    else:
        returns = prices.pct_change()

    if dropna:
        returns = returns.dropna()

    return returns


def aggregate_returns(
    returns: pd.DataFrame,
    horizon_days: int,
) -> pd.DataFrame:
    """
    Aggregate returns over a given time horizon.

    For log-returns, aggregation is additive.

    Parameters
    ----------
    returns : pd.DataFrame
        Daily returns.
    horizon_days : int
        Number of days to aggregate.

    Returns
    -------
    pd.DataFrame
        Aggregated returns.
    """
    if horizon_days < 1:
        raise ValueError("horizon_days must be >= 1")

    return returns.rolling(window=horizon_days).sum().dropna()
