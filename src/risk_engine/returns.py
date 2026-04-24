import numpy as np
import pandas as pd


def compute_returns(
    prices: pd.DataFrame,
    method: str = "log",
    dropna: bool = True,
) -> pd.DataFrame:
    """
    Compute asset returns from price data.
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
    """
    if horizon_days < 1:
        raise ValueError("horizon_days must be >= 1")

    return returns.rolling(window=horizon_days).sum().dropna()
