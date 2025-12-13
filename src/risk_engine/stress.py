import numpy as np


def apply_stress(returns, shock_mean=0.0, shock_vol=1.0):
    """
    Applica uno stress ai rendimenti:
    - shock_mean: shift sulla media
    - shock_vol: moltiplicatore della volatilità
    """
    returns = np.asarray(returns)

    if shock_vol <= 0:
        raise ValueError("shock_vol must be positive")

    stressed = (returns - returns.mean(axis=0)) * shock_vol
    stressed += returns.mean(axis=0) + shock_mean

    return stressed
