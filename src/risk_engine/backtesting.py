import numpy as np
from scipy.stats import chi2


def kupiec_test(returns, var_series, alpha=0.95):
    """
    Kupiec unconditional coverage test for VaR.
    Returns LR statistic and p-value.
    """
    returns = np.asarray(returns)
    var_series = np.asarray(var_series)

    if len(returns) != len(var_series):
        raise ValueError("Returns and VaR series must have same length")

    losses = -returns
    violations = losses > var_series
    x = violations.sum()
    T = len(returns)

    p = 1 - alpha
    p_hat = x / T if T > 0 else 0

    if p_hat in (0, 1):
        return np.inf, 0.0

    lr_uc = -2 * (
        (T - x) * np.log((1 - p) / (1 - p_hat)) +
        x * np.log(p / p_hat)
    )

    p_value = 1 - chi2.cdf(lr_uc, df=1)
    return lr_uc, p_value
