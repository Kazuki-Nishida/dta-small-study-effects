"""The Deeks test on the effective-sample-size scale.

deeks_test  the Deeks test (Deeks, Macaskill and Irwig, 2005): weighted least squares of a
            study-level outcome (the continuity-corrected lnDOR, or any other axis) on
            s_i = 1/sqrt(ESS_i) with weights ESS_i, the slope's t statistic referred to t_{k-2}.
"""
import numpy as np
from scipy.stats import t as tdist


def wls_slope_and_var(y, ess):
    """Slope of the ESS-weighted regression of y on 1/sqrt(ESS) and its model-based variance."""
    x = 1 / np.sqrt(ess)
    X = np.column_stack([np.ones_like(x), x])
    W = ess
    XtWX = X.T @ (X * W[:, None])
    beta = np.linalg.solve(XtWX, X.T @ (W * y))
    resid = y - X @ beta
    s2 = np.sum(W * resid ** 2) / (len(y) - 2)
    var_b = s2 * np.linalg.inv(XtWX)[1, 1]
    return float(beta[1]), float(var_b)


def deeks_test(y, ess):
    """Deeks-form test of the outcome y.  Returns (slope, two-sided p value from t_{k-2})."""
    slope, var_b = wls_slope_and_var(np.asarray(y, float), np.asarray(ess, float))
    tval = slope / np.sqrt(var_b)
    p = 2 * tdist.sf(abs(tval), len(y) - 2)
    return float(slope), float(p)
