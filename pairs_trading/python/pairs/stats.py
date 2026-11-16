"""Regression, stationarity tests and mean-reversion estimates."""
from __future__ import annotations
import numpy as np


def ols(y, x):
    """Fit y = alpha + beta*x and return (alpha, beta)."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    dx = x - x.mean()
    denom = (dx ** 2).sum()
    if denom == 0:
        return np.nan, np.nan
    beta = (dx * (y - y.mean())).sum() / denom
    alpha = y.mean() - beta * x.mean()
    return alpha, beta


def adf(series, regression="c", autolag="AIC"):
    """Return the ADF statistic, p-value and lag count."""
    from statsmodels.tsa.stattools import adfuller

    r = adfuller(np.asarray(series, float), regression=regression, autolag=autolag)
    return r[0], r[1], r[2]


def engle_granger(y, x, trend="c"):
    """Test the null of no cointegration using Engle-Granger critical values."""
    from statsmodels.tsa.stattools import coint

    try:
        stat, p, crit = coint(np.asarray(y, float), np.asarray(x, float),
                              trend=trend, autolag="aic")
        return stat, p, crit
    except (ValueError, np.linalg.LinAlgError):
        return np.nan, 1.0, np.array([np.nan] * 3)


def fit_ar1(s):
    """Fit s[t] = c + phi*s[t-1] + error; return c, phi and residual standard deviation."""
    s = np.asarray(s, float)
    c, phi = ols(s[1:], s[:-1])
    resid = s[1:] - c - phi * s[:-1]
    return c, phi, resid.std(ddof=2)


def ou_params(s, dt=1.0):
    """Estimate OU parameters from AR(1). Non-reverting fits have infinite half-life."""
    c, phi, sd = fit_ar1(s)
    if not (0 < phi < 1) or not np.isfinite(phi):
        return dict(mu=np.nan, kappa=np.nan, sigma=np.nan, sigma_eq=np.nan,
                    half_life=np.inf, phi=phi)
    kappa = -np.log(phi) / dt
    sigma_eq = sd / np.sqrt(1 - phi ** 2)
    return dict(mu=c / (1 - phi), kappa=kappa, sigma=np.sqrt(2 * kappa) * sigma_eq,
                sigma_eq=sigma_eq, half_life=np.log(2) / kappa, phi=phi)


def half_life(s, dt=1.0):
    return ou_params(s, dt)["half_life"]


def benjamini_hochberg(pvals, q=0.05):
    """Return adjusted p-values and rejection flags for the BH procedure."""
    p = np.asarray(pvals, float)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order] * n / np.arange(1, n + 1)
    adj = np.clip(np.minimum.accumulate(ranked[::-1])[::-1], 0, 1)
    out = np.empty(n)
    out[order] = adj
    return out, out <= q
