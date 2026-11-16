"""Close-based signals: +1 buys Y and sells X, -1 reverses the legs. Hedge ratios are notional X / Y."""
from __future__ import annotations
import numpy as np
import pandas as pd
from .stats import ols, ou_params, engle_granger


def _state_machine(z, entry, exit, stop, allowed, max_hold):
    if not 0 <= exit < entry <= stop:
        raise ValueError("Require 0 <= exit < entry <= stop.")
    if np.any(np.asarray(max_hold) <= 0):
        raise ValueError("max_hold must be positive.")
    n = len(z)
    out = np.zeros(n)
    pos, held = 0, 0
    max_hold = np.broadcast_to(np.asarray(max_hold, float), (n,))
    for i in range(n):
        zi = z[i]
        if not np.isfinite(zi):
            pos, held = 0, 0
            continue
        if pos != 0:
            held += 1
            reverted = (pos == 1 and zi >= -exit) or (pos == -1 and zi <= exit)
            if reverted or abs(zi) > stop or held >= max_hold[i] or not allowed[i]:
                pos, held = 0, 0
                continue  # Keep the exit visible to the execution engine.
        if pos == 0 and allowed[i] and entry <= abs(zi) <= stop:
            pos = -1 if zi > 0 else 1
            held = 0
        out[i] = pos
    return out


def _pack(index, signal, hr, z):
    return pd.DataFrame({"signal": signal, "hedge_ratio": hr, "z": z}, index=index)


def _hr(sizing, log, beta, x, y):
    """Return X notional per unit of Y notional."""
    if sizing not in {"dollar", "beta"}:
        raise ValueError("sizing must be dollar or beta.")
    if sizing == "dollar":
        return np.ones_like(beta)
    return beta if log else beta * x / y


def _rolling_model(x, y, window, refit, log, run_coint=True):
    if window < 4 or refit < 1:
        raise ValueError("Require window >= 4 and refit >= 1.")
    xv = np.log(x.values) if log else x.values.astype(float)
    yv = np.log(y.values) if log else y.values.astype(float)
    n = len(xv)
    cols = {k: np.full(n, np.nan) for k in
            ["alpha", "beta", "mu_s", "sd_s", "pval", "hl", "mu_ou", "sd_ou", "spread"]}
    cur = None
    for i in range(window - 1, n):
        if cur is None or (i - (window - 1)) % refit == 0:
            xs, ys = xv[i - window + 1:i + 1], yv[i - window + 1:i + 1]
            a, b = ols(ys, xs)
            s = ys - a - b * xs
            ou_ = ou_params(s)
            p = engle_granger(ys, xs)[1] if run_coint else np.nan
            cur = (a, b, s.mean(), s.std(ddof=1), p, ou_["half_life"], ou_["mu"], ou_["sigma_eq"])
        a, b, mu_s, sd_s, p, hl, mu_o, sd_o = cur
        cols["alpha"][i], cols["beta"][i], cols["mu_s"][i], cols["sd_s"][i] = a, b, mu_s, sd_s
        cols["pval"][i], cols["hl"][i], cols["mu_ou"][i], cols["sd_ou"][i] = p, hl, mu_o, sd_o
        cols["spread"][i] = yv[i] - a - b * xv[i]
    return {k: v for k, v in cols.items()}, xv, yv


def relative_return(close_x, close_y, lookback=20, zwin=120, entry=2.0, exit=0.5,
                    stop=4.0, max_hold=20, **_):
    """Trade the rolling z-score of the difference in cumulative returns; use equal notionals."""
    ry = close_y / close_y.shift(lookback) - 1
    rx = close_x / close_x.shift(lookback) - 1
    d = ry - rx
    z = ((d - d.rolling(zwin).mean()) / d.rolling(zwin).std()).values
    allowed = np.ones(len(z), bool)
    sig = _state_machine(z, entry, exit, stop, allowed, max_hold)
    return _pack(close_x.index, sig, np.ones(len(z)), z)


def cointegration(close_x, close_y, window=252, refit=21, log=True, pmax=0.10,
                  entry=2.0, exit=0.5, stop=4.0, max_hold=60, sizing="beta", **_):
    """Trade rolling OLS residuals with a positive beta and optional Engle-Granger filter."""
    m, xv, yv = _rolling_model(close_x, close_y, window, refit, log, run_coint=pmax < 1.0)
    z = (m["spread"] - m["mu_s"]) / m["sd_s"]
    allowed = m["beta"] > 0
    if pmax < 1.0:
        allowed &= m["pval"] <= pmax
    allowed = np.nan_to_num(allowed.astype(float)).astype(bool)
    sig = _state_machine(z, entry, exit, stop, allowed, max_hold)
    hr = _hr(sizing, log, m["beta"], close_x.values, close_y.values)
    return _pack(close_x.index, sig, np.nan_to_num(hr, nan=1.0), z)


def ou(close_x, close_y, window=252, refit=21, log=True, pmax=1.0, hl_min=2.0,
       hl_max=40.0, entry=2.0, exit=0.5, stop=4.0, hold_mult=3.0, sizing="beta", **_):
    """Trade the OU equilibrium z-score, filtered by half-life; cap holding time by hold_mult."""
    m, xv, yv = _rolling_model(close_x, close_y, window, refit, log, run_coint=pmax < 1.0)
    z = (m["spread"] - m["mu_ou"]) / m["sd_ou"]
    hl = m["hl"]
    allowed = np.isfinite(hl) & (hl >= hl_min) & (hl <= hl_max) & (m["beta"] > 0)
    if pmax < 1.0:
        allowed &= (np.nan_to_num(m["pval"], nan=1.0) <= pmax)
    max_hold = np.ceil(np.nan_to_num(hl, nan=1.0, posinf=1.0) * hold_mult)
    sig = _state_machine(z, entry, exit, stop, allowed, max_hold)
    hr = _hr(sizing, log, m["beta"], close_x.values, close_y.values)
    return _pack(close_x.index, sig, np.nan_to_num(hr, nan=1.0), z)


def kalman(close_x, close_y, delta=1e-7, ve=1e-4, zwin=60, warmup=60, entry=2.0, exit=0.5,
           stop=4.0, max_hold=60, **_):
    """Estimate log-price beta and intercept recursively. Scale innovations by their lagged rolling deviation."""
    if not 0 < delta < 1 or ve <= 0:
        raise ValueError("Require 0 < delta < 1 and ve > 0.")
    x, y = np.log(close_x.values), np.log(close_y.values)
    n = len(x)
    theta = np.zeros(2)
    P = np.zeros((2, 2))
    Vw = delta / (1 - delta) * np.eye(2)
    err, beta = np.full(n, np.nan), np.full(n, np.nan)
    for t in range(n):
        F = np.array([x[t], 1.0])
        R = P + Vw if t > 0 else P
        e = y[t] - F @ theta
        Q = F @ R @ F + ve
        K = R @ F / Q
        theta = theta + K * e
        P = R - np.outer(K, F) @ R
        err[t] = e
        if t >= warmup:
            beta[t] = theta[0]
    e = pd.Series(err)
    z = (e / e.shift(1).rolling(zwin).std()).to_numpy(copy=True)
    z[:warmup] = np.nan
    allowed = np.nan_to_num(beta, nan=-1) > 0
    sig = _state_machine(z, entry, exit, stop, allowed, max_hold)
    return _pack(close_x.index, sig, np.nan_to_num(beta, nan=1.0), z)

STRATEGIES = {
    "relative_return": relative_return,
    "cointegration": cointegration,
    "ou": ou,
    "kalman": kalman,
}
