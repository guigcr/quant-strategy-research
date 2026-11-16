"""Adjusted daily prices and synthetic pairs."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd


def load_universe(tickers, start="2020-01-01", end=None, cache_dir="data_cache"):
    """Load adjusted open and close prices. Fixed date ranges are cached as CSV."""
    import yfinance as yf
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    key = cache / f"{'_'.join(sorted(tickers))}_{start}_{end}.csv"
    if end is not None and key.exists():
        raw = pd.read_csv(key, header=[0, 1], index_col=0, parse_dates=True)
    else:
        raw = yf.download(list(tickers), start=start, end=end, auto_adjust=True, progress=False)
        if raw.empty:
            raise ValueError("No price data returned for the requested tickers and dates.")
        if end is not None:
            raw.to_csv(key)
    o = raw["Open"][list(tickers)]
    c = raw["Close"][list(tickers)]
    ok = o.notna().all(axis=1) & c.notna().all(axis=1)
    return o[ok], c[ok]


def load_pair(x, y, start="2020-01-01", end=None, cache_dir="data_cache") -> pd.DataFrame:
    """Return aligned open_x, close_x, open_y and close_y prices."""
    o, c = load_universe([x, y], start, end, cache_dir)
    return pd.DataFrame({"open_x": o[x], "close_x": c[x], "open_y": o[y], "close_y": c[y]})


def synthetic_pair(n=1500, beta=0.9, kappa=0.08, sigma_eq=0.02, vol=0.018,
                   cointegrated=True, seed=0) -> pd.DataFrame:
    """Generate log prices with an OU residual, or a random-walk residual for the null case."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2020-01-02", periods=n)
    lx = np.log(20) + np.cumsum(rng.normal(0.0002, vol, n))
    phi = np.exp(-kappa)
    s = np.zeros(n)
    sd_eps = sigma_eq * np.sqrt(1 - phi ** 2)
    for t in range(1, n):
        s[t] = (phi * s[t - 1] + rng.normal(0, sd_eps)) if cointegrated else (s[t - 1] + rng.normal(0, 0.004))
    ly = np.log(25) + beta * (lx - lx[0]) + s
    cx, cy = np.exp(lx), np.exp(ly)
    ox = np.r_[cx[0], cx[:-1]] * np.exp(rng.normal(0, 0.003, n))
    oy = np.r_[cy[0], cy[:-1]] * np.exp(rng.normal(0, 0.003, n))
    return pd.DataFrame({"open_x": ox, "close_x": cx, "open_y": oy, "close_y": cy}, index=idx)
