"""Rolling parameter selection and out-of-sample evaluation."""
from __future__ import annotations
import itertools
from dataclasses import replace
import numpy as np
import pandas as pd
from .backtest import BacktestConfig, run_backtest
from .metrics import performance_summary


def _score(res, metric):
    s = performance_summary(res.equity, res.cfg.rf_annual, res.cfg.periods)
    v = s.get(metric, np.nan)
    return -np.inf if (v is None or not np.isfinite(v)) else v


def grid_search(df, strategy, grid: dict, cfg=None, metric="sharpe", fixed=None):
    """Rank parameter combinations by an in-sample performance metric."""
    cfg = cfg or BacktestConfig()
    keys = list(grid)
    rows = []
    for combo in itertools.product(*grid.values()):
        params = {**(fixed or {}), **dict(zip(keys, combo))}
        sig = strategy(df["close_x"], df["close_y"], **params)
        res = run_backtest(df, sig, cfg)
        rows.append({**dict(zip(keys, combo)), metric: _score(res, metric),
                     "n_trades": len(res.trades)})
    return pd.DataFrame(rows).sort_values(metric, ascending=False).reset_index(drop=True)


def walkforward_optimize(df, strategy, grid, cfg=None, train_len=504, test_len=126,
                         warmup=300, metric="sharpe", fixed=None, min_trades=3):
    """Select parameters on each training window and evaluate the following non-overlapping test window."""
    cfg = cfg or BacktestConfig()
    cfg = replace(cfg, liquidate_at_end=True)
    if train_len < 1 or test_len < 2 or warmup < 0:
        raise ValueError("Require train_len >= 1, test_len >= 2 and warmup >= 0.")
    if any(len(values) == 0 for values in grid.values()):
        raise ValueError("Parameter grid entries must not be empty.")
    keys = list(grid)
    n = len(df)
    folds, oos = [], []
    start = warmup
    while start + train_len + test_len <= n:
        tr_end, te_end = start + train_len, start + train_len + test_len
        hist = df.iloc[start - warmup:te_end]
        best, best_score = None, -np.inf
        for combo in itertools.product(*grid.values()):
            params = {**(fixed or {}), **dict(zip(keys, combo))}
            training = hist.iloc[:warmup + train_len]
            sig = strategy(training["close_x"], training["close_y"], **params)
            tr_idx = df.index[start:tr_end]
            res = run_backtest(hist.loc[tr_idx], sig.loc[tr_idx], cfg)
            sc = _score(res, metric) if len(res.trades) >= min_trades else -np.inf
            if sc > best_score:
                best, best_score = params, sc
        selected = best is not None
        if selected:
            sig = strategy(hist["close_x"], hist["close_y"], **best)
        else:
            best = {}
            sig = pd.DataFrame({"signal": 0, "hedge_ratio": 1.0}, index=hist.index)
        te_idx = df.index[tr_end:te_end]
        res = run_backtest(hist.loc[te_idx], sig.loc[te_idx], cfg)
        oos.append(res.returns)
        folds.append({"train_start": df.index[start], "test_start": te_idx[0],
                      "test_end": te_idx[-1], "train_score": best_score,
                      "oos_return": res.equity.iloc[-1] / res.equity.iloc[0] - 1,
                      "oos_trades": len(res.trades), "selected": selected, **{k: best.get(k) for k in keys}})
        start += test_len
    if not oos:
        raise ValueError("Not enough data for one fold; reduce train_len, test_len or warmup.")
    r = pd.concat(oos)
    equity = cfg.capital * (1 + r).cumprod()
    return {"oos_returns": r, "oos_equity": equity, "folds": pd.DataFrame(folds),
            "summary": performance_summary(equity, cfg.rf_annual, cfg.periods)}
