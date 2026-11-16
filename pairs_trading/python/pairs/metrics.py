"""Equity and closed-trade performance statistics."""
from __future__ import annotations
import numpy as np
import pandas as pd


def drawdown(equity: pd.Series) -> pd.Series:
    return equity / equity.cummax() - 1


def performance_summary(equity: pd.Series, rf_annual=0.0, periods=252) -> dict:
    r = equity.pct_change().dropna()
    if len(r) < 2:
        return {}
    rf = (1 + rf_annual) ** (1 / periods) - 1
    ex = r - rf
    years = len(r) / periods
    total = equity.iloc[-1] / equity.iloc[0] - 1
    cagr = (equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1 if equity.iloc[-1] > 0 else -1.0
    vol = r.std() * np.sqrt(periods)
    down = np.sqrt(np.mean(np.minimum(ex, 0) ** 2)) * np.sqrt(periods)
    mdd = drawdown(equity).min()
    return {
        "total_return": total, "cagr": cagr, "annual_volatility": vol,
        "sharpe": ex.mean() / r.std() * np.sqrt(periods) if r.std() > 0 else np.nan,
        "sortino": ex.mean() * periods / down if down and down > 0 else np.nan,
        "max_drawdown": mdd, "calmar": cagr / abs(mdd) if mdd < 0 else np.nan,
    }


def trade_summary(trades: pd.DataFrame) -> dict:
    if trades is None or len(trades) == 0:
        return {"n_trades": 0, "win_rate": np.nan, "profit_factor": np.nan,
                "expectancy": np.nan, "average_holding_days": np.nan}
    w, l = trades.loc[trades.pnl > 0, "pnl"], trades.loc[trades.pnl <= 0, "pnl"]
    p = len(w) / len(trades)
    G = w.mean() if len(w) else 0.0
    L = abs(l.mean()) if len(l) else 0.0
    return {"n_trades": len(trades), "win_rate": p,
            "profit_factor": w.sum() / abs(l.sum()) if len(l) and l.sum() != 0 else np.inf,
            "expectancy": p * G - (1 - p) * L,
            "average_holding_days": trades.days.mean()}
