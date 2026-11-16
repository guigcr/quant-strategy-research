"""Daily execution with fixed trade quantities and explicit costs."""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
from .metrics import performance_summary, trade_summary


@dataclass

class BacktestConfig:
    """Costs in bps per traded notional; leverage is gross exposure / equity."""

    capital: float = 100_000.0
    leverage: float = 1.0
    commission_bps: float = 3.0
    slippage_bps: float = 5.0
    borrow_bps_annual: float = 200.0
    fill: str = "next_open"
    lot_size: int = 1
    compound: bool = True
    rf_annual: float = 0.0
    periods: int = 252
    liquidate_at_end: bool = True

    def __post_init__(self):
        if self.fill not in {"next_open", "next_close"}:
            raise ValueError("fill must be next_open or next_close.")
        if not np.isfinite(self.capital) or self.capital <= 0:
            raise ValueError("capital must be finite and positive.")
        if self.lot_size < 1 or int(self.lot_size) != self.lot_size or self.periods <= 0:
            raise ValueError("lot_size must be a positive integer; periods must be positive.")
        for value in (self.leverage, self.commission_bps, self.slippage_bps, self.borrow_bps_annual):
            if not np.isfinite(value) or value < 0:
                raise ValueError("Exposure and cost parameters must be finite and nonnegative.")


@dataclass

class BacktestResult:
    equity: pd.Series
    trades: pd.DataFrame
    positions: pd.DataFrame
    traded_notional: pd.Series
    cfg: BacktestConfig
    signals: pd.DataFrame | None = field(default=None, repr=False)

    @property
    def returns(self):
        return self.equity.pct_change().fillna(0.0)

    def summary(self) -> dict:
        s = performance_summary(self.equity, self.cfg.rf_annual, self.cfg.periods)
        s.update(trade_summary(self.trades))
        s["annual_turnover"] = (self.traded_notional.sum() / self.cfg.capital
                               * self.cfg.periods / max(len(self.equity), 1))
        s["time_in_market"] = float((self.positions["qx"] != 0).mean())
        return s


def _round_lot(q, lot):
    return np.trunc(q / lot) * lot


def run_backtest(df: pd.DataFrame, signals: pd.DataFrame, cfg: BacktestConfig | None = None):
    """Execute close signals on the next bar with fixed quantities per trade."""
    cfg = cfg or BacktestConfig()
    required = ["open_x", "close_x", "open_y", "close_y"]
    if df.empty or not isinstance(df.index, pd.DatetimeIndex) or not df.index.is_unique or not df.index.is_monotonic_increasing:
        raise ValueError("Prices require a nonempty, unique, increasing DatetimeIndex.")
    prices = df[required].to_numpy(dtype=float)
    if not np.isfinite(prices).all() or (prices <= 0).any():
        raise ValueError("Prices must be finite and positive.")
    if not signals.index.equals(df.index):
        raise ValueError("Signals must have exactly the same index as prices.")
    if not signals["signal"].isin([-1, 0, 1]).all():
        raise ValueError("Signals must be -1, 0 or 1.")
    active = signals["signal"] != 0
    hr = signals.loc[active, "hedge_ratio"]
    if not np.isfinite(hr).all() or (hr <= 0).any():
        raise ValueError("Active signals require a finite, positive hedge ratio.")
    ox, cx = df["open_x"].values, df["close_x"].values
    oy, cy = df["open_y"].values, df["close_y"].values
    px_fill, py_fill = (ox, oy) if cfg.fill == "next_open" else (cx, cy)
    sg = np.nan_to_num(signals["signal"].values)
    hrs = np.nan_to_num(signals["hedge_ratio"].values, nan=1.0)

    comm, slip = cfg.commission_bps / 1e4, cfg.slippage_bps / 1e4
    borrow_d = cfg.borrow_bps_annual / 1e4 / cfg.periods
    n = len(df)
    cash = cfg.capital
    qx = qy = 0.0
    cur_side = 0
    entry_cash, entry_date = cash, None
    equity = np.zeros(n)
    Qx, Qy, traded = np.zeros(n), np.zeros(n), np.zeros(n)
    trades = []

    def trade(q, price):
        nonlocal cash
        if q == 0:
            return 0.0
        exec_price = price * (1 + np.sign(q) * slip)
        cash -= q * exec_price + comm * abs(q) * price
        return abs(q) * price

    for i in range(n):
        target = int(sg[i - 1]) if i > 0 else 0
        if target != cur_side:

            if cur_side != 0:
                traded[i] += trade(-qx, px_fill[i]) + trade(-qy, py_fill[i])
                trades.append(dict(entry_date=entry_date, exit_date=df.index[i], side=cur_side,
                                   days=(i - entry_i), pnl=cash - entry_cash,
                                   ret=(cash - entry_cash) / entry_cash))
                qx = qy = 0.0
                cur_side = 0

            if target != 0 and cash > 0 and not (cfg.liquidate_at_end and i == n - 1):
                hr = hrs[i - 1]
                G = cfg.leverage * (cash if cfg.compound else cfg.capital)
                Ny = G / (1 + hr)
                Nx = hr * Ny
                qy = _round_lot(target * Ny / py_fill[i], cfg.lot_size)
                qx = _round_lot(-target * Nx / px_fill[i], cfg.lot_size)
                if qx == 0 or qy == 0:
                    qx = qy = 0.0
                else:
                    entry_cash, entry_date, entry_i = cash, df.index[i], i
                    traded[i] += trade(qx, px_fill[i]) + trade(qy, py_fill[i])
                    cur_side = target

        short_val = max(-qx * cx[i], 0) + max(-qy * cy[i], 0)
        cash -= borrow_d * short_val
        if cfg.liquidate_at_end and i == n - 1 and cur_side != 0:
            # Scheduled end-of-sample liquidation at the final close.
            traded[i] += trade(-qx, cx[i]) + trade(-qy, cy[i])
            trades.append(dict(entry_date=entry_date, exit_date=df.index[i], side=cur_side,
                               days=i - entry_i, pnl=cash - entry_cash,
                               ret=(cash - entry_cash) / entry_cash))
            qx = qy = 0.0
            cur_side = 0
        equity[i] = cash + qx * cx[i] + qy * cy[i]
        Qx[i], Qy[i] = qx, qy

    idx = df.index
    return BacktestResult(
        equity=pd.Series(equity, idx, name="equity"),
        trades=pd.DataFrame(trades, columns=["entry_date", "exit_date", "side", "days", "pnl", "ret"]),
        positions=pd.DataFrame({"qx": Qx, "qy": Qy}, idx),
        traded_notional=pd.Series(traded, idx), cfg=cfg, signals=signals)
