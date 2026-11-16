"""Equity, signal and transaction-cost charts."""
from __future__ import annotations
import matplotlib.pyplot as plt
import numpy as np
from .metrics import drawdown


def _finish(fig, path):
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=130)
        plt.close(fig)


def plot_equity(results: dict, path=None, title="Equity curve"):
    fig, ax = plt.subplots(2, 1, figsize=(12, 7), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
    for name, r in results.items():
        ax[0].plot(r.equity.index, r.equity / r.equity.iloc[0], label=name)
        ax[1].plot(r.equity.index, drawdown(r.equity), label=name)
    ax[0].set_title(title)
    ax[0].legend()
    ax[0].grid(alpha=.3)
    ax[0].set_ylabel("Equity (base 1)")
    ax[1].set_ylabel("Drawdown")
    ax[1].grid(alpha=.3)
    _finish(fig, path)


def plot_signals(df, res, name="", entry=2.0, path=None):
    sig = res.signals
    fig, ax = plt.subplots(2, 1, figsize=(13, 7), sharex=True)
    ax[0].plot(sig.index, sig["z"], lw=.9)
    for lv, c in ((0, "k"), (entry, "r"), (-entry, "g")):
        ax[0].axhline(lv, color=c, ls="--", lw=.8)
    pos = sig["signal"]
    ax[0].fill_between(sig.index, -5, 5, where=pos > 0, color="g", alpha=.12, label="long spread")
    ax[0].fill_between(sig.index, -5, 5, where=pos < 0, color="r", alpha=.12, label="short spread")
    ax[0].set_ylim(-5, 5)
    ax[0].set_title(f"Z-score and target positions — {name}")
    ax[0].legend()
    ax[0].grid(alpha=.3)
    ax[1].plot(res.equity.index, res.equity)
    ax[1].set_title("Equity")
    ax[1].grid(alpha=.3)
    _finish(fig, path)


def plot_cost_sensitivity(table, path=None):
    fig, ax = plt.subplots(figsize=(9, 5))
    for col in table.columns:
        ax.plot(table.index, table[col], marker="o", label=col)
    ax.axhline(0, color="k", lw=.8)
    ax.set_xlabel("One-way cost (bps: commission + slippage)")
    ax.set_ylabel("Sharpe")
    ax.set_title("Cost sensitivity")
    ax.legend()
    ax.grid(alpha=.3)
    _finish(fig, path)
