#!/usr/bin/env python
"""Run four pair strategies and write performance tables and charts."""
import argparse
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import pandas as pd
from pairs import (STRATEGIES, BacktestConfig, run_backtest, load_pair, synthetic_pair,
                   walkforward_optimize)
from pairs.plots import plot_equity, plot_signals, plot_cost_sensitivity

GRIDS = {
    "relative_return": dict(lookback=[10, 20, 40], entry=[1.5, 2.0, 2.5]),
    "cointegration": dict(window=[126, 252], entry=[1.5, 2.0, 2.5]),
    "ou": dict(window=[126, 252], entry=[1.5, 2.0, 2.5]),
    "kalman": dict(delta=[1e-8, 1e-7, 1e-6], entry=[1.5, 2.0, 2.5]),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--x", default="PETR3.SA")
    ap.add_argument("--y", default="PETR4.SA")
    ap.add_argument("--start", default="2020-01-01")
    ap.add_argument("--end", default=None)
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--walkforward", action="store_true")
    ap.add_argument("--capital", type=float, default=100_000)
    ap.add_argument("--commission-bps", type=float, default=3)
    ap.add_argument("--slippage-bps", type=float, default=5)
    ap.add_argument("--borrow-bps", type=float, default=200)
    ap.add_argument("--lot", type=int, default=1)
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    df = synthetic_pair() if a.synthetic else load_pair(a.x, a.y, a.start, a.end)
    print(f"{len(df)} sessions: {df.index[0].date()} -> {df.index[-1].date()}")
    cfg = BacktestConfig(capital=a.capital, commission_bps=a.commission_bps,
                         slippage_bps=a.slippage_bps, borrow_bps_annual=a.borrow_bps, lot_size=a.lot)

    results, rows = {}, {}
    for name, fn in STRATEGIES.items():
        sig = fn(df["close_x"], df["close_y"])
        res = run_backtest(df, sig, cfg)
        results[name] = res
        rows[name] = res.summary()
        res.trades.to_csv(out / f"trades_{name}.csv", index=False)
        plot_signals(df, res, name, path=out / f"signals_{name}.png")
    table = pd.DataFrame(rows).T
    table.to_csv(out / "summary.csv")
    plot_equity(results, out / "equity.png")
    print("\n=== Summary (default parameters, after costs) ===")
    print(table.round(3).to_string())

    sens = {}
    for bps in [0, 2.5, 5, 10, 20]:
        c = BacktestConfig(**{**cfg.__dict__, "commission_bps": bps / 2, "slippage_bps": bps / 2})
        sens[bps] = {n: run_backtest(df, results[n].signals, c).summary().get("sharpe")
                     for n in STRATEGIES}
    sens = pd.DataFrame(sens).T
    sens.to_csv(out / "cost_sensitivity.csv")
    plot_cost_sensitivity(sens, out / "cost_sensitivity.png")
    print("\nSharpe vs one-way cost (bps):\n", sens.round(2).to_string())

    if a.walkforward:
        print("\n=== Walk-forward (504 training / 126 test sessions) ===")
        wf_rows, wf_res = {}, {}
        for name, fn in STRATEGIES.items():
            wf = walkforward_optimize(df, fn, GRIDS[name], cfg)
            wf["folds"].to_csv(out / f"wf_folds_{name}.csv", index=False)
            wf_rows[name] = wf["summary"]
            wf_res[name] = wf
        print(pd.DataFrame(wf_rows).T.round(3).to_string())
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(12, 5))
        for n, w in wf_res.items():
            ax.plot(w["oos_equity"].index, w["oos_equity"] / w["oos_equity"].iloc[0], label=n)
        ax.set_title("Walk-forward: out-of-sample equity")
        ax.legend()
        ax.grid(alpha=.3)
        fig.savefig(out / "walkforward_equity.png", dpi=130)
        plt.close(fig)
    print(f"\nFiles saved to {out}/")

if __name__ == "__main__":
    main()
