"""Pair selection with multiple-testing adjustments."""
from __future__ import annotations
import itertools
import numpy as np
import pandas as pd
from .stats import ols, engle_granger, ou_params, benjamini_hochberg


def screen_pairs(close: pd.DataFrame, window=504, log=True, q=0.05) -> pd.DataFrame:
    """Screen the final formation window, correcting direction selection before applying BH."""
    c = np.log(close.tail(window)) if log else close.tail(window)
    rets = close.tail(window).pct_change(fill_method=None).dropna()
    rows = []
    for a, b in itertools.combinations(c.columns, 2):
        for y, x in ((a, b), (b, a)):
            stat, p, _ = engle_granger(c[y].values, c[x].values)
            if y == a:
                best = (y, x, stat, p)
            elif p < best[3]:
                best = (y, x, stat, p)
        y, x, stat, p = best
        alpha, beta = ols(c[y].values, c[x].values)
        s = c[y].values - alpha - beta * c[x].values
        o = ou_params(s)
        rows.append(dict(y=y, x=x, corr_ret=rets[y].corr(rets[x]), beta=beta,
                         eg_stat=stat, p_value=p, half_life=o["half_life"]))
    out = pd.DataFrame(rows, columns=["y", "x", "corr_ret", "beta", "eg_stat", "p_value", "half_life"])
    # Bonferroni accounts for selecting the better of two regression directions.
    out["p_pair"] = np.minimum(2 * out["p_value"], 1.0)

    out["p_adj_bh"], out["reject_fdr"] = benjamini_hochberg(out["p_pair"].values, q)
    return out.sort_values("p_value").reset_index(drop=True)
