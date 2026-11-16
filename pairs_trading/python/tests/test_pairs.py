import numpy as np
import pandas as pd
import pytest
from pairs import (STRATEGIES, BacktestConfig, run_backtest, synthetic_pair, ols,
                   ou_params, benjamini_hochberg, engle_granger)

DF = synthetic_pair(n=700, seed=1)
ZERO = BacktestConfig(commission_bps=0, slippage_bps=0, borrow_bps_annual=0)


def test_ols_recovers_parameters():
    rng = np.random.default_rng(0)
    x = rng.normal(size=500)
    y = 0.5 + 1.2 * x + rng.normal(0, 0.01, 500)
    a, b = ols(y, x)
    assert abs(a - 0.5) < 0.01 and abs(b - 1.2) < 0.01


def test_ou_recovers_half_life():
    rng = np.random.default_rng(1)
    kappa, n = 0.1, 20000
    phi = np.exp(-kappa)
    s = np.zeros(n)
    for t in range(1, n):
        s[t] = phi * s[t - 1] + rng.normal(0, 0.1)
    est = ou_params(s)
    assert abs(est["half_life"] - np.log(2) / kappa) < 0.5
    assert est["sigma_eq"] == pytest.approx(0.1 / np.sqrt(1 - phi ** 2), rel=0.05)


def test_eg_distinguishes_cointegration_from_random_walk():
    a = synthetic_pair(n=1000, cointegrated=True, seed=3)
    b = synthetic_pair(n=1000, cointegrated=False, seed=3)
    pa = engle_granger(np.log(a.close_y), np.log(a.close_x))[1]
    pb = engle_granger(np.log(b.close_y), np.log(b.close_x))[1]
    assert pa < 0.05 < pb


def test_benjamini_hochberg_monotonic():
    adj, rej = benjamini_hochberg([0.001, 0.01, 0.04, 0.5, 0.9])
    assert (adj >= np.array([0.001, 0.01, 0.04, 0.5, 0.9])).all() and rej[0]

@pytest.mark.parametrize("name", list(STRATEGIES))

def test_strategies_are_causal(name):
    """Verify strategies are causal."""
    fn = STRATEGIES[name]
    cut = 500
    full = fn(DF.close_x, DF.close_y)
    alt = DF.copy()
    alt.iloc[cut:, alt.columns.get_indexer(["close_x", "close_y"])] *= 3.0
    cut_sig = fn(alt.close_x, alt.close_y)
    pd.testing.assert_frame_equal(full.iloc[:cut].fillna(0), cut_sig.iloc[:cut].fillna(0))


def test_dollar_neutral_trade_pnl():

    idx = pd.bdate_range("2024-01-01", periods=4)
    df = pd.DataFrame({"open_x": [50, 50, 48, 48.], "close_x": [50, 50, 48, 48.],
                       "open_y": [20, 20, 22, 22.], "close_y": [20, 20, 22, 22.]}, idx)
    sig = pd.DataFrame({"signal": [1, 0, 0, 0.], "hedge_ratio": [1.0] * 4, "z": 0.0}, idx)

    sig.loc[idx[1], "signal"] = 1
    sig.loc[idx[2], "signal"] = 0
    cfg = BacktestConfig(capital=5000, leverage=2.0, commission_bps=0, slippage_bps=0,
                         borrow_bps_annual=0, compound=False)
    res = run_backtest(df, sig, cfg)
    tr = res.trades.iloc[0]
    assert tr.pnl == pytest.approx(700)


def test_costs_reduce_equity():
    sig = STRATEGIES["cointegration"](DF.close_x, DF.close_y)
    g = run_backtest(DF, sig, ZERO).equity.iloc[-1]
    n = run_backtest(DF, sig, BacktestConfig()).equity.iloc[-1]
    assert n < g


def test_sem_signals_patrimonio_constante():
    sig = pd.DataFrame({"signal": 0.0, "hedge_ratio": 1.0, "z": 0.0}, DF.index)
    res = run_backtest(DF, sig, BacktestConfig())
    assert res.equity.nunique() == 1 and len(res.trades) == 0


def test_dollar_sizing_is_neutral():
    sig = STRATEGIES["cointegration"](DF.close_x, DF.close_y, sizing="dollar", pmax=1.0)
    res = run_backtest(DF, sig, ZERO)
    pos = res.positions
    entries = pos.index[(pos.qx != 0) & (pos.qx.shift(1).fillna(0) == 0)]
    assert len(entries) > 0
    r = (-pos.loc[entries, "qx"] * DF.loc[entries, "open_x"]) / (pos.loc[entries, "qy"] * DF.loc[entries, "open_y"])
    assert np.allclose(r, 1.0, atol=1e-3)



def test_holding_limit_emits_flat_signal():
    from pairs.strategies import _state_machine
    signals = _state_machine(np.full(6, 2.5), 2.0, 0.5, 4.0,
                             np.ones(6, dtype=bool), 2)
    np.testing.assert_array_equal(signals, [-1, -1, 0, -1, -1, 0])



def test_screening_adjusts_direction_selection(monkeypatch):
    import pairs.screening as screening
    monkeypatch.setattr(screening, 'engle_granger', lambda y, x: (-3, 0.03, None))
    result = screening.screen_pairs(DF[['close_x', 'close_y']])
    assert result.p_pair.iloc[0] == pytest.approx(0.06)
    assert result.p_adj_bh.iloc[0] == pytest.approx(0.06)
    assert not result.reject_fdr.iloc[0]



def test_screening_single_asset_is_empty():
    from pairs.screening import screen_pairs
    assert screen_pairs(DF[['close_x']]).empty
