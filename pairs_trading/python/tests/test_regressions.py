"""Regression coverage for execution, risk metrics and fold boundaries."""
from dataclasses import replace
import unittest
import numpy as np
import pandas as pd
from pairs.backtest import BacktestConfig, run_backtest
from pairs.metrics import performance_summary
from pairs.walkforward import walkforward_optimize



class ExecutionTests(unittest.TestCase):
    def setUp(self):
        self.index = pd.bdate_range('2024-01-01', periods=6)
        self.prices = pd.DataFrame(100.0, index=self.index,
                                   columns=['open_x', 'close_x', 'open_y', 'close_y'])
        self.signals = pd.DataFrame({'signal': 1, 'hedge_ratio': 1.0}, index=self.index)
        self.cfg = BacktestConfig(capital=10000, commission_bps=10,
                                  slippage_bps=0, borrow_bps_annual=0)

    def test_final_liquidation_charges_both_sides(self):
        result = run_backtest(self.prices, self.signals, self.cfg)
        self.assertAlmostEqual(result.equity.iloc[-1], 9980)
        self.assertEqual(len(result.trades), 1)
        self.assertAlmostEqual(result.trades.pnl.sum(), -20)
        self.assertEqual(result.positions.iloc[-1].abs().sum(), 0)
        self.assertEqual(result.traded_notional.sum(), 20000)

    def test_optional_open_position(self):
        result = run_backtest(self.prices, self.signals,
                              replace(self.cfg, liquidate_at_end=False))
        self.assertEqual(len(result.trades), 0)
        self.assertAlmostEqual(result.equity.iloc[-1], 9990)
        self.assertGreater(result.positions.iloc[-1].abs().sum(), 0)

    def test_no_same_bar_execution(self):
        result = run_backtest(self.prices, self.signals, self.cfg)
        self.assertEqual(result.positions.iloc[0].abs().sum(), 0)
        self.assertGreater(result.positions.iloc[1].abs().sum(), 0)

    def test_last_bar_does_not_open_a_trade(self):
        self.signals['signal'] = 0
        self.signals.iloc[-2, 0] = 1
        result = run_backtest(self.prices, self.signals, self.cfg)
        self.assertTrue(result.trades.empty)
        self.assertEqual(result.equity.iloc[-1], 10000)

    def test_reject_invalid_prices(self):
        self.prices.iloc[1, 0] = np.nan
        with self.assertRaises(ValueError):
            run_backtest(self.prices, self.signals, self.cfg)

    def test_reject_misaligned_signals(self):
        with self.assertRaises(ValueError):
            run_backtest(self.prices, self.signals.iloc[1:], self.cfg)

    def test_reject_invalid_signals(self):
        self.signals.iloc[1, 0] = 2
        with self.assertRaises(ValueError):
            run_backtest(self.prices, self.signals, self.cfg)

    def test_reject_negative_hedge(self):
        self.signals['hedge_ratio'] = -1.0
        with self.assertRaises(ValueError):
            run_backtest(self.prices, self.signals, self.cfg)

    def test_reject_unknown_fill(self):
        with self.assertRaises(ValueError):
            BacktestConfig(fill='open')

    def test_sortino_uses_all_observations(self):
        returns = np.array([0.02, -0.01, 0.03, -0.02])
        equity = pd.Series(np.r_[100, 100 * np.cumprod(1 + returns)])
        expected = returns.mean() / np.sqrt(np.mean(np.minimum(returns, 0)**2)) * np.sqrt(252)
        self.assertAlmostEqual(performance_summary(equity)['sortino'], expected)

    def test_no_candidate_leaves_fold_in_cash(self):
        def strategy(x, y):
            return pd.DataFrame({'signal': 1, 'hedge_ratio': 1.0}, index=x.index)
        result = walkforward_optimize(self.prices, strategy, {}, self.cfg,
                                      train_len=2, test_len=2, warmup=0, min_trades=5)
        self.assertTrue((result['oos_equity'] == self.cfg.capital).all())
        self.assertFalse(result['folds']['selected'].any())

    def test_training_does_not_receive_test_prices(self):
        seen = []
        def strategy(x, y):
            seen.append(x.index[-1])
            return pd.DataFrame({'signal': 1, 'hedge_ratio': 1.0}, index=x.index)
        walkforward_optimize(self.prices, strategy, {}, self.cfg,
                             train_len=3, test_len=3, warmup=0, min_trades=0)
        self.assertEqual(seen[0], self.index[2])


if __name__ == '__main__':
    unittest.main()
