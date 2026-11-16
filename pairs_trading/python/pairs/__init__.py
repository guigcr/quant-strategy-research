"""Pairs trading research utilities."""
from .stats import ols, adf, engle_granger, fit_ar1, ou_params, benjamini_hochberg
from .strategies import STRATEGIES, relative_return, cointegration, ou, kalman
from .backtest import BacktestConfig, BacktestResult, run_backtest
from .metrics import performance_summary, trade_summary
from .walkforward import walkforward_optimize, grid_search
from .screening import screen_pairs
from .data import load_pair, load_universe, synthetic_pair
