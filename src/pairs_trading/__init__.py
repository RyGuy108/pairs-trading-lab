"""Educational pairs-trading research and paper-execution package."""

from pairs_trading.backtest import BacktestResult, run_backtest
from pairs_trading.model import PairModel, fit_pair, select_candidate_pairs
from pairs_trading.strategy import StrategyConfig, generate_target_positions

__all__ = [
    "BacktestResult",
    "PairModel",
    "StrategyConfig",
    "fit_pair",
    "generate_target_positions",
    "run_backtest",
    "select_candidate_pairs",
]
