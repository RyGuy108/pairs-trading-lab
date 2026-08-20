"""Train/test orchestration for an honest pairs-trading experiment."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from pairs_trading.backtest import BacktestResult, run_backtest
from pairs_trading.data import validate_prices
from pairs_trading.model import PairModel, select_candidate_pairs
from pairs_trading.strategy import StrategyConfig


@dataclass(frozen=True)
class ResearchResult:
    training_prices: pd.DataFrame
    testing_prices: pd.DataFrame
    candidates: list[PairModel]
    backtests: list[BacktestResult]

    def summary(self) -> pd.DataFrame:
        records: list[dict[str, float | int | str]] = []
        for backtest in self.backtests:
            model = backtest.model
            records.append(
                {
                    "pair": f"{model.symbol_y}/{model.symbol_x}",
                    "symbol_y": model.symbol_y,
                    "symbol_x": model.symbol_x,
                    "train_correlation": model.correlation,
                    "cointegration_pvalue": model.cointegration_pvalue,
                    "hedge_ratio": model.hedge_ratio,
                    "half_life_days": model.half_life_days,
                    **backtest.metrics,
                }
            )
        return pd.DataFrame.from_records(records)


def chronological_split(
    prices: pd.DataFrame, train_fraction: float = 0.60
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split by time; random train/test splits are invalid for this problem."""

    clean = validate_prices(prices, minimum_rows=100)
    if not 0.50 <= train_fraction <= 0.85:
        raise ValueError("train_fraction must be between 0.50 and 0.85")
    split_at = int(len(clean) * train_fraction)
    training = clean.iloc[:split_at].copy()
    testing = clean.iloc[split_at:].copy()
    if len(testing) < 30:
        raise ValueError("the requested split leaves fewer than 30 test observations")
    return training, testing


def run_research(
    prices: pd.DataFrame,
    *,
    train_fraction: float = 0.60,
    minimum_correlation: float = 0.50,
    maximum_pvalue: float = 0.05,
    maximum_pairs: int = 5,
    config: StrategyConfig | None = None,
) -> ResearchResult:
    """Select and estimate pairs in-sample, then backtest only out-of-sample."""

    if maximum_pairs <= 0:
        raise ValueError("maximum_pairs must be positive")
    training, testing = chronological_split(prices, train_fraction)
    candidates = select_candidate_pairs(
        training,
        minimum_correlation=minimum_correlation,
        maximum_pvalue=maximum_pvalue,
    )[:maximum_pairs]
    backtests = [run_backtest(testing, model, config) for model in candidates]
    return ResearchResult(
        training_prices=training,
        testing_prices=testing,
        candidates=candidates,
        backtests=backtests,
    )
