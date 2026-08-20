"""Leakage-aware two-leg backtesting and performance metrics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from pairs_trading.data import validate_prices
from pairs_trading.model import PairModel
from pairs_trading.strategy import StrategyConfig, generate_target_positions


@dataclass(frozen=True)
class BacktestResult:
    model: PairModel
    config: StrategyConfig
    history: pd.DataFrame
    metrics: dict[str, float]


def _performance_metrics(history: pd.DataFrame, annualization_factor: int) -> dict[str, float]:
    returns = history["net_return"]
    equity = history["equity"]
    total_return = float(equity.iloc[-1] - 1.0)
    years = len(returns) / annualization_factor
    annual_return = float(equity.iloc[-1] ** (1.0 / years) - 1.0) if years > 0 else 0.0
    annual_volatility = float(returns.std(ddof=1) * np.sqrt(annualization_factor))
    sharpe = (
        float(returns.mean() / returns.std(ddof=1) * np.sqrt(annualization_factor))
        if returns.std(ddof=1) > 0
        else 0.0
    )
    entries = int(
        (
            (history["target_position"] != 0)
            & (history["target_position"].shift(1).fillna(0) == 0)
        ).sum()
    )
    active_returns = returns[returns != 0]

    return {
        "total_return": total_return,
        "annualized_return": annual_return,
        "annualized_volatility": annual_volatility,
        "sharpe_ratio": sharpe,
        "max_drawdown": float(history["drawdown"].min()),
        "trades": float(entries),
        "exposure_fraction": float((history["held_position"] != 0).mean()),
        "positive_day_fraction": (
            float((active_returns > 0).mean()) if not active_returns.empty else 0.0
        ),
        "total_cost": float(history["trading_cost"].sum()),
    }


def run_backtest(
    prices: pd.DataFrame,
    model: PairModel,
    config: StrategyConfig | None = None,
) -> BacktestResult:
    """Backtest fixed training parameters on unseen prices.

    A z-score observed at timestamp ``t`` becomes a held position at ``t+1``.
    This one-bar lag prevents the strategy from earning the same return that
    generated its signal.
    """

    config = config or StrategyConfig()
    clean = validate_prices(prices[[model.symbol_y, model.symbol_x]], minimum_rows=3)
    zscore = model.zscore(clean)
    spread = model.spread(clean)
    target = generate_target_positions(zscore, config)
    held = target.shift(1).fillna(0).astype(int).rename("held_position")

    simple_returns = clean.pct_change().fillna(0.0)
    gross_scale = 1.0 + abs(model.hedge_ratio)
    weight_y = 1.0 / gross_scale
    weight_x = abs(model.hedge_ratio) / gross_scale
    x_direction = -np.sign(model.hedge_ratio) if model.hedge_ratio != 0 else 0.0

    pair_return = (
        weight_y * simple_returns[model.symbol_y]
        + x_direction * weight_x * simple_returns[model.symbol_x]
    )
    gross_return = (held * pair_return).rename("gross_return")
    turnover = target.diff().abs().fillna(target.abs()).astype(float).rename("turnover")
    trading_cost = (turnover * config.total_cost_rate).rename("trading_cost")
    net_return = (gross_return - trading_cost).rename("net_return")
    equity = (1.0 + net_return).cumprod().rename("equity")
    drawdown = (equity / equity.cummax() - 1.0).rename("drawdown")

    history = pd.concat(
        [
            clean.rename(
                columns={model.symbol_y: "price_y", model.symbol_x: "price_x"}
            ),
            spread,
            zscore,
            target,
            held,
            pair_return.rename("unpositioned_pair_return"),
            gross_return,
            turnover,
            trading_cost,
            net_return,
            equity,
            drawdown,
        ],
        axis=1,
    )
    metrics = _performance_metrics(history, config.annualization_factor)
    return BacktestResult(model=model, config=config, history=history, metrics=metrics)
