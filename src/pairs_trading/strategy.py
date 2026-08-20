"""Translate spread z-scores into a stateful trading signal."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class StrategyConfig:
    """Strategy and cost assumptions.

    Position ``+1`` means long the spread (long Y, short the beta-adjusted X leg).
    Position ``-1`` means short the spread.
    """

    entry_z: float = 2.0
    exit_z: float = 0.5
    stop_z: float = 4.0
    transaction_cost_bps: float = 1.0
    slippage_bps: float = 2.0
    annualization_factor: int = 252

    def __post_init__(self) -> None:
        if not 0 <= self.exit_z < self.entry_z < self.stop_z:
            raise ValueError("thresholds must satisfy 0 <= exit_z < entry_z < stop_z")
        if self.transaction_cost_bps < 0 or self.slippage_bps < 0:
            raise ValueError("cost assumptions cannot be negative")
        if self.annualization_factor <= 0:
            raise ValueError("annualization_factor must be positive")

    @property
    def total_cost_rate(self) -> float:
        return (self.transaction_cost_bps + self.slippage_bps) / 10_000.0


def signal_for_flat_position(zscore: float, config: StrategyConfig) -> int:
    """Return a possible entry signal when no pair position is currently held."""

    if not np.isfinite(zscore) or abs(zscore) >= config.stop_z:
        return 0
    if zscore >= config.entry_z:
        return -1
    if zscore <= -config.entry_z:
        return 1
    return 0


def generate_target_positions(
    zscores: pd.Series,
    config: StrategyConfig,
    *,
    force_final_exit: bool = True,
) -> pd.Series:
    """Run the entry/exit state machine without using future observations."""

    current = 0
    positions: list[int] = []
    for zscore in zscores.astype(float):
        if not np.isfinite(zscore):
            current = 0
        elif current == 0:
            current = signal_for_flat_position(zscore, config)
        elif abs(zscore) >= config.stop_z or abs(zscore) <= config.exit_z:
            current = 0
        positions.append(current)

    target = pd.Series(positions, index=zscores.index, dtype=int, name="target_position")
    if force_final_exit and not target.empty:
        target.iloc[-1] = 0
    return target
