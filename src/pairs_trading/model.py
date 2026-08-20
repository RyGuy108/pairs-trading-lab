"""Pair discovery and hedge-ratio estimation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import combinations

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.stattools import coint

from pairs_trading.data import validate_prices


@dataclass(frozen=True)
class PairModel:
    """Parameters estimated on a training window only.

    The log-price relationship is

    ``log(Y) = intercept + hedge_ratio * log(X) + stationary residual``.
    """

    symbol_y: str
    symbol_x: str
    intercept: float
    hedge_ratio: float
    correlation: float
    cointegration_pvalue: float
    spread_mean: float
    spread_std: float
    half_life_days: float
    observations: int

    def spread(self, prices: pd.DataFrame) -> pd.Series:
        log_y = np.log(prices[self.symbol_y])
        log_x = np.log(prices[self.symbol_x])
        return (log_y - self.intercept - self.hedge_ratio * log_x).rename("spread")

    def zscore(self, prices: pd.DataFrame) -> pd.Series:
        return ((self.spread(prices) - self.spread_mean) / self.spread_std).rename("zscore")

    def to_dict(self) -> dict[str, float | int | str]:
        return asdict(self)


def estimate_half_life(spread: pd.Series) -> float:
    """Estimate OU-style mean-reversion half-life from an AR(1) approximation."""

    lagged = spread.shift(1).dropna()
    changes = spread.diff().dropna()
    aligned = pd.concat([lagged.rename("lagged"), changes.rename("change")], axis=1).dropna()
    if len(aligned) < 3:
        return float("inf")

    x = np.column_stack([np.ones(len(aligned)), aligned["lagged"].to_numpy()])
    slope = float(np.linalg.lstsq(x, aligned["change"].to_numpy(), rcond=None)[0][1])
    if slope >= 0:
        return float("inf")
    return float(-np.log(2.0) / slope)


def fit_pair(prices: pd.DataFrame, symbol_y: str, symbol_x: str) -> PairModel:
    """Fit one pair using regression and the Engle-Granger cointegration test."""

    if symbol_y == symbol_x:
        raise ValueError("a pair requires two different symbols")
    missing = {symbol_y, symbol_x} - set(prices.columns)
    if missing:
        raise KeyError(f"missing price columns: {', '.join(sorted(missing))}")

    clean = validate_prices(prices[[symbol_y, symbol_x]], minimum_rows=30)
    log_prices = np.log(clean)
    log_y = log_prices[symbol_y]
    log_x = log_prices[symbol_x]

    regression = LinearRegression().fit(log_x.to_numpy().reshape(-1, 1), log_y.to_numpy())
    intercept = float(regression.intercept_)
    hedge_ratio = float(regression.coef_[0])
    spread = log_y - intercept - hedge_ratio * log_x
    spread_std = float(spread.std(ddof=1))
    if spread_std <= 1e-12:
        raise ValueError("spread variance is too small to produce meaningful z-scores")

    returns = log_prices.diff().dropna()
    correlation = float(returns[symbol_y].corr(returns[symbol_x]))
    _, pvalue, _ = coint(log_y, log_x, trend="c", autolag="aic")

    return PairModel(
        symbol_y=symbol_y,
        symbol_x=symbol_x,
        intercept=intercept,
        hedge_ratio=hedge_ratio,
        correlation=correlation,
        cointegration_pvalue=float(pvalue),
        spread_mean=float(spread.mean()),
        spread_std=spread_std,
        half_life_days=estimate_half_life(spread),
        observations=len(clean),
    )


def select_candidate_pairs(
    prices: pd.DataFrame,
    *,
    minimum_correlation: float = 0.50,
    maximum_pvalue: float = 0.05,
) -> list[PairModel]:
    """Prefilter on return correlation, then retain cointegrated pairs.

    Correlation is a similarity measure, not evidence of a stable spread. The
    Engle-Granger test is therefore applied after the cheaper correlation filter.
    Both decisions must be made on training data, never the test period.
    """

    if not 0 <= minimum_correlation <= 1:
        raise ValueError("minimum_correlation must be between 0 and 1")
    if not 0 < maximum_pvalue < 1:
        raise ValueError("maximum_pvalue must be between 0 and 1")

    clean = validate_prices(prices)
    log_returns = np.log(clean).diff().dropna()
    candidates: list[PairModel] = []
    for symbol_y, symbol_x in combinations(clean.columns, 2):
        correlation = float(log_returns[symbol_y].corr(log_returns[symbol_x]))
        if not np.isfinite(correlation) or abs(correlation) < minimum_correlation:
            continue
        model = fit_pair(clean, symbol_y, symbol_x)
        if model.cointegration_pvalue <= maximum_pvalue:
            candidates.append(model)

    return sorted(
        candidates,
        key=lambda model: (model.cointegration_pvalue, -abs(model.correlation)),
    )
