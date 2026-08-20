import numpy as np

from pairs_trading.backtest import run_backtest
from pairs_trading.data import simulate_prices
from pairs_trading.model import fit_pair
from pairs_trading.strategy import StrategyConfig


def test_backtest_is_delayed_and_includes_costs() -> None:
    prices = simulate_prices(700, seed=7)
    model = fit_pair(prices.iloc[:400], "AAA", "BBB")
    config = StrategyConfig(transaction_cost_bps=1, slippage_bps=2)
    result = run_backtest(prices.iloc[400:], model, config)

    first_target = result.history.index[result.history["target_position"] != 0][0]
    target_location = result.history.index.get_loc(first_target)
    assert result.history["held_position"].iloc[target_location] == 0
    assert result.history["held_position"].iloc[target_location + 1] != 0
    assert np.isclose(
        result.history["trading_cost"].sum(),
        result.history["turnover"].sum() * 3 / 10_000,
    )
    assert np.isfinite(list(result.metrics.values())).all()
