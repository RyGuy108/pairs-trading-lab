import numpy as np

from pairs_trading.data import simulate_prices
from pairs_trading.execution import build_entry_order_plan
from pairs_trading.model import fit_pair


def test_order_plan_has_opposite_sides_and_requested_gross_notional() -> None:
    prices = simulate_prices(400, seed=7)
    model = fit_pair(prices.iloc[:300], "AAA", "BBB")
    plan = build_entry_order_plan(model, prices.iloc[-1], signal=1, gross_notional=1_000)
    assert [order.side for order in plan] == ["buy", "sell"]
    assert np.isclose(sum(order.target_notional for order in plan), 1_000)
    assert all(order.quantity > 0 for order in plan)
