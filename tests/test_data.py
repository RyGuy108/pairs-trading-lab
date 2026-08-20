import numpy as np
import pandas as pd
import pytest

from pairs_trading.data import simulate_prices, validate_prices


def test_simulated_prices_are_positive_and_reproducible() -> None:
    first = simulate_prices(200, seed=11)
    second = simulate_prices(200, seed=11)
    pd.testing.assert_frame_equal(first, second)
    assert np.isfinite(first.to_numpy()).all()
    assert (first > 0).all().all()


def test_validation_rejects_nonpositive_prices() -> None:
    prices = simulate_prices(100)[["AAA", "BBB"]]
    prices.iloc[5, 0] = 0
    with pytest.raises(ValueError, match="strictly positive"):
        validate_prices(prices)
