from pairs_trading.data import simulate_prices
from pairs_trading.model import fit_pair, select_candidate_pairs


def test_fit_recovers_known_cointegrating_relationship() -> None:
    prices = simulate_prices(800, seed=7)
    model = fit_pair(prices.iloc[:500], "AAA", "BBB")
    assert 0.90 < model.hedge_ratio < 1.25
    assert model.cointegration_pvalue < 0.05
    assert model.half_life_days > 0


def test_pair_selection_finds_the_known_pair() -> None:
    prices = simulate_prices(800, seed=7)
    candidates = select_candidate_pairs(prices.iloc[:500])
    symbols = [{candidate.symbol_y, candidate.symbol_x} for candidate in candidates]
    assert {"AAA", "BBB"} in symbols
