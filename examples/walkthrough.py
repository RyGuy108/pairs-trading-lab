"""A readable, executable tour of the statistical-arbitrage workflow."""

from pairs_trading.backtest import run_backtest
from pairs_trading.data import simulate_prices
from pairs_trading.model import select_candidate_pairs
from pairs_trading.research import chronological_split
from pairs_trading.strategy import StrategyConfig


def main() -> None:
    # Step 1: Work with synthetic data first. We know AAA/BBB is cointegrated,
    # which lets us debug the method before blaming imperfect market data.
    prices = simulate_prices(observations=1_000, seed=7)
    training, testing = chronological_split(prices, train_fraction=0.60)
    print(f"1. Observations: {len(training)} train, {len(testing)} test")

    # Step 2: Pair selection happens on training data only. Looking at the test
    # window while choosing a pair would introduce selection bias.
    candidates = select_candidate_pairs(training)
    print("2. Training-period candidates:")
    for model in candidates:
        print(
            f"   {model.symbol_y}/{model.symbol_x}: corr={model.correlation:.3f}, "
            f"coint p={model.cointegration_pvalue:.4f}, beta={model.hedge_ratio:.3f}"
        )

    if not candidates:
        print("No candidate passed. That is a valid research outcome.")
        return

    # Step 3: Freeze the estimated relationship, then trade only the unseen test
    # period. Cost assumptions are expressed in basis points of gross turnover.
    config = StrategyConfig(entry_z=2.0, exit_z=0.5, stop_z=4.0)
    backtest = run_backtest(testing, candidates[0], config)
    print("3. Out-of-sample metrics:")
    for metric, value in backtest.metrics.items():
        print(f"   {metric}: {value:.4f}")

    # Step 4: Inspect individual rows. target_position is today's decision;
    # held_position is shifted by one bar and earns the next observed return.
    columns = ["zscore", "target_position", "held_position", "net_return", "equity"]
    print("4. First signal rows:")
    active = backtest.history.loc[backtest.history["target_position"] != 0, columns]
    print(active.head(8).to_string())


if __name__ == "__main__":
    main()
