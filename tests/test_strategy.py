import pandas as pd

from pairs_trading.strategy import StrategyConfig, generate_target_positions


def test_state_machine_enters_holds_exits_and_stops() -> None:
    zscores = pd.Series([0.0, -2.1, -1.2, -0.4, 0.0, 2.3, 1.0, 0.4, 4.2])
    positions = generate_target_positions(
        zscores, StrategyConfig(), force_final_exit=False
    )
    assert positions.tolist() == [0, 1, 1, 0, 0, -1, -1, 0, 0]


def test_final_position_is_closed_for_backtest_accounting() -> None:
    zscores = pd.Series([0.0, -2.1, -1.5])
    positions = generate_target_positions(zscores, StrategyConfig())
    assert positions.tolist() == [0, 1, 0]
