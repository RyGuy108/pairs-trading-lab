"""Persist research tables and compact diagnostic plots."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

# Sandboxed or containerized environments may have a read-only home directory.
# A stable temporary cache avoids repeated font discovery and noisy warnings.
_matplotlib_cache = Path(tempfile.gettempdir()) / "pairs-trading-matplotlib"
_matplotlib_cache.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_matplotlib_cache))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from pairs_trading.backtest import BacktestResult  # noqa: E402
from pairs_trading.research import ResearchResult  # noqa: E402


def plot_backtest(backtest: BacktestResult, path: str | Path) -> Path:
    """Plot the spread signal, position, and out-of-sample equity curve."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    history = backtest.history
    config = backtest.config
    pair_name = f"{backtest.model.symbol_y}/{backtest.model.symbol_x}"

    figure, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
    axes[0].plot(history.index, history["zscore"], color="#235789", linewidth=1.2)
    axes[0].axhline(config.entry_z, color="#c1292e", linestyle="--", linewidth=0.9)
    axes[0].axhline(-config.entry_z, color="#2e8b57", linestyle="--", linewidth=0.9)
    axes[0].axhline(0, color="black", linewidth=0.7)
    axes[0].set_ylabel("Spread z-score")
    axes[0].set_title(f"{pair_name}: fixed-parameter out-of-sample backtest")

    axes[1].step(
        history.index,
        history["held_position"],
        where="post",
        color="#f18f01",
        linewidth=1.2,
    )
    axes[1].set_ylabel("Position")
    axes[1].set_yticks([-1, 0, 1])

    axes[2].plot(history.index, history["equity"], color="#2e8b57", linewidth=1.4)
    axes[2].set_ylabel("Growth of $1")
    axes[2].set_xlabel("Date")
    axes[2].grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(destination, dpi=150)
    plt.close(figure)
    return destination


def write_research_outputs(result: ResearchResult, output_dir: str | Path) -> list[Path]:
    """Write inspectable CSVs and one chart for each selected pair."""

    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    summary_path = directory / "summary.csv"
    result.summary().to_csv(summary_path, index=False)
    written.append(summary_path)

    for backtest in result.backtests:
        pair_slug = f"{backtest.model.symbol_y}_{backtest.model.symbol_x}".lower()
        history_path = directory / f"{pair_slug}_history.csv"
        chart_path = directory / f"{pair_slug}_diagnostics.png"
        backtest.history.to_csv(history_path)
        plot_backtest(backtest, chart_path)
        written.extend([history_path, chart_path])
    return written


def format_summary(summary: pd.DataFrame) -> str:
    if summary.empty:
        return "No pair passed the training-period filters."
    display = summary.copy()
    percentage_columns = [
        "total_return",
        "annualized_return",
        "annualized_volatility",
        "max_drawdown",
        "exposure_fraction",
    ]
    for column in percentage_columns:
        display[column] = display[column].map(lambda value: f"{value:.2%}")
    numeric_columns = [
        "train_correlation",
        "cointegration_pvalue",
        "hedge_ratio",
        "half_life_days",
        "sharpe_ratio",
    ]
    for column in numeric_columns:
        display[column] = display[column].map(lambda value: f"{value:.3f}")
    return display.to_string(index=False)
