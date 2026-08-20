"""Command-line interface for the learning workflow."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

from pairs_trading.data import (
    download_alpaca_daily_bars,
    load_price_csv,
    save_prices,
    simulate_prices,
)
from pairs_trading.execution import (
    build_entry_order_plan,
    order_plan_frame,
    submit_paper_order_plan,
)
from pairs_trading.model import fit_pair
from pairs_trading.research import run_research
from pairs_trading.strategy import StrategyConfig, signal_for_flat_position


def _strategy_config(args: argparse.Namespace) -> StrategyConfig:
    return StrategyConfig(
        entry_z=args.entry_z,
        exit_z=args.exit_z,
        stop_z=args.stop_z,
        transaction_cost_bps=args.transaction_cost_bps,
        slippage_bps=args.slippage_bps,
    )


def _add_strategy_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--entry-z", type=float, default=2.0)
    parser.add_argument("--exit-z", type=float, default=0.5)
    parser.add_argument("--stop-z", type=float, default=4.0)
    parser.add_argument("--transaction-cost-bps", type=float, default=1.0)
    parser.add_argument("--slippage-bps", type=float, default=2.0)


def _run_research_command(args: argparse.Namespace, *, simulated: bool) -> int:
    # Plotting is imported only for research commands. Data downloads and paper
    # order planning should not pay Matplotlib's startup/font-cache cost.
    from pairs_trading.reporting import format_summary, write_research_outputs

    if simulated:
        prices = simulate_prices(observations=args.observations, seed=args.seed)
        data_path = Path(args.output_dir) / "synthetic_prices.csv"
        save_prices(prices, data_path)
        print(f"Saved deterministic learning data to {data_path}")
    else:
        prices = load_price_csv(args.csv)

    result = run_research(
        prices,
        train_fraction=args.train_fraction,
        minimum_correlation=args.minimum_correlation,
        maximum_pvalue=args.maximum_pvalue,
        maximum_pairs=args.maximum_pairs,
        config=_strategy_config(args),
    )
    print("\nOut-of-sample research summary")
    print(format_summary(result.summary()))
    written = write_research_outputs(result, args.output_dir)
    print(f"\nWrote {len(written)} research artifacts to {args.output_dir}")
    if not result.candidates:
        print(
            "Try a larger universe/window, but do not relax filters merely "
            "to manufacture a trade."
        )
        return 2
    return 0


def _download(args: argparse.Namespace) -> int:
    prices = download_alpaca_daily_bars(
        args.symbols,
        datetime.fromisoformat(args.start),
        datetime.fromisoformat(args.end),
        feed=args.feed,
    )
    destination = save_prices(prices, args.output)
    print(f"Saved {len(prices)} complete rows for {len(prices.columns)} symbols to {destination}")
    return 0


def _paper_plan(args: argparse.Namespace) -> int:
    prices = load_price_csv(args.csv)[[args.symbol_y, args.symbol_x]]
    if len(prices) < args.lookback + 1:
        raise ValueError(f"need at least {args.lookback + 1} rows for the requested lookback")

    training = prices.iloc[-(args.lookback + 1) : -1]
    latest = prices.iloc[-1]
    model = fit_pair(training, args.symbol_y, args.symbol_x)
    latest_z = float(model.zscore(prices.iloc[[-1]]).iloc[0])
    config = _strategy_config(args)
    signal = signal_for_flat_position(latest_z, config)
    if args.signal == "long":
        signal = 1
    elif args.signal == "short":
        signal = -1

    print(f"Latest fixed-parameter z-score: {latest_z:.3f}")
    if signal == 0:
        print("No entry: the latest spread is inside the configured entry/stop boundaries.")
        return 0

    plan = build_entry_order_plan(model, latest, signal, args.gross_notional)
    print("\nPaper order plan")
    print(order_plan_frame(plan).to_string(index=False))
    print("\nWarning: pair legs are sequential orders and are not an atomic transaction.")

    if args.submit_paper:
        responses = submit_paper_order_plan(plan)
        for response in responses:
            print(
                f"submitted paper order id={response.id} symbol={response.symbol} "
                f"status={response.status}"
            )
    else:
        print("Dry run only. Add --submit-paper to send these orders to an Alpaca paper account.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pairs-trading",
        description="Learn pair selection, signal design, backtesting, and paper execution.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    demo = subparsers.add_parser("demo", help="run the entire workflow on known synthetic data")
    demo.add_argument("--observations", type=int, default=1_000)
    demo.add_argument("--seed", type=int, default=7)
    demo.add_argument("--output-dir", default="outputs/demo")
    demo.add_argument("--train-fraction", type=float, default=0.60)
    demo.add_argument("--minimum-correlation", type=float, default=0.50)
    demo.add_argument("--maximum-pvalue", type=float, default=0.05)
    demo.add_argument("--maximum-pairs", type=int, default=5)
    _add_strategy_arguments(demo)
    demo.set_defaults(handler=lambda args: _run_research_command(args, simulated=True))

    research = subparsers.add_parser("research", help="research a wide timestamp-by-symbol CSV")
    research.add_argument("--csv", required=True)
    research.add_argument("--output-dir", default="outputs/research")
    research.add_argument("--train-fraction", type=float, default=0.60)
    research.add_argument("--minimum-correlation", type=float, default=0.50)
    research.add_argument("--maximum-pvalue", type=float, default=0.05)
    research.add_argument("--maximum-pairs", type=int, default=5)
    _add_strategy_arguments(research)
    research.set_defaults(handler=lambda args: _run_research_command(args, simulated=False))

    download = subparsers.add_parser("download", help="download adjusted daily bars from Alpaca")
    download.add_argument("--symbols", nargs="+", required=True)
    download.add_argument("--start", required=True, help="ISO date, for example 2021-01-01")
    download.add_argument("--end", required=True, help="ISO date, for example 2025-01-01")
    download.add_argument("--feed", choices=["iex", "sip"], default="iex")
    download.add_argument("--output", default="data/downloaded/prices.csv")
    download.set_defaults(handler=_download)

    paper = subparsers.add_parser(
        "paper-plan", help="create, and optionally submit, a paper-only two-leg entry"
    )
    paper.add_argument("--csv", required=True)
    paper.add_argument("--symbol-y", required=True)
    paper.add_argument("--symbol-x", required=True)
    paper.add_argument("--lookback", type=int, default=252)
    paper.add_argument("--gross-notional", type=float, default=1_000.0)
    paper.add_argument(
        "--signal",
        choices=["auto", "long", "short"],
        default="auto",
        help="long/short overrides are for paper execution experiments only",
    )
    paper.add_argument("--submit-paper", action="store_true")
    _add_strategy_arguments(paper)
    paper.set_defaults(handler=_paper_plan)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
