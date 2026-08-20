"""Paper-only Alpaca order planning and submission.

Pairs do not execute atomically: one leg may fill before the other. This module is
intentionally limited to Alpaca paper accounts so that leg risk can be studied
without exposing real capital.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from datetime import UTC, datetime

import numpy as np
import pandas as pd

from pairs_trading.model import PairModel


@dataclass(frozen=True)
class OrderInstruction:
    symbol: str
    side: str
    quantity: float
    reference_price: float
    target_notional: float

    def to_dict(self) -> dict[str, float | str]:
        return asdict(self)


def build_entry_order_plan(
    model: PairModel,
    latest_prices: pd.Series,
    signal: int,
    gross_notional: float,
) -> list[OrderInstruction]:
    """Size a beta-adjusted two-leg entry with normalized gross exposure."""

    if signal not in {-1, 1}:
        raise ValueError("entry signal must be +1 (long spread) or -1 (short spread)")
    if gross_notional <= 0:
        raise ValueError("gross_notional must be positive")
    for symbol in (model.symbol_y, model.symbol_x):
        if symbol not in latest_prices or latest_prices[symbol] <= 0:
            raise ValueError(f"a positive latest price is required for {symbol}")

    gross_scale = 1.0 + abs(model.hedge_ratio)
    y_notional = gross_notional / gross_scale
    x_notional = gross_notional * abs(model.hedge_ratio) / gross_scale
    y_direction = signal
    x_direction = -signal * int(np.sign(model.hedge_ratio) or 1)

    instructions = []
    for symbol, direction, notional in (
        (model.symbol_y, y_direction, y_notional),
        (model.symbol_x, x_direction, x_notional),
    ):
        price = float(latest_prices[symbol])
        instructions.append(
            OrderInstruction(
                symbol=symbol,
                side="buy" if direction > 0 else "sell",
                quantity=round(notional / price, 6),
                reference_price=price,
                target_notional=float(notional),
            )
        )
    return instructions


def order_plan_frame(plan: list[OrderInstruction]) -> pd.DataFrame:
    return pd.DataFrame([instruction.to_dict() for instruction in plan])


def submit_paper_order_plan(plan: list[OrderInstruction]) -> list[object]:
    """Submit sequential market orders to an Alpaca paper account only."""

    try:
        from dotenv import load_dotenv

        load_dotenv()
        from alpaca.trading.client import TradingClient
        from alpaca.trading.enums import OrderSide, TimeInForce
        from alpaca.trading.requests import MarketOrderRequest
    except ImportError as exc:
        raise RuntimeError(
            'Alpaca support is not installed. Run: pip install -e ".[alpaca]"'
        ) from exc

    api_key = os.getenv("ALPACA_API_KEY")
    secret_key = os.getenv("ALPACA_SECRET_KEY")
    if not api_key or not secret_key:
        raise RuntimeError("set ALPACA_API_KEY and ALPACA_SECRET_KEY before paper submission")
    if len(plan) != 2:
        raise ValueError("a pair entry must contain exactly two order instructions")

    client = TradingClient(api_key, secret_key, paper=True)
    batch_id = datetime.now(UTC).strftime("%Y%m%d%H%M%S")
    responses: list[object] = []
    for index, instruction in enumerate(plan):
        request = MarketOrderRequest(
            symbol=instruction.symbol,
            qty=instruction.quantity,
            side=OrderSide.BUY if instruction.side == "buy" else OrderSide.SELL,
            time_in_force=TimeInForce.DAY,
            client_order_id=f"pairs-{batch_id}-{index}-{instruction.symbol.lower()}",
        )
        responses.append(client.submit_order(order_data=request))
    return responses
