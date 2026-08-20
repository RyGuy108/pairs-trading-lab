"""Data validation, deterministic sample data, and optional Alpaca downloads."""

from __future__ import annotations

import os
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


def validate_prices(prices: pd.DataFrame, minimum_rows: int = 60) -> pd.DataFrame:
    """Return clean prices or raise an actionable error.

    Pairs models use logarithms, so prices must be strictly positive. Missing rows are
    removed jointly: both legs must have a price at every timestamp used in a test.
    """

    if not isinstance(prices, pd.DataFrame):
        raise TypeError("prices must be a pandas DataFrame")
    if prices.shape[1] < 2:
        raise ValueError("at least two price columns are required")
    if prices.columns.duplicated().any():
        raise ValueError("price columns must have unique symbol names")

    clean = prices.copy()
    clean.index = pd.to_datetime(clean.index, utc=True)
    clean = clean.sort_index()
    clean = clean[~clean.index.duplicated(keep="last")]
    clean = clean.apply(pd.to_numeric, errors="coerce").dropna(how="any")

    if len(clean) < minimum_rows:
        raise ValueError(f"need at least {minimum_rows} complete observations; found {len(clean)}")
    if not np.isfinite(clean.to_numpy()).all():
        raise ValueError("prices contain infinite values")
    if (clean <= 0).any().any():
        raise ValueError("all prices must be strictly positive because the model uses log prices")
    return clean.astype(float)


def simulate_prices(observations: int = 1_000, seed: int = 7) -> pd.DataFrame:
    """Create a known cointegrated pair plus two distractor assets.

    ``AAA`` and ``BBB`` share a stochastic trend while their log-price residual is
    mean reverting. ``CCC`` and ``DDD`` are independent random walks. Because the
    truth is known, this dataset is useful for learning and automated tests.
    """

    if observations < 100:
        raise ValueError("observations must be at least 100")

    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-02", periods=observations, tz="UTC")

    log_bbb = np.log(80.0) + np.cumsum(rng.normal(0.00025, 0.009, observations))
    residual = np.zeros(observations)
    for index in range(1, observations):
        residual[index] = 0.90 * residual[index - 1] + rng.normal(0.0, 0.009)

    beta = 1.08
    intercept = np.log(100.0) - beta * np.log(80.0)
    log_aaa = intercept + beta * log_bbb + residual
    log_ccc = np.log(55.0) + np.cumsum(rng.normal(0.00015, 0.013, observations))
    log_ddd = np.log(120.0) + np.cumsum(rng.normal(0.00010, 0.011, observations))

    return pd.DataFrame(
        {
            "AAA": np.exp(log_aaa),
            "BBB": np.exp(log_bbb),
            "CCC": np.exp(log_ccc),
            "DDD": np.exp(log_ddd),
        },
        index=dates,
    ).rename_axis("timestamp")


def load_price_csv(path: str | Path) -> pd.DataFrame:
    """Load a wide CSV whose first column is a timestamp and remaining columns are prices."""

    frame = pd.read_csv(path, index_col=0, parse_dates=True)
    return validate_prices(frame)


def save_prices(prices: pd.DataFrame, path: str | Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    prices.to_csv(destination)
    return destination


def download_alpaca_daily_bars(
    symbols: Sequence[str],
    start: datetime,
    end: datetime,
    *,
    feed: str = "iex",
) -> pd.DataFrame:
    """Download adjusted daily closes with the official ``alpaca-py`` SDK.

    Credentials are read from ``ALPACA_API_KEY`` and ``ALPACA_SECRET_KEY``. The
    optional dependency is imported here so the entire offline lab works without
    an Alpaca account.
    """

    try:
        from dotenv import load_dotenv

        load_dotenv()
        from alpaca.data.enums import Adjustment, DataFeed
        from alpaca.data.historical import StockHistoricalDataClient
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame
    except ImportError as exc:
        raise RuntimeError(
            'Alpaca support is not installed. Run: pip install -e ".[alpaca]"'
        ) from exc

    api_key = os.getenv("ALPACA_API_KEY")
    secret_key = os.getenv("ALPACA_SECRET_KEY")
    if not api_key or not secret_key:
        raise RuntimeError("set ALPACA_API_KEY and ALPACA_SECRET_KEY before downloading data")

    requested_symbols = [symbol.upper() for symbol in symbols]
    if len(requested_symbols) < 2:
        raise ValueError("provide at least two symbols")

    feed_enum = DataFeed.IEX if feed.lower() == "iex" else DataFeed.SIP
    client = StockHistoricalDataClient(api_key, secret_key)
    request = StockBarsRequest(
        symbol_or_symbols=requested_symbols,
        timeframe=TimeFrame.Day,
        start=start,
        end=end,
        adjustment=Adjustment.ALL,
        feed=feed_enum,
    )
    bars = client.get_stock_bars(request).df.reset_index()
    if bars.empty:
        raise RuntimeError("Alpaca returned no bars for the requested symbols and dates")

    prices = bars.pivot(index="timestamp", columns="symbol", values="close")
    missing = set(requested_symbols) - set(prices.columns)
    if missing:
        raise RuntimeError(f"Alpaca returned no data for: {', '.join(sorted(missing))}")
    return validate_prices(prices[requested_symbols])
