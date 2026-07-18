"""
data_loader.py
==============
Loads and preprocesses the Nasdaq-100 daily CSV into a clean price matrix
and a returns matrix, both indexed by date with ticker columns.
"""

import pandas as pd
import numpy as np
from pathlib import Path

DEFAULT_DATA_PATH = Path(__file__).parent / "data" / "nasdaq100_daily_5y.csv"


def load_price_matrix(path: str | Path = DEFAULT_DATA_PATH) -> pd.DataFrame:
    """
    Load the raw CSV and pivot to a (dates x tickers) close-price matrix.

    Returns
    -------
    prices : pd.DataFrame
        Index = trading dates (datetime), columns = ticker symbols.
        Missing values are forward-filled then backward-filled so every
        cell is finite after the first trading day for each ticker.
    """
    raw = pd.read_csv(path, parse_dates=["date"])
    raw = raw.sort_values(["ticker", "date"])

    # Keep only close prices; pivot to wide format
    prices = raw.pivot(index="date", columns="ticker", values="close")
    prices.index.name = "date"
    prices.columns.name = None

    # Fill any gaps (e.g., ticker listed later, occasional missing days)
    prices = prices.ffill().bfill()

    return prices


def load_returns_matrix(prices: pd.DataFrame | None = None,
                        path: str | Path = DEFAULT_DATA_PATH) -> pd.DataFrame:
    """
    Compute daily log returns from the price matrix.

    Returns
    -------
    returns : pd.DataFrame
        Same shape as prices, first row is NaN (dropped by callers).
    """
    if prices is None:
        prices = load_price_matrix(path)
    returns = np.log(prices / prices.shift(1))
    return returns


def load_volume_matrix(path: str | Path = DEFAULT_DATA_PATH) -> pd.DataFrame:
    """Load daily volume data (dates x tickers)."""
    raw = pd.read_csv(path, parse_dates=["date"])
    raw = raw.sort_values(["ticker", "date"])
    volume = raw.pivot(index="date", columns="ticker", values="volume")
    volume.index.name = "date"
    volume.columns.name = None
    volume = volume.ffill().bfill()
    return volume


def dataset_summary(path: str | Path = DEFAULT_DATA_PATH) -> None:
    """Print a quick summary of the loaded dataset."""
    prices = load_price_matrix(path)
    print("=" * 50)
    print("Dataset Summary")
    print("=" * 50)
    print(f"  Date range : {prices.index[0].date()} → {prices.index[-1].date()}")
    print(f"  Trading days: {len(prices)}")
    print(f"  Tickers     : {prices.shape[1]}")
    print(f"  Missing cells (after fill): {prices.isna().sum().sum()}")
    print("=" * 50)
