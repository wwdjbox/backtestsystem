"""
strategies/rsi_reversion.py
============================
RSI-based mean-reversion strategy.

Drop this file into strategies/ — it is auto-discovered and immediately
available as --strategy rsi_reversion.  No other file was modified.

Strategy logic
--------------
Compute the 14-day RSI for every stock.  Buy (equal weight) the stocks
that are most "oversold" — those with the lowest RSI values — up to a
configurable maximum number of positions.  When no stocks fall below the
oversold threshold the portfolio sits in cash.

RSI formula (Wilder, 1978)
    RS  = avg_gain / avg_loss  (over the lookback window)
    RSI = 100 − 100 / (1 + RS)
"""

import numpy as np
import pandas as pd
from strategy import Strategy, register


@register(
    key="rsi_reversion",
    default_params={"rsi_period": 14, "oversold_threshold": 40, "top_n": 10},
)
class RSIMeanReversionStrategy(Strategy):
    """
    Buy the most-oversold stocks (lowest RSI) in equal weight.

    Parameters
    ----------
    rsi_period : int
        Window for RSI calculation (default 14 — the classic Wilder period).
    oversold_threshold : float
        Only consider stocks with RSI below this level (default 40).
        Set to 100 to always hold top_n stocks regardless of RSI level.
    top_n : int
        Maximum positions to hold simultaneously (default 10).
    """

    def __init__(
        self,
        rsi_period: int = 14,
        oversold_threshold: float = 40.0,
        top_n: int = 10,
    ):
        super().__init__(
            name=f"RSI Reversion (period={rsi_period}, threshold={oversold_threshold})"
        )
        self.rsi_period = rsi_period
        self.oversold_threshold = oversold_threshold
        self.top_n = top_n

    # ── RSI helper ────────────────────────────────────────────────────

    @staticmethod
    def _compute_rsi(prices: pd.DataFrame, period: int) -> pd.Series:
        """Return the current RSI for each ticker (scalar per column)."""
        delta = prices.diff().iloc[1:]          # daily price changes
        gain = delta.clip(lower=0)
        loss = (-delta).clip(lower=0)

        # Wilder's smoothed average (equivalent to EMA with alpha=1/period)
        avg_gain = gain.ewm(alpha=1.0 / period, adjust=False).mean().iloc[-1]
        avg_loss = loss.ewm(alpha=1.0 / period, adjust=False).mean().iloc[-1]

        rs = avg_gain / avg_loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        rsi = rsi.fillna(100)   # if avg_loss == 0, RSI = 100 (overbought)
        return rsi

    # ── Strategy interface ────────────────────────────────────────────

    def generate_weights(
        self, prices_so_far: pd.DataFrame, current_date: pd.Timestamp
    ) -> pd.Series:
        tickers = prices_so_far.columns.tolist()
        weights = pd.Series(0.0, index=tickers)

        # Need at least period+1 rows for a meaningful RSI
        if len(prices_so_far) < self.rsi_period + 1:
            return weights

        rsi = self._compute_rsi(prices_so_far, self.rsi_period)

        # Filter to oversold stocks; sort ascending (most oversold first)
        oversold = rsi[rsi < self.oversold_threshold].sort_values()
        if oversold.empty:
            return weights   # nothing oversold → hold cash

        selected = oversold.head(self.top_n).index.tolist()
        weights[selected] = 1.0 / len(selected)
        return weights
