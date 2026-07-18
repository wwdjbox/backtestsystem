"""
Momentum strategy.

Ranks stocks by total return over a lookback window and holds the top-N
in equal weight.  A short skip at the end of the window avoids the
well-documented 1-month short-term reversal effect.
"""

import pandas as pd
import numpy as np
from strategy import Strategy, register


@register(key="momentum", default_params={"lookback": 60, "top_n": 10, "skip_last": 5})
class MomentumStrategy(Strategy):
    """
    Buy the top-`top_n` stocks ranked by `lookback`-day return,
    skipping the most recent `skip_last` days.

    Parameters
    ----------
    lookback : int
        Trading days used to measure momentum (default 60 ≈ 3 months).
    top_n : int
        Number of stocks to hold (default 10).
    skip_last : int
        Days excluded from the end of the window to reduce reversal
        (default 5 ≈ one week).
    """

    def __init__(self, lookback: int = 60, top_n: int = 10, skip_last: int = 5):
        super().__init__(name=f"Momentum (lookback={lookback}, top_n={top_n})")
        self.lookback = lookback
        self.top_n = top_n
        self.skip_last = skip_last

    def generate_weights(
        self, prices_so_far: pd.DataFrame, current_date: pd.Timestamp
    ) -> pd.Series:
        tickers = prices_so_far.columns.tolist()
        weights = pd.Series(0.0, index=tickers)

        required = self.lookback + self.skip_last + 1
        if len(prices_so_far) < required:
            return weights   # insufficient history → hold cash

        start_px = prices_so_far.iloc[-(self.lookback + self.skip_last)]
        end_px = (
            prices_so_far.iloc[-self.skip_last]
            if self.skip_last > 0
            else prices_so_far.iloc[-1]
        )
        momentum = (end_px - start_px) / start_px
        top = momentum.nlargest(self.top_n).index.tolist()
        weights[top] = 1.0 / self.top_n
        return weights
