"""
Volatility-weighted (minimum-volatility tilt) strategy.

Allocates to all stocks in inverse proportion to their recent realised
volatility.  Lower-vol stocks get a higher allocation, naturally
reducing portfolio risk vs. equal weight.
"""

import numpy as np
import pandas as pd
from strategy import Strategy, register


@register(key="vol_weighted", default_params={"vol_lookback": 60})
class VolatilityWeightedStrategy(Strategy):
    """
    Weight ∝ 1 / σ_i, where σ_i is the trailing daily return volatility.

    Parameters
    ----------
    vol_lookback : int
        Days used to estimate each stock's daily return std (default 60).
    """

    def __init__(self, vol_lookback: int = 60):
        super().__init__(name=f"Volatility Weighted (lookback={vol_lookback})")
        self.vol_lookback = vol_lookback

    def generate_weights(
        self, prices_so_far: pd.DataFrame, current_date: pd.Timestamp
    ) -> pd.Series:
        tickers = prices_so_far.columns.tolist()
        weights = pd.Series(0.0, index=tickers)

        if len(prices_so_far) < self.vol_lookback + 1:
            return weights

        recent_returns = prices_so_far.iloc[-self.vol_lookback - 1:].pct_change().dropna()
        vol = recent_returns.std()

        # Assign very high vol to zero-vol stocks so they get negligible weight
        vol = vol.replace(0, np.nan).fillna(vol.max() * 10)

        inv_vol = 1.0 / vol
        weights = inv_vol / inv_vol.sum()
        return weights
