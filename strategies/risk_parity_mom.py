"""
Risk-parity momentum strategy.

Selects stocks by momentum ranking (same as MomentumStrategy) but sizes
them inversely proportional to recent volatility, so each position
contributes roughly equally to overall portfolio risk.
"""

import numpy as np
import pandas as pd
from strategy import Strategy, register


@register(
    key="risk_parity_mom",
    default_params={
        "momentum_lookback": 60,
        "top_n": 15,
        "vol_lookback": 20,
        "skip_last": 5,
    },
)
class RiskParityMomentumStrategy(Strategy):
    """
    Top-N momentum stocks, weighted by inverse-volatility (risk parity sizing).

    Parameters
    ----------
    momentum_lookback : int
        Days used for momentum ranking (default 60).
    top_n : int
        Stocks selected after ranking (default 15).
    vol_lookback : int
        Days used to estimate per-stock volatility for sizing (default 20).
    skip_last : int
        Short-term reversal skip applied to momentum signal (default 5).
    """

    def __init__(
        self,
        momentum_lookback: int = 60,
        top_n: int = 15,
        vol_lookback: int = 20,
        skip_last: int = 5,
    ):
        super().__init__(
            name=f"Risk-Parity Momentum (mom={momentum_lookback}, top={top_n})"
        )
        self.momentum_lookback = momentum_lookback
        self.top_n = top_n
        self.vol_lookback = vol_lookback
        self.skip_last = skip_last

    def generate_weights(
        self, prices_so_far: pd.DataFrame, current_date: pd.Timestamp
    ) -> pd.Series:
        tickers = prices_so_far.columns.tolist()
        weights = pd.Series(0.0, index=tickers)

        required = max(self.momentum_lookback + self.skip_last, self.vol_lookback) + 1
        if len(prices_so_far) < required:
            return weights

        # Step 1: rank by momentum
        start_px = prices_so_far.iloc[-(self.momentum_lookback + self.skip_last)]
        end_px = (
            prices_so_far.iloc[-self.skip_last]
            if self.skip_last > 0
            else prices_so_far.iloc[-1]
        )
        momentum = (end_px - start_px) / start_px
        top_tickers = momentum.nlargest(self.top_n).index.tolist()

        # Step 2: inverse-vol sizing within selected stocks
        recent = prices_so_far[top_tickers].iloc[-self.vol_lookback - 1:]
        vol = recent.pct_change().dropna().std().replace(0, np.nan)
        vol = vol.fillna(vol.max() * 10)

        inv_vol = 1.0 / vol
        w = inv_vol / inv_vol.sum()
        weights[top_tickers] = w.values
        return weights
