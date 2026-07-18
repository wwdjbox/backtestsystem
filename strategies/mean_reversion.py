"""
Mean-reversion (contrarian) strategy.

Buys the worst recent performers, betting they will bounce back.
"""

import pandas as pd
from strategy import Strategy, register


@register(key="mean_reversion", default_params={"lookback": 20, "bottom_n": 10})
class MeanReversionStrategy(Strategy):
    """
    Hold the `bottom_n` stocks with the worst `lookback`-day return.

    Parameters
    ----------
    lookback : int
        Days over which underperformance is measured (default 20 ≈ 1 month).
    bottom_n : int
        Number of worst-performers to hold (default 10).
    """

    def __init__(self, lookback: int = 20, bottom_n: int = 10):
        super().__init__(
            name=f"Mean Reversion (lookback={lookback}, bottom_n={bottom_n})"
        )
        self.lookback = lookback
        self.bottom_n = bottom_n

    def generate_weights(
        self, prices_so_far: pd.DataFrame, current_date: pd.Timestamp
    ) -> pd.Series:
        tickers = prices_so_far.columns.tolist()
        weights = pd.Series(0.0, index=tickers)

        if len(prices_so_far) <= self.lookback:
            return weights

        ret = (prices_so_far.iloc[-1] - prices_so_far.iloc[-self.lookback - 1]) / \
              prices_so_far.iloc[-self.lookback - 1]
        bottom = ret.nsmallest(self.bottom_n).index.tolist()
        weights[bottom] = 1.0 / self.bottom_n
        return weights
