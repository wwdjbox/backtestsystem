"""
Moving-average crossover strategy.

Holds stocks whose short-term SMA is above their long-term SMA (a
classic trend-following / bullish signal), sized equally.
"""

import pandas as pd
from strategy import Strategy, register


@register(
    key="ma_cross",
    default_params={"short_window": 20, "long_window": 60, "max_positions": 20},
)
class MovingAverageCrossStrategy(Strategy):
    """
    Enter a stock when its short SMA > long SMA; hold equally weighted.

    Parameters
    ----------
    short_window : int
        Short SMA window in days (default 20).
    long_window : int
        Long SMA window in days (default 60).
    max_positions : int
        Cap on simultaneous positions.  If more stocks qualify, the ones
        with the largest MA gap are preferred (default 20).
    """

    def __init__(
        self,
        short_window: int = 20,
        long_window: int = 60,
        max_positions: int = 20,
    ):
        super().__init__(
            name=f"MA Cross (short={short_window}, long={long_window})"
        )
        self.short_window = short_window
        self.long_window = long_window
        self.max_positions = max_positions

    def generate_weights(
        self, prices_so_far: pd.DataFrame, current_date: pd.Timestamp
    ) -> pd.Series:
        tickers = prices_so_far.columns.tolist()
        weights = pd.Series(0.0, index=tickers)

        if len(prices_so_far) < self.long_window:
            return weights

        short_ma = prices_so_far.iloc[-self.short_window:].mean()
        long_ma = prices_so_far.iloc[-self.long_window:].mean()

        bullish = short_ma[short_ma > long_ma].index.tolist()
        if not bullish:
            return weights

        if len(bullish) > self.max_positions:
            gap = (short_ma - long_ma) / long_ma
            bullish = gap[bullish].nlargest(self.max_positions).index.tolist()

        weights[bullish] = 1.0 / len(bullish)
        return weights
