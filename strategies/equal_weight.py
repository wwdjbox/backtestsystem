"""
Equal-weight benchmark strategy.
Allocates 1/N to every ticker and rebalances daily.
"""

import pandas as pd
from strategy import Strategy, register


@register(key="equal_weight")
class EqualWeightStrategy(Strategy):
    """
    Benchmark: equal weight across all available tickers, rebalanced daily.
    """

    def __init__(self):
        super().__init__(name="Equal Weight (Benchmark)")

    def generate_weights(
        self, prices_so_far: pd.DataFrame, current_date: pd.Timestamp
    ) -> pd.Series:
        tickers = prices_so_far.columns.tolist()
        return pd.Series(1.0 / len(tickers), index=tickers)
