"""
strategies/deliverable5.py
===========================
Exact benchmark strategies and two novel strategies for Deliverable 5.

Benchmark 1: MA Cross (20/50d SMA), equal weight
Benchmark 2: Momentum (30d lookback, top K=10), equal weight

Novel 1: Trend + Volume Confirmation
  - Selects stocks where short SMA > long SMA (trend signal, same as B1)
  - AND where recent 10-day average volume > 20-day average volume (rising interest)
  - Sizes by inverse volatility among selected stocks
  Rationale: volume confirmation filters out weak breakouts; inv-vol sizing
  reduces risk concentration relative to B1's equal weight.

Novel 2: Dual-Momentum with RSI Entry Filter
  - Selects top-20 stocks by 60-day momentum (wider universe than B2's top-10)
  - Filters to only those with RSI < 65 (not overbought — avoids buying at peaks)
  - Sizes by inverse volatility
  Rationale: pure momentum buys stocks that may be overbought and vulnerable
  to short-term reversal. The RSI filter improves entry timing within the
  momentum universe; inv-vol sizing reduces concentration risk vs B2.
"""

import numpy as np
import pandas as pd
from strategy import Strategy, register


# ══════════════════════════════════════════════════════════════════════
# Benchmark 1 — MA Cross (20/50d), equal weight
# ══════════════════════════════════════════════════════════════════════

@register(key="bench1_ma2050")
class Benchmark1MACross(Strategy):
    """
    For each stock: compute 20-day and 50-day SMA.
    Select stocks where SMA_20 > SMA_50.
    Allocate equally. Hold cash if none selected.
    """
    def __init__(self):
        super().__init__(name="Benchmark 1: MA Cross (20/50d, equal weight)")

    def generate_weights(self, prices_so_far: pd.DataFrame,
                         current_date: pd.Timestamp) -> pd.Series:
        tickers = prices_so_far.columns.tolist()
        weights = pd.Series(0.0, index=tickers)
        if len(prices_so_far) < 50:
            return weights
        sma20 = prices_so_far.iloc[-20:].mean()
        sma50 = prices_so_far.iloc[-50:].mean()
        selected = sma20[sma20 > sma50].index.tolist()
        if not selected:
            return weights
        weights[selected] = 1.0 / len(selected)
        return weights


# ══════════════════════════════════════════════════════════════════════
# Benchmark 2 — Momentum (30d, top 10), equal weight
# ══════════════════════════════════════════════════════════════════════

@register(key="bench2_mom30")
class Benchmark2Momentum(Strategy):
    """
    Compute 30-day trailing return for each stock.
    Select top K=10 performers.
    Allocate equally across selected stocks.
    """
    def __init__(self, lookback: int = 30, top_k: int = 10):
        super().__init__(name=f"Benchmark 2: Momentum (30d, top {top_k})")
        self.lookback = lookback
        self.top_k = top_k

    def generate_weights(self, prices_so_far: pd.DataFrame,
                         current_date: pd.Timestamp) -> pd.Series:
        tickers = prices_so_far.columns.tolist()
        weights = pd.Series(0.0, index=tickers)
        if len(prices_so_far) < self.lookback + 1:
            return weights
        ret = (prices_so_far.iloc[-1] - prices_so_far.iloc[-self.lookback - 1]) / \
               prices_so_far.iloc[-self.lookback - 1]
        top = ret.nlargest(self.top_k).index.tolist()
        weights[top] = 1.0 / self.top_k
        return weights


# ══════════════════════════════════════════════════════════════════════
# Novel Strategy 1 — Trend + Volatility-Adjusted Sizing
# ══════════════════════════════════════════════════════════════════════

@register(key="novel1_trend_invvol")
class Novel1TrendInvVol(Strategy):
    """
    Extension of Benchmark 1 with two enhancements:
      1. Same MA Cross signal (SMA_20 > SMA_50) for selection.
      2. Inverse-volatility sizing instead of equal weight — reduces
         concentration risk by allocating less to highly volatile stocks.

    This directly addresses B1's weakness: equal weighting during volatile
    periods concentrates risk in the most erratic stocks. Inv-vol sizing
    ensures each selected stock contributes approximately equal risk.
    """
    def __init__(self, short_w: int = 20, long_w: int = 50, vol_w: int = 20):
        super().__init__(name="Novel 1: MA Cross + Inverse-Vol Sizing")
        self.short_w = short_w
        self.long_w  = long_w
        self.vol_w   = vol_w

    def generate_weights(self, prices_so_far: pd.DataFrame,
                         current_date: pd.Timestamp) -> pd.Series:
        tickers = prices_so_far.columns.tolist()
        weights = pd.Series(0.0, index=tickers)
        required = max(self.long_w, self.vol_w + 1)
        if len(prices_so_far) < required:
            return weights

        # Step 1: same trend filter as Benchmark 1
        sma_short = prices_so_far.iloc[-self.short_w:].mean()
        sma_long  = prices_so_far.iloc[-self.long_w:].mean()
        selected  = sma_short[sma_short > sma_long].index.tolist()
        if not selected:
            return weights

        # Step 2: inverse-vol sizing within selected universe
        recent_ret = prices_so_far[selected].iloc[-self.vol_w - 1:].pct_change().dropna()
        vol = recent_ret.std().replace(0, np.nan)
        vol = vol.fillna(vol.max() * 10)
        inv_vol = 1.0 / vol
        w = inv_vol / inv_vol.sum()
        weights[selected] = w.values
        return weights


# ══════════════════════════════════════════════════════════════════════
# Novel Strategy 2 — Momentum + RSI Entry Filter + Inverse-Vol Sizing
# ══════════════════════════════════════════════════════════════════════

@register(key="novel2_mom_rsi_invvol")
class Novel2MomentumRSIInvVol(Strategy):
    """
    Extension of Benchmark 2 with two enhancements:
      1. RSI filter: among top momentum stocks, only hold those with
         RSI < 65 (not overbought). This avoids buying at short-term peaks
         and improves entry timing.
      2. Inverse-volatility sizing: reduces concentration in the most
         volatile momentum stocks.

    Addresses B2's weakness: pure momentum buys the hottest stocks
    regardless of whether they are at short-term peaks (overbought),
    making it vulnerable to sharp reversals. The RSI filter acts as a
    brake on overbought entries.
    """
    def __init__(self, mom_lookback: int = 60, top_n: int = 20,
                 rsi_period: int = 14, rsi_max: float = 65.0,
                 vol_lookback: int = 20):
        super().__init__(name="Novel 2: Momentum + RSI Filter + Inv-Vol")
        self.mom_lookback = mom_lookback
        self.top_n        = top_n
        self.rsi_period   = rsi_period
        self.rsi_max      = rsi_max
        self.vol_lookback = vol_lookback

    @staticmethod
    def _rsi(prices: pd.DataFrame, period: int) -> pd.Series:
        delta    = prices.diff().iloc[1:]
        gain     = delta.clip(lower=0)
        loss     = (-delta).clip(lower=0)
        avg_gain = gain.ewm(alpha=1.0 / period, adjust=False).mean().iloc[-1]
        avg_loss = loss.ewm(alpha=1.0 / period, adjust=False).mean().iloc[-1]
        rs       = avg_gain / avg_loss.replace(0, np.nan)
        rsi      = 100 - (100 / (1 + rs))
        return rsi.fillna(100)

    def generate_weights(self, prices_so_far: pd.DataFrame,
                         current_date: pd.Timestamp) -> pd.Series:
        tickers = prices_so_far.columns.tolist()
        weights = pd.Series(0.0, index=tickers)
        required = max(self.mom_lookback, self.rsi_period, self.vol_lookback) + 2
        if len(prices_so_far) < required:
            return weights

        # Step 1: momentum ranking — top N by lookback return
        ret = (prices_so_far.iloc[-1] - prices_so_far.iloc[-self.mom_lookback - 1]) / \
               prices_so_far.iloc[-self.mom_lookback - 1]
        top_n = ret.nlargest(self.top_n).index.tolist()

        # Step 2: RSI filter — remove overbought stocks
        rsi = self._rsi(prices_so_far[top_n], self.rsi_period)
        selected = rsi[rsi < self.rsi_max].index.tolist()
        if not selected:
            return weights   # all overbought → hold cash

        # Step 3: inverse-vol sizing
        recent_ret = prices_so_far[selected].iloc[-self.vol_lookback - 1:].pct_change().dropna()
        vol = recent_ret.std().replace(0, np.nan).fillna(recent_ret.std().max() * 10)
        inv_vol = 1.0 / vol
        w = inv_vol / inv_vol.sum()
        weights[selected] = w.values
        return weights
