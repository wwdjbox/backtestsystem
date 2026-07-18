"""
strategy.py
===========
Abstract base class + auto-discovery registry for all daily-close strategies.

Plugin contract
---------------
To add a new strategy, create any .py file inside the strategies/ folder and:

  1. Import Strategy from this module.
  2. Subclass Strategy and implement generate_weights().
  3. Decorate the class with @register(key="my_key").

That is all.  The runner discovers and registers it automatically at startup —
no edits to run_backtest.py, engine.py, or any other file are needed.

Example (strategies/my_new_strategy.py)
----------------------------------------
    from strategy import Strategy, register
    import pandas as pd

    @register(key="my_strat")
    class MyStrategy(Strategy):
        def __init__(self):
            super().__init__(name="My Strategy")

        def generate_weights(self, prices_so_far, current_date):
            tickers = prices_so_far.columns.tolist()
            # ... your logic ...
            return pd.Series(1 / len(tickers), index=tickers)
"""

from __future__ import annotations
from abc import ABC, abstractmethod
import importlib
import pkgutil
import sys
from pathlib import Path
from typing import Type
import numpy as np
import pandas as pd


# ── Global registry: key → Strategy instance ─────────────────────────
_REGISTRY: dict[str, "Strategy"] = {}


def register(key: str, default_params: dict | None = None):
    """
    Class decorator that registers a Strategy subclass in the global registry.

    Parameters
    ----------
    key : str
        Short CLI/lookup key, e.g. "momentum".  Must be unique.
    default_params : dict, optional
        Constructor keyword arguments to use when auto-instantiating.
        If omitted the class is instantiated with no arguments (requires
        a no-arg __init__).

    Usage
    -----
        @register(key="my_strat")
        class MyStrategy(Strategy): ...

        @register(key="momentum_fast", default_params={"lookback": 20, "top_n": 5})
        class MomentumStrategy(Strategy): ...
    """
    def decorator(cls: Type["Strategy"]) -> Type["Strategy"]:
        if key in _REGISTRY:
            raise ValueError(
                f"Strategy key '{key}' is already registered by "
                f"{type(_REGISTRY[key]).__name__}. Choose a unique key."
            )
        params = default_params or {}
        _REGISTRY[key] = cls(**params)
        return cls
    return decorator


def get_registry() -> dict[str, "Strategy"]:
    """Return a copy of the current strategy registry."""
    return dict(_REGISTRY)


def discover_strategies(strategies_dir: str | Path | None = None) -> None:
    """
    Import every .py module inside the strategies/ package so their
    @register decorators execute and populate the registry.

    Called once at startup by run_backtest.py.  Safe to call multiple times.

    Parameters
    ----------
    strategies_dir : path, optional
        Override the default location (sibling 'strategies/' folder).
    """
    if strategies_dir is None:
        strategies_dir = Path(__file__).parent / "strategies"

    strategies_dir = Path(strategies_dir)
    if not strategies_dir.exists():
        return

    # Add parent to sys.path so `import strategies.X` works from anywhere
    root = str(strategies_dir.parent)
    if root not in sys.path:
        sys.path.insert(0, root)

    pkg_name = strategies_dir.name   # "strategies"
    for finder, module_name, _ in pkgutil.iter_modules([str(strategies_dir)]):
        full_name = f"{pkg_name}.{module_name}"
        if full_name not in sys.modules:
            importlib.import_module(full_name)


# ══════════════════════════════════════════════════════════════════════
# Abstract base class
# ══════════════════════════════════════════════════════════════════════

class Strategy(ABC):
    """
    Base class for all daily-close trading strategies.

    Subclass this, implement generate_weights(), and decorate with
    @register(key="...") to make the strategy available system-wide.
    """

    def __init__(self, name: str = "Unnamed Strategy"):
        self.name = name

    @abstractmethod
    def generate_weights(
        self,
        prices_so_far: pd.DataFrame,
        current_date: pd.Timestamp,
    ) -> pd.Series:
        """
        Compute target portfolio weights for the close of `current_date`.

        Parameters
        ----------
        prices_so_far : pd.DataFrame
            Close prices from the first available date up to and including
            `current_date`.  Shape = (days_elapsed, n_tickers).
            **Never contains future data.**
        current_date : pd.Timestamp
            The trading date for which weights are being computed.

        Returns
        -------
        weights : pd.Series
            Target weights indexed by ticker symbol.
            Constraints enforced automatically by the engine:
              - All values ≥ 0  (no short selling)
              - Sum ≤ 1         (no leverage; remainder is cash)
        """
        ...

    # ── Called by engine; strategy authors do not need to override ────

    def _validate_weights(self, weights: pd.Series, tickers: list[str]) -> pd.Series:
        """
        Sanitise weights returned by generate_weights.

        Reindexes to the full ticker universe (filling missing with 0),
        clips negatives to 0, and rescales to sum=1 if the strategy
        returned a leveraged allocation.
        """
        weights = weights.reindex(tickers).fillna(0.0)
        weights = weights.clip(lower=0.0)
        total = weights.sum()
        if total > 1.0:
            weights = weights / total
        return weights

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r})"
