"""
engine.py
=========
Core backtesting engine.

At each trading day's close the engine:
  1. Asks the strategy for target weights.
  2. Computes required trades (difference from current weights).
  3. Applies transaction costs.
  4. Updates portfolio state and records a rich DailyRecord.

BacktestResult exposes:
  • nav           – daily Net Asset Value series (dollar portfolio value)
  • pnl           – daily dollar P&L (gross and net of costs)
  • daily_returns – daily simple returns (net)
  • drawdown      – daily drawdown series
  • weight_history – (dates × tickers) weight DataFrame
  • ledger        – full daily ledger with all accounting columns
  • metrics       – comprehensive scalar metric dict
  • periodic      – monthly and yearly return tables
"""

from __future__ import annotations
import hashlib
import json
import os
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from strategy import Strategy


# ══════════════════════════════════════════════════════════════════════
# Configuration
# ══════════════════════════════════════════════════════════════════════

@dataclass
class BacktestConfig:
    """All parameters governing one backtest run."""
    initial_capital: float       = 1_000_000.0  # Starting NAV ($)
    transaction_cost_bps: float  = 5.0           # One-way cost (basis points)
    start_date: str | None       = None          # "YYYY-MM-DD" or None
    end_date:   str | None       = None          # "YYYY-MM-DD" or None
    warmup_days: int             = 0             # Days strategy observes before trading

    def to_dict(self) -> dict:
        return asdict(self)

    def fingerprint(self) -> str:
        """Short hash of this config — used in run IDs."""
        blob = json.dumps(self.to_dict(), sort_keys=True).encode()
        return hashlib.md5(blob).hexdigest()[:8]


# ══════════════════════════════════════════════════════════════════════
# Internal per-day record
# ══════════════════════════════════════════════════════════════════════

@dataclass
class DailyRecord:
    date:              pd.Timestamp
    nav:               float          # portfolio value after costs
    cash:              float
    stock_value:       float          # total equity position value
    gross_exposure:    float          # stock_value / nav
    daily_pnl:         float          # dollar P&L vs previous day (net of costs)
    daily_pnl_gross:   float          # dollar P&L before transaction costs
    transaction_cost:  float          # dollar costs this day
    turnover:          float          # one-way traded value / nav
    n_positions:       int            # number of non-zero positions
    weights:           pd.Series = field(repr=False)


# ══════════════════════════════════════════════════════════════════════
# Engine
# ══════════════════════════════════════════════════════════════════════

class BacktestEngine:
    """
    Runs a daily-close backtest of a Strategy on a price matrix.

    Parameters
    ----------
    prices   : pd.DataFrame   Full (dates × tickers) close-price matrix.
    strategy : Strategy       Any registered Strategy subclass instance.
    config   : BacktestConfig Run parameters; defaults if None.
    """

    def __init__(
        self,
        prices: pd.DataFrame,
        strategy: Strategy,
        config: BacktestConfig | None = None,
    ):
        self.prices   = prices
        self.strategy = strategy
        self.config   = config or BacktestConfig()

    def run(self) -> "BacktestResult":
        cfg     = self.config
        prices  = self._slice_prices()
        tickers = prices.columns.tolist()
        n       = len(tickers)

        cash   = cfg.initial_capital
        shares = np.zeros(n)
        cost_rate = cfg.transaction_cost_bps / 10_000.0

        records: list[DailyRecord] = []
        prev_nav = cfg.initial_capital

        for t, (date, row) in enumerate(prices.iterrows()):
            close = row.values.astype(float)

            # ── Value at today's close (before any trading) ───────────
            stock_value = (shares * close).sum()
            nav_before  = cash + stock_value

            if t < cfg.warmup_days:
                records.append(DailyRecord(
                    date=date, nav=nav_before, cash=cash,
                    stock_value=stock_value,
                    gross_exposure=stock_value / nav_before if nav_before > 0 else 0.0,
                    daily_pnl=nav_before - prev_nav,
                    daily_pnl_gross=nav_before - prev_nav,
                    transaction_cost=0.0, turnover=0.0, n_positions=0,
                    weights=pd.Series(0.0, index=tickers),
                ))
                prev_nav = nav_before
                continue

            # ── Ask strategy for target weights ────────────────────────
            prices_so_far = prices.iloc[: t + 1]
            target_w  = self.strategy.generate_weights(prices_so_far, date)
            target_w  = self.strategy._validate_weights(target_w, tickers)
            target_arr = target_w.values.astype(float)

            # ── Current weights ────────────────────────────────────────
            per_share_value = shares * close
            current_w = per_share_value / nav_before if nav_before > 0 else np.zeros(n)

            # ── Required trades ────────────────────────────────────────
            delta_w     = target_arr - current_w
            delta_value = delta_w * nav_before
            safe_close  = np.where(close > 0, close, np.nan)
            delta_shares = np.where(np.isfinite(safe_close), delta_value / safe_close, 0.0)

            # ── Costs ──────────────────────────────────────────────────
            gross_traded    = np.abs(delta_shares * close).sum()
            transaction_cost = gross_traded * cost_rate
            turnover         = gross_traded / nav_before if nav_before > 0 else 0.0

            # ── Update state ───────────────────────────────────────────
            shares = shares + delta_shares
            cash  -= (delta_shares * close).sum() + transaction_cost

            stock_value_after = (shares * close).sum()
            nav_after         = cash + stock_value_after

            daily_pnl_gross = nav_before - prev_nav                  # before costs
            daily_pnl_net   = nav_after  - prev_nav                  # after costs

            records.append(DailyRecord(
                date=date,
                nav=nav_after,
                cash=cash,
                stock_value=stock_value_after,
                gross_exposure=stock_value_after / nav_after if nav_after > 0 else 0.0,
                daily_pnl=daily_pnl_net,
                daily_pnl_gross=daily_pnl_gross,
                transaction_cost=transaction_cost,
                turnover=turnover,
                n_positions=int((target_arr > 1e-8).sum()),
                weights=target_w,
            ))
            prev_nav = nav_after

        return BacktestResult(
            records=records,
            strategy_name=self.strategy.name,
            strategy_key=getattr(self.strategy, '_registry_key', self.strategy.name),
            config=cfg,
            tickers=tickers,
        )

    def _slice_prices(self) -> pd.DataFrame:
        prices = self.prices.copy()
        if self.config.start_date:
            prices = prices[prices.index >= pd.Timestamp(self.config.start_date)]
        if self.config.end_date:
            prices = prices[prices.index <= pd.Timestamp(self.config.end_date)]
        return prices


# ══════════════════════════════════════════════════════════════════════
# Result
# ══════════════════════════════════════════════════════════════════════

class BacktestResult:
    """
    Complete output of one backtest run.

    Public attributes
    -----------------
    nav            : pd.Series    Daily NAV (dollar portfolio value)
    pnl            : pd.Series    Daily net dollar P&L
    pnl_gross      : pd.Series    Daily gross dollar P&L (before costs)
    daily_returns  : pd.Series    Daily simple returns (net)
    drawdown       : pd.Series    Daily drawdown from running peak
    weight_history : pd.DataFrame (dates × tickers) target weight matrix
    ledger         : pd.DataFrame Full daily accounting table
    metrics        : dict         Scalar performance metrics
    periodic       : dict         Monthly and yearly return tables
    run_id         : str          Unique identifier for this run
    """

    TDA = 252   # trading days per year

    def __init__(
        self,
        records: list[DailyRecord],
        strategy_name: str,
        strategy_key: str,
        config: BacktestConfig,
        tickers: list[str],
    ):
        self.strategy_name = strategy_name
        self.strategy_key  = strategy_key
        self.config        = config
        self.tickers       = tickers
        self.run_timestamp = time.strftime("%Y-%m-%dT%H:%M:%S")
        self.run_id        = f"{strategy_key}__{config.fingerprint()}__{int(time.time())}"

        self._build_series(records)
        self.metrics  = self._compute_metrics()
        self.periodic = self._compute_periodic()

    # ── Build all time series ─────────────────────────────────────────

    def _build_series(self, records: list[DailyRecord]) -> None:
        rows = []
        weight_rows = []
        for r in records:
            rows.append({
                "date":             r.date,
                "nav":              r.nav,
                "cash":             r.cash,
                "stock_value":      r.stock_value,
                "gross_exposure":   r.gross_exposure,
                "daily_pnl":        r.daily_pnl,
                "daily_pnl_gross":  r.daily_pnl_gross,
                "transaction_cost": r.transaction_cost,
                "turnover":         r.turnover,
                "n_positions":      r.n_positions,
            })
            weight_rows.append(r.weights.rename(r.date))

        ledger = pd.DataFrame(rows).set_index("date")
        ledger.index = pd.to_datetime(ledger.index)

        # Cumulative P&L
        ledger["cum_pnl"]       = ledger["daily_pnl"].cumsum()
        ledger["cum_pnl_gross"] = ledger["daily_pnl_gross"].cumsum()
        ledger["cum_costs"]     = ledger["transaction_cost"].cumsum()

        # Drawdown
        nav = ledger["nav"]
        peak = nav.cummax()
        ledger["drawdown"] = (nav - peak) / peak

        self.ledger        = ledger
        self.nav           = ledger["nav"]
        self.pnl           = ledger["daily_pnl"]
        self.pnl_gross     = ledger["daily_pnl_gross"]
        self.drawdown      = ledger["drawdown"]
        self.daily_returns = nav.pct_change().fillna(0.0)

        # Weight history matrix
        self.weight_history = pd.DataFrame(weight_rows)
        self.weight_history.index = pd.to_datetime(self.weight_history.index)
        self.weight_history = self.weight_history.reindex(columns=self.tickers).fillna(0.0)

    # ── Scalar metrics ─────────────────────────────────────────────────

    def _compute_metrics(self) -> dict:
        r       = self.daily_returns
        T       = self.TDA
        initial = self.config.initial_capital
        final   = self.nav.iloc[-1]
        n_years = len(r) / T

        total_ret = (final - initial) / initial
        cagr      = (final / initial) ** (1 / n_years) - 1 if n_years > 0 else np.nan
        ann_vol   = r.std() * np.sqrt(T)
        sharpe    = r.mean() / r.std() * np.sqrt(T) if r.std() > 0 else np.nan

        neg_r    = r[r < 0]
        downside = neg_r.std() * np.sqrt(T) if len(neg_r) > 0 else np.nan
        sortino  = r.mean() * T / downside if (downside and downside > 0) else np.nan

        max_dd = self.drawdown.min()
        calmar = cagr / abs(max_dd) if max_dd != 0 else np.nan

        # Drawdown duration (longest streak of consecutive drawdown days)
        in_dd      = (self.drawdown < 0).astype(int)
        dd_streaks = in_dd * (in_dd.groupby((in_dd != in_dd.shift()).cumsum()).cumcount() + 1)
        max_dd_dur = int(dd_streaks.max())

        # Win rate
        win_rate = float((r > 0).mean())

        # Best / worst days
        best_day  = float(r.max())
        worst_day = float(r.min())

        # Cost drag
        total_costs   = self.ledger["transaction_cost"].sum()
        avg_daily_to  = self.ledger["turnover"].mean()
        ann_cost_drag = self.ledger["transaction_cost"].mean() * T / initial * 100

        # Value at Risk (parametric, 95%)
        var_95 = float(r.mean() - 1.645 * r.std())

        return {
            # Identity
            "strategy":          self.strategy_name,
            "strategy_key":      self.strategy_key,
            "run_id":            self.run_id,
            "run_timestamp":     self.run_timestamp,
            "start_date":        str(self.nav.index[0].date()),
            "end_date":          str(self.nav.index[-1].date()),
            "n_trading_days":    len(r),
            # Capital
            "initial_capital":   initial,
            "final_nav":         round(final, 2),
            "total_pnl_usd":     round(final - initial, 2),
            # Returns
            "total_return_pct":  round(total_ret * 100, 4),
            "cagr_pct":          round(cagr * 100, 4),
            "ann_volatility_pct":round(ann_vol * 100, 4),
            "sharpe_ratio":      round(sharpe, 4),
            "sortino_ratio":     round(sortino, 4),
            "calmar_ratio":      round(calmar, 4),
            "win_rate_pct":      round(win_rate * 100, 2),
            "best_day_pct":      round(best_day * 100, 4),
            "worst_day_pct":     round(worst_day * 100, 4),
            "var_95_pct":        round(var_95 * 100, 4),
            # Risk
            "max_drawdown_pct":  round(max_dd * 100, 4),
            "max_dd_duration_days": max_dd_dur,
            # Trading activity
            "avg_daily_turnover_pct":    round(avg_daily_to * 100, 4),
            "ann_cost_drag_pct":         round(ann_cost_drag, 4),
            "total_transaction_costs_usd": round(total_costs, 2),
            "avg_n_positions":           round(self.ledger["n_positions"].mean(), 1),
            # Config echo
            "transaction_cost_bps":      self.config.transaction_cost_bps,
            "warmup_days":               self.config.warmup_days,
        }

    # ── Periodic return tables ─────────────────────────────────────────

    def _compute_periodic(self) -> dict:
        r = self.daily_returns.copy()
        r.index = pd.to_datetime(r.index)

        # Monthly returns (compound)
        monthly = (1 + r).resample("ME").prod() - 1
        monthly_table = monthly.copy()
        monthly_table.index = monthly_table.index.to_period("M")

        # Pivot to Year × Month matrix
        pivot = pd.DataFrame({
            "year":  monthly_table.index.year,
            "month": monthly_table.index.month,
            "ret":   monthly_table.values,
        }).pivot(index="year", columns="month", values="ret")
        pivot.columns = [
            "Jan","Feb","Mar","Apr","May","Jun",
            "Jul","Aug","Sep","Oct","Nov","Dec"
        ][:len(pivot.columns)]
        pivot["Annual"] = (1 + monthly).groupby(monthly.index.year).prod() - 1

        # Yearly returns
        yearly = (1 + r).resample("YE").prod() - 1
        yearly.index = yearly.index.year

        return {
            "monthly_pivot": pivot,           # Year x Month DataFrame
            "monthly_series": monthly,        # DatetimeSeries of monthly returns
            "yearly_series":  yearly,         # Series indexed by year
        }

    # ── Console output ────────────────────────────────────────────────

    def print_summary(self) -> None:
        m = self.metrics
        w = 58
        print(f"\n{'═' * w}")
        print(f"  {self.strategy_name}")
        print(f"  Run: {self.run_id}")
        print(f"{'─' * w}")
        print(f"  Period          : {m['start_date']} → {m['end_date']}  ({m['n_trading_days']} days)")
        print(f"  Initial Capital : ${m['initial_capital']:>14,.0f}")
        print(f"  Final NAV       : ${m['final_nav']:>14,.2f}")
        print(f"  Total P&L       : ${m['total_pnl_usd']:>+14,.2f}")
        print(f"{'─' * w}")
        print(f"  Total Return    : {m['total_return_pct']:>+9.2f}%")
        print(f"  CAGR            : {m['cagr_pct']:>+9.2f}%")
        print(f"  Ann. Volatility : {m['ann_volatility_pct']:>9.2f}%")
        print(f"  Sharpe Ratio    : {m['sharpe_ratio']:>9.4f}")
        print(f"  Sortino Ratio   : {m['sortino_ratio']:>9.4f}")
        print(f"  Calmar Ratio    : {m['calmar_ratio']:>9.4f}")
        print(f"  Win Rate        : {m['win_rate_pct']:>9.2f}%")
        print(f"  Best Day        : {m['best_day_pct']:>+9.2f}%")
        print(f"  Worst Day       : {m['worst_day_pct']:>+9.2f}%")
        print(f"  VaR (95%)       : {m['var_95_pct']:>+9.2f}%/day")
        print(f"{'─' * w}")
        print(f"  Max Drawdown    : {m['max_drawdown_pct']:>+9.2f}%")
        print(f"  Max DD Duration : {m['max_dd_duration_days']:>9} days")
        print(f"{'─' * w}")
        print(f"  Avg Positions   : {m['avg_n_positions']:>9.1f}")
        print(f"  Avg Daily T/O   : {m['avg_daily_turnover_pct']:>9.2f}%")
        print(f"  Ann. Cost Drag  : {m['ann_cost_drag_pct']:>9.4f}%")
        print(f"  Total Costs ($) : ${m['total_transaction_costs_usd']:>14,.2f}")
        print(f"{'═' * w}\n")

    # ── Persist outputs ───────────────────────────────────────────────

    def save_results(self, output_dir: str = "results") -> dict[str, str]:
        """
        Write all standardised output files for this result.

        Returns a dict mapping label → file path for every file written.
        """
        run_dir = Path(output_dir) / self.run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        paths: dict[str, str] = {}

        # 1. NAV / equity curve
        p = run_dir / "nav.csv"
        self.nav.to_csv(p, header=["nav"])
        paths["nav"] = str(p)

        # 2. Full daily ledger
        p = run_dir / "ledger.csv"
        self.ledger.to_csv(p)
        paths["ledger"] = str(p)

        # 3. Daily P&L
        p = run_dir / "pnl.csv"
        pnl_df = self.ledger[["daily_pnl", "daily_pnl_gross",
                               "transaction_cost", "cum_pnl"]].copy()
        pnl_df.to_csv(p)
        paths["pnl"] = str(p)

        # 4. Drawdown series
        p = run_dir / "drawdown.csv"
        self.drawdown.to_csv(p, header=["drawdown"])
        paths["drawdown"] = str(p)

        # 5. Weight history
        p = run_dir / "weights.csv"
        self.weight_history.to_csv(p)
        paths["weights"] = str(p)

        # 6. Scalar metrics
        p = run_dir / "metrics.json"
        with open(p, "w") as f:
            json.dump(self.metrics, f, indent=2)
        paths["metrics"] = str(p)

        # 7. Monthly return pivot table
        p = run_dir / "monthly_returns.csv"
        self.periodic["monthly_pivot"].to_csv(p, float_format="%.4f")
        paths["monthly_returns"] = str(p)

        # 8. Yearly returns
        p = run_dir / "yearly_returns.csv"
        self.periodic["yearly_series"].to_csv(p, header=["annual_return"])
        paths["yearly_returns"] = str(p)

        # 9. Manifest (links all files with metadata)
        manifest = {
            "run_id":        self.run_id,
            "strategy":      self.strategy_name,
            "strategy_key":  self.strategy_key,
            "timestamp":     self.run_timestamp,
            "config":        self.config.to_dict(),
            "metrics":       self.metrics,
            "files":         {k: Path(v).name for k, v in paths.items()},
        }
        p = run_dir / "manifest.json"
        with open(p, "w") as f:
            json.dump(manifest, f, indent=2)
        paths["manifest"] = str(p)

        print(f"  Outputs → {run_dir}/")
        for label, path in paths.items():
            print(f"    {label:<20} {Path(path).name}")

        return paths
