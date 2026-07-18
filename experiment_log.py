"""
experiment_log.py
=================
Persistent, append-only experiment log.

Every completed backtest run is recorded as one row in
  results/experiment_log.csv

and one entry in
  results/experiment_log.jsonl   (one JSON object per line)

This makes it trivial to:
  • Compare any two past runs by their run_id
  • Track how a strategy's metrics changed as you tuned hyperparameters
  • Reproduce any past run (config is stored verbatim)

Public API
----------
  ExperimentLog.record(result)          Append one BacktestResult to the log
  ExperimentLog.load() -> pd.DataFrame  Read the full log into a DataFrame
  ExperimentLog.print_log()             Pretty-print the log to stdout
  ExperimentLog.compare_runs(run_ids)   Pull specific runs for comparison
  ExperimentLog.last_n(n) -> list       Most recent n run rows as list[dict]
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path

import pandas as pd

from engine import BacktestResult

# ── Metric columns saved to the flat CSV log ──────────────────────────
_LOG_METRICS = [
    "total_return_pct",
    "cagr_pct",
    "ann_volatility_pct",
    "sharpe_ratio",
    "sortino_ratio",
    "calmar_ratio",
    "win_rate_pct",
    "max_drawdown_pct",
    "max_dd_duration_days",
    "final_nav",
    "total_pnl_usd",
    "avg_n_positions",
    "avg_daily_turnover_pct",
    "ann_cost_drag_pct",
    "total_transaction_costs_usd",
    "var_95_pct",
]


class ExperimentLog:
    """
    Persistent append-only log of backtest runs.

    Parameters
    ----------
    log_dir : Path-like
        Directory that holds the log files (default: ``results/``).
        Created automatically if it does not exist.
    """

    CSV_FILE   = "experiment_log.csv"
    JSONL_FILE = "experiment_log.jsonl"

    def __init__(self, log_dir: str | Path = "results"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.csv_path   = self.log_dir / self.CSV_FILE
        self.jsonl_path = self.log_dir / self.JSONL_FILE

    # ── Write ──────────────────────────────────────────────────────────

    def record(self, result: BacktestResult) -> None:
        """
        Append one BacktestResult to both the CSV and JSONL log files.

        A full manifest (config + all metrics) is stored in JSONL for
        exact reproducibility; a subset of key metrics is stored in CSV
        for quick tabular inspection.
        """
        m   = result.metrics
        cfg = result.config.to_dict()

        # ── Flat row for CSV ──────────────────────────────────────────
        row: dict = {
            "run_id":        result.run_id,
            "timestamp":     result.run_timestamp,
            "strategy_key":  result.strategy_key,
            "strategy_name": result.strategy_name,
            "start_date":    m["start_date"],
            "end_date":      m["end_date"],
            "n_trading_days": m["n_trading_days"],
            "initial_capital": cfg["initial_capital"],
            "transaction_cost_bps": cfg["transaction_cost_bps"],
            "warmup_days":   cfg["warmup_days"],
        }
        for key in _LOG_METRICS:
            row[key] = m.get(key, "")

        # Append to CSV with proper quoting (strategy names may contain commas)
        import csv as _csv
        write_header = not self.csv_path.exists()
        with open(self.csv_path, "a", newline="") as f:
            writer = _csv.DictWriter(f, fieldnames=list(row.keys()))
            if write_header:
                writer.writeheader()
            writer.writerow(row)

        # ── Full manifest for JSONL ───────────────────────────────────
        manifest = {
            "run_id":        result.run_id,
            "timestamp":     result.run_timestamp,
            "strategy_key":  result.strategy_key,
            "strategy_name": result.strategy_name,
            "config":        cfg,
            "metrics":       m,
        }
        with open(self.jsonl_path, "a") as f:
            f.write(json.dumps(manifest) + "\n")

        print(f"  Experiment log   → {self.csv_path}  (run_id: {result.run_id})")

    # ── Read ───────────────────────────────────────────────────────────

    def load(self) -> pd.DataFrame:
        """
        Load the full experiment log as a DataFrame.

        Returns an empty DataFrame (with expected columns) if the log
        does not exist yet.
        """
        if not self.csv_path.exists():
            return pd.DataFrame(columns=["run_id", "timestamp", "strategy_name"])
        df = pd.read_csv(self.csv_path)
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        return df.sort_values("timestamp", ascending=False).reset_index(drop=True)

    def load_manifest(self, run_id: str) -> dict | None:
        """
        Retrieve the full manifest (config + all metrics) for a specific run_id
        from the JSONL log.

        Returns None if not found.
        """
        if not self.jsonl_path.exists():
            return None
        with open(self.jsonl_path) as f:
            for line in f:
                obj = json.loads(line)
                if obj.get("run_id") == run_id:
                    return obj
        return None

    def last_n(self, n: int = 10) -> list[dict]:
        """Return the most recent `n` log rows as a list of dicts."""
        df = self.load()
        return df.head(n).to_dict("records")

    def compare_runs(self, run_ids: list[str]) -> pd.DataFrame:
        """
        Pull specific runs from the log and return a comparison DataFrame
        (run_id as index, metrics as columns).
        """
        df = self.load()
        mask = df["run_id"].isin(run_ids)
        return df[mask].set_index("run_id")

    # ── Display ────────────────────────────────────────────────────────

    def print_log(self, n: int = 20) -> None:
        """
        Pretty-print the most recent `n` runs in a compact table.
        Columns: run_id, strategy, period, CAGR, Sharpe, MaxDD, T/O, Costs
        """
        df = self.load()
        if df.empty:
            print("  [ExperimentLog] No runs recorded yet.")
            return

        df = df.head(n)
        W = 105
        print()
        print("═" * W)
        print("  EXPERIMENT LOG  (most recent first)")
        print("─" * W)
        hdr = (
            f"  {'Run ID':<42}  {'Strategy':<22}  "
            f"{'Period':<24}  {'CAGR%':>6}  {'Sharpe':>7}  "
            f"{'MaxDD%':>7}  {'Costs$':>9}"
        )
        print(hdr)
        print("─" * W)

        for _, row in df.iterrows():
            period = f"{row.get('start_date','?')} → {row.get('end_date','?')}"
            cagr   = row.get("cagr_pct", float("nan"))
            sharpe = row.get("sharpe_ratio", float("nan"))
            maxdd  = row.get("max_drawdown_pct", float("nan"))
            costs  = row.get("total_transaction_costs_usd", float("nan"))
            strat  = str(row.get("strategy_name", ""))[:22]
            rid    = str(row.get("run_id", ""))[:42]

            try:
                cost_str = f"{costs:>9,.0f}"
            except (ValueError, TypeError):
                cost_str = f"{'N/A':>9}"

            print(
                f"  {rid:<42}  {strat:<22}  {period:<24}  "
                f"{cagr:>+6.2f}%  {sharpe:>7.4f}  {maxdd:>+7.2f}%  {cost_str}"
            )

        if len(self.load()) > n:
            print(f"  ... ({len(self.load()) - n} older runs not shown)")
        print("═" * W)
        print()

    def summary_stats(self) -> pd.DataFrame:
        """
        Aggregate statistics per strategy_key across all recorded runs.

        Returns a DataFrame: strategy_key as index, mean/std of key metrics.
        """
        df = self.load()
        if df.empty:
            return pd.DataFrame()
        numeric_cols = [c for c in _LOG_METRICS if c in df.columns]
        return (
            df.groupby("strategy_key")[numeric_cols]
            .agg(["mean", "std", "count"])
            .round(4)
        )
