#!/usr/bin/env python3
"""
run_backtest.py
===============
ENTRY POINT – run this script to reproduce all backtesting results.

Strategies are discovered automatically from the strategies/ folder.
To add a new strategy, drop a .py file there — no changes here needed.

Usage
-----
    python run_backtest.py                         # run all strategies
    python run_backtest.py --strategy momentum     # one strategy by key
    python run_backtest.py --start 2023-01-01      # custom date range
    python run_backtest.py --capital 500000        # custom capital
    python run_backtest.py --costs 10              # 10 bps transaction cost
    python run_backtest.py --list                  # list discovered strategies
    python run_backtest.py --show-log              # print experiment log
    python run_backtest.py --no-dashboard          # skip per-strategy dashboard

All outputs are written to ./results/
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from strategy import discover_strategies, get_registry
from data_loader import load_price_matrix, dataset_summary
from engine import BacktestConfig, BacktestEngine
from reporting import (
    generate_report,
    generate_comparison_report,
    print_comparison_table,
    save_comparison_csv,
)
from experiment_log import ExperimentLog

# ── Auto-discover all strategies ──────────────────────────────────────
discover_strategies()
STRATEGIES = get_registry()


# ── CLI ───────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Nasdaq-100 Daily-Close Backtesting System",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=(
            "To add a new strategy: create strategies/<name>.py, subclass Strategy,\n"
            "implement generate_weights(), and decorate with @register(key='<name>').\n"
            "No other file needs editing."
        ),
    )
    p.add_argument(
        "--strategy", "-s",
        choices=list(STRATEGIES.keys()) + ["all"],
        default="all",
        metavar="KEY",
        help=f"Strategy key (default: all). Available: {', '.join(sorted(STRATEGIES))}, all",
    )
    p.add_argument("--start",    default=None, help="Start date YYYY-MM-DD")
    p.add_argument("--end",      default=None, help="End date YYYY-MM-DD")
    p.add_argument("--capital",  type=float, default=1_000_000.0,
                   help="Initial capital USD (default: 1,000,000)")
    p.add_argument("--costs",    type=float, default=5.0,
                   help="One-way transaction cost bps (default: 5)")
    p.add_argument("--warmup",   type=int, default=0,
                   help="Warm-up days before first trade (default: 0)")
    p.add_argument("--data",     default=None, help="Override CSV data path")
    p.add_argument("--list",     action="store_true", help="List discovered strategies and exit")
    p.add_argument("--show-log", action="store_true", help="Print experiment log and exit")
    p.add_argument("--no-dashboard", action="store_true",
                   help="Skip per-strategy dashboard (faster for large runs)")
    p.add_argument("--no-plot",  action="store_true", help="Skip all chart generation")
    return p.parse_args()


# ── Main ──────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    log  = ExperimentLog(log_dir="results")

    if args.list:
        print(f"\nDiscovered strategies ({len(STRATEGIES)}):")
        for key in sorted(STRATEGIES):
            print(f"  {key:<22} → {STRATEGIES[key].name}")
        print()
        return

    if args.show_log:
        log.print_log()
        return

    # ── Load data ─────────────────────────────────────────────────────
    data_path = args.data or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "data", "nasdaq100_daily_5y.csv"
    )
    print(f"\nLoading data from: {data_path}")
    dataset_summary(data_path)
    prices = load_price_matrix(data_path)

    # ── Config ────────────────────────────────────────────────────────
    config = BacktestConfig(
        initial_capital=args.capital,
        transaction_cost_bps=args.costs,
        start_date=args.start,
        end_date=args.end,
        warmup_days=args.warmup,
    )
    print(
        f"Config: capital=${config.initial_capital:,.0f} | "
        f"costs={config.transaction_cost_bps} bps | "
        f"warmup={config.warmup_days} days"
    )

    # ── Select strategies ─────────────────────────────────────────────
    keys_to_run = sorted(STRATEGIES.keys()) if args.strategy == "all" \
                  else [args.strategy]
    strategies_to_run = [STRATEGIES[k] for k in keys_to_run]

    # ── Run ───────────────────────────────────────────────────────────
    results = []
    for strat in strategies_to_run:
        print(f"\nRunning: {strat.name} ...", end="", flush=True)
        engine = BacktestEngine(prices=prices, strategy=strat, config=config)
        result = engine.run()
        print(" done.")

        # Console summary
        result.print_summary()

        # Per-run file outputs (ledger, NAV, PnL, weights, metrics, manifest)
        result.save_results(output_dir="results")

        # Per-run visual dashboard
        if not args.no_plot and not args.no_dashboard:
            generate_report(result, output_dir="results")

        # Append to experiment log
        log.record(result)

        results.append(result)

    # ── Cross-run outputs ─────────────────────────────────────────────
    if len(results) > 1:
        print_comparison_table(results)
        save_comparison_csv(results, "results/comparison_table.csv")

    if not args.no_plot and len(results) >= 1:
        generate_comparison_report(
            results,
            output_dir="results",
            title="Nasdaq-100 Backtesting System – Strategy Comparison",
        )

    # Show log tail
    print()
    log.print_log(n=len(results) + 2)

    print("Done!  All outputs written to ./results/\n")


if __name__ == "__main__":
    main()
