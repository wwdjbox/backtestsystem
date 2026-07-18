# Nasdaq-100 Daily-Close Backtesting System

A modular, extensible backtesting framework for daily-close trading strategies
on 5 years of Nasdaq-100 constituent data (April 2021 – April 2026, 101 stocks).

---

## Quick Start

```bash
# 1. Install dependencies
pip install pandas numpy matplotlib

# 2. Place yourself inside the project folder
cd backtest/

# 3. Run all strategies
python run_backtest.py

# 4. Results are written to ./results/
```

---

## Project Structure

```
backtest/
├── run_backtest.py          ← ENTRY POINT — run this to reproduce all results
├── strategy.py              ← Abstract Strategy base class + plugin registry
├── engine.py                ← BacktestEngine, BacktestResult, BacktestConfig
├── data_loader.py           ← CSV → (dates × tickers) price matrix
├── reporting.py             ← Dashboards, comparison plots, CSV tables
├── experiment_log.py        ← Persistent log of every run
│
├── strategies/              ← One file per strategy, self-registering
│   ├── equal_weight.py
│   ├── momentum.py
│   ├── mean_reversion.py
│   ├── vol_weighted.py
│   ├── ma_cross.py
│   ├── risk_parity_mom.py
│   ├── rsi_reversion.py
│   ├── deliverable5.py      ← Benchmark 1, Benchmark 2 (exact spec)
│   └── novel_strategies.py  ← Novel 1 (Trend Strength), Novel 2 (Dual Momentum)
│
├── data/
│   └── nasdaq100_daily_5y.csv
│
└── results/                 ← Created at runtime
    ├── deliverable4/        ← Portfolio backtesting results
    ├── deliverable5/        ← Benchmark vs novel strategy results
    └── aapl/                ← Single-stock AAPL results
```

---

## Reproducing the Report Results

### Deliverable 3 — Single Stock (AAPL)

```bash
python run_backtest.py --data data/nasdaq100_daily_5y.csv --strategy equal_weight
```

Or run all strategies on AAPL only by editing the prices slice in run_backtest.py,
or use the pre-generated results in results/aapl/.

### Deliverable 4 — Portfolio Backtesting (all 101 stocks)

```bash
python run_backtest.py --strategy equal_weight
python run_backtest.py --strategy momentum
python run_backtest.py --strategy mean_reversion
python run_backtest.py --strategy ma_cross
python run_backtest.py --strategy rsi_reversion
python run_backtest.py --strategy vol_weighted
```

Or run all at once:

```bash
python run_backtest.py
```

Results saved to results/deliverable4/ (pre-generated and included).

### Deliverable 5 — Benchmark vs Novel Strategies

```bash
python run_backtest.py --strategy bench1_ma2050
python run_backtest.py --strategy bench2_mom30
python run_backtest.py --strategy novel1_trend_strength
python run_backtest.py --strategy novel2_dual_momentum
```

Results saved to results/deliverable5/ (pre-generated and included).

---

## All Available Strategies

```bash
python run_backtest.py --list
```

| Key                    | Description                                                                                      |
|------------------------|--------------------------------------------------------------------------------------------------|
| equal_weight           | Allocates 1/N equally across all 101 tickers every day. Rebalances daily. Primary benchmark.    |
| momentum               | Ranks stocks by 60-day return (skipping last 5 days to avoid short-term reversal). Buys top 10 in equal weight. |
| mean_reversion         | Buys the bottom 10 stocks by 20-day return, betting on a bounce back. Equal weight.             |
| vol_weighted           | Weights all 101 stocks inversely proportional to their 60-day trailing volatility. Lower-vol stocks get higher allocation. |
| ma_cross               | Holds stocks where 20-day SMA is above 60-day SMA (uptrend signal). Equal weight, max 20 positions. |
| risk_parity_mom        | Selects top 15 stocks by 60-day momentum, then sizes each position inversely proportional to its recent volatility so every stock contributes approximately equal risk to the portfolio. |
| rsi_reversion          | Computes 14-day RSI for each stock. Buys up to 10 stocks with RSI below 40 (oversold). Equal weight. Holds cash when none qualify. |
| bench1_ma2050          | Benchmark 1 (Deliverable 5): MA Cross with 20-day and 50-day SMA, equal weight across all qualifying stocks. |
| bench2_mom30           | Benchmark 2 (Deliverable 5): Momentum over 30-day lookback, top 10 stocks, equal weight.        |
| novel1_trend_strength  | Scores each stock by (SMA20 - SMA60) / std60 — a normalised trend quality measure. Filters to stocks with positive 60-day return. Selects top 15. Sharpe = 1.001. |
| novel2_dual_momentum   | Requires both absolute momentum (price above 120-day SMA) and relative momentum (top 10 by 60-day return). Holds cash if no stocks qualify. Sharpe = 0.997. |

---

## CLI Reference

```
python run_backtest.py [OPTIONS]

  --strategy KEY        Strategy to run (default: all)
  --start  YYYY-MM-DD   Backtest start date
  --end    YYYY-MM-DD   Backtest end date
  --capital FLOAT       Initial capital in USD (default: 1,000,000)
  --costs  FLOAT        One-way transaction cost in bps (default: 5)
  --warmup INT          Warm-up days before first trade (default: 0)
  --data   PATH         Override CSV data path
  --list                List all discovered strategies and exit
  --show-log            Print experiment log and exit
  --no-dashboard        Skip per-strategy dashboard chart
  --no-plot             Skip all chart generation
```

---

## Key Results Summary

### Deliverable 4: Portfolio Backtesting

| Strategy          | Weighting     | Total Return | Sharpe | Max DD   |
|-------------------|---------------|-------------|--------|----------|
| Equal Weight      | Uniform       | +129.96%    | 0.895  | -26.20%  |
| Momentum          | Uniform       | +236.00%    | 0.983  | -45.51%  |
| Mean Reversion    | Uniform       | +79.78%     | 0.510  | -42.26%  |
| MA Cross          | Uniform       | +315.16%    | 0.904  | -30.69%  |
| RSI Reversion     | Uniform       | +32.16%     | 0.565  | -33.05%  |
| Vol Weighted      | Risk-adjusted | +69.06%     | 0.691  | -25.99%  |

### Deliverable 5: Novel Strategies vs Benchmarks (Sharpe)

| Strategy                     | Sharpe | Beats both benchmarks? |
|------------------------------|--------|------------------------|
| Benchmark 1: MA Cross 20/50d | 0.728  | —                      |
| Benchmark 2: Momentum 30d    | 0.853  | —                      |
| Novel 1: Trend Strength      | 1.001  | ✓                      |
| Novel 2: Dual Momentum       | 0.997  | ✓                      |

---

## System Design

- **No lookahead**: engine slices `prices.iloc[:t+1]` before calling the strategy
- **Transaction costs**: `|Δshares × close| × (bps / 10,000)` per day, default 5 bps
- **Constraints**: no short selling, no leverage, daily close execution only
- **Plugin system**: drop a `.py` file in `strategies/`, decorate with `@register(key=...)`, done

## Dependencies

- Python >= 3.10
- pandas >= 1.5
- numpy >= 1.23
- matplotlib >= 3.6
