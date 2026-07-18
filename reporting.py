"""
reporting.py
============
Standardised visual and tabular outputs for one or many BacktestResult objects.

Public API
----------
  generate_report(result, output_dir)
      Full single-strategy dashboard (5-panel figure).

  generate_comparison_report(results, output_dir)
      Multi-strategy comparison dashboard (5 panels).

  print_comparison_table(results)
      Console-formatted side-by-side metrics table (sectioned).

  compare_results(results) -> pd.DataFrame
      (strategy × metric) DataFrame.

  save_comparison_csv(results, path)
      Persist comparison table as CSV.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
from matplotlib.colors import TwoSlopeNorm
from matplotlib.gridspec import GridSpec
import numpy as np
import pandas as pd

from engine import BacktestResult

# ── Palette & micro-style helpers ──────────────────────────────────────
COLORS = [
    "#2563eb", "#16a34a", "#dc2626", "#d97706",
    "#7c3aed", "#0891b2", "#db2777", "#65a30d",
    "#ea580c", "#6366f1",
]
_GRID  = dict(alpha=0.22, linewidth=0.65, color="#9ca3af")
_SPINE = "#e5e7eb"


def _ax_style(ax: plt.Axes) -> None:
    ax.set_facecolor("#fafafa")
    for s in ax.spines.values():
        s.set_edgecolor(_SPINE)
    ax.grid(True, **_GRID)
    ax.tick_params(labelsize=8, colors="#374151")


def _date_fmt(ax: plt.Axes, months: int = 6) -> None:
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=months))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right", fontsize=7)


def _pct_fmt(ax: plt.Axes, axis: str = "y") -> None:
    fmt = mticker.FuncFormatter(lambda v, _: f"{v:.0f}%")
    getattr(ax, f"{axis}axis").set_major_formatter(fmt)


# ══════════════════════════════════════════════════════════════════════
# 1.  Single-strategy dashboard
# ══════════════════════════════════════════════════════════════════════

def generate_report(
    result: BacktestResult,
    output_dir: str | Path | None = None,
) -> str:
    """
    Produce a 5-panel single-strategy dashboard.

    Panels
    ------
    1 (full-width) NAV curve (normalised) with cumulative cost-drag shading
    2  Daily P&L bars + 30-day rolling mean
    3  Drawdown area chart
    4  Monthly return heatmap  (Year × Month)
    5  Rolling 60-day Sharpe + position count overlay

    Returns path to saved PNG.
    """
    if output_dir is None:
        output_dir = Path("results")
    run_dir = Path(output_dir) / result.run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    m   = result.metrics
    fig = plt.figure(figsize=(16, 18), facecolor="white")
    fig.suptitle(
        f"{result.strategy_name}\n"
        f"Run {result.run_id}  ·  {m['start_date']} → {m['end_date']}",
        fontsize=13, fontweight="bold", y=0.995,
    )
    gs = GridSpec(3, 2, figure=fig, hspace=0.44, wspace=0.30)
    ax1 = fig.add_subplot(gs[0, :])
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[1, 1])
    ax4 = fig.add_subplot(gs[2, 0])
    ax5 = fig.add_subplot(gs[2, 1])

    col     = COLORS[0]
    r       = result.daily_returns
    nav     = result.nav
    dates   = nav.index
    initial = result.config.initial_capital

    # ── Panel 1: NAV + cost drag ───────────────────────────────────────
    norm = nav / initial
    gross_norm = norm + result.ledger["cum_costs"] / initial
    ax1.plot(dates, norm, color=col, linewidth=2.0, label="Net NAV", zorder=3)
    ax1.plot(dates, gross_norm, color="#9ca3af", linewidth=0.9,
             linestyle="--", label="Gross NAV (before costs)", zorder=2)
    ax1.fill_between(dates, norm, gross_norm,
                     alpha=0.28, color="#ef4444", label="Cumulative cost drag")
    ax1.axhline(1.0, color="#6b7280", linewidth=0.7, linestyle=":")
    ax1.set_title("NAV (normalised · start = 1.0)", fontsize=10, fontweight="bold")
    ax1.set_ylabel("Portfolio Value ×", fontsize=8)
    ax1.legend(fontsize=8, loc="upper left")
    _ax_style(ax1)
    _date_fmt(ax1, months=6)
    # Final-value annotation
    ax1.annotate(
        f" {norm.iloc[-1]:.2f}× ({m['total_return_pct']:+.1f}%)",
        xy=(dates[-1], norm.iloc[-1]), fontsize=8, color=col, va="center",
    )

    # ── Panel 2: Daily P&L ─────────────────────────────────────────────
    pnl_k = result.pnl / 1_000
    bar_c = ["#16a34a" if v >= 0 else "#dc2626" for v in pnl_k]
    ax2.bar(dates, pnl_k, color=bar_c, width=1, alpha=0.75)
    ax2.plot(dates, pnl_k.rolling(30).mean(), color="#1d4ed8",
             linewidth=1.6, label="30-day mean", zorder=3)
    ax2.axhline(0, color="#6b7280", linewidth=0.7)
    ax2.set_title("Daily P&L — net ($ thousands)", fontsize=10, fontweight="bold")
    ax2.set_ylabel("P&L ($k)", fontsize=8)
    ax2.legend(fontsize=7)
    _ax_style(ax2)
    _date_fmt(ax2)

    # ── Panel 3: Drawdown ──────────────────────────────────────────────
    dd = result.drawdown * 100
    ax3.fill_between(dates, dd, 0, alpha=0.50, color="#ef4444")
    ax3.plot(dates, dd, color="#dc2626", linewidth=0.9)
    idx_min = dd.idxmin()
    ax3.annotate(
        f"  {dd.min():.1f}%",
        xy=(idx_min, dd.min()), fontsize=8, color="#991b1b",
    )
    ax3.set_title("Drawdown (%)", fontsize=10, fontweight="bold")
    ax3.set_ylabel("Drawdown %", fontsize=8)
    _pct_fmt(ax3)
    _ax_style(ax3)
    _date_fmt(ax3)

    # ── Panel 4: Monthly return heatmap ───────────────────────────────
    pivot      = result.periodic["monthly_pivot"]
    month_cols = [c for c in pivot.columns if c != "Annual"]
    data       = pivot[month_cols].values.astype(float) * 100
    vmax       = max(np.nanmax(np.abs(data)), 1.0)
    norm_c     = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

    im = ax4.imshow(data, aspect="auto", cmap="RdYlGn", norm=norm_c)
    ax4.set_xticks(range(len(month_cols)))
    ax4.set_xticklabels(month_cols, fontsize=7)
    ax4.set_yticks(range(len(pivot.index)))
    ax4.set_yticklabels(pivot.index.astype(str), fontsize=7)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            v = data[i, j]
            if not np.isnan(v):
                txt_c = "white" if abs(v) > vmax * 0.65 else "black"
                ax4.text(j, i, f"{v:.1f}", ha="center", va="center",
                         fontsize=5.5, color=txt_c)
    plt.colorbar(im, ax=ax4, shrink=0.85, label="Return %")
    ax4.set_title("Monthly Returns (%)", fontsize=10, fontweight="bold")
    for sp in ax4.spines.values():
        sp.set_edgecolor(_SPINE)

    # ── Panel 5: Rolling Sharpe + position count ──────────────────────
    rs = r.rolling(60).mean() / r.rolling(60).std() * np.sqrt(252)
    ax5.plot(dates, rs, color=col, linewidth=1.5, label="60-day Sharpe", zorder=3)
    ax5.axhline(0, color="#6b7280", linewidth=0.7, linestyle="--")
    ax5.axhline(1, color="#16a34a", linewidth=0.6, linestyle=":", alpha=0.7)
    ax5.set_ylabel("Sharpe Ratio", fontsize=8)

    ax5b = ax5.twinx()
    n_pos = result.ledger["n_positions"]
    ax5b.fill_between(dates, n_pos, alpha=0.12, color=COLORS[3])
    ax5b.plot(dates, n_pos, color=COLORS[3], linewidth=0.7, alpha=0.7, label="# Positions")
    ax5b.set_ylabel("# Positions", fontsize=7, color=COLORS[3])
    ax5b.tick_params(axis="y", labelsize=7, colors=COLORS[3])

    lines  = ax5.get_lines() + ax5b.get_lines()
    labels = [l.get_label() for l in lines]
    ax5.legend(lines, labels, fontsize=7, loc="upper left")
    ax5.set_title("Rolling Sharpe (60d) & Positions", fontsize=10, fontweight="bold")
    _ax_style(ax5)
    _date_fmt(ax5)

    fig.tight_layout(rect=[0, 0, 1, 0.99])
    out = run_dir / "dashboard.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Dashboard        → {out}")
    return str(out)


# ══════════════════════════════════════════════════════════════════════
# 2.  Multi-strategy comparison dashboard
# ══════════════════════════════════════════════════════════════════════

def generate_comparison_report(
    results: list[BacktestResult],
    output_dir: str | Path = "results",
    title: str = "Strategy Comparison",
) -> str:
    """
    5-panel multi-strategy figure:
      1 (full) Normalised NAV curves
      2        Drawdown
      3        Rolling 60-day Sharpe
      4        Metrics heatmap  (strategies × KPIs, colour = relative rank)
      5        Annual return bar chart

    Saved to <output_dir>/comparison_dashboard.png.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(16, 20), facecolor="white")
    fig.suptitle(title, fontsize=14, fontweight="bold", y=0.995)
    gs = GridSpec(3, 2, figure=fig, hspace=0.44, wspace=0.30)
    ax_nav  = fig.add_subplot(gs[0, :])
    ax_dd   = fig.add_subplot(gs[1, 0])
    ax_shr  = fig.add_subplot(gs[1, 1])
    ax_heat = fig.add_subplot(gs[2, 0])
    ax_yr   = fig.add_subplot(gs[2, 1])

    initial = results[0].config.initial_capital

    # ── 1. NAV curves ─────────────────────────────────────────────────
    for i, res in enumerate(results):
        ax_nav.plot(res.nav.index, res.nav / initial,
                    color=COLORS[i % len(COLORS)], linewidth=1.8,
                    label=res.strategy_name)
    ax_nav.axhline(1.0, color="#9ca3af", linewidth=0.7, linestyle=":")
    ax_nav.set_title("NAV (normalised · start = 1.0)", fontsize=10, fontweight="bold")
    ax_nav.set_ylabel("Portfolio Value ×", fontsize=8)
    ax_nav.legend(fontsize=8, loc="upper left", ncol=2)
    _ax_style(ax_nav)
    _date_fmt(ax_nav, months=6)

    # ── 2. Drawdown ───────────────────────────────────────────────────
    for i, res in enumerate(results):
        dd = res.drawdown * 100
        ax_dd.fill_between(res.nav.index, dd, 0,
                           alpha=0.18, color=COLORS[i % len(COLORS)])
        ax_dd.plot(res.nav.index, dd, color=COLORS[i % len(COLORS)],
                   linewidth=1.1, label=res.strategy_name)
    ax_dd.set_title("Drawdown (%)", fontsize=10, fontweight="bold")
    ax_dd.set_ylabel("Drawdown %", fontsize=8)
    _pct_fmt(ax_dd)
    ax_dd.legend(fontsize=7)
    _ax_style(ax_dd)
    _date_fmt(ax_dd)

    # ── 3. Rolling Sharpe ─────────────────────────────────────────────
    for i, res in enumerate(results):
        r  = res.daily_returns
        rs = r.rolling(60).mean() / r.rolling(60).std() * np.sqrt(252)
        ax_shr.plot(res.nav.index, rs, color=COLORS[i % len(COLORS)],
                    linewidth=1.4, label=res.strategy_name)
    ax_shr.axhline(0, color="#9ca3af", linewidth=0.7, linestyle="--")
    ax_shr.axhline(1, color="#16a34a", linewidth=0.6, linestyle=":", alpha=0.7)
    ax_shr.set_title("Rolling 60-day Sharpe", fontsize=10, fontweight="bold")
    ax_shr.set_ylabel("Sharpe Ratio", fontsize=8)
    ax_shr.legend(fontsize=7)
    _ax_style(ax_shr)
    _date_fmt(ax_shr)

    # ── 4. Metrics heatmap ────────────────────────────────────────────
    kpi_keys = [
        ("CAGR %",      "cagr_pct",                   False),
        ("Sharpe",      "sharpe_ratio",                False),
        ("Sortino",     "sortino_ratio",               False),
        ("Calmar",      "calmar_ratio",                False),
        ("MaxDD %",     "max_drawdown_pct",            True),   # lower = better
        ("Win %",       "win_rate_pct",                False),
        ("VaR95 %",     "var_95_pct",                  True),   # lower (less negative) = better
        ("AvgT/O %",    "avg_daily_turnover_pct",      True),
        ("CostDrag %",  "ann_cost_drag_pct",           True),
        ("AvgPos",      "avg_n_positions",             False),
    ]
    kpi_labels = [k for k, _, _ in kpi_keys]
    matrix     = np.array([[res.metrics[mk] for _, mk, _ in kpi_keys]
                            for res in results], dtype=float)

    # Rank-normalise each column  (0 = worst in column, 1 = best)
    ranked = np.zeros_like(matrix)
    for col_i, (_, _, invert) in enumerate(kpi_keys):
        col = matrix[:, col_i]
        lo, hi = np.nanmin(col), np.nanmax(col)
        span = hi - lo if hi > lo else 1.0
        norm_col = (col - lo) / span
        ranked[:, col_i] = (1 - norm_col) if invert else norm_col

    im = ax_heat.imshow(ranked.T, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
    strat_lbls = [r.strategy_name[:14] for r in results]
    ax_heat.set_xticks(range(len(results)))
    ax_heat.set_xticklabels(strat_lbls, fontsize=6.5, rotation=15, ha="right")
    ax_heat.set_yticks(range(len(kpi_labels)))
    ax_heat.set_yticklabels(kpi_labels, fontsize=8)
    for j in range(len(results)):
        for i in range(len(kpi_keys)):
            v = matrix[j, i]
            lbl = f"{v:.2f}" if abs(v) < 1000 else f"{v:,.0f}"
            ax_heat.text(j, i, lbl, ha="center", va="center",
                         fontsize=6, color="black")
    ax_heat.set_title("Metrics Heatmap (green = better within row)",
                      fontsize=9, fontweight="bold")
    plt.colorbar(im, ax=ax_heat, shrink=0.8, label="Relative rank")
    for sp in ax_heat.spines.values():
        sp.set_edgecolor(_SPINE)

    # ── 5. Annual returns ─────────────────────────────────────────────
    all_years = sorted({y for res in results
                        for y in res.periodic["yearly_series"].index})
    x     = np.arange(len(all_years))
    width = 0.8 / len(results)
    for i, res in enumerate(results):
        yr   = res.periodic["yearly_series"]
        vals = [yr.get(y, np.nan) * 100 for y in all_years]
        off  = (i - len(results) / 2 + 0.5) * width
        ax_yr.bar(x + off, vals, width=width * 0.88,
                  color=COLORS[i % len(COLORS)], alpha=0.85,
                  label=res.strategy_name)
    ax_yr.axhline(0, color="#6b7280", linewidth=0.7)
    ax_yr.set_xticks(x)
    ax_yr.set_xticklabels(all_years, fontsize=8)
    ax_yr.set_title("Annual Returns (%)", fontsize=10, fontweight="bold")
    ax_yr.set_ylabel("Return %", fontsize=8)
    _pct_fmt(ax_yr)
    ax_yr.legend(fontsize=7, loc="upper left")
    _ax_style(ax_yr)

    fig.tight_layout(rect=[0, 0, 1, 0.995])
    out = output_dir / "comparison_dashboard.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Comparison dashboard → {out}")
    return str(out)


# ══════════════════════════════════════════════════════════════════════
# 3.  Console comparison table
# ══════════════════════════════════════════════════════════════════════

_COL_SPEC = [
    # (metrics_key,                  display_label,    fmt)
    ("total_return_pct",            "Total Ret %",    "{:>+8.2f}%"),
    ("cagr_pct",                    "CAGR %",         "{:>+7.2f}%"),
    ("ann_volatility_pct",          "Vol %",          "{:>6.2f}%"),
    ("sharpe_ratio",                "Sharpe",         "{:>8.4f}"),
    ("sortino_ratio",               "Sortino",        "{:>8.4f}"),
    ("calmar_ratio",                "Calmar",         "{:>8.4f}"),
    ("win_rate_pct",                "Win %",          "{:>6.1f}%"),
    ("best_day_pct",                "Best Day",       "{:>+8.2f}%"),
    ("worst_day_pct",               "Worst Day",      "{:>+8.2f}%"),
    ("var_95_pct",                  "VaR 95%",        "{:>+8.2f}%"),
    ("max_drawdown_pct",            "Max DD %",       "{:>+7.2f}%"),
    ("max_dd_duration_days",        "DD Dur (d)",     "{:>9d}"),
    ("final_nav",                   "Final NAV $",    "{:>12,.0f}"),
    ("total_pnl_usd",               "Total P&L $",    "{:>+12,.0f}"),
    ("avg_n_positions",             "Avg Pos",        "{:>7.1f}"),
    ("avg_daily_turnover_pct",      "Avg T/O %",      "{:>8.2f}%"),
    ("ann_cost_drag_pct",           "CostDrag %",     "{:>9.4f}%"),
    ("total_transaction_costs_usd", "Costs $",        "{:>10,.0f}"),
]

_SECTIONS = [
    ("RETURNS",       ["total_return_pct","cagr_pct","ann_volatility_pct",
                        "win_rate_pct","best_day_pct","worst_day_pct","var_95_pct"]),
    ("RISK-ADJUSTED", ["sharpe_ratio","sortino_ratio","calmar_ratio"]),
    ("DRAWDOWN",      ["max_drawdown_pct","max_dd_duration_days"]),
    ("CAPITAL",       ["final_nav","total_pnl_usd"]),
    ("ACTIVITY",      ["avg_n_positions","avg_daily_turnover_pct",
                        "ann_cost_drag_pct","total_transaction_costs_usd"]),
]


def compare_results(results: list[BacktestResult]) -> pd.DataFrame:
    """Return a (strategy × metric) DataFrame with display labels as columns."""
    key_to_lbl = {k: lbl for k, lbl, _ in _COL_SPEC}
    rows = {}
    for res in results:
        rows[res.strategy_name] = {
            key_to_lbl[k]: res.metrics.get(k, np.nan)
            for k, _, _ in _COL_SPEC
            if k in res.metrics
        }
    df = pd.DataFrame(rows).T
    df.index.name = "Strategy"
    return df


def print_comparison_table(results: list[BacktestResult]) -> None:
    """Sectioned, formatted side-by-side comparison table to stdout."""
    n      = len(results)
    W_LBL  = 15          # metric label column width
    W_COL  = 16          # per-strategy column width
    total  = W_LBL + n * W_COL
    SEP    = "─" * total

    # Truncate long strategy names
    headers = [res.strategy_name[:W_COL - 2] for res in results]
    hdr_row = f"{'Metric':<{W_LBL}}" + "".join(f" {h:<{W_COL-1}}" for h in headers)

    key_to_spec = {k: (lbl, fmt) for k, lbl, fmt in _COL_SPEC}

    lines = [
        "",
        "═" * total,
        "  STRATEGY COMPARISON",
        f"  Period : {results[0].metrics['start_date']} → {results[0].metrics['end_date']}",
        f"  Capital: ${results[0].config.initial_capital:,.0f}   "
        f"Costs: {results[0].config.transaction_cost_bps} bps",
        "═" * total,
        hdr_row,
    ]

    for section_name, keys in _SECTIONS:
        lines.append(f"\n  ── {section_name} {'─'*(total - len(section_name) - 6)}")
        for key in keys:
            if key not in key_to_spec:
                continue
            lbl, fmt = key_to_spec[key]
            row = f"  {lbl:<{W_LBL-2}}"
            for res in results:
                val = res.metrics.get(key, float("nan"))
                try:
                    cell = fmt.format(int(val) if "d}" in fmt else val)
                except (ValueError, TypeError):
                    cell = "N/A"
                row += f" {cell:>{W_COL-1}}"
            lines.append(row)

    lines += ["", "═" * total, ""]
    print("\n".join(lines))


# ══════════════════════════════════════════════════════════════════════
# 4.  CSV output
# ══════════════════════════════════════════════════════════════════════

def save_comparison_csv(
    results: list[BacktestResult],
    output_path: str | Path = "results/comparison_table.csv",
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    compare_results(results).to_csv(output_path)
    print(f"  Comparison CSV   → {output_path}")


# ── Backwards-compat shim ──────────────────────────────────────────────
def plot_equity_curves(
    results: list[BacktestResult],
    output_path: str = "results/equity_curves.png",
    title: str = "Strategy Comparison",
) -> None:
    src = generate_comparison_report(
        results,
        output_dir=str(Path(output_path).parent),
        title=title,
    )
    if src != str(output_path):
        shutil.copy(src, output_path)
