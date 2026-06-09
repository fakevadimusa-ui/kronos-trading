"""
run_research.py — Master research runner for NY Open Liquidity Trap strategy.

Runs all variants, grid search, prop sim, stress tests, and saves full report.
This is the single entry point. Run this file to get all results.

Usage:
  cd research/liquidity_trap
  python run_research.py

Output:
  results/trades_variant_A.csv
  results/trades_variant_B.csv
  results/trades_variant_C.csv
  results/grid_results.csv
  results/prop_sim_results.json
  results/summary.json
  (then results_summary.md written manually from the output)
"""

from __future__ import annotations

import json
import sys
import os
from pathlib import Path

# Ensure local modules importable
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd

from data_loader import prepare_data
from backtest_liquidity_trap import run_variant, TOTAL_FRICTION
from metrics import compute_metrics, print_metrics, grade_strategy
from prop_sim import run_all_modes, print_sim_results

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def run_slippage_stress(trades: list, variant: str, base_r: float = 2.0) -> dict:
    """
    Test strategy robustness to higher friction.
    Re-applies different friction levels to existing trades.
    Returns metrics at each friction level.
    """
    friction_levels = [0.0, 0.5, 1.0, 2.0, 3.0]  # extra pts beyond base
    stress = {}
    for extra in friction_levels:
        modified_pnl = np.array([
            t.pnl_pts_net - extra for t in trades
        ])
        gross = modified_pnl[modified_pnl > 0].sum() if any(modified_pnl > 0) else 0
        loss  = abs(modified_pnl[modified_pnl <= 0].sum()) if any(modified_pnl <= 0) else 0
        pf    = gross / loss if loss > 0 else float("nan")
        exp   = modified_pnl.mean()
        stress[f"+{extra}pts"] = {
            "profit_factor": round(pf, 3),
            "expectancy_pts": round(exp, 3),
            "total_pts": round(modified_pnl.sum(), 2),
        }
    return stress


def run_time_window_test(df: pd.DataFrame, variant: str) -> dict:
    """Test different NY window definitions."""
    windows = {
        "9:30-10:00": (9.5, 10.0),
        "9:30-10:30": (9.5, 10.5),
        "9:30-11:00": (9.5, 11.0),
        "9:30-11:30": (9.5, 11.5),
    }
    results = {}
    original_ny = df["is_ny_window"].copy()

    for label, (start, end) in windows.items():
        df["is_ny_window"] = (df["tod_et"] >= start) & (df["tod_et"] < end)
        trades = run_variant(df, variant, target_r=2.0, min_sweep_pts=0.25)
        m = compute_metrics(trades) if trades else {"n_trades": 0}
        results[label] = {
            "n_trades":      m.get("n_trades", 0),
            "win_rate":      m.get("win_rate", None),
            "profit_factor": m.get("profit_factor", None),
            "expectancy_pts": m.get("expectancy_pts", None),
        }

    df["is_ny_window"] = original_ny  # restore
    return results


def run_in_out_of_sample(df: pd.DataFrame, variant: str) -> dict:
    """
    Split data: first 60% in-sample, last 40% out-of-sample.
    Compare metrics.
    """
    dates = sorted(df["date_et"].unique())
    split_idx = int(len(dates) * 0.60)
    in_dates  = set(dates[:split_idx])
    out_dates = set(dates[split_idx:])

    df_in  = df[df["date_et"].isin(in_dates)]
    df_out = df[df["date_et"].isin(out_dates)]

    trades_in  = run_variant(df_in,  variant, target_r=2.0, min_sweep_pts=0.25)
    trades_out = run_variant(df_out, variant, target_r=2.0, min_sweep_pts=0.25)

    m_in  = compute_metrics(trades_in)  if trades_in  else {"n_trades": 0}
    m_out = compute_metrics(trades_out) if trades_out else {"n_trades": 0}

    return {
        "in_sample": {
            "n_days":    len(in_dates),
            "n_trades":  m_in.get("n_trades", 0),
            "win_rate":  m_in.get("win_rate", None),
            "profit_factor": m_in.get("profit_factor", None),
            "expectancy_pts": m_in.get("expectancy_pts", None),
            "total_pts": m_in.get("total_pts", None),
        },
        "out_of_sample": {
            "n_days":    len(out_dates),
            "n_trades":  m_out.get("n_trades", 0),
            "win_rate":  m_out.get("win_rate", None),
            "profit_factor": m_out.get("profit_factor", None),
            "expectancy_pts": m_out.get("expectancy_pts", None),
            "total_pts": m_out.get("total_pts", None),
        },
    }


def main():
    print("\n" + "=" * 70)
    print("  NY OPEN LIQUIDITY TRAP — RESEARCH BACKTEST")
    print("  Full analysis: 3 variants × grid search × prop sim × stress test")
    print("=" * 70)

    # ── Load data ─────────────────────────────────────────────────────────
    print("\n[1/7] Loading and preprocessing data...")
    df = prepare_data()
    n_bars  = len(df)
    n_days  = df["date_et"].nunique()
    src     = df["_source"].iloc[0]
    date_lo = df["date_et"].min()
    date_hi = df["date_et"].max()
    print(f"  Source  : {src}")
    print(f"  Bars    : {n_bars:,}")
    print(f"  Days    : {n_days}")
    print(f"  Range   : {date_lo} to {date_hi}")
    print(f"  WARNING : {n_days} trading days is below ideal (need 200+ for high confidence)")

    # ── Run all variants at baseline ──────────────────────────────────────
    print("\n[2/7] Running baseline backtest (target=2R, min_sweep=0.25pts)...")
    all_trades = {}
    all_metrics = {}

    for variant in ["A", "B", "C"]:
        trades = run_variant(df, variant, target_r=2.0, min_sweep_pts=0.25)
        m = compute_metrics(trades) if trades else {"n_trades": 0}
        all_trades[variant]  = trades
        all_metrics[variant] = m

        # Save trade log
        if trades:
            trade_rows = []
            for t in trades:
                row = {
                    "variant": t.variant,
                    "date": t.date,
                    "direction": t.direction,
                    "entry_price": t.entry_price,
                    "stop_price": t.stop_price,
                    "target_price": t.target_price,
                    "exit_price": t.exit_price,
                    "exit_reason": t.exit_reason,
                    "risk_pts": t.risk_pts,
                    "reward_pts": t.reward_pts,
                    "pnl_pts": t.pnl_pts,
                    "pnl_pts_net": t.pnl_pts_net,
                    "hold_bars": t.hold_bars,
                    "swept_level": t.swept_level,
                    "sweep_extreme": t.sweep_extreme,
                    "tod_entry_et": t.tod_entry_et,
                    "ref_level": t.ref_level,
                }
                trade_rows.append(row)
            pd.DataFrame(trade_rows).to_csv(
                RESULTS_DIR / f"trades_variant_{variant}.csv", index=False
            )

        print_metrics(m, f"Variant {variant}")
        print(f"  Grade: {grade_strategy(m)}")

    # ── Grid search ───────────────────────────────────────────────────────
    print("\n[3/7] Running grid search (3 variants × 3 R-values × 3 sweep sizes)...")
    grid_rows = []
    for variant in ["A", "B", "C"]:
        for target_r in [1.5, 2.0, 3.0]:
            for min_sweep_pts in [0.25, 0.50, 1.0]:
                trades = run_variant(df, variant, target_r=target_r,
                                     min_sweep_pts=min_sweep_pts)
                m = compute_metrics(trades) if trades else {"n_trades": 0}
                grid_rows.append({
                    "variant": variant,
                    "target_r": target_r,
                    "min_sweep_pts": min_sweep_pts,
                    "n_trades": m.get("n_trades", 0),
                    "win_rate": m.get("win_rate"),
                    "profit_factor": m.get("profit_factor"),
                    "expectancy_pts": m.get("expectancy_pts"),
                    "total_pts": m.get("total_pts"),
                    "max_dd_pts": m.get("max_dd_pts"),
                    "trades_per_day": m.get("trades_per_day"),
                    "grade": grade_strategy(m),
                })

    grid_df = pd.DataFrame(grid_rows)
    grid_df.to_csv(RESULTS_DIR / "grid_results.csv", index=False)

    # Show best combos
    valid_grid = grid_df[grid_df["profit_factor"].notna()].sort_values(
        "profit_factor", ascending=False
    )
    print("\n  Top 10 grid combinations (by profit factor):")
    print(valid_grid.head(10)[[
        "variant", "target_r", "min_sweep_pts", "n_trades",
        "win_rate", "profit_factor", "expectancy_pts", "total_pts", "grade"
    ]].to_string(index=False))

    # ── In/out of sample ─────────────────────────────────────────────────
    print("\n[4/7] In-sample vs out-of-sample validation...")
    ios_results = {}
    for variant in ["A", "B", "C"]:
        ios = run_in_out_of_sample(df, variant)
        ios_results[variant] = ios
        print(f"\n  Variant {variant}:")
        ins = ios["in_sample"]
        oos = ios["out_of_sample"]
        print(f"    In-sample  ({ins['n_days']} days): {ins['n_trades']} trades | "
              f"WR={ins['win_rate']} PF={ins['profit_factor']} E={ins['expectancy_pts']}")
        print(f"    Out-sample ({oos['n_days']} days): {oos['n_trades']} trades | "
              f"WR={oos['win_rate']} PF={oos['profit_factor']} E={oos['expectancy_pts']}")

    # ── Slippage stress test ──────────────────────────────────────────────
    print("\n[5/7] Slippage stress test...")
    stress_results = {}
    for variant in ["A", "B", "C"]:
        if all_trades[variant]:
            stress = run_slippage_stress(all_trades[variant], variant)
            stress_results[variant] = stress
            print(f"\n  Variant {variant} — extra friction impact on P&L:")
            for lvl, s in stress.items():
                print(f"    {lvl}: PF={s['profit_factor']} | E={s['expectancy_pts']:.3f} | "
                      f"Total={s['total_pts']:.1f}pts")

    # ── Time window test ──────────────────────────────────────────────────
    print("\n[6/7] Time window sensitivity (best variant)...")
    # Use best variant by profit factor
    best_variant = max(all_metrics, key=lambda v: all_metrics[v].get("profit_factor") or 0)
    window_results = run_time_window_test(df, best_variant)
    print(f"\n  Variant {best_variant} across different NY windows:")
    for label, r in window_results.items():
        print(f"    {label}: trades={r['n_trades']} WR={r['win_rate']} "
              f"PF={r['profit_factor']} E={r['expectancy_pts']}")

    # ── Prop firm simulation ──────────────────────────────────────────────
    print("\n[7/7] Monte Carlo prop firm simulation (2,000 trials per mode)...")
    prop_results = {}

    for variant in ["A", "B", "C"]:
        trades = all_trades[variant]
        if not trades or len(trades) < 5:
            print(f"\n  Variant {variant}: insufficient trades for simulation")
            continue

        pnl_pts  = np.array([t.pnl_pts_net for t in trades])
        stop_pts = np.array([t.risk_pts     for t in trades])
        tpd      = all_metrics[variant].get("trades_per_day", 0.5)

        results = run_all_modes(pnl_pts, stop_pts, tpd, n_sims=2000)
        prop_results[variant] = results
        print_sim_results(results, f"Variant {variant}")

    # ── Save full summary ─────────────────────────────────────────────────
    summary = {
        "data_source": src,
        "date_range": {"start": str(date_lo), "end": str(date_hi)},
        "n_trading_days": n_days,
        "n_bars": n_bars,
        "variants": {},
        "grid_top_5": valid_grid.head(5).to_dict("records") if len(valid_grid) > 0 else [],
        "in_out_of_sample": ios_results,
        "slippage_stress": stress_results,
        "time_window_test": {best_variant: window_results},
    }

    for variant in ["A", "B", "C"]:
        m = {k: v for k, v in all_metrics[variant].items() if k != "pnl_series"}
        summary["variants"][variant] = {
            "metrics": m,
            "grade": grade_strategy(all_metrics[variant]),
            "prop_sim": prop_results.get(variant, []),
        }

    with open(RESULTS_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print("\n" + "=" * 70)
    print("  FINAL SUMMARY")
    print("=" * 70)
    print(f"\n  Data: {n_days} days of real ES 5-min data")
    print(f"  WARNING: {n_days} days is INSUFFICIENT for high-confidence results.")
    print(f"  Need 200+ trading days (about 10 months) minimum.\n")

    best_pf   = -1
    best_name = None
    for variant in ["A", "B", "C"]:
        pf = all_metrics[variant].get("profit_factor") or -1
        g  = grade_strategy(all_metrics[variant])
        nt = all_metrics[variant].get("n_trades", 0)
        print(f"  Variant {variant}: {nt} trades | PF={pf} | Grade: {g}")
        if pf > best_pf:
            best_pf = pf
            best_name = variant

    if best_name:
        print(f"\n  Best variant: {best_name} (PF={best_pf:.3f})")
        print(f"  Grade: {grade_strategy(all_metrics[best_name])}")

    print(f"\n  Results saved to: {RESULTS_DIR}")
    print("=" * 70)

    return summary


if __name__ == "__main__":
    main()
