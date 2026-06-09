"""
backtest_liquidity_trap.py — NY Open Liquidity Trap Strategy Backtester

Tests three variants of the same core pattern:
  - Price sweeps a known reference level (liquidity grab)
  - Sweep FAILS (price cannot accept beyond the level)
  - Price reclaims back inside prior range
  - We enter in the direction of the reclaim
  - Stop beyond sweep extreme, target fixed R

Variants:
  A — Overnight High/Low Sweep Reclaim
  B — Opening Range Breakout Failure (15-min OR)
  C — Previous Day High/Low Sweep Reclaim

NO lookahead bias:
  - All signals computed on bar CLOSE only
  - Entry at CLOSE of signal bar (worst-case; see ENTRY_AT_CLOSE flag)
  - Levels (overnight H/L, PDH/PDL, OR) computed from data available at signal time
  - VWAP computed rolling from 09:30 — no daily-close contamination

Commissions & slippage:
  COMMISSION_PER_SIDE_PTS : 0.10 pts  (approx $0.50/MES per side = $1/RT)
  SLIPPAGE_PER_SIDE_PTS   : 0.50 pts  (2 ticks, conservative)
  TOTAL_FRICTION_PTS      : 1.20 pts  roundtrip (both sides)
"""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ─── Instrument parameters ────────────────────────────────────────────────────
ES_PTS_PER_DOLLAR = 50.0   # ES: $50/point
MES_PTS_PER_DOLLAR = 5.0   # MES: $5/point
TICK_SIZE = 0.25           # minimum price increment

# ─── Friction (roundtrip) ─────────────────────────────────────────────────────
COMMISSION_RT_PTS = 0.10   # 2 × $0.25 commission per MES = $0.50 RT ÷ $5/pt
SLIPPAGE_RT_PTS   = 0.50   # 2 ticks each side = 4 ticks = 1.0 pt... use 0.5 conservative
TOTAL_FRICTION    = COMMISSION_RT_PTS + SLIPPAGE_RT_PTS  # 0.60 pts per trade

# ─── Session filter ────────────────────────────────────────────────────────────
NY_OPEN_HOUR   = 9.5    # 09:30 ET
NY_CLOSE_HOUR  = 11.5   # 11:30 ET — hard stop, close all positions

# ─── Strategy parameters ──────────────────────────────────────────────────────
MIN_SWEEP_TICKS  = 1       # sweep must exceed level by at least N ticks (0.25 pts each)
MAX_SWEEP_TICKS  = 40      # sweep that is too large → likely gap, skip
STOP_EXTRA_TICKS = 1       # stop set 1 tick beyond sweep extreme
MAX_HOLD_BARS    = 18      # 90 minutes at 5min bars — force close
SIGNAL_COOLDOWN  = 6       # min bars between signals on same day (30 min)

# ─── Target R values to test ──────────────────────────────────────────────────
TARGET_R_LIST = [1.5, 2.0, 3.0]

# ─── Dead market filter ───────────────────────────────────────────────────────
MIN_ATR = 2.0   # skip if 14-bar ATR < 2 pts (dead market)


@dataclass
class Trade:
    """Represents a single completed trade."""
    variant: str
    date: str
    entry_bar_idx: int
    direction: str           # 'long' or 'short'
    entry_price: float
    stop_price: float
    target_price: float
    risk_pts: float          # stop distance in points
    reward_pts: float        # target distance in points
    rr_ratio: float
    exit_price: float = 0.0
    exit_reason: str = ""    # 'TP', 'SL', 'TIME', 'EOD'
    pnl_pts: float = 0.0     # net P&L in ES points (before × $5 or $50)
    pnl_pts_net: float = 0.0 # after friction
    hold_bars: int = 0
    swept_level: float = 0.0
    sweep_extreme: float = 0.0
    tod_entry_et: float = 0.0
    ref_level: str = ""       # 'overnight_low', 'or_low', 'pdl', etc.


def _round_tick(price: float) -> float:
    return round(price / TICK_SIZE) * TICK_SIZE


def _detect_sweep_reclaim(
    bars: pd.DataFrame,
    level: float,
    direction: str,  # 'long' = sweep below, then reclaim above; 'short' = sweep above, reclaim below
    min_sweep: float,
    max_sweep: float,
    bar_idx: int,
) -> tuple[bool, float]:
    """
    Check if bar at bar_idx contains or just completed a sweep+reclaim signal.

    For LONG (bullish reclaim):
      - bar.low < level - min_sweep  (swept below level)
      - bar.close > level            (reclaimed above level — same bar OK)
      - sweep size not excessive

    Returns (signal_triggered, sweep_extreme_price)
    """
    bar = bars.iloc[bar_idx]
    if direction == "long":
        swept = bar["Low"] <= level - min_sweep
        reclaimed = bar["Close"] > level
        extreme = bar["Low"]
        sweep_size = level - extreme
    else:  # short
        swept = bar["High"] >= level + min_sweep
        reclaimed = bar["Close"] < level
        extreme = bar["High"]
        sweep_size = extreme - level

    if not swept or not reclaimed:
        return False, np.nan
    if sweep_size > max_sweep:
        return False, np.nan

    return True, extreme


def run_variant(
    df: pd.DataFrame,
    variant: str,
    target_r: float = 2.0,
    min_sweep_pts: float = 0.25,
    max_sweep_pts: float = 10.0,
    verbose: bool = False,
) -> list[Trade]:
    """
    Core backtesting loop for one variant at one parameter set.

    variant: 'A' (overnight H/L), 'B' (opening range), 'C' (prev day H/L)
    """
    trades: list[Trade] = []
    dates = sorted(df["date_et"].unique())

    for date in dates:
        day_df = df[df["date_et"] == date].copy()
        day_df = day_df.reset_index(drop=True)

        # We only trade the NY window
        ny_bars = day_df[day_df["is_ny_window"]].copy()
        if len(ny_bars) < 3:
            continue

        # Get reference levels for this day
        row0 = day_df.iloc[0]

        # ── Pull reference levels depending on variant ─────────────────────
        if variant == "A":
            # Variant A: Overnight session high/low
            long_level  = row0["overnight_low"]
            short_level = row0["overnight_high"]
            long_ref    = "overnight_low"
            short_ref   = "overnight_high"
        elif variant == "B":
            # Variant B: Opening range (first 15 min)
            long_level  = row0["or_low"]
            short_level = row0["or_high"]
            long_ref    = "or_low"
            short_ref   = "or_high"
        elif variant == "C":
            # Variant C: Previous day RTH high/low
            long_level  = row0["pdl"]
            short_level = row0["pdh"]
            long_ref    = "pdl"
            short_ref   = "pdh"
        else:
            raise ValueError(f"Unknown variant: {variant}")

        # Dead market filter: check ATR
        atr = day_df["atr14"].median()
        if atr < MIN_ATR:
            continue

        # Track one trade per direction per day
        long_fired  = False
        short_fired = False
        last_signal_bar = -SIGNAL_COOLDOWN - 1
        in_trade = False
        active_trade: Optional[Trade] = None
        active_start_idx = -1

        # Iterate through NY window bars
        ny_indices = ny_bars.index.tolist()

        for pos, bar_idx in enumerate(ny_indices):
            bar = day_df.iloc[bar_idx]
            tod = bar["tod_et"]

            # ── Manage active trade ───────────────────────────────────────
            if in_trade and active_trade is not None:
                t = active_trade
                hold = bar_idx - active_start_idx

                if t.direction == "long":
                    stop_hit   = bar["Low"]  <= t.stop_price
                    target_hit = bar["High"] >= t.target_price
                else:
                    stop_hit   = bar["High"] >= t.stop_price
                    target_hit = bar["Low"]  <= t.target_price

                time_out = (tod >= NY_CLOSE_HOUR) or (hold >= MAX_HOLD_BARS)

                if target_hit and stop_hit:
                    # Conservative: assume stop hit first (avoid cherry-pick)
                    exit_price  = t.stop_price
                    exit_reason = "SL"
                elif target_hit:
                    exit_price  = t.target_price
                    exit_reason = "TP"
                elif stop_hit:
                    exit_price  = t.stop_price
                    exit_reason = "SL"
                elif time_out:
                    exit_price  = bar["Close"]
                    exit_reason = "TIME" if hold >= MAX_HOLD_BARS else "EOD"
                else:
                    continue

                t.exit_price    = exit_price
                t.exit_reason   = exit_reason
                t.hold_bars     = hold
                t.pnl_pts       = (exit_price - t.entry_price) if t.direction == "long" \
                                   else (t.entry_price - exit_price)
                t.pnl_pts_net   = t.pnl_pts - TOTAL_FRICTION
                trades.append(t)
                in_trade     = False
                active_trade = None
                continue

            # ── Check for new signals (one trade per day per direction) ───
            if in_trade:
                continue

            # Cooldown between signals
            if bar_idx - last_signal_bar < SIGNAL_COOLDOWN:
                continue

            # ── LONG setup ────────────────────────────────────────────────
            if not long_fired and not np.isnan(long_level):
                signal, sweep_ext = _detect_sweep_reclaim(
                    day_df, long_level, "long",
                    min_sweep_pts, max_sweep_pts, bar_idx
                )
                if signal:
                    stop  = _round_tick(sweep_ext - STOP_EXTRA_TICKS * TICK_SIZE)
                    risk  = bar["Close"] - stop
                    if risk <= 0:
                        pass  # malformed signal
                    else:
                        target  = bar["Close"] + risk * target_r
                        entry_p = bar["Close"]

                        t = Trade(
                            variant=variant,
                            date=str(date),
                            entry_bar_idx=bar_idx,
                            direction="long",
                            entry_price=entry_p,
                            stop_price=stop,
                            target_price=target,
                            risk_pts=risk,
                            reward_pts=risk * target_r,
                            rr_ratio=target_r,
                            swept_level=long_level,
                            sweep_extreme=sweep_ext,
                            tod_entry_et=tod,
                            ref_level=long_ref,
                        )
                        in_trade       = True
                        active_trade   = t
                        active_start_idx = bar_idx
                        long_fired     = True
                        last_signal_bar = bar_idx
                        if verbose:
                            print(f"  [{date} {tod:.2f}h] LONG entry @ {entry_p:.2f} "
                                  f"stop {stop:.2f} target {target:.2f} "
                                  f"risk {risk:.2f}pts")
                    continue

            # ── SHORT setup ───────────────────────────────────────────────
            if not short_fired and not np.isnan(short_level):
                signal, sweep_ext = _detect_sweep_reclaim(
                    day_df, short_level, "short",
                    min_sweep_pts, max_sweep_pts, bar_idx
                )
                if signal:
                    stop  = _round_tick(sweep_ext + STOP_EXTRA_TICKS * TICK_SIZE)
                    risk  = stop - bar["Close"]
                    if risk <= 0:
                        pass
                    else:
                        target  = bar["Close"] - risk * target_r
                        entry_p = bar["Close"]

                        t = Trade(
                            variant=variant,
                            date=str(date),
                            entry_bar_idx=bar_idx,
                            direction="short",
                            entry_price=entry_p,
                            stop_price=stop,
                            target_price=target,
                            risk_pts=risk,
                            reward_pts=risk * target_r,
                            rr_ratio=target_r,
                            swept_level=short_level,
                            sweep_extreme=sweep_ext,
                            tod_entry_et=tod,
                            ref_level=short_ref,
                        )
                        in_trade       = True
                        active_trade   = t
                        active_start_idx = bar_idx
                        short_fired    = True
                        last_signal_bar = bar_idx
                        if verbose:
                            print(f"  [{date} {tod:.2f}h] SHORT entry @ {entry_p:.2f} "
                                  f"stop {stop:.2f} target {target:.2f} "
                                  f"risk {risk:.2f}pts")
                    continue

        # End of day: close any still-open trade
        if in_trade and active_trade is not None:
            t = active_trade
            last_bar = day_df[day_df["is_rth"]].iloc[-1] if day_df["is_rth"].any() else day_df.iloc[-1]
            exit_price = last_bar["Close"]
            t.exit_price  = exit_price
            t.exit_reason = "EOD"
            t.hold_bars   = last_bar.name - active_start_idx
            t.pnl_pts     = (exit_price - t.entry_price) if t.direction == "long" \
                             else (t.entry_price - exit_price)
            t.pnl_pts_net = t.pnl_pts - TOTAL_FRICTION
            trades.append(t)

    return trades


def run_grid_search(df: pd.DataFrame) -> pd.DataFrame:
    """
    Systematic grid search across all variants and key parameters.
    Returns DataFrame with one row per (variant, target_r, min_sweep) combo.
    """
    results = []

    for variant in ["A", "B", "C"]:
        for target_r in TARGET_R_LIST:
            for min_sweep_pts in [0.25, 0.50, 1.0]:
                trades = run_variant(
                    df, variant, target_r=target_r,
                    min_sweep_pts=min_sweep_pts
                )
                if not trades:
                    results.append({
                        "variant": variant,
                        "target_r": target_r,
                        "min_sweep_pts": min_sweep_pts,
                        "n_trades": 0,
                        "win_rate": np.nan,
                        "avg_win_pts": np.nan,
                        "avg_loss_pts": np.nan,
                        "profit_factor": np.nan,
                        "expectancy_pts": np.nan,
                        "total_pts": np.nan,
                        "max_dd_pts": np.nan,
                        "sharpe": np.nan,
                    })
                    continue

                from metrics import compute_metrics
                m = compute_metrics(trades)
                m.update({
                    "variant": variant,
                    "target_r": target_r,
                    "min_sweep_pts": min_sweep_pts,
                })
                results.append(m)

    return pd.DataFrame(results)


if __name__ == "__main__":
    from data_loader import prepare_data
    from metrics import compute_metrics, print_metrics

    df = prepare_data()
    print(f"\nData: {len(df)} bars | {df['date_et'].nunique()} trading days")
    print(f"Range: {df['date_et'].min()} to {df['date_et'].max()}")
    print("=" * 70)

    # Baseline test: each variant at 2R
    for variant in ["A", "B", "C"]:
        print(f"\n{'='*70}")
        print(f"VARIANT {variant} — 2R target, min_sweep=0.25pts")
        print(f"{'='*70}")
        trades = run_variant(df, variant, target_r=2.0, min_sweep_pts=0.25, verbose=False)
        if not trades:
            print("  No trades generated.")
            continue
        m = compute_metrics(trades)
        print_metrics(m, variant)

    # Grid search
    print(f"\n{'='*70}")
    print("GRID SEARCH — All variants × R values × min_sweep")
    print(f"{'='*70}")
    grid = run_grid_search(df)
    grid_sorted = grid.sort_values("profit_factor", ascending=False)
    print(grid_sorted[[
        "variant","target_r","min_sweep_pts","n_trades","win_rate",
        "profit_factor","expectancy_pts","total_pts","max_dd_pts"
    ]].to_string(index=False))

    # Save results
    out_path = Path(__file__).parent / "results" / "grid_results.csv"
    grid.to_csv(out_path, index=False)
    print(f"\nGrid results saved to {out_path}")
