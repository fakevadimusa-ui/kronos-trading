#!/usr/bin/env python3
"""
forward_tracker.py — 50-Trade Forward Test Validation Tracker

Reads ~/logs/liquidity_trap/signals.jsonl and produces a full
performance report for the ongoing paper forward test.

IMPORTANT: Before 50 trades this ALWAYS prints a data-insufficient warning.
Do not make any live deployment decision based on fewer than 50 trades.

Usage:
  python3 research/liquidity_trap/forward_tracker.py

  # Or from the research directory:
  python3 forward_tracker.py

Grading after 50+ trades:
  PROMISING : WR ≥ 55% AND PF ≥ 1.4 AND avg R ≥ 0
  WEAK      : WR 45–55% OR PF 1.2–1.4 (needs more data)
  REJECT    : PF < 1.0 OR avg R < -0.5 after friction
  NEEDS MORE DATA: < 50 trades
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

ET = ZoneInfo("America/New_York")

SIGNALS_FILE = Path.home() / "logs" / "liquidity_trap" / "signals.jsonl"
SUMMARY_FILE = Path.home() / "logs" / "liquidity_trap" / "daily_summary.jsonl"

VALIDATION_THRESHOLD = 50   # trades needed before making any judgment

# Grading thresholds (matches original research spec)
GRADE_PROMISING_WR = 0.55
GRADE_PROMISING_PF = 1.40
GRADE_WEAK_PF      = 1.20
GRADE_REJECT_PF    = 1.00


def load_signals() -> list[dict]:
    if not SIGNALS_FILE.exists():
        return []
    trades = []
    with open(SIGNALS_FILE) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                t = json.loads(line)
                # Only include completed trades (have a status set)
                if t.get("status") in ("WIN", "LOSS", "SCRATCH"):
                    trades.append(t)
            except json.JSONDecodeError:
                continue
    return trades


def load_daily_summaries() -> list[dict]:
    if not SUMMARY_FILE.exists():
        return []
    rows = []
    with open(SUMMARY_FILE) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return rows


def compute_forward_metrics(trades: list[dict]) -> dict:
    if not trades:
        return {}

    pnl     = np.array([t.get("pnl_pts_net", 0.0) for t in trades])
    r_vals  = np.array([t.get("r_result", 0.0)     for t in trades])
    wins    = pnl[pnl > 0]
    losses  = pnl[pnl <= 0]

    n       = len(pnl)
    n_wins  = len(wins)
    n_loss  = len(losses)
    wr      = n_wins / n if n > 0 else 0.0

    avg_win  = wins.mean()   if len(wins)   > 0 else 0.0
    avg_loss = losses.mean() if len(losses) > 0 else 0.0
    gross_p  = wins.sum()    if len(wins)   > 0 else 0.0
    gross_l  = abs(losses.sum()) if len(losses) > 0 else 0.0
    pf       = gross_p / gross_l if gross_l > 0 else np.nan
    exp_pts  = pnl.mean()

    # R-based metrics
    avg_r    = r_vals.mean()
    total_r  = r_vals.sum()

    # Drawdown (in R)
    cum_r    = np.cumsum(r_vals)
    peak_r   = np.maximum.accumulate(cum_r)
    dd_r     = cum_r - peak_r
    max_dd_r = dd_r.min()

    # Drawdown in pts
    cum_pts  = np.cumsum(pnl)
    peak_pts = np.maximum.accumulate(cum_pts)
    dd_pts   = cum_pts - peak_pts
    max_dd_pts = dd_pts.min()

    # Consecutive losses
    max_consec = curr_consec = 0
    for p in pnl:
        if p <= 0:
            curr_consec += 1
            max_consec = max(max_consec, curr_consec)
        else:
            curr_consec = 0

    # Exit breakdown
    exits = {}
    for t in trades:
        exits[t.get("exit_reason", "?")] = exits.get(t.get("exit_reason", "?"), 0) + 1

    # Direction breakdown
    longs  = [t for t in trades if t.get("direction") == "long"]
    shorts = [t for t in trades if t.get("direction") == "short"]
    l_wr   = sum(1 for t in longs  if t.get("pnl_pts_net", 0) > 0) / len(longs)  if longs  else np.nan
    s_wr   = sum(1 for t in shorts if t.get("pnl_pts_net", 0) > 0) / len(shorts) if shorts else np.nan

    # Date range
    dates  = sorted(set(t.get("date", "") for t in trades))

    return {
        "n_trades":        n,
        "n_wins":          n_wins,
        "n_losses":        n_loss,
        "win_rate":        round(wr, 4),
        "avg_win_pts":     round(avg_win, 3),
        "avg_loss_pts":    round(avg_loss, 3),
        "profit_factor":   round(pf, 3) if not np.isnan(pf) else None,
        "expectancy_pts":  round(exp_pts, 3),
        "avg_r":           round(avg_r, 3),
        "total_r":         round(total_r, 3),
        "total_pts":       round(pnl.sum(), 2),
        "max_dd_r":        round(max_dd_r, 3),
        "max_dd_pts":      round(max_dd_pts, 2),
        "max_consec_loss": max_consec,
        "exits":           exits,
        "long_wr":         round(l_wr, 4) if not np.isnan(l_wr) else None,
        "short_wr":        round(s_wr, 4) if not np.isnan(s_wr) else None,
        "n_long":          len(longs),
        "n_short":         len(shorts),
        "date_first":      dates[0]  if dates else "n/a",
        "date_last":       dates[-1] if dates else "n/a",
        "n_trading_days":  len(dates),
        "pnl_series":      pnl.tolist(),
        "r_series":        r_vals.tolist(),
    }


def grade(m: dict) -> str:
    n  = m.get("n_trades", 0)
    pf = m.get("profit_factor") or 0
    wr = m.get("win_rate", 0)
    ar = m.get("avg_r", -999)

    if n < VALIDATION_THRESHOLD:
        return "NEEDS MORE DATA"
    if pf < GRADE_REJECT_PF or ar < -0.5:
        return "REJECT"
    if wr >= GRADE_PROMISING_WR and pf >= GRADE_PROMISING_PF:
        return "PROMISING"
    if pf >= GRADE_WEAK_PF:
        return "WEAK"
    return "REJECT"


def _bar(value: float, scale: float = 5.0, width: int = 20) -> str:
    """Simple ASCII bar chart for equity curve indication."""
    norm = int(min(abs(value) / scale * width, width))
    if value >= 0:
        return "[" + "█" * norm + " " * (width - norm) + "] +" + f"{value:.2f}R"
    else:
        return "[" + " " * (width - norm) + "█" * norm + "] " + f"{value:.2f}R"


def print_report(trades: list[dict], m: dict, summaries: list[dict]) -> None:
    now = datetime.now(ET).strftime("%Y-%m-%d %H:%M ET")
    n   = m.get("n_trades", 0)
    remaining = max(0, VALIDATION_THRESHOLD - n)
    verdict   = grade(m)

    sep = "═" * 62

    print(f"\n{sep}")
    print(f"  LIQUIDITY TRAP VARIANT A — FORWARD TEST TRACKER")
    print(f"  Generated: {now}")
    print(f"{sep}")

    if n == 0:
        print(f"\n  No completed trades logged yet.")
        print(f"  Log file: {SIGNALS_FILE}")
        print(f"\n  Run the paper bot during 09:30–10:30 ET to collect data.")
        print(f"{sep}\n")
        return

    # ── Status banner ─────────────────────────────────────────────────────
    print(f"\n  ⚠  PAPER TEST STATUS")
    if n < VALIDATION_THRESHOLD:
        print(f"  {'INSUFFICIENT DATA — FORWARD TESTING ONLY':^56}")
        print(f"  Trades logged   : {n} / {VALIDATION_THRESHOLD} needed")
        print(f"  Remaining       : {remaining} more trades before grading")
    else:
        print(f"  Trades logged   : {n} (validation threshold reached)")
        verdict_label = {
            "PROMISING":       "✅  PROMISING — consider live integration",
            "WEAK":            "⚠   WEAK — needs more data or parameter review",
            "REJECT":          "❌  REJECT — do not deploy",
            "NEEDS MORE DATA": "⏳  NEEDS MORE DATA",
        }.get(verdict, verdict)
        print(f"  Verdict         : {verdict_label}")

    print(f"\n{'─'*62}")
    print(f"  PERFORMANCE SUMMARY  ({m['date_first']} to {m['date_last']})")
    print(f"{'─'*62}")
    print(f"  Trading days    : {m['n_trading_days']}")
    print(f"  Trades          : {n}  ({n/max(m['n_trading_days'],1):.1f}/day avg)")
    print(f"  Wins / Losses   : {m['n_wins']}W / {m['n_losses']}L")
    print(f"  Win Rate        : {m['win_rate']*100:.1f}%")
    print(f"  Avg Win         : +{m['avg_win_pts']:.2f} pts")
    print(f"  Avg Loss        :  {m['avg_loss_pts']:.2f} pts")
    print(f"  Profit Factor   : {m['profit_factor']}")
    print(f"  Expectancy      : {m['expectancy_pts']:+.3f} pts/trade")
    print(f"  Avg R           : {m['avg_r']:+.3f} R/trade")
    print(f"  Total R         : {m['total_r']:+.2f} R")
    print(f"  Total pts (net) : {m['total_pts']:+.2f} pts")
    print(f"    = ${m['total_pts']*5:+,.0f} MES (1 contract)")
    print(f"  Max Drawdown    : {m['max_dd_r']:.2f}R | {m['max_dd_pts']:.2f} pts")
    print(f"  Max Consec Loss : {m['max_consec_loss']}")

    # ── Direction breakdown ───────────────────────────────────────────────
    print(f"\n{'─'*62}")
    print(f"  DIRECTION BREAKDOWN")
    print(f"{'─'*62}")
    print(f"  Long  trades    : {m['n_long']}  | WR: {(m['long_wr'] or 0)*100:.1f}%")
    print(f"  Short trades    : {m['n_short']} | WR: {(m['short_wr'] or 0)*100:.1f}%")

    # ── Exit reasons ──────────────────────────────────────────────────────
    exits = m.get("exits", {})
    print(f"\n{'─'*62}")
    print(f"  EXIT REASONS")
    print(f"{'─'*62}")
    for reason, count in sorted(exits.items()):
        pct = count / n * 100
        print(f"  {reason:<8}: {count:3d} trades ({pct:.0f}%)")

    # ── Equity curve (R) ─────────────────────────────────────────────────
    r_series = m.get("r_series", [])
    if r_series:
        print(f"\n{'─'*62}")
        print(f"  RUNNING R TOTAL (last 20 trades)")
        print(f"{'─'*62}")
        cum = 0.0
        recent = r_series[-20:]
        for i, r in enumerate(recent):
            cum += r
            label = "W" if r > 0 else ("L" if r < 0 else "S")
            print(f"  {i+max(0,len(r_series)-20)+1:3d}. [{label}] r={r:+.2f} | cumR={cum:+.2f}")

    # ── Individual trades ─────────────────────────────────────────────────
    if trades:
        print(f"\n{'─'*62}")
        print(f"  RECENT TRADES (last 10)")
        print(f"{'─'*62}")
        print(f"  {'Date':<12} {'Dir':<6} {'Entry':>8} {'Exit':>8} {'R':>6} {'Status':<8}")
        for t in trades[-10:]:
            print(f"  {t.get('date',''):<12} "
                  f"{t.get('direction','').upper():<6} "
                  f"{t.get('entry_price',0):>8.2f} "
                  f"{t.get('exit_price',0):>8.2f} "
                  f"{t.get('r_result',0):>+6.2f} "
                  f"{t.get('status',''):<8}")

    # ── Verdict ───────────────────────────────────────────────────────────
    print(f"\n{sep}")
    print(f"  VERDICT: {verdict}")
    print(f"{sep}")

    if n < VALIDATION_THRESHOLD:
        print(f"\n  ⚠  DATA IS INSUFFICIENT FOR ANY CONCLUSION")
        print(f"     {remaining} more trades needed before grading.")
        print(f"     Win rates and PF at n={n} have confidence intervals")
        print(f"     so wide they cannot support any trading decision.")

    elif verdict == "PROMISING":
        print(f"\n  ✅  FORWARD TEST PASSED — Minimum criteria met:")
        print(f"     WR={m['win_rate']*100:.1f}% (min {GRADE_PROMISING_WR*100:.0f}%)")
        print(f"     PF={m['profit_factor']} (min {GRADE_PROMISING_PF})")
        print(f"\n  NEXT STEPS if PROMISING:")
        print(f"  1. Add news filter and trend-day filter")
        print(f"  2. Continue paper test for 30 more days (more data = better)")
        print(f"  3. Review challenge_mode.py integration requirements")
        print(f"  4. Get approval from Phase lead before live deployment")

    elif verdict == "WEAK":
        print(f"\n  ⚠  WEAK EDGE — Continue paper testing:")
        print(f"     PF={m['profit_factor']} is above 1.2 but below 1.4")
        print(f"     Run 50 more trades before re-evaluating")
        print(f"     Do not deploy to live or challenge account")

    elif verdict == "REJECT":
        print(f"\n  ❌  REJECTED — Strategy does not meet minimum standards:")
        print(f"     PF={m['profit_factor']} (below {GRADE_REJECT_PF})")
        print(f"     Do not deploy. Review parameters or abandon strategy.")
        print(f"     Consider: different sweep threshold, different time window,")
        print(f"     or a fundamentally different approach.")

    # ── Data warning always shown ─────────────────────────────────────────
    print(f"\n  DATA NOTE:")
    print(f"  Backtest basis: 51 days / 14 trades — statistically insufficient")
    print(f"  This forward test is the primary source of truth")
    print(f"  Do not trust backtest numbers over forward-test results")
    print(f"{sep}\n")


def main() -> None:
    trades    = load_signals()
    summaries = load_daily_summaries()

    if not trades:
        print(f"\n  No trades found in {SIGNALS_FILE}")
        print(f"  Run worker_liquidity_trap_paper.py during 09:30–10:30 ET")
        print(f"  to start collecting forward test data.\n")
        return

    m = compute_forward_metrics(trades)
    print_report(trades, m, summaries)


if __name__ == "__main__":
    main()
