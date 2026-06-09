#!/usr/bin/env python3
"""
challenge_tracker.py — Challenge Progress Dashboard
=====================================================
Reads challenge state + trade log and displays a terminal dashboard.
Runs Monte Carlo simulation to estimate pass probability.

Usage:
    python challenge_tracker.py           # terminal dashboard
    python challenge_tracker.py json      # JSON output for scripts
    python challenge_tracker.py APEX      # override firm params
"""

from __future__ import annotations

import json
import math
import os
import random
import sys
from dataclasses import dataclass, field
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import Optional
from zoneinfo import ZoneInfo

ET = ZoneInfo("America/New_York")

# ── Config ─────────────────────────────────────────────────────────────────────
_STATE_PATH     = Path(os.environ.get("CHALLENGE_STATE",     "/tmp/challenge_mode_state.json"))
_TRADE_LOG_PATH = Path(os.environ.get("CHALLENGE_TRADE_LOG", "/root/logs/challenge/trades.jsonl"))

# Fallback assumptions when < MIN_TRADES_FOR_STATS trades available
MIN_TRADES_FOR_STATS = 10
FALLBACK_WR          = 0.55
FALLBACK_AVG_WIN     = 3_000.0
FALLBACK_AVG_LOSS    =   900.0


# ── Data ───────────────────────────────────────────────────────────────────────

@dataclass
class ChallengeProgress:
    mode:                str   = "NORMAL"
    stage:               str   = "CHALLENGE"
    total_pnl:           float = 0.0
    daily_pnl:           float = 0.0
    target:              float = 6_000.0
    max_loss:            float = 3_000.0
    dll:                 float = 1_800.0
    dll_used_today:      float = 0.0    # fraction 0-1
    pct_to_target:       float = 0.0    # fraction 0-1
    max_loss_used_pct:   float = 0.0    # fraction 0-1
    trades_today:        int   = 0
    total_trades:        int   = 0
    win_rate:            float = 0.0
    profit_factor:       float = 0.0
    avg_win:             float = 0.0
    avg_loss:            float = 0.0
    consecutive_losses:  int   = 0
    suspended:           bool  = False
    suspension_until:    Optional[str] = None
    pass_probability:    float = 0.0
    trading_days_active: int   = 0
    k7_paused:           bool  = False
    k8_disabled:         bool  = False


def _load_trades(n: int = 500) -> list[dict]:
    if not _TRADE_LOG_PATH.exists():
        return []
    trades: list[dict] = []
    try:
        with open(_TRADE_LOG_PATH) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        trades.append(json.loads(line))
                    except Exception:
                        pass
    except Exception:
        pass
    return trades[-n:]


def _load_state() -> dict:
    if _STATE_PATH.exists():
        try:
            return json.loads(_STATE_PATH.read_text())
        except Exception:
            pass
    return {}


# ── Monte Carlo ────────────────────────────────────────────────────────────────

def monte_carlo_pass_probability(
    current_pnl:  float,
    target:       float,
    max_loss:     float,
    wr:           float,
    avg_win:      float,
    avg_loss:     float,
    simulations:  int = 5_000,
) -> float:
    """
    Simulate prop-firm challenge outcome.
    Returns P(reach target before losing max_loss).
    """
    if wr <= 0:
        return 0.0
    if wr >= 1.0:
        return 1.0
    if current_pnl >= target:
        return 1.0
    if current_pnl <= -max_loss:
        return 0.0
    if avg_win <= 0 or avg_loss <= 0:
        return 0.0

    passed = 0
    # Add small variance so it's not perfectly deterministic
    win_std  = avg_win  * 0.30
    loss_std = avg_loss * 0.20

    for _ in range(simulations):
        pnl = current_pnl
        for _ in range(200):    # max trades per sim
            if pnl >= target:
                passed += 1
                break
            if pnl <= -max_loss:
                break
            if random.random() < wr:
                pnl += max(0.0, random.gauss(avg_win, win_std))
            else:
                pnl -= max(0.0, random.gauss(avg_loss, loss_std))

    return passed / simulations


# ── Main calculator ────────────────────────────────────────────────────────────

def calculate_progress(
    equity:       float = 100_000.0,
    target_pct:   float = 0.06,
    max_loss_pct: float = 0.03,
    dll_pct:      float = 0.018,
) -> ChallengeProgress:
    state  = _load_state()
    trades = _load_trades()

    target   = equity * target_pct
    max_loss = equity * max_loss_pct
    dll      = equity * dll_pct

    total_pnl  = state.get("total_pnl", 0.0)
    today_str  = datetime.now(ET).date().isoformat()
    daily_pnl  = state.get("daily_pnl", {}).get(today_str, 0.0)

    dll_used_today    = abs(daily_pnl) / dll if dll and daily_pnl < 0 else 0.0
    pct_to_target     = total_pnl / target if target else 0.0
    max_loss_used_pct = abs(total_pnl) / max_loss if total_pnl < 0 and max_loss else 0.0

    # Trade stats
    today_trades = [t for t in trades if t.get("date") == today_str]
    wins  = [t for t in trades if t.get("win", t.get("pnl", 0) > 0)]
    losses = [t for t in trades if not t.get("win", t.get("pnl", 0) > 0)]

    win_rate = len(wins) / len(trades) if trades else 0.0
    avg_win  = sum(t.get("pnl", 0) for t in wins)  / len(wins)  if wins  else FALLBACK_AVG_WIN
    avg_loss = sum(abs(t.get("pnl", 0)) for t in losses) / len(losses) if losses else FALLBACK_AVG_LOSS

    gross_profit = sum(t.get("pnl", 0) for t in wins)
    gross_loss   = sum(abs(t.get("pnl", 0)) for t in losses)
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0.0

    # Use fallbacks if insufficient data
    if len(trades) < MIN_TRADES_FOR_STATS:
        wr_for_mc  = FALLBACK_WR
        win_for_mc = FALLBACK_AVG_WIN
        los_for_mc = FALLBACK_AVG_LOSS
    else:
        wr_for_mc  = win_rate
        win_for_mc = avg_win
        los_for_mc = avg_loss

    pass_prob = monte_carlo_pass_probability(
        current_pnl = total_pnl,
        target      = target,
        max_loss    = max_loss,
        wr          = wr_for_mc,
        avg_win     = win_for_mc,
        avg_loss    = los_for_mc,
        simulations = 5_000,
    )

    # Trading days active
    dates_seen: set[str] = set()
    for t in trades:
        d = t.get("date", "")
        if d:
            dates_seen.add(d)
    trading_days_active = len(dates_seen)

    suspended = bool(state.get("suspension_until")) and (
        datetime.now(ET).date().isoformat() <= str(state.get("suspension_until", ""))
    )

    # K7/K8 state
    k7_paused   = bool(state.get("k7_pause_until")) and (
        datetime.now(ET).date().isoformat() <= str(state.get("k7_pause_until", ""))
    )
    k8_disabled = bool(state.get("k8_disable_until")) and (
        datetime.now(ET).date().isoformat() <= str(state.get("k8_disable_until", ""))
    )

    return ChallengeProgress(
        mode                = state.get("mode", "NORMAL"),
        stage               = state.get("stage", "CHALLENGE"),
        total_pnl           = total_pnl,
        daily_pnl           = daily_pnl,
        target              = target,
        max_loss            = max_loss,
        dll                 = dll,
        dll_used_today      = dll_used_today,
        pct_to_target       = pct_to_target,
        max_loss_used_pct   = max_loss_used_pct,
        trades_today        = len(today_trades),
        total_trades        = len(trades),
        win_rate            = win_rate,
        profit_factor       = profit_factor,
        avg_win             = avg_win,
        avg_loss            = avg_loss,
        consecutive_losses  = state.get("consecutive_loss_days", 0),
        suspended           = suspended,
        suspension_until    = state.get("suspension_until"),
        pass_probability    = pass_prob,
        trading_days_active = trading_days_active,
        k7_paused           = k7_paused,
        k8_disabled         = k8_disabled,
    )


# ── Display ────────────────────────────────────────────────────────────────────

def _bar(fraction: float, width: int = 20) -> str:
    filled = max(0, min(width, round(fraction * width)))
    return "█" * filled + "░" * (width - filled)


def _dll_bar(fraction: float, width: int = 16) -> str:
    filled = max(0, min(width, round(fraction * width)))
    char   = "█" if fraction < 0.7 else "▓" if fraction < 0.9 else "░"
    return char * filled + "░" * (width - filled)


def display(p: ChallengeProgress) -> None:
    W = 58   # inner width

    def row(left: str, right: str) -> str:
        gap = W - len(left) - len(right)
        return f"║  {left}{' ' * max(1, gap)}{right}  ║"

    def divider() -> str:
        return "╠" + "═" * (W + 4) + "╣"

    def header(title: str) -> str:
        pad = (W + 4 - len(title)) // 2
        return "║" + " " * pad + title + " " * (W + 4 - pad - len(title)) + "║"

    # ── Build progress bar line ───────────────────────────────────────────────
    prog_pct    = max(-99.9, min(999.9, p.pct_to_target * 100))
    prog_bar    = _bar(max(0.0, p.pct_to_target))
    prog_line   = f"{prog_bar} {prog_pct:+.1f}%"
    pnl_detail  = f"${p.total_pnl:+,.0f} of ${p.target:,.0f}"

    # Days at pace
    if p.trading_days_active > 0 and p.total_pnl > 0:
        pace       = p.total_pnl / p.trading_days_active
        days_left  = math.ceil((p.target - p.total_pnl) / pace) if pace > 0 else 999
        pace_str   = f"~{days_left}d at pace"
    else:
        pace_str   = "no pace yet"

    dll_bar_str = _dll_bar(p.dll_used_today)
    dll_pct_str = f"{p.dll_used_today:.0%}"

    # Mode alerts
    alerts: list[str] = []
    if p.suspended:
        alerts.append(f"⛔ SUSPENDED until {p.suspension_until}")
    if p.k7_paused:
        alerts.append("⛔ K7: PAUSED (PF < 1.0)")
    if p.k8_disabled:
        alerts.append("⛔ K8: DISABLED (5 consec losses)")
    if p.dll_used_today >= 0.80:
        alerts.append(f"⚠️  DLL {p.dll_used_today:.0%} USED — careful")
    if p.max_loss_used_pct >= 0.50:
        alerts.append(f"⚠️  MAX LOSS {p.max_loss_used_pct:.0%} CONSUMED")

    pass_color = "🟢" if p.pass_probability >= 0.65 else "🟡" if p.pass_probability >= 0.45 else "🔴"

    lines = [
        "╔" + "═" * (W + 4) + "╗",
        header("CHALLENGE PROGRESS TRACKER"),
        divider(),
        row(f"Stage: {p.stage:<12} Mode: {p.mode}", f"{datetime.now(ET).strftime('%Y-%m-%d %H:%M ET')}"),
        divider(),
        row("Balance:", f"${100_000 + p.total_pnl:>10,.2f}"),
        row("Target:", f"${100_000 + p.target:>10,.0f}"),
        row("Total P&L:", f"${p.total_pnl:>+10,.2f}"),
        row(f"Progress: {prog_bar}", f"{prog_pct:+.1f}%  ({pace_str})"),
        row("Remaining to target:", f"${max(0.0, p.target - p.total_pnl):>8,.0f}"),
        divider(),
        row(f"Daily P&L:", f"${p.daily_pnl:>+8,.0f}"),
        row(f"DLL used: {dll_bar_str}", f"{dll_pct_str}  (limit ${p.dll:,.0f})"),
        row("Max loss used:", f"{p.max_loss_used_pct:.1%}  (limit ${p.max_loss:,.0f})"),
        row("Drawdown room:", f"${max(0.0, p.max_loss + p.total_pnl):>8,.0f}"),
        divider(),
        row(f"Win Rate: {p.win_rate:.1%} ({int(p.win_rate*p.total_trades)}/{p.total_trades})",
            f"PF: {p.profit_factor:.2f}"),
        row(f"Avg Win:  ${p.avg_win:>6,.0f}", f"Avg Loss: ${p.avg_loss:>6,.0f}"),
        row(f"Consec Loss Days: {p.consecutive_losses}",
            f"Suspended: {'YES' if p.suspended else 'No'}"),
        row(f"Trades today: {p.trades_today}", f"Active days: {p.trading_days_active}"),
        divider(),
        row(f"{pass_color} Pass Probability:", f"{p.pass_probability:.1%}"),
        row("Current Mode:", f"{p.mode} ({['0.3','0.5','0.8'][['SAFE','NORMAL','AGGRESSIVE'].index(p.mode)] if p.mode in ['SAFE','NORMAL','AGGRESSIVE'] else '?'}% risk/trade)"),
        "╚" + "═" * (W + 4) + "╝",
    ]

    for line in lines:
        print(line)

    if alerts:
        print()
        for a in alerts:
            print(f"  {a}")

    print()


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    # Optional: override firm params via CLI arg
    firm_name = sys.argv[1].upper() if len(sys.argv) > 1 else ""

    if firm_name == "JSON":
        p = calculate_progress()
        print(json.dumps(p.__dict__, indent=2, default=str))
        return

    equity       = 100_000.0
    target_pct   = 0.06
    max_loss_pct = 0.03
    dll_pct      = 0.018

    if firm_name and firm_name not in ("", "JSON"):
        try:
            from prop_firms import get_firm
            firm         = get_firm(firm_name.lower())
            equity       = float(firm.account_size)
            target_pct   = firm.profit_target_pct
            max_loss_pct = firm.max_loss_pct
            dll_pct      = firm.safe_dll_pct
            print(f"Using {firm.name} parameters")
        except Exception as e:
            print(f"Unknown firm '{firm_name}' ({e}) — using Lucid defaults")

    p = calculate_progress(equity=equity, target_pct=target_pct,
                           max_loss_pct=max_loss_pct, dll_pct=dll_pct)
    display(p)


if __name__ == "__main__":
    main()
