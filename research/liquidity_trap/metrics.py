"""
metrics.py — Performance metrics for trade list analysis.

All P&L in ES points unless stated. Convert to $ by multiplying:
  × 5.0  for MES
  × 50.0 for ES

Prop firm target: Lucid $100K
  Target      : +$6,000  (+120 MES pts at 1 contract, or +60 pts at 2 contracts)
  Max loss    : -$3,000  (-600 pts × MES single contract — practically 6 pts/trade × 5 = $30/trade)
  Daily limit : -$1,800
"""

from __future__ import annotations
from typing import Any
import numpy as np
import pandas as pd


def compute_metrics(trades: list) -> dict[str, Any]:
    """Full metrics from a list of Trade objects."""
    if not trades:
        return {"n_trades": 0}

    pnl = np.array([t.pnl_pts_net for t in trades])
    wins = pnl[pnl > 0]
    losses = pnl[pnl <= 0]

    n = len(pnl)
    n_wins = len(wins)
    n_losses = len(losses)
    win_rate = n_wins / n if n > 0 else 0.0

    avg_win = wins.mean() if len(wins) > 0 else 0.0
    avg_loss = losses.mean() if len(losses) > 0 else 0.0
    gross_profit = wins.sum() if len(wins) > 0 else 0.0
    gross_loss = abs(losses.sum()) if len(losses) > 0 else 0.0
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.nan

    expectancy = pnl.mean()
    total_pts = pnl.sum()

    # Drawdown
    cumulative = np.cumsum(pnl)
    running_max = np.maximum.accumulate(cumulative)
    dd_series = cumulative - running_max
    max_dd = dd_series.min()

    # Max consecutive losses
    max_consec_loss = 0
    curr_consec = 0
    for p in pnl:
        if p <= 0:
            curr_consec += 1
            max_consec_loss = max(max_consec_loss, curr_consec)
        else:
            curr_consec = 0

    # Sharpe (simple, annualized from per-trade)
    if pnl.std() > 0:
        sharpe = (pnl.mean() / pnl.std()) * np.sqrt(252 * 1.5)  # ~1.5 trades/day avg
    else:
        sharpe = np.nan

    # Avg hold time in bars
    avg_hold = np.mean([t.hold_bars for t in trades])

    # Exit reason breakdown
    exits = {}
    for t in trades:
        exits[t.exit_reason] = exits.get(t.exit_reason, 0) + 1

    # Direction breakdown
    longs  = [t for t in trades if t.direction == "long"]
    shorts = [t for t in trades if t.direction == "short"]
    long_wr  = sum(1 for t in longs  if t.pnl_pts_net > 0) / len(longs)  if longs  else np.nan
    short_wr = sum(1 for t in shorts if t.pnl_pts_net > 0) / len(shorts) if shorts else np.nan

    # Avg risk and avg reward
    avg_risk_pts   = np.mean([t.risk_pts   for t in trades])
    avg_reward_pts = np.mean([t.reward_pts for t in trades])
    realized_rr    = avg_win / abs(avg_loss) if avg_loss != 0 else np.nan

    # Dates
    dates = sorted(set(t.date for t in trades))
    n_days = len(dates)
    trades_per_day = n / n_days if n_days > 0 else 0.0

    return {
        "n_trades":        n,
        "n_days":          n_days,
        "trades_per_day":  round(trades_per_day, 2),
        "n_wins":          n_wins,
        "n_losses":        n_losses,
        "win_rate":        round(win_rate, 4),
        "avg_win_pts":     round(avg_win, 3),
        "avg_loss_pts":    round(avg_loss, 3),
        "profit_factor":   round(profit_factor, 3) if not np.isnan(profit_factor) else np.nan,
        "expectancy_pts":  round(expectancy, 3),
        "total_pts":       round(total_pts, 2),
        "max_dd_pts":      round(max_dd, 2),
        "max_consec_loss": max_consec_loss,
        "sharpe":          round(sharpe, 3) if not np.isnan(sharpe) else np.nan,
        "avg_hold_bars":   round(avg_hold, 1),
        "avg_risk_pts":    round(avg_risk_pts, 2),
        "avg_reward_pts":  round(avg_reward_pts, 2),
        "realized_rr":     round(realized_rr, 2) if not np.isnan(realized_rr) else np.nan,
        "long_wr":         round(long_wr, 4)  if not np.isnan(long_wr)  else np.nan,
        "short_wr":        round(short_wr, 4) if not np.isnan(short_wr) else np.nan,
        "exit_tp":         exits.get("TP", 0),
        "exit_sl":         exits.get("SL", 0),
        "exit_time":       exits.get("TIME", 0),
        "exit_eod":        exits.get("EOD", 0),
        "gross_profit_pts": round(gross_profit, 2),
        "gross_loss_pts":   round(gross_loss, 2),
        "pnl_series":      pnl.tolist(),
    }


def print_metrics(m: dict, label: str = "") -> None:
    """Human-readable metrics print."""
    if m.get("n_trades", 0) == 0:
        print(f"  {label}: No trades")
        return

    sep = "─" * 55
    print(sep)
    if label:
        print(f"  Strategy: {label}")
    print(f"  Trades      : {m['n_trades']}  ({m['n_days']} days, {m['trades_per_day']:.1f}/day)")
    print(f"  Win Rate    : {m['win_rate']*100:.1f}%  ({m['n_wins']}W / {m['n_losses']}L)")
    print(f"  Avg Win     : +{m['avg_win_pts']:.2f} pts")
    print(f"  Avg Loss    : {m['avg_loss_pts']:.2f} pts")
    print(f"  Realized RR : {m.get('realized_rr','n/a')}")
    print(f"  Profit Factor: {m['profit_factor']}")
    print(f"  Expectancy  : {m['expectancy_pts']:.3f} pts/trade")
    print(f"  Total P&L   : {m['total_pts']:.1f} pts")
    print(f"    = ${m['total_pts']*MES:,.0f} MES (1 contract)")
    print(f"    = ${m['total_pts']*ES:,.0f} ES  (1 contract)")
    print(f"  Max Drawdown: {m['max_dd_pts']:.1f} pts")
    print(f"  Max Consec L: {m['max_consec_loss']}")
    print(f"  Sharpe      : {m.get('sharpe','n/a')}")
    print(f"  Avg Hold    : {m['avg_hold_bars']:.1f} bars ({m['avg_hold_bars']*5:.0f} min)")
    print(f"  Exits: TP={m['exit_tp']} SL={m['exit_sl']} TIME={m['exit_time']} EOD={m['exit_eod']}")
    print(f"  Long WR: {m['long_wr']*100:.1f}%   Short WR: {m['short_wr']*100:.1f}%" if
          m['long_wr'] is not np.nan else "  Direction WR: n/a")
    print(sep)


MES = 5.0
ES  = 50.0


def grade_strategy(m: dict) -> str:
    """
    Grade A: 60%+ WR, PF 1.6+, survives slippage, 1+ trade/day
    Grade B: 55%+ WR, PF 1.4+, 0.5+ trade/day
    Grade C: Edge unclear or too few trades
    Reject : Overfit, loses to friction, or breaks risk rules
    """
    if m.get("n_trades", 0) < 10:
        return "C — Insufficient trades (<10)"
    if m.get("profit_factor", 0) <= 1.0 or m.get("expectancy_pts", -1) <= 0:
        return "REJECT — No edge after friction"
    wr = m.get("win_rate", 0)
    pf = m.get("profit_factor", 0)
    tpd = m.get("trades_per_day", 0)

    if wr >= 0.60 and pf >= 1.6 and tpd >= 1.0:
        return "A — Strong edge, viable for live trading"
    if wr >= 0.55 and pf >= 1.4 and tpd >= 0.5:
        return "B — Moderate edge, needs more validation"
    if wr >= 0.50 and pf >= 1.2:
        return "C — Weak edge, research-only"
    return "REJECT — Below minimum thresholds"
