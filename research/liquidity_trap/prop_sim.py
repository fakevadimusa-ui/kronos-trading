"""
prop_sim.py — Monte Carlo prop firm challenge simulator.

Simulates a Lucid $100K futures challenge using observed trade P&L
from the backtest as a sampling distribution.

Challenge rules (Lucid $100K):
  Account      : $100,000
  Target       : +$6,000 (6%)
  Max loss     : -$3,000 (3%)
  Daily limit  : -$1,800 (1.8%)
  MES value    : $5/point
  ES  value    : $50/point

Sizing: conservative fixed-risk approach.
  We risk X% of initial $100K per trade, not a % of current equity.
  This prevents the position from ballooning after wins (which often
  causes challenge blowups via a single losing day).

Three risk modes:
  SAFE       : 0.3% = $300 risk/trade
  NORMAL     : 0.5% = $500 risk/trade
  AGGRESSIVE : 0.8% = $800 risk/trade

For MES sizing given risk and stop distance:
  contracts = floor(risk_dollars / (stop_pts × $5))
  capped at 20 MES per Lucid rules

Monte Carlo procedure:
  1. Sample trade P&L (in pts) from observed distribution
  2. Scale to dollar P&L using contracts × $5
  3. Apply DLL (if day loss ≥ -$1,800, halt for rest of day)
  4. Check max loss breach → challenge BLOWN
  5. Check target reached → challenge PASSED
  6. Run N_SIMS simulations, report pass rate / avg days / avg max DD
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np


# ─── Lucid $100K challenge constants ─────────────────────────────────────────
ACCOUNT_EQUITY   = 100_000.0
CHALLENGE_TARGET = 6_000.0   # need +$6K to pass
MAX_LOSS         = -3_000.0  # below this → blown
DAILY_LOSS_LIMIT = -1_800.0  # daily halt threshold
MES_PER_PT       = 5.0       # $5/point per MES contract
MAX_MES          = 20        # Lucid max contracts
MAX_CHALLENGE_DAYS = 180     # safety cutoff

RISK_MODES = {
    "SAFE":       300.0,  # $300 per trade
    "NORMAL":     500.0,  # $500 per trade
    "AGGRESSIVE": 800.0,  # $800 per trade
}


@dataclass
class SimResult:
    passed: bool
    blown: bool
    days: int
    final_equity: float
    max_drawdown: float
    peak_equity: float
    n_trades: int
    daily_halts: int


def _size_trade(risk_dollars: float, stop_pts: float, mode: str) -> int:
    """Calculate MES contracts for a given risk budget and stop size."""
    max_risk = RISK_MODES[mode]
    actual_risk = min(risk_dollars, max_risk)
    if stop_pts <= 0:
        return 0
    contracts = int(actual_risk / (stop_pts * MES_PER_PT))
    return max(1, min(contracts, MAX_MES))


def simulate_challenge(
    trade_pnl_pts: np.ndarray,       # per-trade net P&L in points (from backtest)
    trade_stop_pts: np.ndarray,      # per-trade stop distance in points
    trades_per_day: float,           # average signal frequency
    risk_mode: str = "NORMAL",
    n_sims: int = 2000,
    seed: int = 42,
) -> dict:
    """Run Monte Carlo challenge simulation. Returns summary stats."""
    rng = np.random.default_rng(seed)

    results: list[SimResult] = []
    risk_per_trade = RISK_MODES[risk_mode]

    for sim in range(n_sims):
        equity      = ACCOUNT_EQUITY
        peak        = ACCOUNT_EQUITY
        max_dd      = 0.0
        day         = 0
        total_trades = 0
        daily_halts  = 0
        passed       = False
        blown        = False

        while day < MAX_CHALLENGE_DAYS:
            day += 1
            # Sample number of trades today (Poisson distributed)
            n_today = max(0, int(rng.poisson(trades_per_day)))
            if n_today == 0:
                continue

            daily_pnl = 0.0
            for _ in range(n_today):
                # Sample one trade from observed distribution
                idx = rng.integers(0, len(trade_pnl_pts))
                pnl_pts  = trade_pnl_pts[idx]
                stop_pts = trade_stop_pts[idx]

                contracts = _size_trade(risk_per_trade, stop_pts, risk_mode)
                pnl_dollars = pnl_pts * contracts * MES_PER_PT

                daily_pnl += pnl_dollars
                equity    += pnl_dollars
                total_trades += 1

                # Check daily loss limit (halt rest of day)
                if daily_pnl <= DAILY_LOSS_LIMIT:
                    daily_halts += 1
                    break  # no more trades today

                # Check max loss (blown)
                if equity - ACCOUNT_EQUITY <= MAX_LOSS:
                    blown = True
                    break

                # Check target (passed)
                if equity - ACCOUNT_EQUITY >= CHALLENGE_TARGET:
                    passed = True
                    break

            # Track drawdown
            peak   = max(peak, equity)
            dd     = equity - peak
            max_dd = min(max_dd, dd)

            if blown or passed:
                break

        results.append(SimResult(
            passed=passed,
            blown=blown,
            days=day,
            final_equity=equity,
            max_drawdown=max_dd,
            peak_equity=peak,
            n_trades=total_trades,
            daily_halts=daily_halts,
        ))

    # Aggregate stats
    pass_results  = [r for r in results if r.passed]
    blown_results = [r for r in results if r.blown]
    time_out      = [r for r in results if not r.passed and not r.blown]

    pass_rate = len(pass_results)  / n_sims
    blow_rate = len(blown_results) / n_sims
    timeout_rate = len(time_out)   / n_sims

    avg_days_to_pass = np.mean([r.days for r in pass_results])  if pass_results  else np.nan
    avg_days_to_blow = np.mean([r.days for r in blown_results]) if blown_results else np.nan
    avg_max_dd_all   = np.mean([r.max_drawdown for r in results])
    avg_max_dd_pass  = np.mean([r.max_drawdown for r in pass_results]) if pass_results else np.nan
    median_days      = np.median([r.days for r in pass_results]) if pass_results else np.nan

    # Monthly P&L estimate (after passing)
    # Assume funded account: same strategy, same risk mode
    # Typical payout: 80% profit split
    PAYOUT_SPLIT = 0.80
    monthly_trades = trades_per_day * 21  # ~21 trading days/month
    from metrics import MES
    avg_trade_pnl = trade_pnl_pts.mean()
    # Estimate using median contract size (rough: use $300 risk, 4pt avg stop → 15 contracts)
    avg_contracts = _size_trade(risk_per_trade, 4.0, risk_mode)  # 4pt avg stop assumption
    monthly_gross = monthly_trades * avg_trade_pnl * avg_contracts * MES_PER_PT
    monthly_net   = monthly_gross * PAYOUT_SPLIT

    return {
        "risk_mode":        risk_mode,
        "n_sims":           n_sims,
        "pass_rate":        round(pass_rate, 4),
        "blow_rate":        round(blow_rate, 4),
        "timeout_rate":     round(timeout_rate, 4),
        "avg_days_to_pass": round(avg_days_to_pass, 1) if not np.isnan(avg_days_to_pass) else None,
        "median_days_pass": round(median_days, 1)       if not np.isnan(median_days)      else None,
        "avg_days_to_blow": round(avg_days_to_blow, 1)  if not np.isnan(avg_days_to_blow) else None,
        "avg_max_dd":       round(avg_max_dd_all, 2),
        "avg_max_dd_pass":  round(avg_max_dd_pass, 2)   if not np.isnan(avg_max_dd_pass)  else None,
        "avg_contracts":    avg_contracts,
        "monthly_gross_est": round(monthly_gross, 0),
        "monthly_net_est":   round(monthly_net, 0),
        "risk_per_trade":   risk_per_trade,
        "trades_per_day_input": trades_per_day,
    }


def run_all_modes(
    trade_pnl_pts: np.ndarray,
    trade_stop_pts: np.ndarray,
    trades_per_day: float,
    n_sims: int = 2000,
) -> list[dict]:
    """Run sim for all three risk modes."""
    results = []
    for mode in ["SAFE", "NORMAL", "AGGRESSIVE"]:
        r = simulate_challenge(
            trade_pnl_pts, trade_stop_pts, trades_per_day,
            risk_mode=mode, n_sims=n_sims
        )
        results.append(r)
    return results


def print_sim_results(results: list[dict], label: str = "") -> None:
    """Pretty-print simulation results."""
    print(f"\n{'='*65}")
    print(f"  PROP FIRM SIMULATION — Lucid $100K   [{label}]")
    print(f"  {results[0]['n_sims']:,} Monte Carlo simulations per mode")
    print(f"{'='*65}")

    for r in results:
        mode = r["risk_mode"]
        print(f"\n  ── {mode} mode (${r['risk_per_trade']:.0f}/trade risk) ──")
        print(f"  Pass rate    : {r['pass_rate']*100:.1f}%")
        print(f"  Blow rate    : {r['blow_rate']*100:.1f}%")
        print(f"  Timeout rate : {r['timeout_rate']*100:.1f}%")
        print(f"  Days to pass : avg {r['avg_days_to_pass']} | median {r['median_days_pass']}")
        print(f"  Avg max DD   : ${r['avg_max_dd']:.0f} (all sims)")
        if r['avg_max_dd_pass']:
            print(f"  Avg max DD   : ${r['avg_max_dd_pass']:.0f} (passing sims only)")
        print(f"  Avg contracts: {r['avg_contracts']} MES")
        print(f"  Monthly gross: ~${r['monthly_gross_est']:,.0f}  (1 funded account, before split)")
        print(f"  Monthly net  : ~${r['monthly_net_est']:,.0f}  (80% payout split)")

    # Scaling analysis
    print(f"\n{'─'*65}")
    print(f"  SCALING TO $100K/MONTH ANALYSIS")
    print(f"{'─'*65}")
    best = max(results, key=lambda r: r['monthly_net_est'])
    if best['monthly_net_est'] > 0:
        accounts_needed = int(np.ceil(100_000 / best['monthly_net_est']))
        print(f"  Best mode: {best['risk_mode']} — ${best['monthly_net_est']:,.0f}/month per account")
        print(f"  Funded accounts needed for $100K/month: {accounts_needed}")
        print(f"  Pass rate per account: {best['pass_rate']*100:.1f}%")
        if best['pass_rate'] > 0:
            challenges_needed = accounts_needed / best['pass_rate']
            print(f"  Expected challenges to fund {accounts_needed} accounts: "
                  f"~{challenges_needed:.0f}")
        print(f"\n  REALITY CHECK:")
        if accounts_needed > 10:
            print(f"  WARNING: {accounts_needed} accounts is unrealistic for one trader.")
            print(f"  Correlated strategy → all accounts lose simultaneously on bad days.")
            print(f"  Prop firms may flag/ban for running identical algos on many accounts.")
            print(f"  Practical limit: 3-5 accounts for one strategy.")
        elif accounts_needed <= 3:
            print(f"  FEASIBLE: {accounts_needed} accounts is manageable.")
            print(f"  Still: correlate risk matters — 3 accounts = 3× drawdown on bad days.")
        else:
            print(f"  POSSIBLE but challenging: {accounts_needed} accounts needs capital + management.")
    else:
        print(f"  Monthly P&L is NEGATIVE — strategy not viable for prop challenge.")

    print(f"{'='*65}")
