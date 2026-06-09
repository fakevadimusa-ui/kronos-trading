#!/usr/bin/env python3
"""
prop_firms.py — Prop Firm Configuration Library
=================================================
Centralises all prop firm rule sets so challenge_mode.py can be
instantiated with per-firm parameters rather than Lucid-only hardcodes.

WARNING: Prop firm rules change without notice. Verify these values
against the firm's current Terms & Conditions before every challenge
purchase. Last audited: 2026-06.

Firms supported:
  lucid, topstep, apex, takeprofittrader, tpt, myfundedfutures, mff, tradeday
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class PropFirmConfig:
    name:                  str
    account_size:          int           # dollars
    profit_target_pct:     float         # e.g. 0.06 = 6%
    profit_target_abs:     float         # dollars
    max_loss_pct:          float         # fraction of starting balance
    max_loss_abs:          float         # dollars (trailing or static per field below)
    trailing_drawdown:     bool          # True = trailing (worst-case), False = static
    dll_pct:               float         # 0.0 if no daily loss limit
    dll_abs:               float         # dollars; 0 if no DLL
    min_trading_days:      int           # minimum calendar/trading days before payout
    has_consistency_rule:  bool          # True if any day > X% of total profit is flagged
    max_contracts_mes:     int           # Micro E-mini S&P 500
    max_contracts_es:      int           # Full E-mini S&P 500
    notes:                 str           # key rules not captured above

    # Derived safety buffers — used by ChallengeGuard
    @property
    def safe_dll_pct(self) -> float:
        """DLL expressed as fraction of account_size. 0 → use max_loss_pct / 5."""
        if self.dll_pct > 0:
            return self.dll_pct
        return self.max_loss_pct / 5.0   # conservative proxy when no formal DLL

    @property
    def challenge_guard_kwargs(self) -> dict:
        """Return kwargs suitable for ChallengeGuard(equity=..., ...)."""
        return dict(
            equity        = float(self.account_size),
            target_pct    = self.profit_target_pct,
            max_loss_pct  = self.max_loss_pct,
            dll_pct       = self.safe_dll_pct,
        )


# ── Firm definitions ──────────────────────────────────────────────────────────

_FIRMS: dict[str, PropFirmConfig] = {

    "lucid": PropFirmConfig(
        name                 = "Lucid Trading $100K",
        account_size         = 100_000,
        profit_target_pct    = 0.06,
        profit_target_abs    = 6_000,
        max_loss_pct         = 0.03,
        max_loss_abs         = 3_000,
        trailing_drawdown    = False,   # static from starting balance
        dll_pct              = 0.018,
        dll_abs              = 1_800,
        min_trading_days     = 0,       # no minimum
        has_consistency_rule = False,
        max_contracts_mes    = 60,
        max_contracts_es     = 6,
        notes=(
            "Static drawdown from starting balance. "
            "No minimum trading days. No consistency rule. "
            "Payout split 80/20."
        ),
    ),

    "topstep": PropFirmConfig(
        name                 = "Topstep $100K",
        account_size         = 100_000,
        profit_target_pct    = 0.06,
        profit_target_abs    = 6_000,
        max_loss_pct         = 0.045,   # $4,500 trailing from peak equity
        max_loss_abs         = 4_500,
        trailing_drawdown    = True,    # CRITICAL: trails with highest account equity
        dll_pct              = 0.02,
        dll_abs              = 2_000,
        min_trading_days     = 10,
        has_consistency_rule = False,
        max_contracts_mes    = 50,
        max_contracts_es     = 5,
        notes=(
            "TRAILING drawdown — as equity rises, the floor rises with it. "
            "If account reaches $103K, max loss floor becomes $98.5K. "
            "This is the most dangerous rule for straddle strategies. "
            "Winning trades reduce your effective max loss buffer. "
            "Min 10 trading days required before payout request."
        ),
    ),

    "apex": PropFirmConfig(
        name                 = "Apex Trader Funding $100K",
        account_size         = 100_000,
        profit_target_pct    = 0.09,
        profit_target_abs    = 9_000,
        max_loss_pct         = 0.025,   # $2,500 trailing
        max_loss_abs         = 2_500,
        trailing_drawdown    = True,    # CRITICAL: trailing
        dll_pct              = 0.0,     # no DLL
        dll_abs              = 0,
        min_trading_days     = 0,
        has_consistency_rule = True,    # no single day > 30% of total profit
        max_contracts_mes    = 999,     # no contract limit stated
        max_contracts_es     = 99,
        notes=(
            "No daily loss limit — only trailing max drawdown. "
            "CONSISTENCY RULE: no single day can be >30% of total profit. "
            "If you make $9K in one trade you fail the consistency rule. "
            "Straddle strategy MUST spread gains across multiple events. "
            "Trailing drawdown is tight at $2,500 — one bad trade can bust."
        ),
    ),

    "takeprofittrader": PropFirmConfig(
        name                 = "TakeProfitTrader $100K",
        account_size         = 100_000,
        profit_target_pct    = 0.10,
        profit_target_abs    = 10_000,
        max_loss_pct         = 0.06,
        max_loss_abs         = 6_000,
        trailing_drawdown    = False,   # static
        dll_pct              = 0.02,
        dll_abs              = 2_000,
        min_trading_days     = 10,
        has_consistency_rule = False,
        max_contracts_mes    = 50,
        max_contracts_es     = 5,
        notes=(
            "Static $6K max loss — most forgiving drawdown rule of the group. "
            "10% profit target is the highest — needs ~5-6 straddle TP hits. "
            "DLL $2,000 is moderate. Min 10 trading days."
        ),
    ),

    "myfundedfutures": PropFirmConfig(
        name                 = "MyFundedFutures $100K",
        account_size         = 100_000,
        profit_target_pct    = 0.08,
        profit_target_abs    = 8_000,
        max_loss_pct         = 0.05,
        max_loss_abs         = 5_000,
        trailing_drawdown    = True,
        dll_pct              = 0.02,
        dll_abs              = 2_000,
        min_trading_days     = 5,
        has_consistency_rule = False,
        max_contracts_mes    = 50,
        max_contracts_es     = 5,
        notes=(
            "Trailing drawdown $5K. Moderate target at 8%. "
            "5 min trading days is the most lenient. "
            "Good balance of risk/reward for the straddle approach."
        ),
    ),

    "tradeday": PropFirmConfig(
        name                 = "TradeDay $100K",
        account_size         = 100_000,
        profit_target_pct    = 0.06,
        profit_target_abs    = 6_000,
        max_loss_pct         = 0.04,
        max_loss_abs         = 4_000,
        trailing_drawdown    = False,
        dll_pct              = 0.01,
        dll_abs              = 1_000,
        min_trading_days     = 10,
        has_consistency_rule = False,
        max_contracts_mes    = 50,
        max_contracts_es     = 5,
        notes=(
            "TIGHTEST DLL in the group at $1,000/day. "
            "One straddle SL on NFP at default sizing could hit 80% of DLL. "
            "Requires SAFE mode with reduced contract sizing. "
            "Static $4K drawdown is generous relative to target."
        ),
    ),
}

# Aliases
_FIRMS["tpt"]  = _FIRMS["takeprofittrader"]
_FIRMS["mff"]  = _FIRMS["myfundedfutures"]


# ── Public API ────────────────────────────────────────────────────────────────

def get_firm(name: str) -> PropFirmConfig:
    """Return config for the named firm. Case-insensitive. Raises KeyError if unknown."""
    key = name.lower().replace(" ", "").replace("-", "")
    if key not in _FIRMS:
        available = sorted(set(_FIRMS.keys()) - {"tpt", "mff"})
        raise KeyError(f"Unknown firm '{name}'. Available: {available}")
    return _FIRMS[key]


def list_firms() -> None:
    """Print a comparison table of all supported prop firms."""
    firms = [_FIRMS[k] for k in ("lucid", "topstep", "apex", "takeprofittrader",
                                  "myfundedfutures", "tradeday")]
    header = f"{'Firm':<28} {'Target':>7} {'Max DD':>7} {'DD Type':<10} {'DLL':>7} {'Min Days':>8} {'Consistency':>12}"
    print("\n" + "=" * len(header))
    print(header)
    print("=" * len(header))
    for f in firms:
        dd_type = "TRAILING" if f.trailing_drawdown else "static  "
        dll_str = f"${f.dll_abs:,.0f}" if f.dll_abs else "none"
        consist = "YES" if f.has_consistency_rule else "no"
        print(
            f"{f.name:<28} "
            f"${f.profit_target_abs:>5,.0f}  "
            f"${f.max_loss_abs:>5,.0f}  "
            f"{dd_type:<10} "
            f"{dll_str:>7}  "
            f"{f.min_trading_days:>8}  "
            f"{consist:>12}"
        )
    print("=" * len(header))
    print("WARNING: Rules change without notice. Verify before purchasing.\n")


def challenge_guard_kwargs(firm_name: str) -> dict:
    """Return dict of kwargs for ChallengeGuard() for the named firm."""
    return get_firm(firm_name).challenge_guard_kwargs


def best_firm_for_straddle() -> str:
    """
    Return the firm name with the best combination of:
      - Highest max_loss_abs (most forgiving on single bad trade)
      - No trailing drawdown (trailing is deadly for straddle)
      - No consistency rule (one big TP shouldn't fail the challenge)
      - Reasonable DLL
    Score: +10 static, -10 trailing, -15 consistency rule,
           +1 per $100 max_loss_abs / target_abs ratio
    """
    firms = [_FIRMS[k] for k in ("lucid", "topstep", "apex",
                                  "takeprofittrader", "myfundedfutures", "tradeday")]
    scores: list[tuple[float, str]] = []
    for f in firms:
        score = 0.0
        if not f.trailing_drawdown:
            score += 10
        else:
            score -= 10
        if f.has_consistency_rule:
            score -= 15
        # Buffer ratio: max_loss / target (higher = more room to lose before target)
        score += (f.max_loss_abs / f.profit_target_abs) * 10
        # DLL penalty: tighter DLL = more likely to fail on a single bad straddle
        if f.dll_abs > 0:
            score -= (f.profit_target_abs / f.dll_abs)  # lower DLL relative to target = penalty
        scores.append((score, f.name))
    scores.sort(reverse=True)
    return scores[0][1]


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        try:
            firm = get_firm(sys.argv[1])
            print(f"\n{firm.name}")
            print("-" * 50)
            print(f"  Account size:      ${firm.account_size:,.0f}")
            print(f"  Profit target:     ${firm.profit_target_abs:,.0f}  ({firm.profit_target_pct:.0%})")
            print(f"  Max drawdown:      ${firm.max_loss_abs:,.0f}  "
                  f"({'TRAILING — danger' if firm.trailing_drawdown else 'static'})")
            print(f"  Daily loss limit:  {'none' if not firm.dll_abs else f'${firm.dll_abs:,.0f}'}")
            print(f"  Min trading days:  {firm.min_trading_days}")
            print(f"  Consistency rule:  {'YES — dangerous for large single wins' if firm.has_consistency_rule else 'none'}")
            print(f"  Max MES contracts: {firm.max_contracts_mes}")
            print(f"  Notes: {firm.notes}")
            print()
        except KeyError as e:
            print(e)
            sys.exit(1)
    else:
        list_firms()
        print(f"Best firm for news straddle strategy: {best_firm_for_straddle()}")
