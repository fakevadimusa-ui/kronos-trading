#!/usr/bin/env python3
"""
Black Box Log Auditor — run after each session to assess bot health.
Usage: python3 audit_logs.py [log_file]
Default log: /root/logs/ict/ict.log (VPS) or ~/freqtrade/user_data/logs/ict_trader.log (local)
"""
import sys
import re
from pathlib import Path
from collections import Counter

LOG_PATHS = [
    Path("/root/logs/ict/ict.log"),
    Path.home() / "freqtrade/user_data/logs/ict_trader.log",
]

def load_log(path_arg=None):
    if path_arg:
        return Path(path_arg).read_text()
    for p in LOG_PATHS:
        if p.exists():
            return p.read_text()
    return None

def audit(text: str):
    lines = text.strip().splitlines()
    last50 = lines[-50:]
    full = "\n".join(last50)

    print(f"\n{'='*60}")
    print("  ICT BOT LOG AUDIT")
    print(f"  Last {len(last50)} lines of {len(lines)} total")
    print(f"{'='*60}\n")

    # ── 1. STATE ────────────────────────────────────────────────
    errors    = [l for l in last50 if "Error" in l or "Exception" in l or "Traceback" in l]
    signals   = [l for l in last50 if "Signal:" in l]
    orders    = [l for l in last50 if "Order submitted" in l or "Placing" in l]
    kills     = [l for l in last50 if "kill zone" in l.lower()]
    holds     = [l for l in last50 if "HOLD" in l or "No ICT setup" in l]
    no_kz     = [l for l in last50 if "Not in kill zone" in l]
    halts     = [l for l in last50 if "HALT" in l]

    if errors:
        state = "🔴 ERRORING"
    elif orders:
        state = "🟢 EXECUTED"
    elif signals and any("BUY" in s or "SELL" in s for s in signals):
        state = "🟡 SIGNAL FOUND (check execution)"
    elif kills or holds:
        state = "🔵 SEARCHING"
    else:
        state = "⚪ IDLE / NOT RUNNING"

    print(f"STATE:  {state}")
    print()

    # ── 2. FILTER BREAKDOWN ──────────────────────────────────────
    reason_pat = re.compile(r"Signal: HOLD \| (.+)")
    reasons    = [m.group(1) for l in last50 for m in [reason_pat.search(l)] if m]
    reason_counts = Counter(reasons)

    print("FILTER BREAKDOWN (why HOLD fired):")
    if reason_counts:
        for reason, count in reason_counts.most_common(8):
            print(f"  {count:>3}x  {reason[:80]}")
    elif no_kz:
        print("  Bot running outside kill zone windows (expected if checked off-hours)")
    else:
        print("  No HOLD reasons found in last 50 lines")
    print()

    # ── 3. SIGNAL / EXECUTION ────────────────────────────────────
    print("SIGNALS & ORDERS:")
    if orders:
        for l in orders:
            print(f"  ✅ {l.strip()}")
    elif signals:
        for l in signals[-5:]:
            print(f"  ⚠️  {l.strip()}")
        print("  No order submission found — signal may have been filtered after generation")
    else:
        print("  No signals or orders in last 50 lines")
    print()

    # ── 4. STABILITY ─────────────────────────────────────────────
    print("STABILITY:")
    if halts:
        for l in halts:
            print(f"  🛑 {l.strip()}")
    if errors:
        for l in errors[:5]:
            print(f"  ❌ {l.strip()}")
    warnings = [l for l in last50 if "WARNING" in l.upper() or "warning" in l.lower()]
    if warnings:
        for l in warnings[:3]:
            print(f"  ⚠️  {l.strip()}")
    if not errors and not halts and not warnings:
        print("  ✅ No errors, warnings, or halts")
    print()

    # ── 5. LAST 5 LINES ──────────────────────────────────────────
    print("LAST 5 LOG LINES:")
    for l in lines[-5:]:
        print(f"  {l}")
    print()

    # ── 6. VERDICT ───────────────────────────────────────────────
    print("VERDICT:")
    if errors:
        print("  Bot is crashing — fix the exception before next session.")
    elif halts:
        print("  Bot is halted (DD or loss limit hit) — check account equity.")
    elif orders:
        print("  Trade was placed — verify fill on Alpaca paper account.")
    elif "No ICT setup" in full and not errors:
        print("  Bot is alive and searching. No A+ setup met criteria yet — normal.")
    elif no_kz and len(no_kz) == len([l for l in last50 if "Signal:" in l]):
        print("  All runs outside kill zone — check cron timing or run during London/NY hours.")
    else:
        print("  Bot appears healthy. Review filter breakdown above for signal blockers.")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else None
    text = load_log(path)
    if not text:
        print("No log file found. Pass path as argument or check VPS.")
        sys.exit(1)
    audit(text)
