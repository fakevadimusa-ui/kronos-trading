# =============================================================================
# execution_guards.py  —  3-Knob Execution Protection Matrix
# =============================================================================
# Knob A:  2-Phase News Embargo   (hard block + spread-gated soft window)
# Knob B:  Magnitude-Weighted Circuit Breaker
# Knob C:  Z-Score Relative Spread Gate
#
# KEY DESIGN PRINCIPLE — SHARED BASELINE:
#   Call refresh_spread_baseline(data_client) ONCE per cron tick.
#   This populates _spread_deque with the current NBBO sample.
#   Both Knob A (soft phase) and Knob C read from that same deque.
#   Result: one API call per tick, zero redundant fetches.
#
# INTEGRATION ORDER IN main():
#   1.  if is_day_locked(): sys.exit(0)              # padlock  — always first
#   2.  refresh_spread_baseline(data_client)          # ONE quote fetch, shared
#   3.  if is_embargo_active(): return               # Knob A   — reads deque
#   4.  ... signal generation ...
#   5.  if not check_spread_gate(): return           # Knob C   — reads deque
#   6.  if check_circuit_breaker_magnitude(n_losses): # Knob B
#           engage_day_lock(...); sys.exit(0)
#   7.  place_order()
# =============================================================================

from __future__ import annotations

import json
import statistics
import sys
from collections import deque
from datetime import datetime, time, timedelta
from pathlib import Path
from typing import Optional


# =============================================================================
# --- TUNABLE CONSTANTS ---
# Adjust these without touching any logic below.
# =============================================================================

NEWS_HARD_POST_MINUTES = 3
# How many minutes AFTER a news release the hard block stays active.
# Range: 2–5. Below 2 = genuine price-discovery chaos still active.
# Above 5 = you're leaving money on the table.

CB_MAGNITUDE_RATIO = 2.5
# Circuit breaker fires only when combined recent losses exceed
# (average_winning_trade * CB_MAGNITUDE_RATIO).
# Range: 1.5–4.0.  At 2.5: three $9 losses on a choppy day won't lock you out
# if your avg win is $50. Three $200 losses will.

CB_MAX_LOSSES = 6
# Absolute hard ceiling — locks out at this many losses regardless of dollar size.
# No magnitude math runs above this. Keep between 4–8.
# Exists so a slow bleed of tiny losses can't run all day undetected.

SPREAD_Z_THRESHOLD = 0.5
# Primary spread gate. Fires when current spread is >50% above rolling median.
# Range: 0.3–0.8.  At 0.5: natural NY-open spread widening passes (because
# the median is also wide at open); a genuine flash-crash spike does not.

SPREAD_Z_SOFT = 1.0
# Softer threshold used only during the post-news soft window.
# Higher = more permissive = bot re-enters sooner after CPI/FOMC.
# Range: 0.6–1.5.  Must always be >= SPREAD_Z_THRESHOLD.

SPREAD_HISTORY_SIZE = 20
# Deque capacity for rolling spread baseline. 20 samples at one per 5m cron
# = ~100 minutes of history. Larger = smoother baseline, slower to adapt.

_SPREAD_COLD_START_CAP = 0.0005
# 5bp absolute fallback used only during cold-start (first 5 samples).
# ~10x normal SPY spread — only fires during genuine dislocations.

_CIRCUIT_BREAKER_LOSSES = 3
# Count-gate minimum. Combined with magnitude gate above.
# Kept as a private constant because CB_MAX_LOSSES is the primary public knob.


# =============================================================================
# --- FILE PATHS ---
# Must match paths used in ict_alpaca_trader.py exactly.
# =============================================================================

_DAY_LOCK_PATH = Path.home() / "freqtrade/user_data/logs/ict_day_lock.json"
_LOSS_LOG_PATH = Path.home() / "freqtrade/user_data/logs/ict_loss_log.jsonl"


# =============================================================================
# --- SHARED STATE ---
# Module-level. Persists across function calls within a single cron process.
# Both Knob A (soft phase) and Knob C read from _spread_deque after one
# call to refresh_spread_baseline(). No second fetch needed.
# =============================================================================

_spread_deque: deque[float] = deque(maxlen=SPREAD_HISTORY_SIZE)
_current_spread: Optional[float] = None  # most recent NBBO sample this tick


# =============================================================================
# --- SHARED BASELINE REFRESH ---
# Call ONCE per cron tick before any guard checks.
# =============================================================================

def refresh_spread_baseline(data_client, symbol: str = "SPY") -> None:
    """
    Fetches the live NBBO quote ONCE and caches it in the shared deque.

    All guard checks after this call are pure computation — no further API calls.
    If the fetch fails, the deque is unchanged and both guards fall back to
    cold-start logic or their previous baseline.

    data_client: a StockHistoricalDataClient instance (already built in main()).
    """
    global _current_spread
    try:
        from alpaca.data.requests import StockLatestQuoteRequest
        q = data_client.get_stock_latest_quote(
            StockLatestQuoteRequest(symbol_or_symbols=symbol)
        )[symbol]
        spread = (q.ask_price - q.bid_price) / q.ask_price
        _spread_deque.append(spread)
        _current_spread = spread
        _log(f"[BASELINE] Spread {spread:.5%} | history {len(_spread_deque)}/{SPREAD_HISTORY_SIZE}")
    except Exception as e:
        _log(f"[WARN] Spread baseline refresh failed ({e}) — deque unchanged, guards use prior state")


def _compute_spread_z() -> Optional[tuple[float, float, float]]:
    """
    Pure computation from the deque. Zero API calls.

    Returns (current_spread, median_baseline, spread_z).
    Returns None if deque is empty (refresh never ran this tick).
    Returns spread_z = -1.0 as a sentinel signalling cold-start state.

    Why exclude the current sample from the baseline?
    The Z-score measures how far THIS sample deviates from PRIOR history.
    Including the current sample in its own baseline would dampen genuine spikes —
    a 10x spread would shift the median up, making itself look less abnormal.
    """
    if not _spread_deque or _current_spread is None:
        return None                                   # refresh never called
    if len(_spread_deque) < 5:
        return _current_spread, 0.0, -1.0            # cold-start sentinel
    history  = list(_spread_deque)[:-1]               # all samples except current
    baseline = statistics.median(history)
    z        = (_current_spread - baseline) / baseline if baseline > 0 else 0.0
    return _current_spread, baseline, z


# =============================================================================
# --- KNOB A: 2-PHASE NEWS EMBARGO ---
# =============================================================================
# Hard phase: absolute block. Pre-news liquidity vacuum + first N minutes of
#   genuine price-discovery chaos. Time-only gate — cannot be unlocked early.
#
# Soft phase: conditional block. Stays active while spread Z-score > SPREAD_Z_SOFT.
#   Self-expires the moment the market microstructure normalizes. Reads from
#   _spread_deque — ZERO extra API calls beyond what Knob C already uses.
#
# Why split the window?
#   The dangerous zone is only the first 2-3 minutes post-release.
#   A flat 30-minute block wastes the entire post-news setup window on days
#   when the market digests the print cleanly in under 7 minutes (common on
#   in-line CPI/NFP prints).
# =============================================================================

# Hard phase boundaries: [pre-release start → release + NEWS_HARD_POST_MINUTES]
_EMBARGO_HARD = [
    (time(8, 15), time(8, 33)),    # CPI/PPI/NFP/Claims: -15min pre, +3min post
    (time(9, 25), time(9, 32)),    # Equities open:        -5min pre, +2min post
    (time(13, 45), time(14, 3)),   # FOMC:                -15min pre, +3min post
]

# Soft phase: post-news tail that self-expires when spread normalizes.
_EMBARGO_SOFT = [
    (time(8, 33), time(8, 45)),    # CPI/PPI/NFP: up to 12min post (spread-gated)
    (time(9, 32), time(9, 35)),    # Open:          up to 3min post (spread-gated)
    (time(14, 3), time(14, 15)),   # FOMC:          up to 12min post (spread-gated)
]


def is_embargo_active(dt_et=None) -> bool:
    """
    Returns True if the current time is inside an active embargo window.

    Hard phase:  time-only check. Cannot be cleared by spread normalization.
    Soft phase:  reads spread Z-score from shared deque (no API call).
                 Clears early when Z-score drops below SPREAD_Z_SOFT.

    Pass dt_et (timezone-aware datetime) for unit testing without wall-clock.
    """
    try:
        from ict_model import ICTModel
        t = (dt_et or datetime.now(ICTModel.ET)).time()
    except Exception:
        t = datetime.utcnow().time()

    # ── Hard phase ────────────────────────────────────────────────────────────
    if any(s <= t <= e for s, e in _EMBARGO_HARD):
        _log(f"[EMBARGO-HARD] {t.strftime('%H:%M')} ET — blocked unconditionally")
        return True

    # ── Soft phase ────────────────────────────────────────────────────────────
    if any(s <= t <= e for s, e in _EMBARGO_SOFT):
        result = _compute_spread_z()

        if result is None:
            # refresh_spread_baseline() not called yet — default to blocked (safe)
            _log("[EMBARGO-SOFT] No baseline data — holding as precaution")
            return True

        spread, baseline, z = result

        if z == -1.0:
            # Cold-start fallback: compare against absolute cap
            if spread > _SPREAD_COLD_START_CAP:
                _log(f"[EMBARGO-SOFT] Cold-start: {spread:.5%} > {_SPREAD_COLD_START_CAP:.5%} — held")
                return True
            _log(f"[EMBARGO-SOFT] Cold-start: {spread:.5%} within cap — cleared early")
            return False

        if z > SPREAD_Z_SOFT:
            _log(f"[EMBARGO-SOFT] Z={z:.2f} > {SPREAD_Z_SOFT} | "
                 f"spread {spread:.5%} vs baseline {baseline:.5%} — held")
            return True

        _log(f"[EMBARGO-SOFT] Z={z:.2f} <= {SPREAD_Z_SOFT} — spread normalized, cleared early")
        return False

    return False


# =============================================================================
# --- KNOB B: MAGNITUDE-WEIGHTED CIRCUIT BREAKER ---
# =============================================================================
# Problem: 3 consecutive $8 losses on a choppy day lock you out for no reason.
#   Count alone cannot distinguish noise (small losses, choppy market) from
#   system failure (large losses, edge is broken).
#
# Solution: two-gate logic.
#   Count gate:     >= _CIRCUIT_BREAKER_LOSSES losses required to evaluate.
#   Magnitude gate: combined damage must exceed avg_win * CB_MAGNITUDE_RATIO.
#   Hard ceiling:   CB_MAX_LOSSES always locks out, no magnitude math.
#
# avg_win is self-calibrating from real trade history in ict_loss_log.jsonl.
# Wins and losses both append to the same file under different "type" fields.
# Until wins accumulate, avg_win defaults to $50 (conservative = tighter gate).
# =============================================================================

def record_ict_loss(pnl_dollars: float, reason: str) -> int:
    """
    Appends one loss entry to the write-once append log.
    Returns today's total loss count after recording.

    Write-once means the count cannot be reset by equity recovery from other bots.
    One line appended per loss. Never subtracted. Kronos NVDA winning money
    mid-day has zero effect on this counter.
    """
    today = datetime.now().date().isoformat()
    _append_log({"date": today, "ts": datetime.now().isoformat(),
                 "pnl": round(pnl_dollars, 2), "type": "loss", "reason": reason})
    total = count_ict_losses_today()
    _log(f"[LOSS LOG] ${pnl_dollars:+.2f} — {total} losses today | reason: {reason}")
    return total


def record_ict_win(pnl_dollars: float) -> None:
    """
    Appends a win entry. Feeds get_avg_ict_win() for CB magnitude calibration.
    Without win history, the threshold defaults to $50 (conservative).
    """
    today = datetime.now().date().isoformat()
    _append_log({"date": today, "ts": datetime.now().isoformat(),
                 "pnl": round(pnl_dollars, 2), "type": "win"})
    _log(f"[WIN LOG] ${pnl_dollars:+.2f} recorded")


def count_ict_losses_today() -> int:
    """
    Reads today's loss count from the append log.
    Zero equity math — completely independent of account balance.
    """
    today = datetime.now().date().isoformat()
    try:
        if not _LOSS_LOG_PATH.exists():
            return 0
        count = 0
        with open(_LOSS_LOG_PATH) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    e = json.loads(line)
                    if e.get("date") == today and e.get("type") == "loss":
                        count += 1
                except json.JSONDecodeError:
                    continue
        return count
    except Exception as ex:
        _log(f"[WARN] Loss count read failed ({ex}) — returning 0")
        return 0


def get_avg_ict_win(lookback: int = 20) -> float:
    """
    Rolling average of the last `lookback` ICT wins from the log.
    Fallback $50 if no win history exists yet — keeps threshold tight
    until real trade data accumulates over time.
    """
    try:
        if not _LOSS_LOG_PATH.exists():
            return 50.0
        wins = []
        with open(_LOSS_LOG_PATH) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    e = json.loads(line)
                    if e.get("type") == "win" and float(e.get("pnl", 0)) > 0:
                        wins.append(float(e["pnl"]))
                except json.JSONDecodeError:
                    continue
        return sum(wins[-lookback:]) / len(wins[-lookback:]) if wins else 50.0
    except Exception:
        return 50.0


def check_circuit_breaker_magnitude(daily_losses: int) -> bool:
    """
    Returns True = lockout warranted  |  False = losses are noise, continue.

    Evaluation order:
    1. Hard ceiling (CB_MAX_LOSSES): always True, no magnitude math.
    2. Count gate (_CIRCUIT_BREAKER_LOSSES): must have minimum before evaluating.
    3. Magnitude gate: damage > avg_win * CB_MAGNITUDE_RATIO → True.
       Reads the last _CIRCUIT_BREAKER_LOSSES loss PnLs from today's log.

    Example at CB_MAGNITUDE_RATIO=2.5, avg_win=$50, threshold=$125:
      - Losses $9/$8/$11 = $28 total → False (noise, $28 < $125)
      - Losses $180/$160/$200 = $540 total → True  (broken, $540 > $125)
    """
    # ── Hard ceiling: unconditional ───────────────────────────────────────────
    if daily_losses >= CB_MAX_LOSSES:
        _log(f"[CB] Hard ceiling: {daily_losses} >= {CB_MAX_LOSSES} — lockout")
        return True

    # ── Count gate: need minimum N losses before magnitude is meaningful ──────
    if daily_losses < _CIRCUIT_BREAKER_LOSSES:
        return False

    # ── Magnitude gate ────────────────────────────────────────────────────────
    today      = datetime.now().date().isoformat()
    loss_pnls  = []
    try:
        if _LOSS_LOG_PATH.exists():
            with open(_LOSS_LOG_PATH) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        e = json.loads(line)
                        if e.get("date") == today and e.get("type") == "loss":
                            loss_pnls.append(abs(float(e.get("pnl", 0))))
                    except json.JSONDecodeError:
                        continue
    except Exception:
        pass

    if not loss_pnls:
        # No records found — fall back to count-only (treat as tripped)
        _log("[CB] No loss records in log — count-only gate active")
        return True

    # Use only the last N losses in the tally
    damage    = sum(loss_pnls[-_CIRCUIT_BREAKER_LOSSES:])
    avg_win   = get_avg_ict_win()
    threshold = avg_win * CB_MAGNITUDE_RATIO

    if damage > threshold:
        _log(f"[CB] LOCKOUT — damage ${damage:.2f} > threshold ${threshold:.2f} "
             f"(avg_win=${avg_win:.2f} × {CB_MAGNITUDE_RATIO})")
        return True

    _log(f"[CB] Count gate hit but magnitude OK — "
         f"${damage:.2f} < ${threshold:.2f} — noise not system failure, continuing")
    return False


def is_day_locked() -> bool:
    """
    Reads the physical day-lock file. Returns True if today is locked.
    Call as the VERY FIRST statement in main() before anything else.
    A locked process should call sys.exit(0) immediately — dead processes
    cannot revenge-trade.
    File is date-stamped: auto-expires at midnight, no manual cleanup needed.
    """
    try:
        if not _DAY_LOCK_PATH.exists():
            return False
        with open(_DAY_LOCK_PATH) as f:
            lock = json.load(f)
        if lock.get("locked_date") == datetime.now().date().isoformat():
            _log(f"[PADLOCK] Active — locked at {lock.get('locked_at')} | "
                 f"reason: {lock.get('reason')} | unlocks: {lock.get('unlock_after')}")
            return True
    except Exception as ex:
        _log(f"[WARN] Day-lock unreadable ({ex}) — allowing execution")
    return False


def engage_day_lock(reason: str, loss_count: int) -> None:
    """
    Writes the padlock file to disk and fires a Telegram alert.

    unlock_after is set to next trading day 09:30 ET — not a timer.
    Why: prevents a cron job running at 23:59 from bypassing a lock written
    moments before midnight. The condition is a calendar date, immune to timing.
    """
    today     = datetime.now()
    next_open = (today.date() + timedelta(days=1)).isoformat() + " 09:30:00 ET"
    lock = {
        "locked_date":  today.date().isoformat(),
        "locked_at":    today.isoformat(),
        "reason":       reason,
        "losses_today": loss_count,
        "unlock_after": next_open,
    }
    try:
        _DAY_LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(_DAY_LOCK_PATH, "w") as f:
            json.dump(lock, f, indent=2)
        _log(f"[PADLOCK] Engaged — {reason}")
        _send_telegram(
            f"🔒 <b>ICT CIRCUIT BREAKER ENGAGED</b>\n"
            f"Reason: {reason}\n"
            f"Losses: {loss_count} | Unlocks: {next_open}"
        )
    except Exception as ex:
        _log(f"[WARN] Day-lock write failed: {ex}")


# =============================================================================
# --- KNOB C: Z-SCORE RELATIVE SPREAD GATE ---
# =============================================================================
# Problem: a fixed 5bp cap rejects valid NY-open entries because SPY spreads
#   legitimately widen to 8–15bp at the open before normalizing.
#   The fixed threshold is blind to session context.
#
# Solution: compare current spread to its own rolling median.
#   NY-open median is also wide → Z-score stays low → gate stays open.
#   Flash-crash or news spike: median is tight → Z spikes → gate fires.
#   The same number means different things depending on recent history.
#
# Reads from _spread_deque populated by refresh_spread_baseline().
# Pure computation — zero API calls.
# =============================================================================

def check_spread_gate(soft_mode: bool = False) -> bool:
    """
    Returns True = spread healthy, safe to submit order.
    Returns False = spread abnormally elevated, abort entry.

    soft_mode=True: uses SPREAD_Z_SOFT threshold (more lenient).
    Called by Knob A soft-phase embargo — same deque, no extra fetch.

    Cold-start (< 5 deque samples): falls back to _SPREAD_COLD_START_CAP.
    Fail-open: any exception returns True + logs warning.
    A gate failure must NEVER block a valid trade.
    """
    try:
        result    = _compute_spread_z()
        threshold = SPREAD_Z_SOFT if soft_mode else SPREAD_Z_THRESHOLD
        tag       = "soft" if soft_mode else "primary"

        if result is None:
            # Deque is empty — refresh_spread_baseline() was skipped this tick.
            _log(f"[SMOKE DETECTOR] [{tag}] Deque empty — fail-open")
            return True

        spread, baseline, z = result

        if z == -1.0:
            # ── Cold-start: Z-score unavailable, use absolute cap ─────────────
            if spread > _SPREAD_COLD_START_CAP:
                _log(f"[SMOKE DETECTOR] [{tag}] Cold-start: "
                     f"{spread:.5%} > {_SPREAD_COLD_START_CAP:.5%} — BLOCKED")
                return False
            _log(f"[SMOKE DETECTOR] [{tag}] Cold-start: {spread:.5%} OK")
            return True

        # ── Z-score gate ──────────────────────────────────────────────────────
        if z > threshold:
            _log(f"[SMOKE DETECTOR] [{tag}] BLOCKED — "
                 f"spread {spread:.5%} | Z={z:.2f} > {threshold} | baseline {baseline:.5%}")
            return False

        _log(f"[SMOKE DETECTOR] [{tag}] OK — "
             f"spread {spread:.5%} | Z={z:.2f} <= {threshold} | baseline {baseline:.5%}")
        return True

    except Exception as ex:
        _log(f"[WARN] Spread gate error ({ex}) — fail-open")
        return True


# =============================================================================
# --- INTERNAL HELPERS ---
# =============================================================================

def _log(msg: str) -> None:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def _append_log(entry: dict) -> None:
    try:
        _LOSS_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(_LOSS_LOG_PATH, "a") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception as ex:
        _log(f"[WARN] Log append failed: {ex}")


def _send_telegram(msg: str) -> None:
    # Late import to avoid circular dependency with ict_alpaca_trader.
    # Only fires on circuit breaker events — not in the hot execution path.
    try:
        import importlib
        bot = importlib.import_module("ict_alpaca_trader")
        bot.send_telegram(msg)
    except Exception:
        pass
