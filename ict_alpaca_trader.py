"""
ICT Alpaca Trader — ES1! Signal / SPY Execution
=================================================
Runs the ICT model on 5-minute data during kill zones and executes
market orders on SPY via Alpaca (paper → live → prop firm).

Data source priority (automatic fallback):
  1. TradingView Desktop (CDP at localhost:9222) — ES1! futures, Mac-local only
  2. Alpaca data API — real-time SPY bars, works 24/7 on VPS with no Mac
  3. yfinance SPY — last-resort fallback (includes pre-market for London kill zone)

SL/TP translation: when data source is TradingView (ES1!), risk % is preserved
and re-applied to current SPY price for Alpaca bracket orders.

Schedule (cron, PT weekdays):
  */5 0,1     * * 1-5   # London kill zone  (3–5 AM ET = 12–2 AM PT)
  30,45 4     * * 1-5   # NY open first bar  (7:30–7:45 AM ET = 4:30–4:45 AM PT)
  */5 5,6,7,8,9,10,11,12 * * 1-5   # NY AM session (8–10 AM ET = 5–7 AM PT)

Credentials (env vars or JSON fallback):
  ALPACA_KEY, ALPACA_SECRET, TELEGRAM_TOKEN, TELEGRAM_CHAT_ID

Risk rules:
  - 1% account risk per trade  (0.5% when DD > 4%)
  - 1:3 RR — SL at setup invalidation, TP 3×risk
  - Max 1 open ICT position at a time
  - Daily loss halt: −4%
  - Total DD halt: −8%
"""

from __future__ import annotations

from typing import Optional

import json
import os
import socket
import subprocess
import sys
import time as _time   # stdlib time — for sleep(); avoid shadowing datetime.time
import warnings
from datetime import datetime, timezone, timedelta, time
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import yfinance as yf
from dotenv import load_dotenv

warnings.filterwarnings("ignore")

# Load .env from repo root so Telegram/Alpaca credentials are available in cron
load_dotenv(Path(__file__).parent / ".env")

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import (
    MarketOrderRequest,
    TakeProfitRequest,
    StopLossRequest,
    GetPortfolioHistoryRequest,
    GetOrdersRequest,
)
from alpaca.trading.enums import OrderSide, TimeInForce, OrderClass, QueryOrderStatus
from alpaca.common.exceptions import APIError
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

from ict_model import ICTModel


# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════

CONFIG_PATH   = Path.home() / "freqtrade/user_data/config_kronos_nvda.json"
_TV_FETCH_PRIMARY  = Path.home() / "tradingview-mcp/tv_fetch.js"   # Mac: full deps
_TV_FETCH_FALLBACK = Path(__file__).parent / "tv_fetch.js"          # repo backup
TV_FETCH = _TV_FETCH_PRIMARY if _TV_FETCH_PRIMARY.exists() else _TV_FETCH_FALLBACK

SIGNAL_SYMBOL = "CME_MINI_DL:ES1!"   # ICT analysis instrument (TradingView)
SYMBOL        = "SPY"                  # Alpaca execution instrument
SYMBOL_ALT    = "QQQ"                  # alternate execution (tech/NQ equivalent)
YF_PERIOD_15M = "5d"
YF_PERIOD_1H  = "60d"

RISK_PER_TRADE       = 0.005   # 0.5% per trade — prop firm safe
RISK_PER_TRADE_SMALL = 0.0025  # 0.25% when DD > 2%
DAILY_DD_LIMIT       = -0.04
TOTAL_DD_LIMIT       = -0.08
DAILY_LOSS_LIMIT     = 2       # halt after 2 losses in one day
DAILY_PROFIT_TARGET  = 0.015   # bank gains and stop at +1.5% for the day
TRAILING_DD_LIMIT    = 0.04    # prop firm trailing drawdown from intraday peak

CORR_SYMBOL     = "QQQ"    # NQ proxy for SMT divergence
ICT_ORDER_PREFIX = "ICT"   # client_order_id prefix — isolates ICT from Kronos orders

# Hard time cutoffs — force-close any open ICT position at session end
SESSION_CUTOFFS_ET = {
    "london":        time(5,  0),
    "new_york":      time(11, 0),    # aligned with new 8:30-11am window
    "london_close":  time(12, 0),
    "silver_bullet": time(16, 0),
}
STAGNATION_BARS  = 12     # 60 min at 5m — close flat/losing position
MAX_ENTRY_DRIFT  = 0.0015  # 0.15% — skip trade if price drifted too far from FVG entry

ICT_TAG = "ICT"   # order tag to identify our positions vs Kronos positions

# Session state — persists within a trading day, resets at midnight
SESSION_STATE_PATH   = Path.home() / "freqtrade/user_data/logs/ict_session.json"
# Security state — tracks last accepted bar + seen signal hashes (Bouncer + Decoy Detector)
SECURITY_STATE_PATH  = Path.home() / "freqtrade/user_data/logs/ict_security_state.json"
# Circuit breaker — date-stamped day-lock file written on 3rd consecutive loss
DAY_LOCK_PATH        = Path.home() / "freqtrade/user_data/logs/ict_day_lock.json"
# Write-once append loss log — immune to equity-recovery flickering from other bots
ICT_LOSS_LOG_PATH    = Path.home() / "freqtrade/user_data/logs/ict_loss_log.jsonl"

# Max consecutive losses before the circuit breaker physical padlock engages
CIRCUIT_BREAKER_LOSSES = 3

# ── Calibration Knobs — tune without touching logic ───────────────────────────
# KNOB A: News windows — hard post-news floor (minutes always blocked after release)
NEWS_HARD_POST_MINUTES  = 3      # range 2–5  | tighter=safer, looser=more setups

# KNOB B: Circuit breaker — magnitude gate (combined loss vs avg win ratio)
CB_MAGNITUDE_RATIO      = 2.5    # range 1.5–4.0 | lower=tighter, higher=more tolerant
CB_MAX_LOSSES           = 6      # absolute ceiling — always locks out regardless of magnitude

# KNOB C: Spread gate — Z-score thresholds and rolling window
SPREAD_Z_THRESHOLD      = 0.5    # range 0.3–0.8 | lower=tighter, higher=more tolerant
SPREAD_Z_SOFT           = 1.0    # used during post-news soft window
SPREAD_HISTORY_SIZE     = 20     # samples for rolling baseline


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def log(msg: str) -> None:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


# ══════════════════════════════════════════════════════════════════════════════
# TOOL 1 — THE NEWS BLACKOUT CLOCK (Embargo Window)
# ══════════════════════════════════════════════════════════════════════════════

# ── Two-phase news windows ────────────────────────────────────────────────────
# Hard phase: always blocked — covers pre-news liquidity pullback AND the first
# NEWS_HARD_POST_MINUTES of genuine price-discovery chaos after release.
# Soft phase: blocked only if spread is still abnormally elevated vs baseline.
#   Self-expires the moment the market normalizes — zero extra API calls (reuses
#   the spread gate that already runs before every order submission).

_EMBARGO_HARD_ET = [
    (time(8, 15),  time(8, 33)),   # CPI/PPI/NFP: -15min pre + 3min post
    (time(9, 25),  time(9, 32)),   # Market open: -5min pre  + 2min post
    (time(13, 45), time(14, 3)),   # FOMC:        -15min pre + 3min post
]
_EMBARGO_SOFT_ET = [
    (time(8, 33),  time(8, 45)),   # CPI/PPI/NFP: up to 12min post (spread-gated)
    (time(9, 32),  time(9, 35)),   # Market open: up to 3min post  (spread-gated)
    (time(14, 3),  time(14, 15)),  # FOMC:        up to 12min post (spread-gated)
]

def is_embargo_active(dt_et=None) -> bool:
    """
    Two-phase embargo — calibrated to minimize opportunity cost:

    Hard phase: absolute block. Pre-news liquidity vacuum + first 3 min of
    post-release chaos. Cannot be overridden by spread normalization.

    Soft phase: conditional block. Stays active only while spread is elevated
    above Z-score threshold (spread gate reused, no extra API call).
    Clears automatically when the market returns to normal microstructure.
    """
    t = (dt_et or datetime.now(ICTModel.ET)).time()

    # Hard phase — always blocked
    if any(s <= t <= e for s, e in _EMBARGO_HARD_ET):
        return True

    # Soft phase — blocked only if spread is still abnormal
    if any(s <= t <= e for s, e in _EMBARGO_SOFT_ET):
        spread_normal = check_spread_gate(None, SYMBOL, soft_mode=True)
        if not spread_normal:
            log("[EMBARGO-SOFT] Post-news spread still elevated — window held")
            return True
        log("[EMBARGO-SOFT] Spread normalized — window cleared early")
    return False


# ══════════════════════════════════════════════════════════════════════════════
# TOOL 2 — THE PHYSICAL PADLOCK (Day-Lock Circuit Breaker)
# ══════════════════════════════════════════════════════════════════════════════

def is_day_locked() -> bool:
    """
    Checks the physical day-lock file on disk.
    Returns True if a lock exists for today — caller should sys.exit(0).
    File is date-stamped so it auto-expires at midnight with no cleanup needed.
    """
    try:
        if not DAY_LOCK_PATH.exists():
            return False
        with open(DAY_LOCK_PATH) as f:
            lock = json.load(f)
        if lock.get("locked_date") == datetime.now().date().isoformat():
            log(f"[PADLOCK] Circuit breaker active — locked at {lock.get('locked_at')} | "
                f"Reason: {lock.get('reason')} | Unlock after: {lock.get('unlock_after')}")
            return True
    except Exception as e:
        log(f"[WARN] Day-lock file unreadable ({e}) — allowing execution")
    return False


def engage_day_lock(reason: str, loss_count: int) -> None:
    """
    Writes the physical padlock file to disk and fires a Telegram alert.
    Called when loss_count reaches CIRCUIT_BREAKER_LOSSES.
    The unlock_after timestamp is set to next trading day 09:30 ET so a
    midnight cron overlap cannot accidentally bypass a lock set at 23:59.
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
        DAY_LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(DAY_LOCK_PATH, "w") as f:
            json.dump(lock, f, indent=2)
        log(f"[PADLOCK] Day-lock engaged — {reason}")
        send_telegram(
            f"🔒 <b>ICT CIRCUIT BREAKER ENGAGED</b>\n"
            f"Reason: {reason}\n"
            f"Consecutive losses: {loss_count}\n"
            f"All entries locked until: {next_open}"
        )
    except Exception as e:
        log(f"[WARN] Could not write day-lock file: {e}")


# ── Loss Append Log — immune to equity-recovery flickering ──────────────────

def record_ict_loss(pnl_dollars: float, reason: str) -> int:
    """
    Appends one JSON line to the write-once loss log when an ICT trade closes
    in the red. Returns today's total ICT loss count after recording.

    This is completely independent of account equity — other bots winning
    money cannot reset or confuse this counter. One line = one loss, always.
    """
    today = datetime.now().date().isoformat()
    entry = {
        "date":   today,
        "ts":     datetime.now().isoformat(),
        "pnl":    round(pnl_dollars, 2),
        "reason": reason,
    }
    try:
        ICT_LOSS_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(ICT_LOSS_LOG_PATH, "a") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception as e:
        log(f"[WARN] Loss log write failed: {e}")
    total = count_ict_losses_today()
    log(f"[LOSS LOG] Recorded loss ${pnl_dollars:+.2f} | Total ICT losses today: {total}/{CIRCUIT_BREAKER_LOSSES}")
    return total


def record_ict_win(pnl_dollars: float) -> None:
    """
    Appends a win entry to the same loss log file.
    Used by get_avg_ict_win() to calibrate the magnitude circuit breaker.
    """
    today = datetime.now().date().isoformat()
    entry = {"date": today, "ts": datetime.now().isoformat(),
             "pnl": round(pnl_dollars, 2), "type": "win"}
    try:
        ICT_LOSS_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(ICT_LOSS_LOG_PATH, "a") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception as e:
        log(f"[WARN] Win log write failed: {e}")


def get_avg_ict_win(lookback_days: int = 10) -> float:
    """
    Returns the average ICT winning trade size over the last lookback_days.
    Falls back to a conservative $50 estimate if no win history exists.
    Used to calibrate the magnitude threshold in the circuit breaker.
    """
    try:
        if not ICT_LOSS_LOG_PATH.exists():
            return 50.0
        wins = []
        with open(ICT_LOSS_LOG_PATH) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    if entry.get("type") == "win" and entry.get("pnl", 0) > 0:
                        wins.append(float(entry["pnl"]))
                except json.JSONDecodeError:
                    continue
        return sum(wins[-lookback_days:]) / len(wins[-lookback_days:]) if wins else 50.0
    except Exception:
        return 50.0


def check_circuit_breaker_magnitude(daily_losses: int) -> bool:
    """
    Magnitude-weighted circuit breaker — prevents lockout from small noise losses.

    Trips only when BOTH conditions are true:
    1. Count gate:     daily_losses >= CIRCUIT_BREAKER_LOSSES
    2. Magnitude gate: combined recent loss damage > avg_win * CB_MAGNITUDE_RATIO

    Hard ceiling: always trips at CB_MAX_LOSSES regardless of magnitude.

    Rationale: 3 losses of $8/$9/$7 on a choppy day is market noise.
    3 losses of $180/$160/$200 on a calm day means the edge is broken.
    The count alone cannot distinguish the two.
    """
    if daily_losses >= CB_MAX_LOSSES:
        log(f"[CB-MAGNITUDE] Hard ceiling hit ({daily_losses} >= {CB_MAX_LOSSES}) — locking out")
        return True

    if daily_losses < CIRCUIT_BREAKER_LOSSES:
        return False

    # Read recent losses from append log
    today = datetime.now().date().isoformat()
    recent_loss_pnls = []
    try:
        if ICT_LOSS_LOG_PATH.exists():
            with open(ICT_LOSS_LOG_PATH) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entry = json.loads(line)
                        if (entry.get("date") == today
                                and entry.get("pnl", 0) < 0
                                and entry.get("type") != "win"):
                            recent_loss_pnls.append(abs(float(entry["pnl"])))
                    except json.JSONDecodeError:
                        continue
    except Exception:
        pass

    if not recent_loss_pnls:
        # No loss records yet — fall back to count-only
        return True

    total_damage = sum(recent_loss_pnls[-CIRCUIT_BREAKER_LOSSES:])
    avg_win      = get_avg_ict_win()
    threshold    = avg_win * CB_MAGNITUDE_RATIO

    if total_damage > threshold:
        log(f"[CB-MAGNITUDE] TRIPS — damage ${total_damage:.2f} > threshold ${threshold:.2f} "
            f"(avg_win=${avg_win:.2f} × {CB_MAGNITUDE_RATIO})")
        return True

    log(f"[CB-MAGNITUDE] Count limit hit but magnitude OK — "
        f"${total_damage:.2f} damage < ${threshold:.2f} threshold — noise, not system failure")
    return False


def count_ict_losses_today() -> int:
    """
    Counts today's ICT losses from the append log.
    Reads the flat file and counts lines matching today's date.
    Zero equity math — cannot be affected by Kronos NVDA or any other bot.
    """
    today = datetime.now().date().isoformat()
    try:
        if not ICT_LOSS_LOG_PATH.exists():
            return 0
        count = 0
        with open(ICT_LOSS_LOG_PATH) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    if entry.get("date") == today and entry.get("pnl", 0) < 0:
                        count += 1
                except json.JSONDecodeError:
                    continue
        return count
    except Exception as e:
        log(f"[WARN] Loss log read failed ({e}) — returning 0")
        return 0


# ── Broker-first loss truth — authoritative SAFETY-CHECK count ────────────────
# The log-based count_ict_losses_today() above still feeds the magnitude gate and
# the audit trail (KEEP LOGS). For the actual circuit-breaker SAFETY decision we
# use Alpaca's own order history as the source of truth: it cannot desync from the
# broker, it survives a missed record_ict_loss() write, and it needs ZERO local
# state (identical result on the VPS and on an ephemeral GitHub runner).

def _et_session_start_utc() -> datetime:
    """Start of the current ET trading day (00:00 ET) as tz-aware UTC. Clock-derived,
    no local file — identical on every host. ICT kill zones all fall after 00:00 ET."""
    et_now      = datetime.now(ICTModel.ET)
    et_midnight = ICTModel.ET.localize(datetime.combine(et_now.date(), time(0, 0)))
    return et_midnight.astimezone(timezone.utc)


def _ict_realized_trades_today(client: TradingClient) -> list[float]:
    """
    Reconstruct today's CLOSED ICT round-trips from Alpaca order history.
    SPY is ICT-exclusive on this account (Kronos=NVDA/USO), so every filled SPY
    order today is ICT. Walk fills in fill-time order — bracket legs share their
    parent's created_at, so we MUST sort by filled_at, never created_at — accumulate
    signed cash, and snapshot realized P&L each time net position returns flat.
    Exit-path agnostic: counts bracket SL/TP fills AND manual time-exit closes.
    """
    orders = client.get_orders(filter=GetOrdersRequest(
        status  = QueryOrderStatus.ALL,
        after   = _et_session_start_utc(),
        symbols = [SYMBOL],
        limit   = 500,
    ))
    fills = [o for o in orders
             if o.filled_at is not None
             and o.filled_qty is not None and float(o.filled_qty) > 0
             and o.filled_avg_price is not None]
    fills.sort(key=lambda o: o.filled_at)

    trades: list[float] = []
    pos_qty = 0.0   # signed shares held within the currently-open round-trip
    cash    = 0.0   # signed cash flow of that round-trip (+ received, - spent)
    for o in fills:
        qty    = float(o.filled_qty)
        px     = float(o.filled_avg_price)
        signed = qty if o.side == OrderSide.BUY else -qty
        cash  -= signed * px        # buy spends cash (-), sell receives cash (+)
        pos_qty += signed
        if abs(pos_qty) < 1e-9:     # net flat → one round-trip realized
            trades.append(cash)
            cash = 0.0
    return trades


def count_ict_losses_today_broker(client: TradingClient) -> int:
    """
    Authoritative count of today's LOSING ICT round-trips, derived from the broker.
    Used for the circuit-breaker SAFETY gate. No try/except — a query failure
    propagates so the caller HALTS rather than silently assuming zero losses.
    """
    return sum(1 for pnl in _ict_realized_trades_today(client) if pnl < 0)


# ══════════════════════════════════════════════════════════════════════════════
# TOOL 3 — THE SPREAD SMOKE DETECTOR (NBBO Spread Gate)
# ══════════════════════════════════════════════════════════════════════════════

# Absolute fallback cap (5bp) — used during cold start while history builds.
# Primary gate is now Z-score relative, so this rarely fires in normal operation.
_MAX_SPREAD_PCT  = 0.0005

# Disk-backed spread history — survives across cron ticks so Z-score baseline
# never resets to cold-start. Cache path mirrors the session/security state paths.
_SPREAD_CACHE_PATH = Path.home() / "freqtrade/user_data/logs/ict_spread_cache.json"
_current_spread: Optional[float] = None

# Load history from disk at startup so Z-score baseline persists across ticks.
_spread_history: list[float] = []
try:
    if _SPREAD_CACHE_PATH.exists():
        with open(_SPREAD_CACHE_PATH) as _f:
            _spread_history = json.load(_f).get("history", [])[-20:]  # cap at SPREAD_HISTORY_SIZE
except Exception:
    pass

def refresh_spread_baseline() -> None:
    """
    Fetches NBBO quote once per cron tick. Both is_embargo_active() (Knob A soft phase)
    and check_spread_gate() (Knob C) read from _spread_history after this runs.
    One API call per tick, zero redundant fetches.
    Call this in main() BEFORE is_embargo_active().
    """
    global _spread_history, _current_spread
    try:
        from alpaca.data.historical import StockHistoricalDataClient
        from alpaca.data.requests import StockLatestQuoteRequest
        key, secret = load_credentials()
        data_client = StockHistoricalDataClient(key, secret)
        q = data_client.get_stock_latest_quote(
            StockLatestQuoteRequest(symbol_or_symbols=SYMBOL)
        )[SYMBOL]
        if q.ask_price <= 0: log("[BASELINE] Market closed — no NBBO (ask=0), skipping"); return
        spread_pct = (q.ask_price - q.bid_price) / q.ask_price
        _current_spread = spread_pct
        _spread_history.append(spread_pct)
        if len(_spread_history) > SPREAD_HISTORY_SIZE:
            _spread_history.pop(0)
        # Persist to disk so Z-score baseline survives across cron ticks
        _SPREAD_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(_SPREAD_CACHE_PATH, "w") as f:
            json.dump({"history": _spread_history, "ts": datetime.now().isoformat()}, f)
        log(f"[BASELINE] Spread {spread_pct:.5%} | history={len(_spread_history)}/{SPREAD_HISTORY_SIZE}")
    except Exception as e:
        log(f"[WARN] Spread baseline refresh failed ({e}) — using stale history")


def check_spread_gate(client, symbol: str = SYMBOL,
                      soft_mode: bool = False) -> bool:
    """
    Volatility-adjusted NBBO spread gate.

    Instead of a fixed 5bp threshold (blind to context), compares the current
    spread against a rolling baseline of recent quotes using a Z-score:
        spread_z = (current_spread - median_baseline) / median_baseline

    When spread_z > SPREAD_Z_THRESHOLD (default 0.5 = 50% above baseline):
        → gate fires, trade aborted

    This means NY open naturally-wide spreads do NOT trigger the gate, because
    the rolling median is also wide during that window. Only a sudden *relative*
    spike (news, flash crash) causes the Z-score to cross threshold.

    soft_mode=True: uses SPREAD_Z_SOFT (more lenient) — called by the post-news
    soft embargo window to decide whether to clear early.

    Fail-open: always returns True on API failure. Gate error ≠ blocked trade.
    """
    global _spread_history, _current_spread
    try:
        if _current_spread is None:
            log("[SMOKE DETECTOR] No spread data yet — refresh_spread_baseline() not called, fail-open")
            return True

        spread_pct = _current_spread
        threshold = SPREAD_Z_SOFT if soft_mode else SPREAD_Z_THRESHOLD

        if len(_spread_history) >= 5:
            import statistics
            baseline  = statistics.median(_spread_history[:-1])
            spread_z  = (spread_pct - baseline) / baseline if baseline > 0 else 0.0
            if spread_z > threshold:
                log(f"[SMOKE DETECTOR] Spread {spread_pct:.5%} | Z={spread_z:.2f} > {threshold} | baseline {baseline:.5%} — aborted")
                return False
            log(f"[SMOKE DETECTOR] Spread {spread_pct:.5%} | Z={spread_z:.2f} | baseline {baseline:.5%} — cleared")
        else:
            if spread_pct > _MAX_SPREAD_PCT:
                log(f"[SMOKE DETECTOR] Cold-start: {spread_pct:.5%} > {_MAX_SPREAD_PCT:.5%} — aborted")
                return False
            log(f"[SMOKE DETECTOR] Cold-start: {spread_pct:.5%} — cleared ({len(_spread_history)}/5 samples)")
        return True
    except Exception as e:
        log(f"[WARN] Spread gate check failed ({e}) — fail-open")
        return True


# ── GUARD 1 + 2: The Bouncer & The Decoy Detector ───────────────────────────

def _load_security_state() -> dict:
    try:
        if SECURITY_STATE_PATH.exists():
            with open(SECURITY_STATE_PATH) as f:
                return json.load(f)
    except Exception:
        pass
    return {"last_bar_ts": "", "seen_hashes": []}


def _save_security_state(state: dict) -> None:
    try:
        SECURITY_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(SECURITY_STATE_PATH, "w") as f:
            json.dump(state, f)
    except Exception as e:
        log(f"[WARN] Security state write failed: {e}")


def is_signal_valid(side: str, symbol: str, bar_ts: str) -> bool:
    """
    Guard 1 — The Bouncer: rejects signals from a bar that's already been
    processed. On a cron-based bot, this prevents re-entering the same FVG
    setup if the 5m bar hasn't changed between two consecutive runs.

    Guard 2 — The Decoy Detector: SHA256 content-addresses every signal.
    If the same symbol+side+bar_ts combination has already been acted on,
    the duplicate is silently dropped. Survives crashes and restarts.
    Keeps only the last 100 hashes so the file stays lightweight.
    """
    import hashlib
    state = _load_security_state()

    # Guard 1: Bouncer — bar must be newer than the last accepted bar
    if bar_ts and bar_ts <= state["last_bar_ts"]:
        log(f"[BOUNCER] Signal dropped — bar {bar_ts} already processed (last: {state['last_bar_ts']})")
        return False

    # Guard 2: Decoy Detector — content-address this exact signal
    signal_hash = hashlib.sha256(f"{symbol}|{side}|{bar_ts}".encode()).hexdigest()[:16]
    if signal_hash in state["seen_hashes"]:
        log(f"[DECOY DETECTOR] Duplicate signal dropped — hash {signal_hash} already executed")
        return False

    # Passed both guards — update state
    state["last_bar_ts"] = bar_ts
    state["seen_hashes"].append(signal_hash)
    if len(state["seen_hashes"]) > 100:
        state["seen_hashes"].pop(0)
    _save_security_state(state)
    log(f"[SIGNAL VERIFIED] {side} on bar {bar_ts} | hash {signal_hash} — cleared for execution")
    return True


# ── GUARD 3: The Orphan Reaper ────────────────────────────────────────────────

def orphan_reaper(client: TradingClient) -> None:
    """
    Runs at the top of every cron tick. Fetches all open SYMBOL orders from
    Alpaca and checks for orphaned stop-loss orders whose take-profit partner
    has already filled. This happens when TP hits while the script is between
    runs — Alpaca auto-closes the position but the SL remains as a ghost.

    Pattern: if open SL exists but no open TP partner → TP already hit →
    cancel the ghost SL and flatten any residual position.

    Wrapped in try/except — a reaper failure must never block a live trade.
    """
    try:
        from alpaca.trading.requests import GetOrdersRequest
        from alpaca.trading.enums import QueryOrderStatus

        open_orders = client.get_orders(GetOrdersRequest(
            status=QueryOrderStatus.OPEN,
            symbols=[SYMBOL],
            limit=20,
        ))

        sl_orders = [o for o in open_orders
                     if getattr(o, "type", None) and o.type.value in ("stop", "stop_limit")]
        tp_orders = [o for o in open_orders
                     if getattr(o, "type", None) and o.type.value == "limit"]

        if sl_orders and not tp_orders:
            log(f"[ORPHAN REAPER] Ghost SL detected — TP already filled, {len(sl_orders)} orphan(s) found")
            for sl in sl_orders:
                try:
                    client.cancel_order_by_id(sl.id)
                    log(f"[ORPHAN REAPER] Cancelled ghost SL id={sl.id}")
                except Exception as e:
                    log(f"[WARN] Could not cancel ghost SL {sl.id}: {e}")
            try:
                client.close_position(SYMBOL)
                log(f"[ORPHAN REAPER] Position flattened — account clean")
            except Exception:
                log(f"[ORPHAN REAPER] Position already flat")
            send_telegram(
                f"👻 <b>ORPHAN REAPER</b>\n"
                f"Ghost SL cancelled ({len(sl_orders)} order(s))\n"
                f"TP had already filled — position flattened cleanly."
            )
    except Exception as e:
        log(f"[WARN] Orphan Reaper scan failed ({e}) — continuing")


def send_telegram(msg: str) -> None:
    token   = os.environ.get("TELEGRAM_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        return
    try:
        requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            json={"chat_id": chat_id, "text": msg, "parse_mode": "HTML"},
            timeout=10,
        )
    except Exception:
        pass


def load_credentials() -> tuple[str, str]:
    key    = os.environ.get("ALPACA_KEY")
    secret = os.environ.get("ALPACA_SECRET")
    if key and secret:
        return key, secret
    with open(CONFIG_PATH) as f:
        cfg = json.load(f)
    ex = cfg["exchange"]
    return ex["key"], ex["secret"]


STATE_PATH = Path.home() / "freqtrade/user_data/logs/ict_state.json"


def _is_tradingview_running() -> bool:
    """Check if TradingView Desktop's CDP port is open (fast, 0.5s max)."""
    try:
        with socket.create_connection(("localhost", 9222), timeout=0.5):
            return True
    except (socket.timeout, ConnectionRefusedError, OSError):
        return False


def _tv_fetch_bars(tf: str, count: int) -> pd.DataFrame | None:
    """
    Call tv_fetch.js to get OHLCV bars for SIGNAL_SYMBOL at the given timeframe.
    Returns a DataFrame with lowercase columns and UTC DatetimeIndex, or None on failure.
    """
    if not TV_FETCH.exists() or not _is_tradingview_running():
        return None
    try:
        result = subprocess.run(
            ["node", str(TV_FETCH), "--symbol", SIGNAL_SYMBOL, "--tf", tf, "--count", str(count)],
            capture_output=True, text=True, timeout=45,
            cwd=str(TV_FETCH.parent),
        )
        if result.returncode != 0 or not result.stdout.strip():
            log(f"[WARN] tv_fetch.js ({tf}m) exit={result.returncode}: {result.stderr.strip()[:200]}")
            return None
        data = json.loads(result.stdout.strip())
        if not data.get("success") or not data.get("bars"):
            log(f"[WARN] tv_fetch.js ({tf}m) returned: {data.get('error', 'no bars')}")
            return None
        df = pd.DataFrame(data["bars"])
        df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
        df = df.set_index("time").sort_index()
        df.columns = [c.lower() for c in df.columns]
        df.dropna(inplace=True)
        return df
    except Exception as e:
        log(f"[WARN] tv_fetch.js ({tf}m) exception: {e}")
        return None


def fetch_data_alpaca() -> tuple[pd.DataFrame | None, pd.DataFrame | None, pd.DataFrame | None]:
    """
    Fetch SPY 5m + 1H + QQQ 5m bars from Alpaca.
    Returns (df_5m, df_1h, df_corr) or (None, None, None) on failure.
    """
    try:
        key, secret = load_credentials()
        data_client = StockHistoricalDataClient(key, secret)
        now = datetime.now(timezone.utc)

        def _get_bars(sym: str, tf_value: int, tf_unit, days_back: int) -> pd.DataFrame:
            req = StockBarsRequest(
                symbol_or_symbols=sym,
                timeframe=TimeFrame(tf_value, tf_unit),
                start=now - timedelta(days=days_back),
                end=now,
                feed="iex",
            )
            bars = data_client.get_stock_bars(req)
            df = bars.df
            if isinstance(df.index, pd.MultiIndex):
                df = df.xs(sym, level="symbol")
            df.index.name = "time"
            df = df[["open", "high", "low", "close", "volume"]].copy()
            df.columns = [c.lower() for c in df.columns]
            df.dropna(inplace=True)
            return df

        df_5m  = _get_bars(SYMBOL,      5, TimeFrameUnit.Minute, 7)
        df_1h  = _get_bars(SYMBOL,      1, TimeFrameUnit.Hour,   60)
        df_corr = _get_bars(CORR_SYMBOL, 5, TimeFrameUnit.Minute, 7)

        if len(df_5m) >= 60 and len(df_1h) >= 20:
            return df_5m, df_1h, df_corr
        log(f"[WARN] Alpaca data API too few bars: 5m={len(df_5m)} 1H={len(df_1h)}")
        return None, None, None
    except Exception as e:
        log(f"[WARN] Alpaca data API failed: {e}")
        return None, None, None


def _is_london_session() -> bool:
    """True if current ET time is in London kill zone (3-5am ET)."""
    now_et = datetime.now(ICTModel.ET).time()
    return time(3, 0) <= now_et <= time(5, 0)


def fetch_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame | None, str]:
    """
    Returns (df_5m, df_1h, df_correlated, data_source).
    df_correlated = QQQ/NQ1! 5m bars for SMT divergence.
    Priority: 1) TradingView ES1!+NQ1!  2) Alpaca API  3) yfinance

    London session note: SPY IEX pre-market opens at 4am ET, not 3am.
    During 3-4am London, only TradingView (ES1!) has live data.
    Alpaca/yfinance are skipped for 3-4am to prevent stale bar analysis.
    """
    log(f"Fetching data — 1) TradingView {SIGNAL_SYMBOL}  2) Alpaca API  3) yfinance {SYMBOL}")

    # 1. TradingView Desktop — ES1! + NQ1!, Mac-only
    df_5m  = _tv_fetch_bars("5",  400)
    df_1h  = _tv_fetch_bars("60", 100)
    df_corr = _tv_fetch_bars("5", 400) if False else None   # placeholder — NQ fetch below
    if df_5m is not None and df_1h is not None and len(df_5m) >= 60:
        # Try to get NQ1! for SMT from TradingView too
        try:
            import subprocess as _sp, json as _json
            r = _sp.run(
                ["node", str(TV_FETCH), "--symbol", "CME_MINI_DL:NQ1!", "--tf", "5", "--count", "400"],
                capture_output=True, text=True, timeout=30, cwd=str(TV_FETCH.parent),
            )
            d = _json.loads(r.stdout.strip())
            if d.get("success") and d.get("bars"):
                import pandas as _pd
                df_corr = _pd.DataFrame(d["bars"])
                df_corr["time"] = _pd.to_datetime(df_corr["time"], unit="s", utc=True)
                df_corr = df_corr.set_index("time").sort_index()
                df_corr.columns = [c.lower() for c in df_corr.columns]
        except Exception:
            df_corr = None
        log(f"[DATA] TradingView {SIGNAL_SYMBOL} | 5m: {len(df_5m)} bars | last: {df_5m['close'].iloc[-1]:.2f} | SMT: {'NQ1!' if df_corr is not None else 'unavailable'}")
        return df_5m, df_1h, df_corr, "tradingview"

    # 2. Alpaca data API — SPY + QQQ
    # Skip during 3-4am London: IEX pre-market opens at 4am ET, not 3am
    # Using stale/empty pre-market bars would create false ICT setups
    if _is_london_session() and datetime.now(ICTModel.ET).time() < time(4, 0):
        log(f"[DATA] Skipping Alpaca/yfinance during 3-4am London — no SPY data yet (IEX opens 4am ET)")
        return None, None, None, "insufficient"

    df_5m, df_1h, df_corr = fetch_data_alpaca()
    if df_5m is not None and len(df_5m) >= 60:
        log(f"[DATA] Alpaca API {SYMBOL} | 5m: {len(df_5m)} bars | last: {df_5m['close'].iloc[-1]:.2f} | SMT: {CORR_SYMBOL}")
        return df_5m, df_1h, df_corr, "alpaca"

    # 3. yfinance — SPY + QQQ
    log(f"[DATA] Falling back to yfinance {SYMBOL}")

    def _yf_bars(sym):
        df = yf.download(sym, period="7d", interval="5m", auto_adjust=True, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.columns = [str(c).lower() for c in df.columns]
        return df.dropna()

    df_5m   = _yf_bars(SYMBOL)
    df_corr = _yf_bars(CORR_SYMBOL)
    df_1h   = yf.download(SYMBOL, period=YF_PERIOD_1H, interval="1h", auto_adjust=True, progress=False)
    if isinstance(df_1h.columns, pd.MultiIndex):
        df_1h.columns = df_1h.columns.get_level_values(0)
    df_1h.columns = [str(c).lower() for c in df_1h.columns]
    df_1h.dropna(inplace=True)

    log(f"[DATA] yfinance {SYMBOL} | 5m: {len(df_5m)} bars | last: {df_5m['close'].iloc[-1]:.2f} | SMT: {CORR_SYMBOL}")
    return df_5m, df_1h, df_corr, "yfinance"


def fetch_htf_futures_1h() -> pd.DataFrame | None:
    """
    Fetch 60 days of 1H bars for ES=F (S&P 500 continuous futures) from yfinance.
    Used ONLY for the HTF trend-bias calculation — NOT for execution or PDH/PDL.
    ES=F trades 24/7 so it captures the overnight Globex structure (London kill
    zone lows, gap opens) that SPY RTH data never sees.
    Returns None on failure — model gracefully falls back to SPY 1H bias.
    """
    try:
        df = yf.download("ES=F", period="60d", interval="1h",
                         auto_adjust=True, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.columns = [str(c).lower() for c in df.columns]
        df = df.dropna()
        if len(df) < 20:
            log("[WARN] ES=F HTF: too few bars — falling back to SPY 1H bias")
            return None
        log(f"[HTF] ES=F 1H futures: {len(df)} bars | last: {df['close'].iloc[-1]:.2f}")
        return df
    except Exception as e:
        log(f"[WARN] ES=F HTF fetch failed ({e}) — falling back to SPY 1H bias")
        return None


def fetch_live_price(client: TradingClient) -> float | None:
    """
    Fetch the absolute latest SPY trade price from Alpaca at the exact moment
    of order submission. Used for both the entry drift gate and position sizing
    so our risk footprint is accurate to the second, not to the last 5m close
    (which can be 1–4 minutes stale — $0.30–$0.80 off during NY open).
    Returns None on failure; callers fall back to the signal entry price.
    """
    try:
        from alpaca.data.historical import StockHistoricalDataClient
        from alpaca.data.requests import StockLatestTradeRequest
        key, secret = load_credentials()
        data_client = StockHistoricalDataClient(key, secret)
        resp        = data_client.get_stock_latest_trade(
            StockLatestTradeRequest(symbol_or_symbols=SYMBOL)
        )
        return float(resp[SYMBOL].price)
    except Exception as e:
        log(f"[WARN] Live price fetch failed ({e}) — using signal entry price")
        return None


def translate_signal_to_spy(signal: dict, client) -> dict:
    """
    Convert ES1! entry/sl/tp prices to SPY prices for Alpaca bracket orders.
    Preserves the risk % from the ES1! ICT setup; applies it to the current SPY price.
    """
    if signal["signal"] == "HOLD":
        return signal

    es1_entry = signal["entry"]
    es1_sl    = signal["sl"]
    risk_pct  = abs(es1_entry - es1_sl) / es1_entry  # e.g. 0.0036 for 0.36%

    # Get current SPY price from Alpaca last trade
    try:
        from alpaca.data.historical import StockHistoricalDataClient
        from alpaca.data.requests import StockLatestTradeRequest
        key, secret = load_credentials()
        data_client = StockHistoricalDataClient(key, secret)
        resp = data_client.get_stock_latest_trade(StockLatestTradeRequest(symbol_or_symbols=SYMBOL))
        spy_price = float(resp[SYMBOL].price)
    except Exception:
        # Fallback: last close from quick yfinance 1m pull
        spy_df = yf.download(SYMBOL, period="1d", interval="1m", progress=False, auto_adjust=True)
        spy_price = float(spy_df["Close"].iloc[-1])

    rr = signal["rr"]
    if signal["signal"] == "BUY":
        spy_sl = round(spy_price * (1 - risk_pct), 2)
        spy_tp = round(spy_price + (spy_price - spy_sl) * rr, 2)
    else:  # SELL
        spy_sl = round(spy_price * (1 + risk_pct), 2)
        spy_tp = round(spy_price - (spy_sl - spy_price) * rr, 2)

    translated = {**signal}
    translated["entry"]  = round(spy_price, 2)
    translated["sl"]     = spy_sl
    translated["tp"]     = spy_tp
    translated["reason"] = f"[ES1!→SPY] {signal['reason']} | ES1!={es1_entry:.2f} risk={risk_pct:.3%}"
    return translated


def save_state(signal: dict, shares: int, equity: float, data_source: str) -> None:
    """Persist SL/TP + metadata so subsequent runs can manage the position."""
    state = {
        "symbol":       SYMBOL,
        "signal":       signal["signal"],
        "entry":        signal["entry"],
        "sl":           signal["sl"],
        "tp":           signal["tp"],
        "shares":       shares,
        "setup":        signal["setup"],
        "kill_zone":    signal["kill_zone"],
        "opened_at":    datetime.now().isoformat(),
        "data_source":  data_source,      # feed active when trade was opened
        "entry_equity": equity,           # frozen at entry — used for exit P&L calculation
        "peak_equity":  equity,           # tracks intraday high for trailing DD
    }
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(STATE_PATH, "w") as f:
        json.dump(state, f, indent=2)
    log(f"[STATE] Saved to {STATE_PATH}")


def load_state() -> dict | None:
    if STATE_PATH.exists():
        with open(STATE_PATH) as f:
            return json.load(f)
    return None


def clear_state() -> None:
    if STATE_PATH.exists():
        STATE_PATH.unlink()
        log("[STATE] Cleared")


# ── Session state (resets daily, persists within a session) ──────────────────

def _load_session() -> dict:
    try:
        if SESSION_STATE_PATH.exists():
            with open(SESSION_STATE_PATH) as f:
                s = json.load(f)
            if s.get("date") == datetime.now().date().isoformat():
                return s
    except Exception:
        pass
    return {}


def _save_session(s: dict) -> None:
    s["date"] = datetime.now().date().isoformat()
    SESSION_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(SESSION_STATE_PATH, "w") as f:
        json.dump(s, f, indent=2)


def get_ict_session_start_equity(client: TradingClient) -> float:
    """
    Return today's ICT-isolated starting equity.
    First call of the day records current equity as baseline so Kronos
    losses from other sessions don't contaminate ICT's daily loss count.
    """
    s = _load_session()
    if "ict_start_equity" in s:
        return float(s["ict_start_equity"])
    eq = get_account_equity(client)
    s["ict_start_equity"] = eq
    _save_session(s)
    return eq


def get_trend_day_bypass(kill_zone: str) -> bool:
    """Return True if trend day bypass was already triggered this session."""
    return bool(_load_session().get(f"trend_bypass_{kill_zone}", False))


def set_trend_day_bypass(kill_zone: str) -> None:
    """Persist trend day bypass flag for this kill zone session."""
    s = _load_session()
    s[f"trend_bypass_{kill_zone}"] = True
    _save_session(s)


# ══════════════════════════════════════════════════════════════════════════════
# RISK MANAGEMENT
# ══════════════════════════════════════════════════════════════════════════════

def get_account_equity(client: TradingClient) -> float:
    acct = client.get_account()
    return float(acct.equity)


def check_daily_dd(client: TradingClient) -> float:
    """
    Returns ICT-isolated daily PnL % vs ICT session start equity.
    Uses get_ict_session_start_equity() — not Alpaca's last_equity — so losses
    from Kronos or other bots running earlier in the day don't consume ICT's
    daily drawdown budget.
    """
    eq   = get_account_equity(client)
    prev = get_ict_session_start_equity(client)
    return (eq - prev) / prev if prev > 0 else 0.0


# count_daily_losses() replaced by count_ict_losses_today() — see TOOL 2 section above.
# The old equity-arithmetic fallback is removed: it caused the "flickering" bug
# where Kronos NVDA profits reset the ICT loss counter mid-day.


def check_total_dd(client: TradingClient) -> float:
    """
    Drawdown % from the all-time equity peak (live equity vs 1-year history peak).
    Fail-CLOSED: a broker error propagates so the caller HALTS rather than assuming
    zero drawdown. (Previously swallowed the error and returned 0.0, silently
    disabling the -8% total-DD breaker whenever the history endpoint hiccuped.)
    An empty history is the one benign case — a fresh account has no peak above
    itself, so 0.0 is correct there.
    """
    hist    = client.get_portfolio_history(GetPortfolioHistoryRequest(period="1A"))
    eq_list = [e for e in hist.equity if e is not None and e > 0]
    current = get_account_equity(client)   # live, includes unrealized P&L
    if not eq_list:
        return 0.0
    peak = max(eq_list + [current])        # peak never below current
    return (current - peak) / peak if peak > 0 else 0.0


def check_trailing_dd(client: TradingClient) -> float:
    """
    Intraday trailing drawdown from the equity peak, read from the BROKER's own
    portfolio history — no local peak_equity file (that read 0 on ephemeral hosts
    and silently disabled prop-firm trailing-DD protection). The5ers/FTMO track
    trailing DD on the whole ACCOUNT, so account-level equity is the right basis
    (all-time peak is separately covered by check_total_dd at -8%). Returns a
    NEGATIVE fraction (-0.04 = 4% below peak). No try/except — a failed query
    propagates so the caller decides explicitly.
    """
    hist = client.get_portfolio_history(GetPortfolioHistoryRequest(
        period="1D", timeframe="5Min", extended_hours=True))
    eqs     = [float(e) for e in hist.equity if e is not None and float(e) > 0]
    current = get_account_equity(client)
    peak    = max(eqs + [current]) if eqs else current   # peak never below current
    return (current - peak) / peak if peak > 0 else 0.0


def notify_bracket_exit(client: TradingClient, state: dict) -> None:
    """
    Called when a state file exists but the position is gone — meaning Alpaca's
    bracket SL or TP leg filled autonomously while the script was not running.
    Queries recent closed orders to identify exit type and price, then sends a
    Telegram notification with the full trade result.
    Wrapped in try/except — a notification failure must never block anything.
    """
    try:
        from alpaca.trading.requests import GetOrdersRequest
        from alpaca.trading.enums import QueryOrderStatus

        entry_px   = float(state.get("entry", 0))
        direction  = state.get("signal", "?")
        shares     = int(state.get("shares", 0))
        sl         = float(state.get("sl", 0))
        tp         = float(state.get("tp", 0))
        setup      = state.get("setup", "?")
        kz         = state.get("kill_zone", "?")
        entry_eq   = float(state.get("entry_equity", 0))
        current_eq = get_account_equity(client)

        # Query last 10 closed SPY orders to find the exit fill
        exit_px    = None
        exit_label = "CLOSED"
        try:
            orders = client.get_orders(GetOrdersRequest(
                status=QueryOrderStatus.CLOSED,
                symbols=[SYMBOL],
                limit=10,
            ))
            # Exit leg: BUY bracket exits via a SELL; SELL bracket exits via a BUY
            expected_exit_side = "sell" if direction == "BUY" else "buy"
            for o in orders:
                if not o.filled_avg_price or float(o.filled_qty or 0) == 0:
                    continue
                if o.side.value == expected_exit_side:
                    exit_px = float(o.filled_avg_price)
                    # Classify SL vs TP by which level the fill is closer to
                    exit_label = "🎯 TP HIT" if abs(exit_px - tp) < abs(exit_px - sl) else "🛑 SL HIT"
                    break
        except Exception:
            pass

        # P&L: use exact fill prices if available, else fall back to equity delta
        if exit_px and entry_px and shares:
            pnl_dollars = (exit_px - entry_px) * shares if direction == "BUY" else (entry_px - exit_px) * shares
            pnl_pct     = pnl_dollars / (entry_px * shares) * 100
        elif entry_eq and current_eq:
            pnl_dollars = current_eq - entry_eq
            pnl_pct     = pnl_dollars / entry_eq * 100 if entry_eq else 0.0
        else:
            pnl_dollars = 0.0
            pnl_pct     = 0.0

        emoji    = "✅" if pnl_dollars >= 0 else "❌"
        px_line  = f"Entry: ${entry_px:.2f} → Exit: ${exit_px:.2f}" if exit_px else f"Entry: ${entry_px:.2f} (exit price unavailable)"

        msg = (
            f"{emoji} <b>ICT {exit_label}</b>\n"
            f"{direction} {shares}x {SYMBOL}\n"
            f"{px_line}\n"
            f"P&amp;L: <b>${pnl_dollars:+.2f} ({pnl_pct:+.2f}%)</b>\n"
            f"Equity now: ${current_eq:,.2f}\n"
            f"Setup: {setup} | Zone: {kz}"
        )
        send_telegram(msg)
        log(f"[EXIT] {exit_label} | P&L: ${pnl_dollars:+.2f} ({pnl_pct:+.2f}%)")

        # Record outcome to the append log + check circuit breaker
        if pnl_dollars < 0:
            total_losses = record_ict_loss(pnl_dollars, exit_label)
            if check_circuit_breaker_magnitude(total_losses):
                engage_day_lock(
                    f"Loss #{total_losses} — magnitude confirmed system failure ({exit_label})",
                    total_losses
                )
        else:
            record_ict_win(pnl_dollars)   # feeds avg_win for CB magnitude calibration

    except Exception as e:
        log(f"[WARN] Exit notification failed: {e}")


def calc_position_size(equity: float, entry: float, sl: float,
                       risk_pct: float = RISK_PER_TRADE) -> int:
    """Shares to buy/sell for given risk %."""
    risk_dollars = equity * risk_pct
    risk_per_share = abs(entry - sl)
    if risk_per_share <= 0:
        return 0
    return max(1, int(risk_dollars / risk_per_share))


# ══════════════════════════════════════════════════════════════════════════════
# POSITION MANAGEMENT
# ══════════════════════════════════════════════════════════════════════════════

def get_ict_position(client: TradingClient) -> dict | None:
    """Returns current ICT position or None."""
    try:
        pos = client.get_open_position(SYMBOL)
        qty = float(pos.qty)
        if qty != 0:
            return {
                "symbol":    SYMBOL,
                "qty":       qty,
                "side":      "long" if qty > 0 else "short",
                "avg_entry": float(pos.avg_entry_price),
                "unrealized_pl": float(pos.unrealized_pl),
            }
    except Exception:
        pass
    return None


def cancel_ict_orders(client: TradingClient) -> None:
    """Cancel all open orders for SYMBOL — removes bracket SL/TP orphans."""
    try:
        from alpaca.trading.requests import GetOrdersRequest
        from alpaca.trading.enums import QueryOrderStatus
        orders = client.get_orders(GetOrdersRequest(
            status=QueryOrderStatus.OPEN,
            symbols=[SYMBOL],
            limit=20,
        ))
        for o in orders:
            try:
                client.cancel_order_by_id(o.id)
            except Exception:
                pass
        if orders:
            _time.sleep(0.5)   # allow exchange to process cancellations
            log(f"[CANCEL] Cancelled {len(orders)} open orders for {SYMBOL}")
    except Exception as e:
        log(f"[WARN] Could not cancel orders: {e}")


def close_ict_position(client: TradingClient) -> None:
    """Cancel bracket orders first, then close position — prevents double-close."""
    cancel_ict_orders(client)
    try:
        client.close_position(SYMBOL)
        log(f"[CLOSE] Closed {SYMBOL} position")
    except Exception as e:
        log(f"[WARN] Could not close position: {e}")


def manage_open_position(client: TradingClient, pos: dict) -> None:
    """
    Check time-based exits first, then log position status.
    Bracket orders handle SL/TP on exchange; time exits are our kill switch.
    """
    ET = ICTModel.ET
    now_et = datetime.now(ET)

    # ── Hard session end cutoff ───────────────────────────────────────────────
    state = load_state()
    if state:
        kz      = state.get("kill_zone")
        cutoff  = SESSION_CUTOFFS_ET.get(kz)
        if cutoff and now_et.time() >= cutoff:
            log(f"[TIME EXIT] {kz} session ended at {cutoff} ET — closing position")
            close_ict_position(client)
            clear_state()
            send_telegram(f"⏰ ICT TIME EXIT: {kz} ended — position closed at session close")
            return

        # ── 60-min stagnation stop (time + structural condition) ─────────────
        try:
            opened_at    = datetime.fromisoformat(state["opened_at"])
            elapsed_bars = (datetime.now() - opened_at).total_seconds() / 300
            if elapsed_bars >= STAGNATION_BARS:
                live_pos      = client.get_open_position(SYMBOL)
                unrealized_pl = float(live_pos.unrealized_pl)
                current_px    = float(live_pos.current_price)
                midline       = (state["entry"] + state["sl"]) / 2   # OB/FVG midline

                # Exit if: not in profit AND price has moved back through the midline
                structure_broken = (
                    (state["signal"] == "BUY"  and current_px < midline) or
                    (state["signal"] == "SELL" and current_px > midline)
                )
                if unrealized_pl <= 0 and structure_broken:
                    log(f"[TIME EXIT] {elapsed_bars:.0f} bars, not in profit, structure broken at midline {midline:.2f}")
                    close_ict_position(client)
                    clear_state()
                    send_telegram(f"⏰ ICT STAGNATION EXIT: {elapsed_bars:.0f} bars + structure broken")
                    return
        except Exception:
            pass

    # ── Regular status log ────────────────────────────────────────────────────
    try:
        live_pos       = client.get_open_position(SYMBOL)
        current        = float(live_pos.current_price)
        unrealized_pct = float(live_pos.unrealized_plpc)
        log(f"[HOLD] {SYMBOL} {pos['side']} @ {current:.2f} | Unrealized: {unrealized_pct:.2%} | Bracket active")
    except Exception as e:
        log(f"[WARN] Could not fetch live position: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# ORDER EXECUTION
# ══════════════════════════════════════════════════════════════════════════════

def _is_duplicate_order(err: Exception) -> bool:
    """
    True if Alpaca rejected an order because its client_order_id already exists —
    i.e. an overlapping run already placed this bar's trade. Alpaca enforces
    client_order_id uniqueness SERVER-SIDE (HTTP 422), so a duplicate can never
    become a second position; this only decides benign-skip vs. failure alert.
    """
    if isinstance(err, APIError) and err.status_code == 422:
        return True
    msg = str(err).lower()
    return "client_order_id" in msg or ("unique" in msg and "order" in msg)


def place_order(client: TradingClient, signal: dict, shares: int,
                equity: float = 0.0, bar_ts: str = "") -> None:
    """
    Bracket order — SL and TP are sent directly to Alpaca.
    Alpaca auto-closes the position when either level is hit,
    even if our script is not running (cloud-safe).

    Idempotent: client_order_id is derived from the signal BAR timestamp, so two
    overlapping runs on the same 5-minute bar build the SAME id and Alpaca rejects
    the duplicate — a hard, server-side complement to is_signal_valid().
    """
    side = OrderSide.BUY if signal["signal"] == "BUY" else OrderSide.SELL
    sl   = round(signal["sl"], 2)
    tp   = round(signal["tp"], 2)

    bar_token = "".join(ch for ch in bar_ts if ch.isalnum()) or datetime.now().strftime("%Y%m%d%H%M%S")
    order_id  = f"{ICT_ORDER_PREFIX}_{SYMBOL}_{bar_token}"
    req = MarketOrderRequest(
        symbol           = SYMBOL,
        qty              = shares,
        side             = side,
        time_in_force    = TimeInForce.GTC,
        order_class      = OrderClass.BRACKET,
        stop_loss        = StopLossRequest(stop_price=sl),
        take_profit      = TakeProfitRequest(limit_price=tp),
        client_order_id  = order_id,
    )
    order = client.submit_order(req)
    if order.status.value not in ("new", "pending_new", "accepted", "filled"):
        raise RuntimeError(f"Order rejected by Alpaca: status={order.status.value} id={order.id}")
    log(f"[ORDER] {signal['signal']} {shares}x {SYMBOL} | {signal['reason']}")
    log(f"        SL={sl} TP={tp} RR=1:{signal['rr']} id={order.id} cid={order_id}")

    direction = "🟢 BUY" if signal["signal"] == "BUY" else "🔴 SELL"
    msg = (
        f"<b>{direction} {SYMBOL}</b>\n"
        f"Setup: {signal['setup']} | Zone: {signal['kill_zone']} | HTF: {signal['htf_bias']}\n"
        f"Entry: <b>${signal['entry']:.2f}</b> | SL: ${sl} | TP: ${tp}\n"
        f"Shares: {shares} | RR: 1:{signal['rr']}\n"
        f"Equity: <b>${equity:,.2f}</b>\n"
        f"{signal['reason']}"
    )
    send_telegram(msg)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    log("═" * 60)
    log("ICT Alpaca Trader — starting")

    # ══════════════════════════════════════════════════════════════════════════
    # TOOL 2 — PHYSICAL PADLOCK: must be the very first check, before anything.
    # A dead process cannot revenge trade. sys.exit(0) cannot be bypassed by
    # downstream logic, exception handlers, or accidental restarts.
    # ══════════════════════════════════════════════════════════════════════════
    if is_day_locked():
        sys.exit(0)

    # Load Alpaca client
    key, secret = load_credentials()
    IS_PAPER    = os.environ.get("ALPACA_PAPER", "true").lower() == "true"
    client      = TradingClient(key, secret, paper=IS_PAPER)
    log(f"Alpaca {'PAPER' if IS_PAPER else 'LIVE'} client connected")

    # ── Priority 0: Session exit always runs first — bypasses all other logic ──
    # This prevents phase transition lock-outs from blocking session closes.
    _pos_early   = get_ict_position(client)
    _state_early = load_state()
    if _pos_early and _state_early:
        _ET     = ICTModel.ET
        _now_et = datetime.now(_ET)
        _kz     = _state_early.get("kill_zone")
        _cutoff = SESSION_CUTOFFS_ET.get(_kz)
        if _cutoff and _now_et.time() >= _cutoff:
            log(f"[PRIORITY EXIT] {_kz} ended — closing before any other checks")
            close_ict_position(client)
            clear_state()
            send_telegram(f"⏰ ICT TIME EXIT: {_kz} session closed")
            return

    # ── Guard 3: Orphan Reaper — sweep ghost orders before anything else ──────
    orphan_reaper(client)

    # ── Spread baseline — fetch NBBO once per tick, shared by all consumers ──
    refresh_spread_baseline()

    # ── Risk checks — broker-derived; fail CLOSED if any cannot be computed ──
    # No silent zero-fills: if a metric can't be read, we do NOT open a trade.
    # daily_losses is now the AUTHORITATIVE broker count (Alpaca order history);
    # the local append-log count stays for the magnitude gate + audit trail.
    try:
        daily_dd     = check_daily_dd(client)
        total_dd     = check_total_dd(client)
        equity       = get_account_equity(client)
        daily_losses = count_ict_losses_today_broker(client)
    except Exception as e:
        log(f"[HALT] Risk metrics unavailable — standing down, no trade: {e}")
        send_telegram(f"⛔ ICT HALT: risk checks failed — {e}")
        return

    log(f"Equity: ${equity:,.2f} | Daily DD: {daily_dd:.2%} | Total DD: {total_dd:.2%} | ICT Losses today: {daily_losses}/{CIRCUIT_BREAKER_LOSSES}")

    if daily_dd <= DAILY_DD_LIMIT:
        log(f"[HALT] Daily DD {daily_dd:.2%} breaches limit {DAILY_DD_LIMIT:.0%}. No trade.")
        send_telegram(f"⛔ ICT HALT: Daily DD {daily_dd:.2%}")
        return

    if total_dd <= TOTAL_DD_LIMIT:
        log(f"[HALT] Total DD {total_dd:.2%} breaches limit {TOTAL_DD_LIMIT:.0%}. No trade.")
        send_telegram(f"⛔ ICT HALT: Total DD {total_dd:.2%}")
        return

    if daily_losses >= CIRCUIT_BREAKER_LOSSES:
        if check_circuit_breaker_magnitude(daily_losses):
            log(f"[HALT] {daily_losses} losses + magnitude confirmed — circuit breaker engaging")
            engage_day_lock(f"{daily_losses} ICT losses — magnitude gate confirmed", daily_losses)
            sys.exit(0)
        else:
            log(f"[CB] {daily_losses} losses hit count limit but magnitude within noise threshold — continuing")

    if daily_dd >= DAILY_PROFIT_TARGET:
        log(f"[HALT] Daily profit target hit: +{daily_dd:.2%} ≥ +{DAILY_PROFIT_TARGET:.0%}. Banking gains.")
        send_telegram(f"🎯 ICT PROFIT TARGET: +{daily_dd:.2%} today — done trading, gains locked")
        return

    # Reduce risk if halfway to daily limit
    risk_pct = RISK_PER_TRADE_SMALL if daily_dd <= DAILY_DD_LIMIT / 2 else RISK_PER_TRADE

    model = ICTModel(tf_minutes=5, fvg_min_pct=0.0002, rr_ratio=3.0, displacement_factor=1.5, strict_filters=False)

    # ── Existing position check ────────────────────────────────────────────
    pos   = get_ict_position(client)
    state = load_state()

    if pos:
        # Trailing drawdown — broker-derived intraday peak (no local peak_equity).
        # Fail SAFE, not closed: if the metric can't be read we HOLD (the exchange
        # bracket still protects) and alert loudly — never force-close on an API blip.
        try:
            trailing_dd = check_trailing_dd(client)
        except Exception as e:
            log(f"[WARN] Trailing-DD unavailable — holding position, bracket intact: {e}")
            send_telegram(f"⚠️ ICT: trailing-DD check failed ({e}) — position held")
            trailing_dd = 0.0
        else:
            if trailing_dd <= -TRAILING_DD_LIMIT:
                log(f"[HALT] Trailing DD {trailing_dd:.2%} from intraday peak — prop firm protection")
                send_telegram(f"⛔ ICT TRAILING DD: {trailing_dd:.2%} from peak — closing position")
                close_ict_position(client)
                clear_state()
                return

        log(f"[POS] Existing ICT position: {pos['side']} {pos['qty']} {SYMBOL} | Trailing DD: {trailing_dd:.2%}")
        manage_open_position(client, pos)
        return

    # ── Bracket exit detection ─────────────────────────────────────────────
    # pos is None here. If a state file exists the bracket SL/TP fired since
    # the last run — Alpaca closed the position autonomously. Notify and reset.
    if state is not None:
        notify_bracket_exit(client, state)
        clear_state()
        return

    # ══════════════════════════════════════════════════════════════════════════
    # TOOL 1 — NEWS BLACKOUT CLOCK: gates before fetch_data() so no signal is
    # ever evaluated during CPI/FOMC/open volatility windows. Existing positions
    # are unaffected — the check only blocks new entry logic below.
    # ══════════════════════════════════════════════════════════════════════════
    if is_embargo_active():
        log("[EMBARGO] High-impact news window active — signal evaluation suppressed")
        return

    # ── Data feed + signal ─────────────────────────────────────────────────
    df_5m, df_1h, df_corr, data_source = fetch_data()
    # Fetch ES=F 1H futures separately for HTF bias — anchors trend to the
    # continuous overnight market, not SPY's RTH-only window.
    df_1h_htf = fetch_htf_futures_1h()

    if data_source == "insufficient" or df_5m is None:
        log("[SKIP] No actionable data for this session window — standing by")
        return

    # Data feed freeze: if source changed mid-session, don't open new positions
    if state and state.get("data_source") and state["data_source"] != data_source:
        log(f"[FREEZE] Data source changed mid-session ({state['data_source']} → {data_source}) — no new entries")
        send_telegram(f"⚠️ ICT DATA FEED CHANGE: {state['data_source']} → {data_source} — entries frozen")
        return

    # Pass trend day bypass state from session file into model
    active_kz      = None   # determined after signal
    trend_bp_state = {}     # {kz: bool} — loaded per kill zone below

    signal = model.get_signal(df_5m, df_1h, df_correlated=df_corr, df_1h_htf=df_1h_htf)

    # Persist trend day bypass if signal reason indicates it
    if signal.get("kill_zone"):
        active_kz = signal["kill_zone"]
        if "[TREND_DAY]" in signal.get("reason", "") or get_trend_day_bypass(active_kz):
            set_trend_day_bypass(active_kz)

    # Freeze the original ICT model entry BEFORE any price translation.
    # translate_signal_to_spy() overwrites signal["entry"] with a fresh live
    # SPY price. If we later compare exec_px (also a fresh live price) against
    # that overwritten value, we get live - live = 0 and the drift gate is blind.
    # _model_entry preserves the original FVG/OB level for the honest comparison.
    _model_entry = signal["entry"]

    # Translate ES1! price levels → SPY prices when TradingView data was used
    if data_source == "tradingview" and signal["signal"] != "HOLD":
        signal = translate_signal_to_spy(signal, client)

    log(f"Signal: {signal['signal']} | {signal['reason']}")

    if signal["signal"] == "HOLD":
        log("No ICT setup — standing by")
        return

    # ── Guard 1+2: Bouncer + Decoy Detector ──────────────────────────────────
    # bar_ts = timestamp of the last 5m bar — the cron-native equivalent of
    # bar_index from a webhook payload. Rejects stale or duplicate signals.
    bar_ts = str(df_5m.index[-1])
    if not is_signal_valid(signal["signal"], SYMBOL, bar_ts):
        return

    # ── Live price fetch — used for both drift gate and position sizing ───────
    # The 5m close is 1–4 min stale at execution time; during NY open that
    # translates to $0.30–$0.80 of untracked slippage baked into every size calc.
    live_px = fetch_live_price(client)
    exec_px = live_px if live_px is not None else float(df_5m["close"].iloc[-1])

    # ── Entry drift gate — active on ALL data sources ─────────────────────────
    # Previously gated behind `data_source == "tradingview"`, which made it
    # completely dead code on the VPS where data_source is always "alpaca".
    entry_drift = abs(exec_px - _model_entry) / _model_entry
    if entry_drift > MAX_ENTRY_DRIFT:
        log(f"[WARN] Trade skipped: Entry drift exceeded ({entry_drift:.3%} > {MAX_ENTRY_DRIFT:.3%}) — price has moved from FVG zone")
        return

    # Stamp live execution price onto the signal so the state file and Telegram
    # notification reflect the actual fill price, not the stale bar close.
    signal = {**signal, "entry": round(exec_px, 2)}

    # ── Position sizing — live entry vs structural SL for exact risk ──────────
    # signal["sl"] is the ICT structural level (unchanged by live price).
    # Risk distance = live fill price → SL, not historical bar close → SL.
    shares = calc_position_size(equity, signal["entry"], signal["sl"], risk_pct)
    if shares == 0:
        log("[WARN] Calculated 0 shares — skip")
        return

    cost = shares * signal["entry"]
    log(f"Live entry: ${signal['entry']:.2f} (drift: {entry_drift:.3%}) | {shares} shares = ${cost:,.2f}")

    # ══════════════════════════════════════════════════════════════════════════
    # TOOL 3 — SPREAD SMOKE DETECTOR: live NBBO check at the exact submission
    # instant. Aborts if bid-ask spread > 5bp — the market's own warning light
    # for news spikes and flash-crash conditions before price even gaps.
    # ══════════════════════════════════════════════════════════════════════════
    if not check_spread_gate(client, SYMBOL):
        log("[SMOKE DETECTOR] Entry aborted — abnormal spread detected")
        return

    # ── Place bracket order ─────────────────────────────────────────────────
    # bar_ts (computed above for is_signal_valid) → deterministic client_order_id.
    save_state(signal, shares, equity, data_source)   # save BEFORE submit — prevent orphan on crash
    try:
        place_order(client, signal, shares, equity, bar_ts)
    except Exception as e:
        if _is_duplicate_order(e):
            # Concurrent run already placed this bar's trade — keep state, no alert.
            log(f"[IDEMPOTENT] Bar {bar_ts} already traded by a concurrent run — no double entry")
            return
        log(f"[ERROR] Order failed — clearing state: {e}")
        clear_state()
        send_telegram(f"❌ ICT ORDER FAILED: {e}")
        return

    log("═" * 60)


if __name__ == "__main__":
    main()
