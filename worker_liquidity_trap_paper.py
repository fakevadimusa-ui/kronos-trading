#!/usr/bin/env python3
"""
worker_liquidity_trap_paper.py — NY Open Liquidity Trap | PAPER ONLY
======================================================================
*** PAPER ONLY — ZERO LIVE ORDERS — NO ACCOUNT CREDENTIALS USED ***

Strategy: Variant A — Overnight High/Low Sweep Reclaim
  - Trade window : 09:30–10:30 AM ET
  - Reference    : Overnight session H/L (17:00 ET prior day → 09:30 ET today)
  - Signal       : Price sweeps below OH/OL by ≥1.0 pt AND closes back inside within same 5-min bar
  - Entry        : Close of signal bar
  - Stop         : Sweep extreme ± 1 tick (0.25 pts)
  - Target       : 1.5R from entry
  - Max 1 trade per direction per day
  - Dead market  : Skip if ATR14 < 2.0 pts
  - News filter  : Skip on CPI / NFP / FOMC days

LOCKED parameters (from Phase 10 grid search, best combo on 51 days ES data):
  MIN_SWEEP_PTS = 1.0     (4 ticks — filters noise)
  TARGET_R      = 1.5     (reward:risk ratio)
  WINDOW        = 9:30–10:30 ET
  FRICTION      = 0.60 pts roundtrip (0.10 commission + 0.50 slippage)

DO NOT change parameters mid-test. The paper test must run consistent rules
to produce a statistically valid forward-test. 50 trades needed before any
conclusion can be drawn.

Cron (run from kronos-trading directory, UTC times):
  */5 13,14 * * 1-5  cd /root/kronos-trading && venv/bin/python3 worker_liquidity_trap_paper.py >> /root/logs/liquidity_trap/paper.log 2>&1

Log files:
  ~/logs/liquidity_trap/paper.log          — timestamped run log
  ~/logs/liquidity_trap/signals.jsonl      — one line per completed paper trade
  ~/logs/liquidity_trap/daily_summary.jsonl — one line per trading day
  ~/logs/liquidity_trap/state.json         — active paper trade state (reset daily)
"""

from __future__ import annotations

import json
import os
import sys
import warnings
from datetime import datetime, date, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests
import yfinance as yf

warnings.filterwarnings("ignore")

# ── Identity ──────────────────────────────────────────────────────────────────
WORKER_NAME = "liquidity_trap_paper"
PAPER_ONLY  = True   # NEVER set this to False

ET = ZoneInfo("America/New_York")

# ── Log paths ─────────────────────────────────────────────────────────────────
LOG_DIR       = Path.home() / "logs" / "liquidity_trap"
SIGNALS_FILE  = LOG_DIR / "signals.jsonl"
SUMMARY_FILE  = LOG_DIR / "daily_summary.jsonl"
STATE_FILE    = LOG_DIR / "state.json"
LOG_FILE      = LOG_DIR / "paper.log"

# ── Strategy parameters (LOCKED — do not change mid-test) ────────────────────
MIN_SWEEP_PTS   = 1.0    # minimum sweep below/above level (4 ticks)
MAX_SWEEP_PTS   = 10.0   # skip if sweep > 10 pts (gap or extreme volatility)
TARGET_R        = 1.5    # reward:risk ratio
STOP_EXTRA_TICKS = 1     # stop is 1 tick beyond sweep extreme
TICK_SIZE       = 0.25
FRICTION_PTS    = 0.60   # roundtrip friction applied to all paper trades

WINDOW_OPEN_H   = 9      # 09:30 ET
WINDOW_OPEN_M   = 30
WINDOW_CLOSE_H  = 10     # 10:30 ET — hard exit
WINDOW_CLOSE_M  = 30
FORCE_EXIT_H    = 10     # force-close open trade at 10:30
FORCE_EXIT_M    = 30

MIN_ATR         = 2.0    # dead market filter
MAX_HOLD_BARS   = 12     # 60 minutes at 5-min bars — force close

# ── News filter ───────────────────────────────────────────────────────────────
NEWS_EVENTS_FILE  = Path(__file__).parent / "news_events.json"
NEWS_FILTER_HOURS = 2   # skip if high-impact event within ±2h of window open

# ── Telegram ──────────────────────────────────────────────────────────────────
_env_path = Path(__file__).parent / ".env"
if _env_path.exists():
    for _line in _env_path.read_text().splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _v = _line.split("=", 1)
            os.environ.setdefault(_k.strip(), _v.strip().strip('"').strip("'"))

TG_TOKEN  = os.environ.get("TELEGRAM_TOKEN", os.environ.get("TELEGRAM_BOT_TOKEN", ""))
TG_CHAT   = os.environ.get("TELEGRAM_CHAT_ID", "")


# ═══════════════════════════════════════════════════════════════════════════════
# LOGGING
# ═══════════════════════════════════════════════════════════════════════════════

def _log(msg: str) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(ET).strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] [PAPER] {msg}"
    print(line, flush=True)


def _tg(msg: str) -> None:
    if not TG_TOKEN or not TG_CHAT:
        return
    try:
        requests.post(
            f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage",
            json={"chat_id": TG_CHAT, "text": f"[PAPER-LT] {msg}", "parse_mode": "HTML"},
            timeout=6,
        )
    except Exception:
        pass


# ═══════════════════════════════════════════════════════════════════════════════
# STATE (persists between 5-min cron runs)
# ═══════════════════════════════════════════════════════════════════════════════

def _today() -> str:
    return datetime.now(ET).strftime("%Y-%m-%d")


def load_state() -> dict:
    if STATE_FILE.exists():
        try:
            s = json.loads(STATE_FILE.read_text())
            if s.get("date") == _today():
                return s
        except Exception:
            pass
    # New day or corrupted state — fresh start
    return {
        "date": _today(),
        "active_trade": None,
        "long_fired": False,
        "short_fired": False,
        "daily_trades": [],
    }


def save_state(state: dict) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps(state, indent=2, default=str))


# ═══════════════════════════════════════════════════════════════════════════════
# DATA
# ═══════════════════════════════════════════════════════════════════════════════

def fetch_bars() -> pd.DataFrame:
    """
    Fetch last 5 days of ES 5-min bars via yfinance.
    Returns DataFrame indexed by naive ET datetime, sorted ascending.
    Drops the last (still-forming) bar.
    """
    raw = yf.download("ES=F", period="5d", interval="5m",
                      auto_adjust=True, progress=False)
    if raw.empty:
        raise RuntimeError("yfinance returned empty DataFrame for ES=F")
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    df = raw[["Open", "High", "Low", "Close", "Volume"]].dropna().copy()
    # Convert to naive ET (matches existing workers)
    df.index = pd.to_datetime(df.index, utc=True).tz_convert(ET).tz_localize(None)
    df.index.name = "Datetime"
    df = df.sort_index()
    # Drop last bar (still forming — no intra-bar signals)
    df = df.iloc[:-1]
    return df


def _tod_dec(dt_naive_et) -> float:
    """Return time-of-day as decimal hours ET (9.5 = 09:30)."""
    return dt_naive_et.hour + dt_naive_et.minute / 60.0


def compute_overnight_levels(df: pd.DataFrame, today: date) -> tuple[float, float]:
    """
    Overnight session high/low for today:
      - Prior trading day bars from 17:00 ET onward
      - Today's bars before 09:30 ET

    Returns (overnight_high, overnight_low). Returns (nan, nan) if insufficient data.
    """
    yesterday = today - timedelta(days=1)
    # Skip to prior TRADING day (skip weekends)
    while yesterday.weekday() >= 5:
        yesterday -= timedelta(days=1)

    prior_evening = df[
        (df.index.date == yesterday) & (df.index.hour >= 17)
    ]
    today_preopen = df[
        (df.index.date == today) & (_tod_dec(df.index) < 9.5)
    ]
    overnight_bars = pd.concat([prior_evening, today_preopen])

    if len(overnight_bars) < 4:
        return np.nan, np.nan

    return overnight_bars["High"].max(), overnight_bars["Low"].min()


def compute_atr(df: pd.DataFrame, period: int = 14) -> float:
    """Current 14-bar ATR from the most recent bars."""
    prev_close = df["Close"].shift(1)
    tr = pd.concat([
        df["High"] - df["Low"],
        (df["High"] - prev_close).abs(),
        (df["Low"]  - prev_close).abs(),
    ], axis=1).max(axis=1)
    atr_series = tr.ewm(span=period, adjust=False).mean()
    return float(atr_series.iloc[-1]) if not atr_series.empty else np.nan


# ═══════════════════════════════════════════════════════════════════════════════
# NEWS FILTER
# ═══════════════════════════════════════════════════════════════════════════════

def is_news_day(today_str: str) -> tuple[bool, str]:
    """Return (True, event_name) if today is a high-impact news day."""
    if not NEWS_EVENTS_FILE.exists():
        return False, ""
    try:
        events = json.loads(NEWS_EVENTS_FILE.read_text())
    except Exception:
        return False, ""
    for ev in events:
        if ev.get("date") == today_str and ev.get("impact") == "HIGH":
            return True, ev.get("event", "UNKNOWN")
    return False, ""


# ═══════════════════════════════════════════════════════════════════════════════
# SIGNAL DETECTION
# ═══════════════════════════════════════════════════════════════════════════════

def _round_tick(price: float) -> float:
    return round(price / TICK_SIZE) * TICK_SIZE


def detect_signal(
    bar: pd.Series,
    overnight_high: float,
    overnight_low: float,
    direction: str,  # 'long' or 'short'
) -> dict | None:
    """
    Check a single closed 5-min bar for a sweep+reclaim signal.

    LONG (bullish):
      bar.low  ≤ overnight_low  - MIN_SWEEP_PTS   → swept below OL
      bar.close > overnight_low                    → reclaimed above OL (same bar)
      sweep size ≤ MAX_SWEEP_PTS

    SHORT (bearish):
      bar.high ≥ overnight_high + MIN_SWEEP_PTS   → swept above OH
      bar.close < overnight_high                   → reclaimed below OH (same bar)
      sweep size ≤ MAX_SWEEP_PTS

    Returns signal dict if triggered, None otherwise.
    """
    if direction == "long":
        if np.isnan(overnight_low):
            return None
        swept     = bar["Low"] <= overnight_low - MIN_SWEEP_PTS
        reclaimed = bar["Close"] > overnight_low
        extreme   = bar["Low"]
        sweep_size = overnight_low - extreme
        level     = overnight_low
        level_name = "overnight_low"
    else:
        if np.isnan(overnight_high):
            return None
        swept     = bar["High"] >= overnight_high + MIN_SWEEP_PTS
        reclaimed = bar["Close"] < overnight_high
        extreme   = bar["High"]
        sweep_size = extreme - overnight_high
        level     = overnight_high
        level_name = "overnight_high"

    if not swept or not reclaimed:
        return None
    if sweep_size > MAX_SWEEP_PTS:
        return None

    # Compute entry, stop, target
    entry_price = bar["Close"]
    if direction == "long":
        stop_price  = _round_tick(extreme - STOP_EXTRA_TICKS * TICK_SIZE)
        risk_pts    = entry_price - stop_price
    else:
        stop_price  = _round_tick(extreme + STOP_EXTRA_TICKS * TICK_SIZE)
        risk_pts    = stop_price - entry_price

    if risk_pts <= 0:
        return None

    reward_pts  = risk_pts * TARGET_R
    if direction == "long":
        target_price = entry_price + reward_pts
    else:
        target_price = entry_price - reward_pts

    return {
        "direction":      direction,
        "ref_level_name": level_name,
        "ref_level":      level,
        "sweep_extreme":  extreme,
        "sweep_size_pts": round(sweep_size, 2),
        "entry_price":    round(entry_price, 2),
        "stop_price":     round(stop_price, 2),
        "target_price":   round(round(target_price / TICK_SIZE) * TICK_SIZE, 2),
        "risk_pts":       round(risk_pts, 2),
        "reward_pts":     round(reward_pts, 2),
        "r_multiple":     TARGET_R,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# TRADE MANAGEMENT
# ═══════════════════════════════════════════════════════════════════════════════

def check_trade_exit(
    active: dict,
    df: pd.DataFrame,
    now_et: datetime,
) -> dict | None:
    """
    Check if the active paper trade has been exited.
    Scans all bars from entry_bar_time onward.
    Returns updated trade dict with status/exit fields, or None if still open.
    """
    entry_dt = pd.Timestamp(active["entry_bar_time"])
    direction = active["direction"]
    stop   = active["stop_price"]
    target = active["target_price"]

    # Bars AFTER entry bar (the entry bar itself is already done)
    candidate_bars = df[df.index > entry_dt]

    force_close_time = now_et.replace(
        hour=FORCE_EXIT_H, minute=FORCE_EXIT_M, second=0, microsecond=0
    )
    force_close = now_et >= force_close_time

    for bar_dt, bar in candidate_bars.iterrows():
        if direction == "long":
            tp_hit = bar["High"] >= target
            sl_hit = bar["Low"]  <= stop
        else:
            tp_hit = bar["Low"]  <= target
            sl_hit = bar["High"] >= stop

        # Conservative: if both hit same bar, assume stop first
        if sl_hit and tp_hit:
            return _close_trade(active, stop, "SL", bar_dt)
        if tp_hit:
            return _close_trade(active, target, "TP", bar_dt)
        if sl_hit:
            return _close_trade(active, stop, "SL", bar_dt)

        # Time-based force close
        if bar_dt >= force_close_time.replace(tzinfo=None):
            return _close_trade(active, bar["Close"], "TIME", bar_dt)

    # Still open: check if we need to force-close right now
    if force_close:
        last_bar = candidate_bars.iloc[-1] if not candidate_bars.empty else None
        close_px = last_bar["Close"] if last_bar is not None else active["entry_price"]
        return _close_trade(active, close_px, "TIME", now_et)

    return None


def _close_trade(trade: dict, exit_price: float, reason: str, exit_time) -> dict:
    direction = trade["direction"]
    entry     = trade["entry_price"]

    raw_pnl   = (exit_price - entry) if direction == "long" else (entry - exit_price)
    net_pnl   = raw_pnl - FRICTION_PTS
    r_result  = net_pnl / trade["risk_pts"] if trade["risk_pts"] > 0 else 0.0

    trade = {**trade}  # copy
    trade["status"]       = "WIN" if net_pnl > 0 else ("LOSS" if net_pnl < 0 else "SCRATCH")
    trade["exit_price"]   = round(exit_price, 2)
    trade["exit_reason"]  = reason
    trade["exit_time"]    = str(exit_time)[:19]
    trade["pnl_pts"]      = round(raw_pnl, 3)
    trade["pnl_pts_net"]  = round(net_pnl, 3)
    trade["r_result"]     = round(r_result, 3)
    return trade


# ═══════════════════════════════════════════════════════════════════════════════
# LOGGING
# ═══════════════════════════════════════════════════════════════════════════════

def log_signal(trade: dict) -> None:
    """Append completed trade to signals.jsonl."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    record = {**trade, "paper_only": True, "worker": WORKER_NAME}
    with open(SIGNALS_FILE, "a") as f:
        f.write(json.dumps(record, default=str) + "\n")


def write_daily_summary(state: dict, skipped: bool, skip_reason: str = "") -> None:
    """Append daily summary to daily_summary.jsonl."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    trades = state.get("daily_trades", [])
    wins       = [t for t in trades if t.get("status") == "WIN"]
    losses     = [t for t in trades if t.get("status") == "LOSS"]
    timeouts   = [t for t in trades if t.get("status") not in ("WIN", "LOSS")]
    total_r    = sum(t.get("r_result", 0) for t in trades)
    total_pts  = sum(t.get("pnl_pts_net", 0) for t in trades)

    record = {
        "date":          _today(),
        "n_signals":     len(trades),
        "wins":          len(wins),
        "losses":        len(losses),
        "timeouts":      len(timeouts),
        "net_r":         round(total_r, 3),
        "net_pts":       round(total_pts, 3),
        "skipped":       skipped,
        "skip_reason":   skip_reason,
        "paper_only":    True,
    }
    with open(SUMMARY_FILE, "a") as f:
        f.write(json.dumps(record, default=str) + "\n")

    # Telegram daily summary
    if trades:
        emoji = "✅" if total_pts >= 0 else "❌"
        msg = (
            f"{emoji} <b>Daily Summary — {_today()}</b>\n"
            f"Trades: {len(trades)} | W:{len(wins)} L:{len(losses)} T:{len(timeouts)}\n"
            f"Net: {total_pts:+.2f} pts ({total_r:+.2f}R)\n"
            f"<i>PAPER ONLY</i>"
        )
    elif skipped:
        msg = f"⏸ {_today()}: No trade — {skip_reason} | PAPER"
    else:
        msg = f"⚪ {_today()}: No valid setup today | PAPER"

    _tg(msg)


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    _log("=" * 60)
    _log("PAPER ONLY — NO LIVE ORDERS — LIQUIDITY TRAP VARIANT A")
    _log("=" * 60)

    now_et    = datetime.now(ET)
    today_str = _today()
    today_d   = now_et.date()

    # ── Time gate: only run during 09:25–10:35 ET ─────────────────────────
    tod = now_et.hour + now_et.minute / 60.0
    in_window_precheck  = tod >= 9.4   # 09:24 — allow early cron runs
    in_window_postcheck = tod <= 10.65  # 10:39 — allow exit checks slightly past close
    if not (in_window_precheck and in_window_postcheck):
        # End-of-day summary write: run once after 10:35 ET
        if tod >= 10.65 and tod < 11.0:
            state = load_state()
            if state.get("active_trade"):
                _log("Force closing active trade at EOD (post-window check)")
                # Will be handled on next run if not already done
            if not state.get("summary_written_today"):
                write_daily_summary(state, skipped=False)
                state["summary_written_today"] = True
                save_state(state)
        else:
            _log(f"Outside trading window (ET={tod:.2f}h). Exiting.")
        return

    _log(f"Running at {now_et.strftime('%H:%M:%S')} ET")

    # ── Load state ────────────────────────────────────────────────────────
    state = load_state()
    _log(f"State: active={bool(state['active_trade'])} long_fired={state['long_fired']} short_fired={state['short_fired']}")

    # ── News filter ───────────────────────────────────────────────────────
    is_news, news_ev = is_news_day(today_str)
    if is_news:
        _log(f"NEWS FILTER: {news_ev} today — no new signals (will manage existing)")

    # ── Fetch live bars ───────────────────────────────────────────────────
    try:
        df = fetch_bars()
        _log(f"Fetched {len(df)} bars | last bar: {df.index[-1]} | last close: {df['Close'].iloc[-1]:.2f}")
    except Exception as e:
        _log(f"ERROR fetching bars: {e}")
        return

    # ── Compute overnight levels ──────────────────────────────────────────
    oh, ol = compute_overnight_levels(df, today_d)
    if np.isnan(oh) or np.isnan(ol):
        _log("WARNING: Insufficient overnight data — cannot compute OH/OL. Skipping signals.")
        if not state.get("active_trade"):
            return
    else:
        _log(f"Overnight range: H={oh:.2f}  L={ol:.2f}  Range={oh-ol:.2f}pts")

    # ── ATR filter ────────────────────────────────────────────────────────
    atr = compute_atr(df)
    _log(f"ATR14: {atr:.2f} pts (min={MIN_ATR})")
    atr_ok = not np.isnan(atr) and atr >= MIN_ATR

    # ── Manage active paper trade ─────────────────────────────────────────
    if state["active_trade"]:
        active = state["active_trade"]
        _log(f"Active paper trade: {active['direction'].upper()} @ {active['entry_price']} | "
             f"stop={active['stop_price']} target={active['target_price']}")

        closed = check_trade_exit(active, df, now_et)
        if closed:
            status = closed["status"]
            pnl    = closed["pnl_pts_net"]
            r      = closed["r_result"]
            reason = closed["exit_reason"]
            _log(f"TRADE CLOSED: {status} | exit={closed['exit_price']} "
                 f"reason={reason} pnl={pnl:+.2f}pts ({r:+.2f}R)")

            log_signal(closed)
            state["daily_trades"].append(closed)
            state["active_trade"] = None

            emoji = "✅" if status == "WIN" else "❌"
            _tg(
                f"{emoji} <b>Paper Trade Closed</b>\n"
                f"Dir: {closed['direction'].upper()} | {reason}\n"
                f"Entry: {closed['entry_price']} → Exit: {closed['exit_price']}\n"
                f"P&L: {pnl:+.2f} pts ({r:+.2f}R)\n"
                f"<i>PAPER ONLY</i>"
            )
            save_state(state)
        else:
            _log(f"Trade still OPEN | current close: {df['Close'].iloc[-1]:.2f}")

    # ── Scan for new signal ───────────────────────────────────────────────
    # Only scan if no active trade and not yet fired for that direction today
    # and not a news-filtered day
    if not state["active_trade"] and atr_ok and not (np.isnan(oh) or np.isnan(ol)):

        # Get the most recently closed bar in the trading window
        window_mask = df.index.map(lambda dt: (
            (dt.date() == today_d) and
            (dt.hour + dt.minute / 60.0 >= 9.5) and
            (dt.hour + dt.minute / 60.0 < 10.5)
        ))
        window_bars = df[window_mask]

        if len(window_bars) == 0:
            _log("No bars in 09:30–10:30 window yet — waiting for market open")
        else:
            last_bar = window_bars.iloc[-1]
            last_bar_tod = last_bar.name.hour + last_bar.name.minute / 60.0
            _log(f"Last window bar: {last_bar.name} | O={last_bar['Open']:.2f} H={last_bar['High']:.2f} "
                 f"L={last_bar['Low']:.2f} C={last_bar['Close']:.2f}")

            # Check LONG signal
            if not state["long_fired"] and not is_news:
                sig = detect_signal(last_bar, oh, ol, "long")
                if sig:
                    _log(f"*** LONG SIGNAL ***  entry={sig['entry_price']} "
                         f"stop={sig['stop_price']} target={sig['target_price']} "
                         f"risk={sig['risk_pts']:.2f}pts sweep={sig['sweep_size_pts']:.2f}pts")

                    trade = {
                        "date":           today_str,
                        "symbol":         "ES=F",
                        "overnight_high": round(oh, 2),
                        "overnight_low":  round(ol, 2),
                        "atr14":          round(atr, 2),
                        "entry_bar_time": str(last_bar.name),
                        "entry_time_et":  last_bar.name.strftime("%H:%M"),
                        "news_filtered":  is_news,
                        **sig,
                    }
                    state["active_trade"] = trade
                    state["long_fired"]   = True

                    _tg(
                        f"🔔 <b>Paper Signal: LONG</b>\n"
                        f"Ref: overnight_low = {ol:.2f}\n"
                        f"Sweep to {sig['sweep_extreme']:.2f} ({sig['sweep_size_pts']:.2f}pts)\n"
                        f"Entry: {sig['entry_price']} | Stop: {sig['stop_price']} | Target: {sig['target_price']}\n"
                        f"Risk: {sig['risk_pts']:.2f}pts | R={TARGET_R}\n"
                        f"<i>PAPER ONLY</i>"
                    )
                    save_state(state)
                    _log("State saved — active paper long trade")
                    return

            # Check SHORT signal
            if not state["short_fired"] and not is_news:
                sig = detect_signal(last_bar, oh, ol, "short")
                if sig:
                    _log(f"*** SHORT SIGNAL ***  entry={sig['entry_price']} "
                         f"stop={sig['stop_price']} target={sig['target_price']} "
                         f"risk={sig['risk_pts']:.2f}pts sweep={sig['sweep_size_pts']:.2f}pts")

                    trade = {
                        "date":           today_str,
                        "symbol":         "ES=F",
                        "overnight_high": round(oh, 2),
                        "overnight_low":  round(ol, 2),
                        "atr14":          round(atr, 2),
                        "entry_bar_time": str(last_bar.name),
                        "entry_time_et":  last_bar.name.strftime("%H:%M"),
                        "news_filtered":  is_news,
                        **sig,
                    }
                    state["active_trade"] = trade
                    state["short_fired"]  = True

                    _tg(
                        f"🔔 <b>Paper Signal: SHORT</b>\n"
                        f"Ref: overnight_high = {oh:.2f}\n"
                        f"Sweep to {sig['sweep_extreme']:.2f} ({sig['sweep_size_pts']:.2f}pts)\n"
                        f"Entry: {sig['entry_price']} | Stop: {sig['stop_price']} | Target: {sig['target_price']}\n"
                        f"Risk: {sig['risk_pts']:.2f}pts | R={TARGET_R}\n"
                        f"<i>PAPER ONLY</i>"
                    )
                    save_state(state)
                    _log("State saved — active paper short trade")
                    return

            if state["long_fired"] and state["short_fired"]:
                _log("Both directions already fired today — no new signals")
            elif is_news:
                _log(f"News filter active ({news_ev}) — no new signals today")
            else:
                _log("No signal on last closed bar — watching for sweep+reclaim")

    elif not atr_ok:
        _log(f"Dead market filter: ATR {atr:.2f} < {MIN_ATR} — no new signals")
    elif state["active_trade"]:
        _log("Already in a paper trade — not scanning for new signals")

    save_state(state)
    _log("Run complete.")


if __name__ == "__main__":
    main()
