"""
ICT MT5 Trader — US500 CFD Execution for The5ers
=================================================
Production execution module for The5ers funded accounts.
Trades US500 (S&P 500 CFD) via MetaTrader 5 with native limit orders.
Signal engine: ICTModel (same as Alpaca paper bot — no changes needed).

Key differences from Alpaca version:
  - Limit orders (not market) — price, SL, TP sent as one request to MT5
  - Lot sizing (not shares) — 1 lot US500 ≈ $1/point, sized via symbol info
  - Magic number (not client_order_id string) — isolates ICT from other bots
  - MT5 polling for position status (not websocket push)
  - No PDT rule — CFD accounts are exempt

VPS Setup:
  Option A (recommended): Windows VPS — Azure B1s ~$13/mo, MT5 runs natively
  Option B: Ubuntu + Wine — sudo apt install wine, then install MT5 .exe via wine

pip install MetaTrader5 pytz requests

Environment variables:
  MT5_LOGIN, MT5_PASSWORD, MT5_SERVER (from The5ers welcome email)
  TELEGRAM_TOKEN, TELEGRAM_CHAT_ID
"""

from __future__ import annotations

import json
import os
import time as _time
import warnings
from datetime import datetime, timezone, timedelta, time
from pathlib import Path

import numpy as np
import pandas as pd
import pytz
import requests

try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError:
    MT5_AVAILABLE = False
    print("[WARN] MetaTrader5 not installed. Run: pip install MetaTrader5")

warnings.filterwarnings("ignore")

from ict_model import ICTModel


# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════

SYMBOL           = "US500"      # S&P 500 index CFD on The5ers
CORR_SYMBOL      = "US100"      # Nasdaq CFD for SMT divergence
ICT_MAGIC        = 1001         # Magic number — identifies ICT bot orders (use 2002 for Kronos)

RISK_PER_TRADE   = 0.005        # 0.5% per trade
DAILY_DD_LIMIT   = -0.04        # -4% daily halt
TOTAL_DD_LIMIT   = -0.08        # -8% total halt
DAILY_LOSS_LIMIT = 2            # halt after 2 losses
DAILY_PROFIT_PCT = 0.015        # bank gains at +1.5%
TRAILING_DD_LIMIT = 0.04        # 4% from intraday peak (prop firm trailing DD)
MAX_ENTRY_DRIFT  = 0.0015       # 0.15% max price drift from signal entry

STATE_PATH   = Path.home() / "freqtrade/user_data/logs/ict_mt5_state.json"
SESSION_PATH = Path.home() / "freqtrade/user_data/logs/ict_mt5_session.json"

SESSION_CUTOFFS_ET = {
    "london":        time(5,  0),
    "new_york":      time(11, 0),
    "london_close":  time(12, 0),
    "silver_bullet": time(16, 0),
}

ET = pytz.timezone("America/New_York")


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


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


def load_state() -> dict | None:
    if STATE_PATH.exists():
        with open(STATE_PATH) as f:
            return json.load(f)
    return None


def save_state(signal: dict, lots: float, equity: float) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(STATE_PATH, "w") as f:
        json.dump({
            "symbol":      SYMBOL,
            "signal":      signal["signal"],
            "entry":       signal["entry"],
            "sl":          signal["sl"],
            "tp":          signal["tp"],
            "lots":        lots,
            "setup":       signal["setup"],
            "kill_zone":   signal["kill_zone"],
            "opened_at":   datetime.now().isoformat(),
            "peak_equity": equity,
        }, f, indent=2)


def clear_state() -> None:
    if STATE_PATH.exists():
        STATE_PATH.unlink()


def _load_session() -> dict:
    try:
        if SESSION_PATH.exists():
            with open(SESSION_PATH) as f:
                s = json.load(f)
            if s.get("date") == datetime.now().date().isoformat():
                return s
    except Exception:
        pass
    return {}


def _save_session(s: dict) -> None:
    s["date"] = datetime.now().date().isoformat()
    SESSION_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(SESSION_PATH, "w") as f:
        json.dump(s, f, indent=2)


# ══════════════════════════════════════════════════════════════════════════════
# MT5 CONNECTION
# ══════════════════════════════════════════════════════════════════════════════

def connect_mt5() -> None:
    if not MT5_AVAILABLE:
        raise RuntimeError("MetaTrader5 package not installed")
    if not mt5.initialize():
        raise RuntimeError(f"MT5 initialize() failed: {mt5.last_error()}")
    login    = int(os.environ.get("MT5_LOGIN", 0))
    password = os.environ.get("MT5_PASSWORD", "")
    server   = os.environ.get("MT5_SERVER", "")
    if login and password and server:
        if not mt5.login(login, password=password, server=server):
            raise RuntimeError(f"MT5 login failed: {mt5.last_error()}")
    log("MT5 connected")


# ══════════════════════════════════════════════════════════════════════════════
# DATA FETCH
# ══════════════════════════════════════════════════════════════════════════════

def _mt5_bars(symbol: str, tf_minutes: int, count: int) -> pd.DataFrame | None:
    """Fetch OHLCV from MT5 broker feed as DataFrame."""
    TF_MAP = {1: mt5.TIMEFRAME_M1, 5: mt5.TIMEFRAME_M5, 15: mt5.TIMEFRAME_M15, 60: mt5.TIMEFRAME_H1}
    tf = TF_MAP.get(tf_minutes)
    if tf is None:
        return None
    rates = mt5.copy_rates_from_pos(symbol, tf, 0, count)
    if rates is None or len(rates) == 0:
        return None
    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    df = df.set_index("time").rename(columns={"tick_volume": "volume"})
    df = df[["open", "high", "low", "close", "volume"]].copy()
    df.dropna(inplace=True)
    return df


def fetch_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame | None]:
    """Returns (df_5m, df_1h, df_corr) from MT5 broker feed."""
    df_5m   = _mt5_bars(SYMBOL,      5,  400)
    df_1h   = _mt5_bars(SYMBOL,      60, 100)
    df_corr = _mt5_bars(CORR_SYMBOL, 5,  400)

    if df_5m is None or df_1h is None or len(df_5m) < 60:
        raise RuntimeError(f"Insufficient MT5 data: 5m={len(df_5m) if df_5m is not None else 0} bars")

    log(f"[DATA] MT5 {SYMBOL} | 5m: {len(df_5m)} bars | last: {df_5m['close'].iloc[-1]:.2f} | SMT: {CORR_SYMBOL}")
    return df_5m, df_1h, df_corr


# ══════════════════════════════════════════════════════════════════════════════
# ACCOUNT & RISK
# ══════════════════════════════════════════════════════════════════════════════

def get_account_equity() -> float:
    info = mt5.account_info()
    if info is None:
        raise RuntimeError("MT5 account_info() returned None")
    return float(info.equity)


def check_daily_dd() -> float:
    """P&L % vs today's starting balance."""
    s = _load_session()
    start = s.get("start_balance")
    if not start:
        info = mt5.account_info()
        start = float(info.balance)
        s["start_balance"] = start
        _save_session(s)
    current = get_account_equity()
    return (current - start) / start if start > 0 else 0.0


def check_trailing_dd(state: dict | None) -> float:
    if not state or "peak_equity" not in state:
        return 0.0
    return (get_account_equity() - state["peak_equity"]) / state["peak_equity"]


def update_peak_equity(state: dict) -> None:
    current = get_account_equity()
    if current > state.get("peak_equity", 0):
        state["peak_equity"] = current
        with open(STATE_PATH, "w") as f:
            json.dump(state, f, indent=2)


def count_daily_losses() -> int:
    """Count closed ICT losing trades today."""
    try:
        today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        deals = mt5.history_deals_get(today, datetime.now())
        if deals is None:
            return 0
        return sum(
            1 for d in deals
            if d.magic == ICT_MAGIC
            and d.profit < 0
            and d.entry == mt5.DEAL_ENTRY_OUT  # closing trade, not opening
        )
    except Exception:
        return 0


# ══════════════════════════════════════════════════════════════════════════════
# POSITION SIZING  (Lots for CFD)
# ══════════════════════════════════════════════════════════════════════════════

def calc_lots(entry: float, sl: float, risk_dollars: float) -> float:
    """
    Calculate lot size for US500 CFD.
    Formula: lots = risk_dollars / (sl_points × tick_value)
    """
    info = mt5.symbol_info(SYMBOL)
    if info is None:
        raise RuntimeError(f"Symbol {SYMBOL} not found on MT5")

    sl_points  = abs(entry - sl) / info.point
    tick_value = info.trade_tick_value   # $ per tick per lot

    if sl_points <= 0 or tick_value <= 0:
        return 0.0

    lots = risk_dollars / (sl_points * tick_value)
    lots = round(lots / info.volume_step) * info.volume_step
    lots = max(info.volume_min, min(lots, info.volume_max))
    return round(lots, 2)


# ══════════════════════════════════════════════════════════════════════════════
# ORDER EXECUTION
# ══════════════════════════════════════════════════════════════════════════════

def place_limit_order(signal: dict, lots: float) -> bool:
    """
    Place MT5 limit order with SL/TP directly attached.
    Limit orders guarantee entry at the FVG/OB level — no market order slippage.
    """
    order_type = mt5.ORDER_TYPE_BUY_LIMIT if signal["signal"] == "BUY" else mt5.ORDER_TYPE_SELL_LIMIT

    request = {
        "action":       mt5.TRADE_ACTION_PENDING,
        "symbol":       SYMBOL,
        "volume":       float(lots),
        "type":         order_type,
        "price":        round(signal["entry"], 2),
        "sl":           round(signal["sl"], 2),
        "tp":           round(signal["tp"], 2),
        "deviation":    20,
        "magic":        ICT_MAGIC,
        "comment":      f"ICT_{signal['setup']}_{signal['kill_zone']}",
        "type_time":    mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }

    result = mt5.order_send(request)

    # Retry with FOK if IOC not supported (retcode 10030)
    if result.retcode == 10030:
        request["type_filling"] = mt5.ORDER_FILLING_FOK
        result = mt5.order_send(request)

    if result.retcode != mt5.TRADE_RETCODE_DONE:
        log(f"[ERROR] Order failed retcode={result.retcode}: {result.comment}")
        send_telegram(f"⛔ ICT ORDER FAILED: {result.retcode} — {result.comment}")
        return False

    log(f"[ORDER] {signal['signal']} {lots:.2f} lots {SYMBOL} @ {signal['entry']:.2f} | SL={signal['sl']:.2f} TP={signal['tp']:.2f}")
    log(f"        Ticket: {result.order} | {signal['reason']}")
    send_telegram(
        f"<b>ICT {signal['signal']}</b> {SYMBOL}\n"
        f"Setup: {signal['setup']} | Zone: {signal['kill_zone']} | HTF: {signal['htf_bias']}\n"
        f"Entry: <b>{signal['entry']:.2f}</b> | SL: {signal['sl']:.2f} | TP: {signal['tp']:.2f}\n"
        f"Lots: {lots:.2f} | {signal['reason']}"
    )
    return True


# ══════════════════════════════════════════════════════════════════════════════
# POSITION MANAGEMENT
# ══════════════════════════════════════════════════════════════════════════════

def get_ict_position():
    """Return open ICT position (by magic number) or None."""
    positions = mt5.positions_get(symbol=SYMBOL)
    if not positions:
        return None
    for pos in positions:
        if pos.magic == ICT_MAGIC:
            return pos
    return None


def get_ict_pending():
    """Return pending ICT limit order or None."""
    orders = mt5.orders_get(symbol=SYMBOL)
    if not orders:
        return None
    for o in orders:
        if o.magic == ICT_MAGIC:
            return o
    return None


def cancel_ict_orders() -> None:
    """Cancel all pending ICT orders before closing — prevents orphan SL/TP."""
    orders = mt5.orders_get(symbol=SYMBOL)
    if not orders:
        return
    for o in orders:
        if o.magic == ICT_MAGIC:
            result = mt5.order_send({"action": mt5.TRADE_ACTION_REMOVE, "order": o.ticket})
            if result.retcode == mt5.TRADE_RETCODE_DONE:
                log(f"[CANCEL] Cancelled order {o.ticket}")
    _time.sleep(0.5)


def close_ict_position() -> None:
    """Close ICT position with cancel-first to prevent double-close."""
    cancel_ict_orders()
    pos = get_ict_position()
    if pos is None:
        return

    tick       = mt5.symbol_info_tick(SYMBOL)
    close_type = mt5.ORDER_TYPE_SELL if pos.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY
    close_px   = tick.bid if close_type == mt5.ORDER_TYPE_SELL else tick.ask

    result = mt5.order_send({
        "action":       mt5.TRADE_ACTION_DEAL,
        "symbol":       SYMBOL,
        "volume":       pos.volume,
        "type":         close_type,
        "position":     pos.ticket,
        "price":        close_px,
        "deviation":    20,
        "magic":        ICT_MAGIC,
        "comment":      "ICT_Exit",
        "type_filling": mt5.ORDER_FILLING_IOC,
    })

    if result.retcode == mt5.TRADE_RETCODE_DONE:
        log(f"[CLOSE] Position {pos.ticket} closed @ {close_px:.2f}")
    else:
        log(f"[WARN] Close failed retcode={result.retcode}")


def manage_open_position(pos, state: dict | None) -> None:
    """Handle time exits and stagnation stops on open MT5 position."""
    now_et = datetime.now(ET)

    # Session time exit
    if state:
        kz     = state.get("kill_zone")
        cutoff = SESSION_CUTOFFS_ET.get(kz)
        if cutoff and now_et.time() >= cutoff:
            log(f"[TIME EXIT] {kz} ended — closing position")
            close_ict_position()
            clear_state()
            send_telegram(f"⏰ ICT TIME EXIT: {kz} ended")
            return

        # Stagnation: 60 min + structure broken
        try:
            opened_at    = datetime.fromisoformat(state["opened_at"])
            elapsed_bars = (datetime.now() - opened_at).total_seconds() / 300
            if elapsed_bars >= 12:
                midline = (state["entry"] + state["sl"]) / 2
                broken  = (
                    (state["signal"] == "BUY"  and pos.price_current < midline) or
                    (state["signal"] == "SELL" and pos.price_current > midline)
                )
                if pos.profit <= 0 and broken:
                    log(f"[STAGNATION] {elapsed_bars:.0f} bars, structure broken at midline {midline:.2f}")
                    close_ict_position()
                    clear_state()
                    send_telegram(f"⏰ ICT STAGNATION EXIT: 60 min, structure broken")
                    return
        except Exception:
            pass

    log(f"[HOLD] {SYMBOL} | P&L: ${pos.profit:.2f} | Lots: {pos.volume}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    log("═" * 60)
    log("ICT MT5 Trader — starting")

    if not MT5_AVAILABLE:
        log("[ERROR] MetaTrader5 not installed. pip install MetaTrader5")
        return

    connect_mt5()

    equity     = get_account_equity()
    daily_dd   = check_daily_dd()
    daily_loss = count_daily_losses()
    state      = load_state()

    log(f"Equity: ${equity:,.2f} | Daily DD: {daily_dd:.2%} | Losses: {daily_loss}/{DAILY_LOSS_LIMIT}")

    # ── Priority 0: Session exit — bypasses all other checks ─────────────────
    pos = get_ict_position()
    if pos and state:
        kz     = state.get("kill_zone")
        cutoff = SESSION_CUTOFFS_ET.get(kz)
        now_et = datetime.now(ET)
        if cutoff and now_et.time() >= cutoff:
            log(f"[PRIORITY EXIT] {kz} session ended")
            close_ict_position()
            clear_state()
            send_telegram(f"⏰ ICT TIME EXIT: {kz}")
            mt5.shutdown()
            return

    # ── Risk halts ────────────────────────────────────────────────────────────
    if daily_dd <= DAILY_DD_LIMIT:
        log(f"[HALT] Daily DD {daily_dd:.2%}")
        send_telegram(f"⛔ ICT HALT: Daily DD {daily_dd:.2%}")
        mt5.shutdown(); return

    if daily_dd >= DAILY_PROFIT_PCT:
        log(f"[HALT] Profit target +{daily_dd:.2%} hit")
        send_telegram(f"🎯 ICT PROFIT: +{daily_dd:.2%} — done for day")
        mt5.shutdown(); return

    if daily_loss >= DAILY_LOSS_LIMIT:
        log(f"[HALT] {daily_loss} losses today")
        send_telegram(f"⛔ ICT HALT: {daily_loss} losses")
        mt5.shutdown(); return

    # ── Open position management ──────────────────────────────────────────────
    if pos:
        if state:
            update_peak_equity(state)
        trailing_dd = check_trailing_dd(state)
        if trailing_dd <= -TRAILING_DD_LIMIT:
            log(f"[HALT] Trailing DD {trailing_dd:.2%} from peak")
            close_ict_position()
            clear_state()
            send_telegram(f"⛔ ICT TRAILING DD: {trailing_dd:.2%}")
            mt5.shutdown(); return
        log(f"[POS] Trailing DD: {trailing_dd:.2%}")
        manage_open_position(pos, state)
        mt5.shutdown(); return

    # Cancel any stale pending limit orders from previous cron run
    pending = get_ict_pending()
    if pending:
        log(f"[STALE] Cancelling unfilled pending order {pending.ticket}")
        cancel_ict_orders()

    # ── Signal generation ─────────────────────────────────────────────────────
    df_5m, df_1h, df_corr = fetch_data()
    model  = ICTModel(tf_minutes=5, fvg_min_pct=0.0002, rr_ratio=3.0, displacement_factor=1.5)
    signal = model.get_signal(df_5m, df_1h, df_correlated=df_corr)

    log(f"Signal: {signal['signal']} | {signal['reason']}")

    if signal["signal"] == "HOLD":
        log("No ICT setup — standing by")
        mt5.shutdown(); return

    # Entry drift gate
    current_px  = float(df_5m["close"].iloc[-1])
    entry_drift = abs(current_px - signal["entry"]) / signal["entry"]
    if entry_drift > MAX_ENTRY_DRIFT:
        log(f"[SKIP] Entry drift {entry_drift:.3%} > {MAX_ENTRY_DRIFT:.3%}")
        mt5.shutdown(); return

    # ── Position sizing ───────────────────────────────────────────────────────
    risk_dollars = equity * RISK_PER_TRADE
    lots         = calc_lots(signal["entry"], signal["sl"], risk_dollars)
    if lots <= 0:
        log("[WARN] 0 lots — skip")
        mt5.shutdown(); return

    log(f"Sizing: {lots:.2f} lots | risk=${risk_dollars:.0f} | SL distance={abs(signal['entry']-signal['sl']):.2f} pts")

    # ── Execute limit order ───────────────────────────────────────────────────
    if place_limit_order(signal, lots):
        save_state(signal, lots, equity)

    mt5.shutdown()
    log("═" * 60)


if __name__ == "__main__":
    main()
