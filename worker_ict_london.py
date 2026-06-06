"""
Worker 3 — ICT London Kill Zone Sniper
========================================
Session  : London Kill Zone — 3:00 AM to 5:00 AM ET
Cron UTC : */5 7,8 * * 1-5   (07:00–09:00 UTC = 3:00–5:00 AM ET)
Account  : ICT-Challenge paper account (.env.ict)
Log dir  : ~/logs/ict_london/
Lock     : /tmp/worker_ict_london.lock

Strategy: ICT London Judas Swing
  - Sweep of overnight accumulation box (midnight–2AM ET)
  - Displaced FVG entry during London open
  - EMA-50 trend guard
  - HTF 1H bias filter (no lookahead)

Risk parameters:
  SL floor  : raw_sl ≥ 0.5 × ATR14
  Max risk  : $250 per trade (slightly tighter than NY — London is choppier)
  Max size  : 3 contracts
  RR target : 3.0 (1:3 — London reversals tend to be shorter lived)
  Stagnation: 8 bars (40 min) — London session is time-pressured

ISOLATION GUARANTEES:
  - Reads credentials from .env.ict (same ICT-Challenge account as NY worker
    but account is separate from Kronos — capital isolation from Kronos is the goal)
  - Writes logs to ~/logs/ict_london/ only
  - State files prefixed ict_london_* — NEVER touches ict_ny_* files
  - Daily loss limit independent counter in ~/logs/ict_london/daily_losses.json
  - Run lock /tmp/worker_ict_london.lock — SEPARATE from NY and Kronos locks
  - Zero imports from worker_ict_ny or worker_kronos
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import yfinance as yf

# ── Worker identity ───────────────────────────────────────────────────────────
WORKER_NAME  = "ict_london"
ENV_FILE     = str(Path(__file__).parent / ".env.ict")
LOG_DIR      = str(Path.home() / "logs" / "ict_london")
LOCK_FILE    = "/tmp/worker_ict_london.lock"
STATE_FILE   = Path.home() / "logs/ict_london/ict_london_state.json"
LOSS_LOG     = Path.home() / "logs/ict_london/daily_losses.json"
ET           = ZoneInfo("America/New_York")

# ── Strategy parameters (London-specific — different from NY worker) ──────────
DISP_THRESHOLD      = 1.5
FVG_LOOKBACK        = 6      # slightly shorter — London FVGs go stale faster
ATR_MULT            = 0.5
RR                  = 3.0    # 1:3 for London (shorter trend bursts vs NY 1:4)
MAX_TRADE_RISK      = 250    # $250 — tighter in London (more choppy)
ES_MULT             = 50.0
MAX_CONTRACTS       = 3      # max 3 in London (vs 4 in NY)
STAGNATION_BARS     = 8      # 40 min max (London close pressure)
MIN_SL_ATR_MULT     = 0.5
SYMBOL              = "SPY"
SIGNAL_TICKER       = "ES=F"

# ── Entry window (ET) ─────────────────────────────────────────────────────────
WINDOW_OPEN  = (3, 0)        # 3:00 AM ET
WINDOW_CLOSE = (5, 0)        # 5:00 AM ET

# ── Independent daily loss limit ─────────────────────────────────────────────
DAILY_LOSS_LIMIT = 2

# ── Bootstrap shared utilities ────────────────────────────────────────────────
from worker_shared import make_logger, load_credentials, send_telegram, RunLock

log  = make_logger(WORKER_NAME, LOG_DIR)
lock = RunLock(LOCK_FILE, log)


# ═══════════════════════════════════════════════════════════
# DAILY LOSS TRACKING  (independent — London only)
# ═══════════════════════════════════════════════════════════

def _today_str() -> str:
    return datetime.now(ET).strftime("%Y-%m-%d")


def get_loss_count() -> int:
    if not LOSS_LOG.exists():
        return 0
    try:
        return json.loads(LOSS_LOG.read_text()).get(_today_str(), 0)
    except Exception:
        return 0


def record_loss() -> int:
    LOSS_LOG.parent.mkdir(parents=True, exist_ok=True)
    data  = json.loads(LOSS_LOG.read_text()) if LOSS_LOG.exists() else {}
    today = _today_str()
    data[today] = data.get(today, 0) + 1
    cutoff = (datetime.now(ET) - timedelta(days=7)).strftime("%Y-%m-%d")
    data   = {k: v for k, v in data.items() if k >= cutoff}
    LOSS_LOG.write_text(json.dumps(data, indent=2))
    return data[today]


# ═══════════════════════════════════════════════════════════
# STATE
# ═══════════════════════════════════════════════════════════

def load_state() -> dict | None:
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text())
        except Exception:
            return None
    return None


def save_state(state: dict) -> None:
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps(state, indent=2, default=str))


def clear_state() -> None:
    if STATE_FILE.exists():
        STATE_FILE.unlink()


# ═══════════════════════════════════════════════════════════
# DATA
# ═══════════════════════════════════════════════════════════

def fetch_bars(bars: int = 80) -> pd.DataFrame:
    """5-min ES=F bars. Needs more bars than NY to cover accum box (midnight–2AM)."""
    raw = yf.download("ES=F", period="2d", interval="5m",
                      auto_adjust=True, progress=False)
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    df = raw[["Open", "High", "Low", "Close", "Volume"]].dropna().copy()
    df.index = (pd.to_datetime(df.index, utc=True)
                .tz_convert("America/New_York").tz_localize(None))
    df.index.name = "Datetime"
    return df.sort_index().tail(bars)


# ═══════════════════════════════════════════════════════════
# SIGNAL ENGINE  (London-specific: uses overnight accum box)
# ═══════════════════════════════════════════════════════════

def _atr(df: pd.DataFrame, p: int = 14) -> pd.Series:
    h, l, c = df["High"], df["Low"], df["Close"].shift(1)
    tr = pd.concat([h-l, (h-c).abs(), (l-c).abs()], axis=1).max(axis=1)
    return tr.ewm(span=p, adjust=False).mean()


def _ffill_cap(arr: np.ndarray, cap: int, idx) -> pd.Series:
    out = np.full(len(arr), np.nan); last = np.nan; cnt = cap + 1
    for i in range(len(arr)):
        if not np.isnan(arr[i]): last = arr[i]; cnt = 0
        if cnt <= cap: out[i] = last; cnt += 1
    return pd.Series(out, index=idx)


def compute_signal(df: pd.DataFrame) -> dict | None:
    if len(df) < 25:
        return None

    df = df.copy()
    df["atr14"] = _atr(df, 14)
    df["ema50"]  = df["Close"].ewm(span=50, adjust=False).mean()
    body         = (df["Close"] - df["Open"]).abs()
    avg_body     = body.rolling(14, min_periods=1).mean()
    df["_date"]  = df.index.date
    h = df.index.hour; m = df.index.minute

    # Window gate: 3:00–5:00 AM ET
    df["in_window"] = (h == 3) | (h == 4)

    # PDH / PDL
    daily = df.groupby("_date").agg(dh=("High","max"), dl=("Low","min"))
    df    = df.join(daily.shift(1).rename(columns={"dh":"pdh","dl":"pdl"}), on="_date")

    # ── London-specific: overnight accumulation box (midnight–2AM ET) ────
    # This is the ICT "true" Judas swing — manipulation of the overnight range.
    # Completely absent from the NY worker which uses PDH/PDL sweeps instead.
    in_accum = (h >= 0) & (h < 2)
    df["accum_h"] = df["High"].where(in_accum).groupby(df["_date"]).transform("max").ffill()
    df["accum_l"] = df["Low"].where(in_accum).groupby(df["_date"]).transform("min").ffill()
    not_accum     = ~in_accum

    accum_bear = (df["High"] > df["accum_h"].shift(1)) & not_accum & df["accum_h"].notna()
    accum_bull = (df["Low"]  < df["accum_l"].shift(1)) & not_accum & df["accum_l"].notna()
    pdh_bear   = df["High"] > df["pdh"].fillna(-np.inf)
    pdl_bull   = df["Low"]  < df["pdl"].fillna(np.inf)

    # Primary: accum box sweep. Fallback: PDH/PDL sweep
    df["judas_bear"] = (accum_bear | pdh_bear).groupby(df["_date"]).transform("cummax")
    df["judas_bull"] = (accum_bull | pdl_bull).groupby(df["_date"]).transform("cummax")

    # HTF 1H bias (no lookahead)
    df_1h = df.resample("1h").agg(High=("High","max"), Low=("Low","min"),
                                  Close=("Close","last")).dropna()
    df_1h["bias"] = "neutral"
    hh = df_1h["High"].shift(1) > df_1h["High"].shift(2)
    hl = df_1h["Low"].shift(1)  > df_1h["Low"].shift(2)
    lh = df_1h["High"].shift(1) < df_1h["High"].shift(2)
    ll = df_1h["Low"].shift(1)  < df_1h["Low"].shift(2)
    df_1h.loc[hh & hl, "bias"] = "bull"
    df_1h.loc[lh & ll, "bias"] = "bear"
    df_1h["bias"] = df_1h["bias"].ffill()
    df["htf_bias"] = df_1h["bias"].reindex(df.index, method="ffill").fillna("neutral")

    # Displaced FVGs
    n = len(df)
    bt = np.full(n, np.nan); bb = np.full(n, np.nan)
    rt = np.full(n, np.nan); rb = np.full(n, np.nan)
    ab = avg_body.values; bv = body.values

    for i in range(2, n):
        mid = bv[i-1]
        if ab[i-1] <= 0 or mid < DISP_THRESHOLD * ab[i-1]:
            continue
        if df["Low"].iloc[i] > df["High"].iloc[i-2]:
            bt[i] = df["Low"].iloc[i]; bb[i] = df["High"].iloc[i-2]
        if df["High"].iloc[i] < df["Low"].iloc[i-2]:
            rt[i] = df["Low"].iloc[i-2]; rb[i] = df["High"].iloc[i]

    idx = df.index
    df["bull_fvg_t"] = _ffill_cap(bt, FVG_LOOKBACK, idx)
    df["bull_fvg_b"] = _ffill_cap(bb, FVG_LOOKBACK, idx)
    df["bear_fvg_t"] = _ffill_cap(rt, FVG_LOOKBACK, idx)
    df["bear_fvg_b"] = _ffill_cap(rb, FVG_LOOKBACK, idx)

    row   = df.iloc[-1]
    price = row["Close"]
    atr   = row["atr14"]
    ha    = atr * ATR_MULT

    if not row["in_window"]:
        log.info(f"Outside London window (3–5 AM ET) — no entry")
        return None

    # LONG
    if (row["judas_bull"] and row["htf_bias"] != "bear" and
            not np.isnan(row["bull_fvg_t"]) and
            row["Low"] <= row["bull_fvg_t"] + ha and
            row["Close"] >= row["bull_fvg_b"] - ha and
            price > row["ema50"]):
        sl   = row["bull_fvg_b"] - ha
        risk = price - sl
        if risk < MIN_SL_ATR_MULT * atr or risk > price * 0.03:
            log.info(f"LONG rejected: risk={risk:.2f} atr_floor={MIN_SL_ATR_MULT*atr:.2f}")
            return None
        contracts = min(int(MAX_TRADE_RISK / (risk * ES_MULT)), MAX_CONTRACTS)
        if contracts < 1:
            return None
        log.info(f"LONG signal price={price:.2f} sl={sl:.2f} tp={price+risk*RR:.2f} sz={contracts}ct")
        return dict(side="BUY", entry=round(price,2), sl=round(sl,2),
                    tp=round(price+risk*RR,2), risk=round(risk,4),
                    contracts=contracts, atr=round(atr,4))

    # SHORT
    if (row["judas_bear"] and row["htf_bias"] != "bull" and
            not np.isnan(row["bear_fvg_b"]) and
            row["High"] >= row["bear_fvg_b"] - ha and
            row["Close"] <= row["bear_fvg_t"] + ha and
            price < row["ema50"]):
        sl   = row["bear_fvg_t"] + ha
        risk = sl - price
        if risk < MIN_SL_ATR_MULT * atr or risk > price * 0.03:
            log.info(f"SHORT rejected: risk={risk:.2f} atr_floor={MIN_SL_ATR_MULT*atr:.2f}")
            return None
        tp = price - risk * RR
        if tp <= 0:
            return None
        contracts = min(int(MAX_TRADE_RISK / (risk * ES_MULT)), MAX_CONTRACTS)
        if contracts < 1:
            return None
        log.info(f"SHORT signal price={price:.2f} sl={sl:.2f} tp={tp:.2f} sz={contracts}ct")
        return dict(side="SELL", entry=round(price,2), sl=round(sl,2),
                    tp=round(tp,2), risk=round(risk,4),
                    contracts=contracts, atr=round(atr,4))

    log.info("No ICT London setup on current bar.")
    return None


# ═══════════════════════════════════════════════════════════
# BROKER  (ICT-Challenge account)
# ═══════════════════════════════════════════════════════════

def get_alpaca_client():
    from alpaca.trading.client import TradingClient
    key, secret = load_credentials(ENV_FILE)
    return TradingClient(key, secret, paper=True)


def get_position(client, symbol: str):
    try:
        return client.get_open_position(symbol)
    except Exception:
        return None


def close_position(client, symbol: str) -> None:
    try:
        client.close_position(symbol)
        log.info(f"Closed position: {symbol}")
    except Exception as e:
        log.error(f"Close failed: {e}")


def place_bracket(client, symbol: str, side: str, qty: int,
                  entry: float, sl: float, tp: float) -> str:
    from alpaca.trading.requests import MarketOrderRequest
    from alpaca.trading.enums   import OrderSide, TimeInForce, OrderClass

    side_enum = OrderSide.BUY if side == "BUY" else OrderSide.SELL
    req = MarketOrderRequest(
        symbol          = symbol,
        qty             = qty,
        side            = side_enum,
        time_in_force   = TimeInForce.GTC,
        order_class     = OrderClass.BRACKET,
        stop_loss       = {"stop_price":  round(sl, 2)},
        take_profit     = {"limit_price": round(tp, 2)},
        client_order_id = f"ICT_LON_{datetime.now(ET).strftime('%Y%m%d%H%M%S')}",
    )
    order = client.submit_order(req)
    log.info(f"Bracket order placed: {order.id}")
    return order.id


# ═══════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════

def main() -> None:
    now_et = datetime.now(ET)

    if not lock.acquire():
        sys.exit(0)

    try:
        log.info("═" * 55)
        log.info(f"ICT London Sniper tick — {now_et.strftime('%Y-%m-%d %H:%M %Z')}")

        losses = get_loss_count()
        if losses >= DAILY_LOSS_LIMIT:
            log.info(f"Daily loss limit ({losses}/{DAILY_LOSS_LIMIT}) — standing down.")
            return

        client = get_alpaca_client()
        acct   = client.get_account()
        equity = float(acct.equity)
        log.info(f"ICT-Challenge equity: ${equity:,.2f}  losses today: {losses}/{DAILY_LOSS_LIMIT}")

        state    = load_state()
        live_pos = get_position(client, SYMBOL)

        if state and not live_pos:
            log.info(f"Bracket exit — {state['side']} @ {state['entry']}")
            send_telegram(
                f"ICT London Bracket Exit\n"
                f"{state['side']} {state['contracts']}x {SYMBOL}\n"
                f"SL: {state['sl']}  TP: {state['tp']}", log
            )
            clear_state()
            state = None

        if live_pos:
            entry_time = datetime.fromisoformat(state["entry_time"]) if state else None
            if entry_time:
                bars_open = int((now_et.replace(tzinfo=None) -
                                 entry_time.replace(tzinfo=None)).total_seconds() / 300)
                log.info(f"Open position: {live_pos.side.value} bars_open={bars_open}/{STAGNATION_BARS}")
                if bars_open >= STAGNATION_BARS:
                    log.warning(f"STAGNATION EXIT after {bars_open} bars")
                    close_position(client, SYMBOL)
                    send_telegram(f"⏰ ICT London STAGNATION EXIT\n{SYMBOL} {bars_open} bars", log)
                    record_loss()
                    clear_state()
            return

        # Window gate
        window_open  = now_et.replace(hour=WINDOW_OPEN[0],  minute=WINDOW_OPEN[1],  second=0)
        window_close = now_et.replace(hour=WINDOW_CLOSE[0], minute=WINDOW_CLOSE[1], second=0)
        if not (window_open <= now_et < window_close):
            log.info("Outside London entry window — standing by.")
            return

        df = fetch_bars(80)
        if len(df) < 25:
            log.warning(f"Insufficient bars ({len(df)})")
            return

        sig = compute_signal(df)
        if not sig:
            return

        dollar_risk = sig["contracts"] * sig["risk"] * ES_MULT
        log.info(f"ENTRY {sig['side']} {sig['contracts']}x {SYMBOL}  "
                 f"entry≈{sig['entry']}  sl={sig['sl']}  tp={sig['tp']}  "
                 f"risk_pts={sig['risk']}  dollar_risk=${dollar_risk:.0f}")

        try:
            order_id = place_bracket(client, SYMBOL, sig["side"],
                                     sig["contracts"], sig["entry"],
                                     sig["sl"], sig["tp"])
            save_state({**sig, "order_id": order_id,
                        "entry_time": now_et.isoformat()})
            send_telegram(
                f"{'🟢' if sig['side']=='BUY' else '🔴'} ICT London {sig['side']}\n"
                f"{sig['contracts']}x {SYMBOL}\n"
                f"Entry: {sig['entry']} | SL: {sig['sl']} | TP: {sig['tp']}\n"
                f"Risk: {sig['risk']} pts = ${dollar_risk:.0f}  RR: 1:{RR}", log
            )
        except Exception as e:
            log.error(f"Order failed: {e}")
            send_telegram(f"❌ ICT London ORDER FAILED\n{e}", log)

    finally:
        lock.release()


if __name__ == "__main__":
    main()
