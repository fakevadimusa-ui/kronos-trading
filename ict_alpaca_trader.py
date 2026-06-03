"""
ICT Alpaca Trader — ES1! Signal / SPY Execution
=================================================
Runs the ICT model on 15-minute ES1! futures data (from TradingView Desktop)
during kill zones and executes market orders on SPY via Alpaca (paper → live → prop firm).

Data source priority:
  1. TradingView Desktop (CDP at localhost:9222) — ES1! futures (authentic ICT instrument)
  2. Fallback: yfinance SPY — used when TradingView is not running

SL/TP translation: ES1! risk % → SPY bracket prices via Alpaca last trade price.

Schedule (cron, PT weekdays):
  */15 0,1    * * 1-5   # London kill zone  (3–5 AM ET = 12–2 AM PT)
  30,45 4     * * 1-5   # NY open first bar  (7:30–7:45 AM ET = 4:30–4:45 AM PT)
  */15 5,6    * * 1-5   # NY AM session      (8–10 AM ET = 5–7 AM PT)

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

import json
import os
import subprocess
import sys
import warnings
from datetime import datetime, timezone, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import yfinance as yf

warnings.filterwarnings("ignore")

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import (
    MarketOrderRequest,
    TakeProfitRequest,
    StopLossRequest,
    GetPortfolioHistoryRequest,
)
from alpaca.trading.enums import OrderSide, TimeInForce, OrderClass

from ict_model import ICTModel


# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════

CONFIG_PATH   = Path.home() / "freqtrade/user_data/config_kronos_nvda.json"
TV_FETCH      = Path(__file__).parent / "tv_fetch.js"

SIGNAL_SYMBOL = "CME_MINI_DL:ES1!"   # ICT analysis instrument (TradingView)
SYMBOL        = "SPY"                  # Alpaca execution instrument
SYMBOL_ALT    = "QQQ"                  # alternate execution (tech/NQ equivalent)
YF_PERIOD_15M = "5d"
YF_PERIOD_1H  = "60d"

RISK_PER_TRADE       = 0.01    # 1% of account
RISK_PER_TRADE_SMALL = 0.005   # reduced when DD > 4%
DAILY_DD_LIMIT       = -0.04
TOTAL_DD_LIMIT       = -0.08

ICT_TAG = "ICT"   # order tag to identify our positions vs Kronos positions


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def log(msg: str) -> None:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


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


def _tv_fetch_bars(tf: str, count: int) -> pd.DataFrame | None:
    """
    Call tv_fetch.js to get OHLCV bars for SIGNAL_SYMBOL at the given timeframe.
    Returns a DataFrame with lowercase columns and UTC DatetimeIndex, or None on failure.
    """
    if not TV_FETCH.exists():
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


def fetch_data() -> tuple[pd.DataFrame, pd.DataFrame, str]:
    """
    Returns (df_15m, df_1h, data_source).
    Tries TradingView ES1! first; falls back to yfinance SPY.
    """
    log(f"Fetching data — primary: TradingView {SIGNAL_SYMBOL}, fallback: yfinance {SYMBOL}")

    df_15m = _tv_fetch_bars("15", 200)
    df_1h  = _tv_fetch_bars("60", 100)

    if df_15m is not None and df_1h is not None and len(df_15m) >= 25:
        log(f"[DATA] TradingView {SIGNAL_SYMBOL} | 15m: {len(df_15m)} bars | 1H: {len(df_1h)} bars | last: {df_15m['close'].iloc[-1]:.2f}")
        return df_15m, df_1h, "tradingview"

    log(f"[DATA] TradingView unavailable — falling back to yfinance {SYMBOL}")
    df_15m = yf.download(SYMBOL, period=YF_PERIOD_15M, interval="15m",
                         auto_adjust=True, progress=False)
    df_1h  = yf.download(SYMBOL, period=YF_PERIOD_1H,  interval="1h",
                         auto_adjust=True, progress=False)
    for df in (df_15m, df_1h):
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.columns = [str(c).lower() for c in df.columns]
        df.dropna(inplace=True)
    log(f"[DATA] yfinance {SYMBOL} | 15m: {len(df_15m)} bars | 1H: {len(df_1h)} bars | last: {df_15m['close'].iloc[-1]:.2f}")
    return df_15m, df_1h, "yfinance"


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


def save_state(signal: dict, shares: int) -> None:
    """Persist SL/TP so subsequent runs can manage the open position."""
    state = {
        "symbol":     SYMBOL,
        "signal":     signal["signal"],
        "entry":      signal["entry"],
        "sl":         signal["sl"],
        "tp":         signal["tp"],
        "shares":     shares,
        "setup":      signal["setup"],
        "kill_zone":  signal["kill_zone"],
        "opened_at":  datetime.now().isoformat(),
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


# ══════════════════════════════════════════════════════════════════════════════
# RISK MANAGEMENT
# ══════════════════════════════════════════════════════════════════════════════

def get_account_equity(client: TradingClient) -> float:
    acct = client.get_account()
    return float(acct.equity)


def check_daily_dd(client: TradingClient) -> float:
    """Returns today's PnL % vs last_equity."""
    acct = client.get_account()
    eq   = float(acct.equity)
    prev = float(acct.last_equity)
    return (eq - prev) / prev if prev > 0 else 0.0


def check_total_dd(client: TradingClient) -> float:
    """
    Returns drawdown % from all-time equity peak.
    Uses live equity (includes intraday unrealized P&L) vs history snapshots for peak.
    """
    try:
        hist    = client.get_portfolio_history(GetPortfolioHistoryRequest(period="1A"))
        eq_list = [e for e in hist.equity if e is not None and e > 0]
        if not eq_list:
            return 0.0
        peak    = max(eq_list)
        current = get_account_equity(client)   # live, includes unrealized P&L
        return (current - peak) / peak if peak > 0 else 0.0
    except Exception as e:
        log(f"[WARN] Could not fetch portfolio history: {e}")
        return 0.0


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


def close_ict_position(client: TradingClient) -> None:
    try:
        client.close_position(SYMBOL)
        log(f"[CLOSE] Closed {SYMBOL} position")
    except Exception as e:
        log(f"[WARN] Could not close position: {e}")


def manage_open_position(client: TradingClient, pos: dict) -> None:
    """
    Position is managed by Alpaca bracket orders — SL/TP auto-execute on exchange.
    We just log current status and skip entering a new trade.
    """
    try:
        live_pos       = client.get_open_position(SYMBOL)
        current        = float(live_pos.current_price)
        unrealized_pct = float(live_pos.unrealized_plpc)
        log(f"[HOLD] {SYMBOL} {pos['side']} @ {current:.2f} | Unrealized: {unrealized_pct:.2%} | Bracket orders active on Alpaca")
    except Exception as e:
        log(f"[WARN] Could not fetch live position: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# ORDER EXECUTION
# ══════════════════════════════════════════════════════════════════════════════

def place_order(client: TradingClient, signal: dict, shares: int) -> None:
    """
    Bracket order — SL and TP are sent directly to Alpaca.
    Alpaca auto-closes the position when either level is hit,
    even if our script is not running (cloud-safe).
    """
    side = OrderSide.BUY if signal["signal"] == "BUY" else OrderSide.SELL
    sl   = round(signal["sl"], 2)
    tp   = round(signal["tp"], 2)

    req = MarketOrderRequest(
        symbol        = SYMBOL,
        qty           = shares,
        side          = side,
        time_in_force = TimeInForce.DAY,
        order_class   = OrderClass.BRACKET,
        stop_loss     = StopLossRequest(stop_price=sl),
        take_profit   = TakeProfitRequest(limit_price=tp),
    )
    client.submit_order(req)
    log(f"[ORDER] {signal['signal']} {shares}x {SYMBOL} | {signal['reason']}")
    log(f"        SL={sl} TP={tp} RR=1:{signal['rr']} (bracket order — Alpaca manages SL/TP)")

    msg = (
        f"<b>ICT {signal['signal']}</b> {SYMBOL}\n"
        f"Setup: {signal['setup']} | Zone: {signal['kill_zone']} | HTF: {signal['htf_bias']}\n"
        f"Entry: <b>{signal['entry']:.2f}</b> | SL: {sl} | TP: {tp}\n"
        f"Shares: {shares} | {signal['reason']}"
    )
    send_telegram(msg)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    log("═" * 60)
    log("ICT Alpaca Trader — starting")

    # Load Alpaca client
    key, secret = load_credentials()
    client      = TradingClient(key, secret, paper=True)
    log("Alpaca paper client connected")

    # ── Risk checks ────────────────────────────────────────────────────────
    daily_dd = check_daily_dd(client)
    total_dd = check_total_dd(client)
    equity   = get_account_equity(client)

    log(f"Equity: ${equity:,.2f} | Daily DD: {daily_dd:.2%} | Total DD: {total_dd:.2%}")

    if daily_dd <= DAILY_DD_LIMIT:
        log(f"[HALT] Daily DD {daily_dd:.2%} breaches limit {DAILY_DD_LIMIT:.0%}. No trade.")
        send_telegram(f"⛔ ICT HALT: Daily DD {daily_dd:.2%}")
        return

    if total_dd <= TOTAL_DD_LIMIT:
        log(f"[HALT] Total DD {total_dd:.2%} breaches limit {TOTAL_DD_LIMIT:.0%}. No trade.")
        send_telegram(f"⛔ ICT HALT: Total DD {total_dd:.2%}")
        return

    # Reduce risk if halfway to daily limit
    risk_pct = RISK_PER_TRADE_SMALL if daily_dd <= DAILY_DD_LIMIT / 2 else RISK_PER_TRADE

    model = ICTModel(swing_lookback=5, fvg_min_pct=0.0003, ob_lookback=30, rr_ratio=3.0)

    # ── Existing position check ────────────────────────────────────────────
    pos = get_ict_position(client)

    if pos:
        log(f"[POS] Existing ICT position: {pos['side']} {pos['qty']} {SYMBOL} — bracket orders managing SL/TP")
        manage_open_position(client, pos)
        return

    # ── ICT signal generation ───────────────────────────────────────────────
    df_15m, df_1h, data_source = fetch_data()
    signal = model.get_signal(df_15m, df_1h)

    # Translate ES1! price levels → SPY prices when TradingView data was used
    if data_source == "tradingview" and signal["signal"] != "HOLD":
        signal = translate_signal_to_spy(signal, client)

    log(f"Signal: {signal['signal']} | {signal['reason']}")

    if signal["signal"] == "HOLD":
        log("No ICT setup — standing by")
        return

    # ── Position sizing ─────────────────────────────────────────────────────
    shares = calc_position_size(equity, signal["entry"], signal["sl"], risk_pct)
    if shares == 0:
        log("[WARN] Calculated 0 shares — skip")
        return

    cost = shares * signal["entry"]
    log(f"Position size: {shares} shares @ ${signal['entry']:.2f} = ${cost:,.2f}")

    # ── Place bracket order — Alpaca manages SL/TP even when script is offline
    place_order(client, signal, shares)

    log("═" * 60)


if __name__ == "__main__":
    main()
