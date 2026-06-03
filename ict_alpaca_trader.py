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

warnings.filterwarnings("ignore")

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import (
    MarketOrderRequest,
    TakeProfitRequest,
    StopLossRequest,
    GetPortfolioHistoryRequest,
)
from alpaca.trading.enums import OrderSide, TimeInForce, OrderClass
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
STAGNATION_BARS = 12   # 60 min at 5m — close flat/losing position

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


def fetch_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame | None, str]:
    """
    Returns (df_5m, df_1h, df_correlated, data_source).
    df_correlated = QQQ/NQ1! 5m bars for SMT divergence.
    Priority: 1) TradingView ES1!+NQ1!  2) Alpaca API  3) yfinance
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


def count_daily_losses(client: TradingClient) -> int:
    """
    Estimate daily loss count from realized equity change.
    Robust vs order-type string parsing (Alpaca fills stops as 'market' type).
    Each loss ≈ RISK_PER_TRADE × equity.
    """
    try:
        acct          = client.get_account()
        eq            = float(acct.equity)
        prev          = float(acct.last_equity)
        daily_pnl     = eq - prev
        if daily_pnl >= 0:
            return 0
        loss_per_trade = max(prev * RISK_PER_TRADE, 1.0)
        return min(DAILY_LOSS_LIMIT + 1, int(abs(daily_pnl) / loss_per_trade))
    except Exception as e:
        log(f"[WARN] Could not count daily losses: {e}")
        return 0


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


def check_trailing_dd(client: TradingClient, state: dict | None) -> float:
    """
    Prop firms (The5ers/FTMO) track trailing drawdown from intraday equity peak.
    Returns drawdown % from the highest equity seen since the position opened.
    """
    if not state or "peak_equity" not in state:
        return 0.0
    current    = get_account_equity(client)
    peak       = state["peak_equity"]
    return (current - peak) / peak if peak > 0 else 0.0


def update_peak_equity(client: TradingClient, state: dict) -> None:
    """Update peak_equity in state file if current equity is higher."""
    try:
        current = get_account_equity(client)
        if current > state.get("peak_equity", 0):
            state["peak_equity"] = current
            with open(STATE_PATH, "w") as f:
                json.dump(state, f, indent=2)
    except Exception:
        pass


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

        # ── 60-min stagnation stop ────────────────────────────────────────────
        try:
            opened_at    = datetime.fromisoformat(state["opened_at"])
            elapsed_bars = (datetime.now() - opened_at).total_seconds() / 300
            if elapsed_bars >= STAGNATION_BARS:
                live_pos = client.get_open_position(SYMBOL)
                if float(live_pos.unrealized_pl) <= 0:
                    log(f"[TIME EXIT] {elapsed_bars:.0f} bars open, not in profit — stagnation stop")
                    close_ict_position(client)
                    clear_state()
                    send_telegram(f"⏰ ICT STAGNATION EXIT: {elapsed_bars:.0f} bars flat/losing")
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

def place_order(client: TradingClient, signal: dict, shares: int) -> None:
    """
    Bracket order — SL and TP are sent directly to Alpaca.
    Alpaca auto-closes the position when either level is hit,
    even if our script is not running (cloud-safe).
    """
    side = OrderSide.BUY if signal["signal"] == "BUY" else OrderSide.SELL
    sl   = round(signal["sl"], 2)
    tp   = round(signal["tp"], 2)

    order_id = f"{ICT_ORDER_PREFIX}_{SYMBOL}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
    req = MarketOrderRequest(
        symbol           = SYMBOL,
        qty              = shares,
        side             = side,
        time_in_force    = TimeInForce.DAY,
        order_class      = OrderClass.BRACKET,
        stop_loss        = StopLossRequest(stop_price=sl),
        take_profit      = TakeProfitRequest(limit_price=tp),
        client_order_id  = order_id,
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

    daily_losses = count_daily_losses(client)
    log(f"Equity: ${equity:,.2f} | Daily DD: {daily_dd:.2%} | Total DD: {total_dd:.2%} | Losses today: {daily_losses}/{DAILY_LOSS_LIMIT}")

    if daily_dd <= DAILY_DD_LIMIT:
        log(f"[HALT] Daily DD {daily_dd:.2%} breaches limit {DAILY_DD_LIMIT:.0%}. No trade.")
        send_telegram(f"⛔ ICT HALT: Daily DD {daily_dd:.2%}")
        return

    if total_dd <= TOTAL_DD_LIMIT:
        log(f"[HALT] Total DD {total_dd:.2%} breaches limit {TOTAL_DD_LIMIT:.0%}. No trade.")
        send_telegram(f"⛔ ICT HALT: Total DD {total_dd:.2%}")
        return

    if daily_losses >= DAILY_LOSS_LIMIT:
        log(f"[HALT] {daily_losses} losses today — 2-loss daily rule. Done for the day.")
        send_telegram(f"⛔ ICT HALT: {daily_losses} losses today — terminal closed")
        return

    if daily_dd >= DAILY_PROFIT_TARGET:
        log(f"[HALT] Daily profit target hit: +{daily_dd:.2%} ≥ +{DAILY_PROFIT_TARGET:.0%}. Banking gains.")
        send_telegram(f"🎯 ICT PROFIT TARGET: +{daily_dd:.2%} today — done trading, gains locked")
        return

    # Reduce risk if halfway to daily limit
    risk_pct = RISK_PER_TRADE_SMALL if daily_dd <= DAILY_DD_LIMIT / 2 else RISK_PER_TRADE

    model = ICTModel(tf_minutes=5, fvg_min_pct=0.0002, rr_ratio=3.0, displacement_factor=1.5)

    # ── Existing position check ────────────────────────────────────────────
    pos   = get_ict_position(client)
    state = load_state()

    if pos:
        # Update intraday equity peak for trailing DD tracking
        if state:
            update_peak_equity(client, state)

        # Trailing drawdown check (prop firm uses peak-to-trough, not daily open)
        trailing_dd = check_trailing_dd(client, state)
        if trailing_dd <= -TRAILING_DD_LIMIT:
            log(f"[HALT] Trailing DD {trailing_dd:.2%} from intraday peak — prop firm protection")
            send_telegram(f"⛔ ICT TRAILING DD: {trailing_dd:.2%} from peak — closing position")
            close_ict_position(client)
            clear_state()
            return

        log(f"[POS] Existing ICT position: {pos['side']} {pos['qty']} {SYMBOL} | Trailing DD: {trailing_dd:.2%}")
        manage_open_position(client, pos)
        return

    # ── Data feed + signal ─────────────────────────────────────────────────
    df_5m, df_1h, df_corr, data_source = fetch_data()

    # Data feed freeze: if source changed mid-session, don't open new positions
    if state and state.get("data_source") and state["data_source"] != data_source:
        log(f"[FREEZE] Data source changed mid-session ({state['data_source']} → {data_source}) — no new entries")
        send_telegram(f"⚠️ ICT DATA FEED CHANGE: {state['data_source']} → {data_source} — entries frozen")
        return

    signal = model.get_signal(df_5m, df_1h, df_correlated=df_corr)

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

    # ── Place bracket order ─────────────────────────────────────────────────
    place_order(client, signal, shares)
    save_state(signal, shares, equity, data_source)

    log("═" * 60)


if __name__ == "__main__":
    main()
