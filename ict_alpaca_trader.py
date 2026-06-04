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
SESSION_STATE_PATH = Path.home() / "freqtrade/user_data/logs/ict_session.json"


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


def save_state(signal: dict, shares: int, data_source: str) -> None:
    """
    Persist position-management metadata (time-exit / stagnation / feed-freeze).

    NOTE: this file is NOT used for risk management any more — daily losses and
    trailing DD are now broker-derived (see count_daily_losses / check_trailing_dd).
    It is still local-file based for position management, which remains ephemeral
    on GitHub Actions (flagged for migration in the next cycle).
    """
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
    """Returns today's PnL % vs last_equity."""
    acct = client.get_account()
    eq   = float(acct.equity)
    prev = float(acct.last_equity)
    return (eq - prev) / prev if prev > 0 else 0.0


def _et_session_start_utc() -> datetime:
    """
    Start of the current ET trading day (00:00 ET) as a tz-aware UTC datetime.
    Stateless anchor for 'today' — derived from the clock, not a local file, so it
    is identical on every ephemeral runner. All ICT kill zones (London 3-5am ET,
    NY 8:30-11am ET, etc.) fall after 00:00 ET, so a round-trip never straddles
    this boundary.
    """
    et_now      = datetime.now(ICTModel.ET)
    et_midnight = ICTModel.ET.localize(datetime.combine(et_now.date(), time(0, 0)))
    return et_midnight.astimezone(timezone.utc)


def _ict_realized_trades_today(client: TradingClient) -> list[float]:
    """
    Reconstruct today's CLOSED ICT round-trips straight from Alpaca order history.

    SPY is ICT-exclusive on this account (Kronos trades NVDA/USO; MT5 is a separate
    platform), so EVERY filled SPY order today belongs to ICT — the same isolation
    status.py already relies on. We walk fills in fill-time order, accumulate signed
    cash flow, and snapshot realized P&L each time the net position returns flat.

    This is exit-path agnostic: it counts bracket SL/TP fills AND manual time-exit
    `close_position` fills, and it needs ZERO local state — so it produces the same
    answer on a fresh GitHub runner as on the Mac. Bracket child legs share their
    parent's created_at, so we MUST order by filled_at (actual fill time), never
    created_at, or an exit could sort before its entry.
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
        cash  -= signed * px          # buy spends cash (-), sell receives cash (+)
        pos_qty += signed
        if abs(pos_qty) < 1e-9:       # net flat → one round-trip complete
            trades.append(cash)
            cash = 0.0
    return trades


def count_daily_losses(client: TradingClient) -> int:
    """
    Number of LOSING ICT round-trips closed today, derived 100% from the broker.

    Replaces the old session-file dollar estimate, which silently returned 0 on
    ephemeral GitHub runners (the local baseline file never persisted) — quietly
    disabling the 2-loss daily halt. No try/except here: if the broker query fails
    the exception propagates so the caller HALTS rather than assuming zero losses.
    """
    return sum(1 for pnl in _ict_realized_trades_today(client) if pnl < 0)


def check_total_dd(client: TradingClient) -> float:
    """
    Drawdown % from the all-time equity peak (live equity vs 1-year history peak).

    Fail-CLOSED: a broker error propagates so the caller HALTS rather than assuming
    zero drawdown. (Previously this swallowed the exception and returned 0.0, which
    silently disabled the -8% total-DD circuit breaker whenever the history endpoint
    hiccuped.) An empty history is the one benign case — a genuinely fresh account
    has no peak above itself, so 0.0 is correct there.
    """
    hist    = client.get_portfolio_history(GetPortfolioHistoryRequest(period="1A"))
    eq_list = [e for e in hist.equity if e is not None and e > 0]
    current = get_account_equity(client)   # live, includes unrealized P&L
    if not eq_list:
        return 0.0
    peak = max(eq_list + [current])        # peak is never below current
    return (current - peak) / peak if peak > 0 else 0.0


def check_trailing_dd(client: TradingClient) -> float:
    """
    Intraday trailing drawdown from the equity peak, read from the BROKER's own
    portfolio history — no local peak_equity file (that silently read 0 on
    ephemeral runners, disabling prop-firm trailing-DD protection entirely).

    The5ers/FTMO track trailing DD on the whole ACCOUNT, so account-level equity
    is the correct basis here (all-time peak-to-trough is separately covered by
    check_total_dd at -8%). Returns a NEGATIVE fraction (-0.04 = 4% below peak).
    No try/except: a failed query propagates so the caller decides explicitly.
    """
    hist = client.get_portfolio_history(GetPortfolioHistoryRequest(
        period="1D", timeframe="5Min", extended_hours=True))
    eqs     = [float(e) for e in hist.equity if e is not None and float(e) > 0]
    current = get_account_equity(client)
    peak    = max(eqs + [current]) if eqs else current   # peak is never below current
    return (current - peak) / peak if peak > 0 else 0.0


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
    True if the broker rejected an order because its client_order_id already exists
    (an overlapping cron run already placed this bar's trade). Alpaca enforces
    client_order_id uniqueness SERVER-SIDE and returns HTTP 422 — which is exactly
    why the deterministic id is a real idempotency guard: a duplicate can never
    become a second position. This only decides log-a-benign-skip vs. alert-failure.
    """
    if isinstance(err, APIError) and err.status_code == 422:
        return True
    msg = str(err).lower()
    return "client_order_id" in msg or ("unique" in msg and "order" in msg)


def place_order(client: TradingClient, signal: dict, shares: int, bar_ts: str) -> None:
    """
    Bracket order — SL and TP are sent directly to Alpaca.
    Alpaca auto-closes the position when either level is hit,
    even if our script is not running (cloud-safe).

    Idempotency: client_order_id is derived from the signal BAR timestamp, so two
    overlapping cron runs analysing the same 5-minute bar build the SAME id and
    Alpaca rejects the second — no double entry.
    """
    side = OrderSide.BUY if signal["signal"] == "BUY" else OrderSide.SELL
    sl   = round(signal["sl"], 2)
    tp   = round(signal["tp"], 2)

    order_id = f"{ICT_ORDER_PREFIX}_{SYMBOL}_{bar_ts}"
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

    # ── Risk checks — broker-derived; fail CLOSED if any cannot be computed ──
    # No silent zero-fills: if a metric can't be read, we do NOT open a trade.
    try:
        daily_dd     = check_daily_dd(client)
        total_dd     = check_total_dd(client)
        equity       = get_account_equity(client)
        daily_losses = count_daily_losses(client)
    except Exception as e:
        log(f"[HALT] Risk metrics unavailable — standing down, no trade: {e}")
        send_telegram(f"⛔ ICT HALT: risk checks failed — {e}")
        return

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
        # Trailing drawdown — broker-derived intraday peak (no local peak_equity).
        # Fail SAFE, not closed: if the metric can't be read we HOLD (the exchange
        # bracket still protects the position) and alert loudly — we never
        # force-close or fall silent on a transient API blip.
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

    # ── Data feed + signal ─────────────────────────────────────────────────
    df_5m, df_1h, df_corr, data_source = fetch_data()

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

    signal = model.get_signal(df_5m, df_1h, df_correlated=df_corr)

    # Persist trend day bypass if signal reason indicates it
    if signal.get("kill_zone"):
        active_kz = signal["kill_zone"]
        if "[TREND_DAY]" in signal.get("reason", "") or get_trend_day_bypass(active_kz):
            set_trend_day_bypass(active_kz)

    # Translate ES1! price levels → SPY prices when TradingView data was used
    if data_source == "tradingview" and signal["signal"] != "HOLD":
        signal = translate_signal_to_spy(signal, client)

    log(f"Signal: {signal['signal']} | {signal['reason']}")

    if signal["signal"] == "HOLD":
        log("No ICT setup — standing by")
        return

    # ── Entry drift gate — skip if price moved too far from FVG entry ────────
    current_px = float(df_5m["close"].iloc[-1])
    if data_source == "tradingview":   # drift only meaningful when translate_signal_to_spy used live price
        entry_drift = abs(current_px - signal["entry"]) / signal["entry"]
        if entry_drift > MAX_ENTRY_DRIFT:
            log(f"[SKIP] Entry drift {entry_drift:.3%} > {MAX_ENTRY_DRIFT:.3%} — price moved from FVG zone")
            return

    # ── Position sizing ─────────────────────────────────────────────────────
    shares = calc_position_size(equity, signal["entry"], signal["sl"], risk_pct)
    if shares == 0:
        log("[WARN] Calculated 0 shares — skip")
        return

    cost = shares * signal["entry"]
    log(f"Position size: {shares} shares @ ${signal['entry']:.2f} = ${cost:,.2f}")

    # ── Place bracket order ─────────────────────────────────────────────────
    # Deterministic id per 5-minute bar → two overlapping cron runs on the same
    # bar build the SAME client_order_id, and Alpaca rejects the duplicate.
    bar_ts = df_5m.index[-1].strftime("%Y%m%d%H%M")
    save_state(signal, shares, data_source)   # save BEFORE submit — prevent orphan on crash
    try:
        place_order(client, signal, shares, bar_ts)
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
