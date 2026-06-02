"""
Kronos Crude Oil — Direct Alpaca Paper Trading Script
======================================================
Signal source : CL=F (WTI Crude Oil futures via yfinance)
Execution     : USO (long) / SCO (2x inverse, short)

Strategy (confirmed OOS 2024-07-01+):
  WR=35.7%, PF=2.054, DD=1.99%, 14 trades

Credentials (env vars or JSON fallback):
  ALPACA_KEY, ALPACA_SECRET, TELEGRAM_TOKEN, TELEGRAM_CHAT_ID
"""

import json
import os
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests
import yfinance as yf

warnings.filterwarnings("ignore")

for _p in (Path("Kronos"), Path.home() / "Kronos"):
    if _p.exists():
        sys.path.insert(0, str(_p))
        break

from model import Kronos, KronosTokenizer, KronosPredictor

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, GetOrdersRequest
from alpaca.trading.enums import OrderSide, TimeInForce, OrderClass, QueryOrderStatus


# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════

CONFIG_PATH = Path.home() / "freqtrade/user_data/config_kronos_nvda.json"

SIGNAL_TICKER   = "CL=F"
EXEC_SYMBOL     = "USO"
EXEC_SYMBOL_INV = "SCO"
LOOKBACK        = 400
PRED_LEN        = 1
THRESHOLD       = 0.010
SL_PCT          = 0.01
TP_PCT          = 0.03
RISK_PER_TRADE  = 0.01
SMA_PERIOD      = 50

DAILY_DD_LIMIT  = -0.04
TOTAL_DD_LIMIT  = -0.08

KRONOS_MODEL_ID     = "NeoQuasar/Kronos-small"
KRONOS_TOKENIZER_ID = "NeoQuasar/Kronos-Tokenizer-base"
KRONOS_MAX_CONTEXT  = 512


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


def load_kronos() -> KronosPredictor:
    log("Loading Kronos model...")
    tokenizer = KronosTokenizer.from_pretrained(KRONOS_TOKENIZER_ID)
    model     = Kronos.from_pretrained(KRONOS_MODEL_ID)
    predictor = KronosPredictor(model, tokenizer, max_context=KRONOS_MAX_CONTEXT)
    log(f"Kronos loaded on {predictor.device}")
    return predictor


def fetch_signal_data() -> pd.DataFrame:
    log(f"Fetching {SIGNAL_TICKER} daily data for signal...")
    raw = yf.download(SIGNAL_TICKER, period="5y", interval="1d",
                      auto_adjust=True, progress=False)
    if raw.empty:
        raise ValueError(f"yfinance returned no data for {SIGNAL_TICKER}")
    df = raw[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.columns = ["open", "high", "low", "close", "volume"]
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df = df.dropna()
    log(f"Got {len(df)} bars — {SIGNAL_TICKER} last close: {df['close'].iloc[-1]:.2f}")
    return df


def fetch_exec_price(symbol: str) -> float:
    raw = yf.download(symbol, period="5d", interval="1d",
                      auto_adjust=True, progress=False)
    price = float(raw["Close"].iloc[-1].item())
    log(f"{symbol} last close (execution price): {price:.2f}")
    return price


def compute_sma(df: pd.DataFrame) -> float:
    return float(df["close"].rolling(SMA_PERIOD).mean().iloc[-1])


def kronos_predicted_return(predictor: KronosPredictor, df: pd.DataFrame) -> float:
    context = df.tail(LOOKBACK)
    x_ts    = pd.Series(context.index)
    last_ts = context.index[-1]
    y_ts    = pd.Series(pd.date_range(start=last_ts, periods=2, freq="B")[1:])

    pred_df       = predictor.predict(context, x_ts, y_ts, pred_len=PRED_LEN, verbose=False)
    pred_close    = float(pred_df["close"].iloc[0])
    current_close = float(context["close"].iloc[-1])
    ret = (pred_close - current_close) / current_close
    log(f"Kronos: {SIGNAL_TICKER} current={current_close:.2f}, predicted={pred_close:.2f}, return={ret:+.3%}")
    return ret


def check_daily_loss(client: TradingClient, equity: float) -> bool:
    today = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    try:
        orders = client.get_orders(GetOrdersRequest(
            status=QueryOrderStatus.CLOSED,
            after=today,
        ))
        daily_pnl = sum(
            float(o.filled_avg_price or 0) * float(o.filled_qty or 0) *
            (1 if o.side == OrderSide.SELL else -1)
            for o in orders
            if o.filled_avg_price and o.filled_qty
        )
        pct = daily_pnl / equity if equity else 0
        log(f"Daily PnL estimate: {pct:+.2%}")
        return pct > DAILY_DD_LIMIT
    except Exception as e:
        log(f"Daily PnL check failed ({e}) — allowing trade")
        return True


def check_total_dd(account) -> bool:
    equity      = float(account.equity)
    last_equity = float(account.last_equity)
    if last_equity <= 0:
        return True
    dd = (equity - last_equity) / last_equity
    log(f"Total drawdown vs last_equity: {dd:+.2%}")
    return dd > TOTAL_DD_LIMIT


def has_open_position(client: TradingClient) -> bool:
    for sym in (EXEC_SYMBOL, EXEC_SYMBOL_INV):
        try:
            pos = client.get_open_position(sym)
            log(f"Existing {sym} position: {float(pos.qty)} shares — skipping entry")
            return True
        except Exception:
            pass
    return False


def place_bracket_order(
    client: TradingClient,
    symbol: str,
    shares: int,
    entry_price: float,
) -> tuple[float, float]:
    tp_price = round(entry_price * (1 + TP_PCT), 2)
    sl_price = round(entry_price * (1 - SL_PCT), 2)

    log(f"Placing BUY {shares}x {symbol} | SL={sl_price} TP={tp_price}")

    request = MarketOrderRequest(
        symbol=symbol,
        qty=shares,
        side=OrderSide.BUY,
        time_in_force=TimeInForce.DAY,
        order_class=OrderClass.BRACKET,
        take_profit={"limit_price": str(tp_price)},
        stop_loss={"stop_price": str(sl_price)},
    )
    order = client.submit_order(request)
    log(f"Order submitted: id={order.id} status={order.status}")
    return sl_price, tp_price


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    log(f"=== Kronos Crude Oil Paper Trading (signal:{SIGNAL_TICKER} exec:{EXEC_SYMBOL}) ===")

    api_key, api_secret = load_credentials()
    client  = TradingClient(api_key, api_secret, paper=True)
    account = client.get_account()
    equity  = float(account.equity)
    log(f"Alpaca paper equity: ${equity:,.2f}")

    if not check_total_dd(account):
        msg = f"🚨 <b>CL=F | HALTED</b>\nTotal DD limit hit\nAccount: ${equity:,.0f}"
        log(msg)
        send_telegram(msg)
        return

    if not check_daily_loss(client, equity):
        msg = f"🚨 <b>CL=F | HALTED</b>\nDaily loss limit hit\nAccount: ${equity:,.0f}"
        log(msg)
        send_telegram(msg)
        return

    if has_open_position(client):
        send_telegram(f"📌 <b>CL=F</b> | Position already open — skipped\nAccount: ${equity:,.0f}")
        return

    df    = fetch_signal_data()
    sma   = compute_sma(df)
    close = float(df["close"].iloc[-1])
    log(f"SMA{SMA_PERIOD}={sma:.2f}  {SIGNAL_TICKER} close={close:.2f}")

    long_ok  = close > sma
    short_ok = close < sma

    if not long_ok and not short_ok:
        log("Price exactly at SMA — no signal")
        send_telegram(f"📊 <b>CL=F</b> | No signal (price at SMA50)\nAccount: ${equity:,.0f}")
        return

    predictor     = load_kronos()
    predicted_ret = kronos_predicted_return(predictor, df)

    if long_ok and predicted_ret >= THRESHOLD:
        direction = "LONG"
    elif short_ok and predicted_ret <= -THRESHOLD:
        direction = "SHORT"
    else:
        needed = f"≥+{THRESHOLD:.1%}" if long_ok else f"≤-{THRESHOLD:.1%}"
        log(f"No trade: SMA50 allows {'long' if long_ok else 'short'}, Kronos predicts {predicted_ret:+.3%} (need {needed})")
        send_telegram(
            f"📊 <b>CL=F</b> | No trade\n"
            f"Kronos: {predicted_ret:+.3%} (need {needed})\n"
            f"CL=F @ ${close:.2f} | Account: ${equity:,.0f}"
        )
        return

    risk_dollars = equity * RISK_PER_TRADE

    if direction == "LONG":
        exec_sym   = EXEC_SYMBOL
        exec_price = fetch_exec_price(EXEC_SYMBOL)
        shares     = max(1, int(risk_dollars / (exec_price * SL_PCT)))
        log(f"Long crude → buying {shares}x {exec_sym} (risk ${risk_dollars:.0f})")
    else:
        exec_sym   = EXEC_SYMBOL_INV
        exec_price = fetch_exec_price(EXEC_SYMBOL_INV)
        shares     = max(1, int((risk_dollars / (exec_price * SL_PCT)) / 2))
        log(f"Short crude → buying {shares}x {exec_sym} (2x inverse, risk ${risk_dollars:.0f})")

    sl_price, tp_price = place_bracket_order(client, exec_sym, shares, exec_price)

    send_telegram(
        f"🔔 <b>CL=F | {direction} FIRED</b>\n"
        f"{shares}x {exec_sym} @ ${exec_price:.2f}\n"
        f"SL=${sl_price} | TP=${tp_price}\n"
        f"Kronos: {predicted_ret:+.3%} | Account: ${equity:,.0f}"
    )

    log("=== Run complete ===")


if __name__ == "__main__":
    main()
