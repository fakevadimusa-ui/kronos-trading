"""
Kronos NVDA — Direct Alpaca Paper Trading Script
=================================================
Signal logic:
  - 1D SMA20 filter: price must be above SMA20 for longs, below for shorts
  - Kronos threshold: predicted next-day move must be >= 0.5%
  - 1% SL, 3% TP (1:3 RR), 1% account risk per trade

Risk limits:
  - Max 1 open position at a time
  - 4% daily loss halt
  - 8% total drawdown halt

Credentials (env vars or JSON fallback):
  ALPACA_KEY, ALPACA_SECRET, TELEGRAM_TOKEN, TELEGRAM_CHAT_ID
"""

import json
import os
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
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
from alpaca.trading.requests import MarketOrderRequest, GetOrdersRequest, ClosePositionRequest
from alpaca.trading.enums import OrderSide, TimeInForce, OrderClass, QueryOrderStatus


# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════

CONFIG_PATH = Path.home() / "freqtrade/user_data/config_kronos_nvda.json"

SYMBOL          = "NVDA"
YF_TICKER       = "NVDA"
LOOKBACK        = 400
PRED_LEN        = 1
THRESHOLD       = 0.005
SL_PCT          = 0.01
TP_PCT          = 0.03
RISK_PER_TRADE  = 0.01
SMA_PERIOD      = 20

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


def fetch_nvda_data() -> pd.DataFrame:
    log(f"Fetching {YF_TICKER} daily data...")
    raw = yf.download(YF_TICKER, period="5y", interval="1d",
                      auto_adjust=True, progress=False)
    if raw.empty:
        raise ValueError("yfinance returned no data")
    df = raw[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.columns = ["open", "high", "low", "close", "volume"]
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df = df.dropna()
    log(f"Got {len(df)} daily bars, last close: {df['close'].iloc[-1]:.2f}")
    return df


def compute_sma20(df: pd.DataFrame) -> float:
    return float(df["close"].rolling(SMA_PERIOD).mean().iloc[-1])


def kronos_predicted_return(predictor: KronosPredictor, df: pd.DataFrame) -> float:
    context  = df.tail(LOOKBACK)
    x_ts     = pd.Series(context.index)
    last_ts  = context.index[-1]
    y_ts     = pd.Series(pd.date_range(start=last_ts, periods=2, freq="B")[1:])

    pred_df       = predictor.predict(context, x_ts, y_ts, pred_len=PRED_LEN, verbose=False)
    pred_close    = float(pred_df["close"].iloc[0])
    current_close = float(context["close"].iloc[-1])
    ret = (pred_close - current_close) / current_close
    log(f"Kronos: current={current_close:.2f}, predicted={pred_close:.2f}, return={ret:+.3%}")
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
    try:
        pos = client.get_open_position(SYMBOL)
        qty = float(pos.qty)
        log(f"Existing {SYMBOL} position: {qty} shares — skipping entry")
        return True
    except Exception:
        return False


def place_bracket_order(
    client: TradingClient,
    side: OrderSide,
    shares: int,
    entry_price: float,
) -> tuple[float, float]:
    if side == OrderSide.BUY:
        tp_price = round(entry_price * (1 + TP_PCT), 2)
        sl_price = round(entry_price * (1 - SL_PCT), 2)
    else:
        tp_price = round(entry_price * (1 - TP_PCT), 2)
        sl_price = round(entry_price * (1 + SL_PCT), 2)

    log(f"Placing {side.value} {shares}x {SYMBOL} | SL={sl_price} TP={tp_price}")

    request = MarketOrderRequest(
        symbol=SYMBOL,
        qty=shares,
        side=side,
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
    log("=== Kronos NVDA Paper Trading — run start ===")

    api_key, api_secret = load_credentials()
    client  = TradingClient(api_key, api_secret, paper=True)
    account = client.get_account()
    equity  = float(account.equity)
    log(f"Alpaca paper account equity: ${equity:,.2f}")

    if not check_total_dd(account):
        msg = f"🚨 <b>NVDA | HALTED</b>\nTotal DD limit hit\nAccount: ${equity:,.0f}"
        log(msg)
        send_telegram(msg)
        return

    if not check_daily_loss(client, equity):
        msg = f"🚨 <b>NVDA | HALTED</b>\nDaily loss limit hit\nAccount: ${equity:,.0f}"
        log(msg)
        send_telegram(msg)
        return

    if has_open_position(client):
        send_telegram(f"📌 <b>NVDA</b> | Position already open — skipped\nAccount: ${equity:,.0f}")
        return

    df     = fetch_nvda_data()
    sma20  = compute_sma20(df)
    close  = float(df["close"].iloc[-1])
    log(f"SMA20={sma20:.2f}  close={close:.2f}")

    long_ok  = close > sma20
    short_ok = close < sma20

    if not long_ok and not short_ok:
        log("Price exactly at SMA20 — no signal")
        send_telegram(f"📊 <b>NVDA</b> | No signal (price at SMA20)\nAccount: ${equity:,.0f}")
        return

    predictor     = load_kronos()
    predicted_ret = kronos_predicted_return(predictor, df)

    if long_ok and predicted_ret >= THRESHOLD:
        side = OrderSide.BUY
    elif short_ok and predicted_ret <= -THRESHOLD:
        side = OrderSide.SELL
    else:
        direction = "long" if long_ok else "short"
        needed    = f"≥+{THRESHOLD:.1%}" if long_ok else f"≤-{THRESHOLD:.1%}"
        log(f"No trade: SMA20 allows {direction}, Kronos predicts {predicted_ret:+.3%} (need {needed})")
        send_telegram(
            f"📊 <b>NVDA</b> | No trade\n"
            f"Kronos: {predicted_ret:+.3%} (need {needed})\n"
            f"NVDA @ ${close:.2f} | Account: ${equity:,.0f}"
        )
        return

    risk_dollars = equity * RISK_PER_TRADE
    shares       = max(1, int(risk_dollars / (close * SL_PCT)))
    log(f"Position size: {shares} shares (risk ${risk_dollars:.0f})")

    sl_price, tp_price = place_bracket_order(client, side, shares, close)

    send_telegram(
        f"🔔 <b>NVDA | {side.value.upper()} FIRED</b>\n"
        f"{shares}x NVDA @ ${close:.2f}\n"
        f"SL=${sl_price} | TP=${tp_price}\n"
        f"Risk: ${risk_dollars:.0f} | Account: ${equity:,.0f}"
    )

    log("=== Run complete ===")


if __name__ == "__main__":
    main()
