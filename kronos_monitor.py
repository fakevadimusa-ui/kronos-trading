"""
Kronos End-of-Day Monitor
=========================
Runs at market close (4 PM ET). Sends a Telegram summary of:
  - Trades opened today
  - Trades closed today (SL/TP hit)
  - Open positions
  - Account P&L
"""

import json
import os
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

import requests

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetOrdersRequest
from alpaca.trading.enums import OrderSide, QueryOrderStatus


CONFIG_PATH = Path.home() / "freqtrade/user_data/config_kronos_nvda.json"

TRACKED = ("NVDA", "USO", "SCO")   # symbols we care about


def send_telegram(msg: str) -> None:
    token   = os.environ.get("TELEGRAM_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        print("No Telegram credentials — skipping")
        return
    try:
        r = requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            json={"chat_id": chat_id, "text": msg, "parse_mode": "HTML"},
            timeout=10,
        )
        print(f"Telegram: {r.status_code}")
    except Exception as e:
        print(f"Telegram error: {e}")


def load_credentials() -> tuple[str, str]:
    key    = os.environ.get("ALPACA_KEY")
    secret = os.environ.get("ALPACA_SECRET")
    if key and secret:
        return key, secret
    with open(CONFIG_PATH) as f:
        cfg = json.load(f)
    ex = cfg["exchange"]
    return ex["key"], ex["secret"]


def main() -> None:
    api_key, api_secret = load_credentials()
    tc = TradingClient(api_key, api_secret, paper=True)

    account     = tc.get_account()
    equity      = float(account.equity)
    last_equity = float(account.last_equity)
    total_pnl   = equity - 100_000   # vs starting $100K

    # Today's window (UTC midnight → now)
    today_start = datetime.now(timezone.utc).replace(
        hour=0, minute=0, second=0, microsecond=0
    )

    all_orders = tc.get_orders(GetOrdersRequest(
        status=QueryOrderStatus.ALL,
        after=today_start,
        limit=50,
    ))

    # Filter to our symbols only
    orders = [o for o in all_orders if o.symbol in TRACKED]

    entries = [o for o in orders if o.side == OrderSide.BUY and o.filled_qty and float(o.filled_qty) > 0]
    exits   = [o for o in orders if o.side == OrderSide.SELL and o.filled_qty and float(o.filled_qty) > 0]

    # Open positions
    positions = tc.get_all_positions()
    tracked_pos = [p for p in positions if p.symbol in TRACKED]

    today = datetime.now(timezone.utc).strftime("%b %-d")
    lines = [f"📊 <b>Daily Summary — {today}</b>"]
    lines.append(f"Account: ${equity:,.0f} ({total_pnl:+,.0f} total)")
    lines.append("")

    # Entries
    if entries:
        lines.append("<b>Opened today:</b>")
        for o in entries:
            lines.append(f"  ▶ {o.symbol} {o.filled_qty}x @ ${float(o.filled_avg_price):.2f}")

    # Exits
    if exits:
        lines.append("<b>Closed today:</b>")
        for o in exits:
            price = float(o.filled_avg_price)
            # Find matching entry to compute P&L
            matching = next(
                (e for e in entries if e.symbol == o.symbol), None
            )
            if matching:
                entry_price = float(matching.filled_avg_price)
                qty         = float(o.filled_qty)
                pnl         = (price - entry_price) * qty
                icon = "✅" if pnl > 0 else "❌"
                lines.append(f"  {icon} {o.symbol} @ ${price:.2f} | P&L: ${pnl:+,.0f}")
            else:
                lines.append(f"  — {o.symbol} @ ${price:.2f}")

    # Open positions
    if tracked_pos:
        lines.append("<b>Open positions:</b>")
        for p in tracked_pos:
            unreal = float(p.unrealized_pl)
            icon   = "🟢" if unreal >= 0 else "🔴"
            lines.append(
                f"  {icon} {p.symbol} {p.qty}x @ ${float(p.avg_entry_price):.2f}"
                f" | now ${float(p.current_price):.2f} | {unreal:+,.0f}"
            )
    elif not entries and not exits:
        lines.append("No activity today.")

    msg = "\n".join(lines)
    print(msg)
    send_telegram(msg)


if __name__ == "__main__":
    main()
