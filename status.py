#!/usr/bin/env python3
"""Quick status dump — run once to resume any trading session."""
import json, os, sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetOrdersRequest

config = json.load(open(Path.home() / "freqtrade/user_data/config_kronos_nvda.json"))
client = TradingClient(config["exchange"]["key"], config["exchange"]["secret"], paper=True)

acct = client.get_account()
equity = float(acct.equity)
start = float(acct.last_equity)
pnl_today = equity - start

print(f"\n{'='*55}")
print(f"  TRADING STATUS  {datetime.now().strftime('%Y-%m-%d %H:%M PT')}")
print(f"{'='*55}")
print(f"  Equity:      ${equity:>10,.2f}")
print(f"  P&L today:   ${pnl_today:>+10,.2f}  ({pnl_today/start*100:+.2f}%)")
print(f"  From $100K:  ${equity-100000:>+10,.2f}  ({(equity/100000-1)*100:+.2f}%)")

positions = client.get_all_positions()
print(f"\n--- OPEN POSITIONS ({len(positions)}) ---")
for p in positions:
    print(f"  {p.symbol:6} {p.qty:>6} @ {float(p.avg_entry_price):.3f} | unrealized: ${float(p.unrealized_pl):+,.2f}")
if not positions:
    print("  None")

req = GetOrdersRequest(status="open", limit=20)
open_orders = client.get_orders(filter=req)
print(f"\n--- OPEN ORDERS ({len(open_orders)}) ---")
for o in open_orders:
    print(f"  {str(o.side.value):4} {o.qty} {o.symbol:6} | {o.type.value:8} lim={o.limit_price} stop={o.stop_price} | {o.time_in_force.value}")
if not open_orders:
    print("  None")

req2 = GetOrdersRequest(status="all", limit=30,
        after=(datetime.now(timezone.utc) - timedelta(days=7)).isoformat())
orders = client.get_orders(filter=req2)
filled = [o for o in orders if o.status.value == "filled"]
print(f"\n--- RECENT FILLS (last 7d, {len(filled)} fills) ---")
for o in filled[-10:]:
    print(f"  {o.created_at.strftime('%m/%d %H:%M')} | {str(o.side.value):4} {o.qty} {o.symbol:6} @ {o.filled_avg_price}")

# ICT paper trade count
spy_fills = [o for o in filled if o.symbol == "SPY"]
print(f"\n--- ICT PAPER TRADES ---")
print(f"  SPY fills: {len(spy_fills)} / 30 needed before challenge")

# Session state
sess_path = Path.home() / "freqtrade/user_data/logs/ict_session.json"
if sess_path.exists():
    s = json.load(open(sess_path))
    sess_start = float(s.get("ict_start_equity", 0))
    print(f"  Session start equity: ${sess_start:,.2f}  date={s.get('date')}")

print(f"\n--- NEXT PRIORITIES ---")
print(f"  1. ICT: accumulate {30 - len(spy_fills)} more paper trades (strict_filters=False)")
print(f"  2. Fix SCO bracket: inverse ETF legs expire DAY — need manual SL monitor")
print(f"  3. MT5 setup for The5ers (Windows VPS, Azure B1s)")
print(f"{'='*55}\n")
