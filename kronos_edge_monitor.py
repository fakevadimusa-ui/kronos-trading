"""
Kronos Edge Monitor — Bayesian Win-Rate Tracker
================================================
Reads all filled NVDA / USO / SCO bracket trades from Alpaca history,
pairs each entry with its exit, computes realized P&L, and updates a
Bayesian Beta(wins, losses) posterior seeded from OOS validation results.

Outputs a Telegram summary showing:
  - Posterior mean WR and 5th-percentile lower bound
  - Rolling 10-trade win rate
  - Streak (current consecutive wins or losses)
  - ALERT if lower-bound WR drops below the breakeven floor

Run: python kronos_edge_monitor.py
     (or add to crontab alongside monitor, e.g. Monday mornings)
"""

import json
import os
from datetime import datetime, timezone, timedelta
from pathlib import Path

import requests
from scipy import stats

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetOrdersRequest
from alpaca.trading.enums import OrderSide, QueryOrderStatus


# ── CONFIG ───────────────────────────────────────────────────────────────────

CONFIG_PATH = Path.home() / "freqtrade/user_data/config_kronos_nvda.json"

TRACKED_SYMBOLS = ("NVDA", "USO", "SCO")

SL_PCT  = 0.01
TP_PCT  = 0.03
RR      = TP_PCT / SL_PCT        # 3.0
WR_BREAKEVEN = 1 / (1 + RR)     # 25.0%

# OOS priors: Beta(wins, losses) seeded from validated OOS results
# NVDA: 15 trades, WR=46.7% → 7 wins, 8 losses
# CL=F (USO/SCO): 14 trades, WR=35.7% → 5 wins, 9 losses
PRIORS = {
    "NVDA": (7, 8),
    "CL=F": (5, 9),
}

ALERT_LOWER_BOUND_THRESHOLD = 0.28   # 5th-pct WR below this → send alert
ROLLING_WINDOW = 10


# ── HELPERS ──────────────────────────────────────────────────────────────────

def load_credentials() -> tuple[str, str]:
    key    = os.environ.get("ALPACA_KEY")
    secret = os.environ.get("ALPACA_SECRET")
    if key and secret:
        return key, secret
    with open(CONFIG_PATH) as f:
        cfg = json.load(f)
    ex = cfg["exchange"]
    return ex["key"], ex["secret"]


def send_telegram(msg: str) -> None:
    token   = os.environ.get("TELEGRAM_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        print(msg)
        return
    try:
        requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            json={"chat_id": chat_id, "text": msg, "parse_mode": "HTML"},
            timeout=10,
        )
    except Exception as e:
        print(f"Telegram error: {e}")


def fetch_closed_trades(client: TradingClient) -> list[dict]:
    """Return list of {symbol, side, entry_price, exit_price, pnl_pct, win} dicts."""
    # Fetch 6 months of orders to cover OOS window
    since = datetime.now(timezone.utc) - timedelta(days=180)
    orders = client.get_orders(GetOrdersRequest(
        status=QueryOrderStatus.CLOSED,
        after=since,
        limit=500,
    ))

    # Filter to our tracked symbols, filled only
    filled = [
        o for o in orders
        if o.symbol in TRACKED_SYMBOLS
        and o.filled_qty
        and float(o.filled_qty) > 0
        and o.filled_avg_price
    ]

    # Pair entries (BUY or SELL-short) with exits
    entries = [o for o in filled if o.side == OrderSide.BUY]
    exits   = [o for o in filled if o.side == OrderSide.SELL]

    trades = []
    for entry in entries:
        # Find the matching exit for this symbol closest after this entry
        entry_time = entry.filled_at or entry.submitted_at
        matching_exits = [
            ex for ex in exits
            if ex.symbol == entry.symbol
            and (ex.filled_at or ex.submitted_at) > entry_time
        ]
        if not matching_exits:
            continue
        exit_order = min(matching_exits, key=lambda ex: ex.filled_at or ex.submitted_at)

        ep  = float(entry.filled_avg_price)
        xp  = float(exit_order.filled_avg_price)
        pnl = (xp - ep) / ep

        trades.append({
            "symbol":      entry.symbol,
            "entry_price": ep,
            "exit_price":  xp,
            "pnl_pct":     pnl,
            "win":         pnl > 0,
            "filled_at":   str(entry.filled_at or entry.submitted_at),
        })

    return sorted(trades, key=lambda t: t["filled_at"])


def bayesian_wr_report(trades: list[dict], label: str, prior_wins: int, prior_losses: int) -> str:
    if not trades:
        return f"<b>{label}</b>: No closed trades yet."

    wins   = sum(1 for t in trades if t["win"])
    losses = len(trades) - wins

    # Update Beta posterior
    a = prior_wins  + wins
    b = prior_losses + losses
    dist = stats.beta(a, b)

    posterior_mean  = dist.mean()
    lower_bound_5th = dist.ppf(0.05)
    ci_95_lo, ci_95_hi = dist.interval(0.90)

    # Rolling WR (last ROLLING_WINDOW trades)
    recent = trades[-ROLLING_WINDOW:]
    rolling_wr = sum(1 for t in recent if t["win"]) / len(recent)

    # Streak
    streak = 0
    streak_type = "W"
    for t in reversed(trades):
        if streak == 0:
            streak_type = "W" if t["win"] else "L"
            streak = 1
        elif (t["win"] and streak_type == "W") or (not t["win"] and streak_type == "L"):
            streak += 1
        else:
            break

    alert = "🚨 " if lower_bound_5th < ALERT_LOWER_BOUND_THRESHOLD else ""

    lines = [
        f"{alert}<b>{label}</b>  ({len(trades)} live trades, {prior_wins+prior_losses} OOS prior)",
        f"  Posterior WR: {posterior_mean:.1%}  [90% CI: {ci_95_lo:.1%}–{ci_95_hi:.1%}]",
        f"  5th-pct lower bound: {lower_bound_5th:.1%}  (breakeven: {WR_BREAKEVEN:.0%})",
        f"  Rolling {ROLLING_WINDOW}: {rolling_wr:.0%}  |  Streak: {streak}{streak_type}",
    ]

    if lower_bound_5th < ALERT_LOWER_BOUND_THRESHOLD:
        lines.append(f"  ⚠️ EDGE WARNING — lower bound below {ALERT_LOWER_BOUND_THRESHOLD:.0%}")

    return "\n".join(lines)


def main() -> None:
    api_key, api_secret = load_credentials()
    client = TradingClient(api_key, api_secret, paper=True)

    all_trades = fetch_closed_trades(client)

    nvda_trades = [t for t in all_trades if t["symbol"] == "NVDA"]
    crude_trades = [t for t in all_trades if t["symbol"] in ("USO", "SCO")]

    nvda_report  = bayesian_wr_report(nvda_trades,  "NVDA",  *PRIORS["NVDA"])
    crude_report = bayesian_wr_report(crude_trades, "CL=F",  *PRIORS["CL=F"])

    today = datetime.now(timezone.utc).strftime("%b %-d")
    msg = (
        f"📈 <b>Kronos Edge Report — {today}</b>\n\n"
        f"{nvda_report}\n\n"
        f"{crude_report}"
    )

    print(msg)
    send_telegram(msg)


if __name__ == "__main__":
    main()
