#!/usr/bin/env python3
"""
news_straddle.py — MES bracket straddle on high-impact news events.

Places a buy-stop + sell-stop N points from current price at T-2 min before the
event. On first fill the other leg is cancelled and TP/SL orders are placed.

Lucid $100K rules   →  DLL $1,800 | Max loss $3,000 | Target $6,000 | 60 MES max
Default sizing      →  20 MES ($100/point) | TP 50 pts (+$5,000) | SL 15 pts (-$1,500)
One TP hit passes the challenge.

Cron (VPS, UTC — self-exits if no news event today):
  0 12 * * 1-5  cd /root/kronos-trading && venv/bin/python3 news_straddle.py >> /root/logs/straddle/straddle.log 2>&1
  0 17 * * 1-5  cd /root/kronos-trading && venv/bin/python3 news_straddle.py >> /root/logs/straddle/straddle.log 2>&1

Environment variables (add to VPS .env):
  TRADOVATE_BASE       https://demo.tradovateapi.com/v1  (change to live when ready)
  TRADOVATE_USER       your Tradovate username
  TRADOVATE_PASS       your Tradovate password
  TRADOVATE_APP_ID     app name from developer.tradovate.com (default: KronosStraddle)
  TRADOVATE_CID        integer client id from developer portal (default: 0 for dev)
  TRADOVATE_SEC        client secret from developer portal (default: "" for dev)
"""

import os
import sys
import json
import time
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import requests
import pytz

# ── Load .env from project directory ─────────────────────────────────────────
_env_path = Path(__file__).parent / ".env"
if _env_path.exists():
    for _line in _env_path.read_text().splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _v = _line.split("=", 1)
            os.environ.setdefault(_k.strip(), _v.strip().strip('"').strip("'"))

# ── Timezone ──────────────────────────────────────────────────────────────────
ET = pytz.timezone("America/New_York")

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stdout,
)
log = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
TV_BASE    = os.environ.get("TRADOVATE_BASE",    "https://demo.tradovateapi.com/v1")
TV_USER    = os.environ.get("TRADOVATE_USER",    "")
TV_PASS    = os.environ.get("TRADOVATE_PASS",    "")
TV_APP_ID  = os.environ.get("TRADOVATE_APP_ID",  "KronosStraddle")
TV_APP_VER = os.environ.get("TRADOVATE_APP_VER", "1.0")
TV_CID     = int(os.environ.get("TRADOVATE_CID", "0"))
TV_SEC     = os.environ.get("TRADOVATE_SEC",     "")

TG_TOKEN   = os.environ.get("TELEGRAM_TOKEN",  os.environ.get("TELEGRAM_BOT_TOKEN", ""))
TG_CHAT    = os.environ.get("TELEGRAM_CHAT_ID", "")

SYMBOL          = "MES"    # Micro E-mini S&P 500 — front month resolved at runtime
CONTRACTS       = 20       # 20 × $5/pt = $100/pt  (Lucid max = 60)
OFFSET_PTS      = 8.0      # points above/below price for entry stops
TP_PTS          = 50.0     # take-profit distance from fill
SL_PTS          = 15.0     # stop-loss distance from fill
TICK            = 0.25     # MES minimum price increment

FILL_TIMEOUT    = 600      # seconds — give up waiting for entry if no fill in 10 min
POS_TIMEOUT     = 45 * 60  # seconds — force-close if no TP/SL in 45 min
POLL            = 4        # seconds between order-status polls

DLL             = 1_800.0  # Lucid daily loss limit
DLL_BUFFER      = 0.90     # skip if daily P&L has used ≥90% of DLL

EVENTS_FILE     = Path(__file__).parent / "news_events.json"
LOG_DIR         = Path("/root/logs/straddle")


# ── Helpers ───────────────────────────────────────────────────────────────────

def tick(price: float) -> float:
    """Round to nearest MES tick (0.25)."""
    return round(round(price / TICK) * TICK, 2)


def notify(msg: str) -> None:
    log.info(msg)
    if TG_TOKEN and TG_CHAT:
        try:
            requests.post(
                f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage",
                json={"chat_id": TG_CHAT, "text": f"[STRADDLE] {msg}"},
                timeout=5,
            )
        except Exception:
            pass


def _is_filled(order: dict) -> bool:
    s = str(order.get("ordStatus", "")).lower()
    return s in ("filled", "completed", "2")


def _is_dead(order: dict) -> bool:
    s = str(order.get("ordStatus", "")).lower()
    return s in ("canceled", "cancelled", "expired", "rejected", "3", "4", "5")


def _front_month() -> str:
    """Compute front-month MES symbol (e.g. MESM6) based on today's date."""
    codes = {1:"F",2:"G",3:"H",4:"J",5:"K",6:"M",7:"N",8:"Q",9:"U",10:"V",11:"X",12:"Z"}
    now   = datetime.now(ET)
    y, m  = now.year, now.month
    # Find 3rd Friday of current month (CME expiry)
    first_day = datetime(y, m, 1, tzinfo=ET)
    offset    = (4 - first_day.weekday()) % 7   # days until first Friday (4=Fri)
    third_fri = first_day.date() + timedelta(days=offset + 14)
    # On/after expiry → roll to next month
    if now.date() >= third_fri:
        m += 1
        if m > 12:
            m, y = 1, y + 1
    return f"MES{codes[m]}{str(y)[-1]}"


# ── Tradovate client ──────────────────────────────────────────────────────────

class TV:
    """Minimal Tradovate REST client."""

    def __init__(self):
        self.s     = requests.Session()
        self.token = ""
        self.acct  = {}   # {id, name}
        self.ctr   = {}   # {id, name}  ← front-month MES

    # ── Transport ──

    def _h(self) -> dict:
        h = {"Content-Type": "application/json"}
        if self.token:
            h["Authorization"] = f"Bearer {self.token}"
        return h

    def _get(self, path: str, **params) -> object:
        r = self.s.get(f"{TV_BASE}{path}", headers=self._h(), params=params, timeout=10)
        r.raise_for_status()
        return r.json()

    def _post(self, path: str, body: dict) -> object:
        r = self.s.post(f"{TV_BASE}{path}", headers=self._h(), json=body, timeout=10)
        r.raise_for_status()
        return r.json()

    def _delete(self, path: str) -> object:
        r = self.s.delete(f"{TV_BASE}{path}", headers=self._h(), timeout=10)
        r.raise_for_status()
        return r.json()

    # ── Auth / setup ──

    def auth(self) -> None:
        resp = self._post("/auth/accesstokenrequest", {
            "name": TV_USER, "password": TV_PASS,
            "appId": TV_APP_ID, "appVersion": TV_APP_VER,
            "cid": TV_CID, "sec": TV_SEC,
        })
        if "accessToken" not in resp:
            raise RuntimeError(f"Auth failed: {resp}")
        self.token = resp["accessToken"]
        log.info(f"Tradovate authenticated | userId={resp.get('userId')}")

    def load_account(self) -> None:
        accts = self._get("/account/list")
        if not accts:
            raise RuntimeError("No Tradovate accounts found")
        a = accts[0]
        self.acct = {"id": a["id"], "name": a["name"]}
        log.info(f"Account: {self.acct['name']} (id={self.acct['id']})")

    def load_contract(self) -> None:
        # Try the API first; fall back to local front-month computation
        sym = _front_month()
        try:
            resp = self._get("/contract/find", name=sym)
            if resp and "id" in resp:
                self.ctr = {"id": resp["id"], "name": resp["name"]}
                log.info(f"Contract: {self.ctr['name']} (id={self.ctr['id']})")
                return
        except Exception as e:
            log.warning(f"contract/find failed ({e}) — using computed symbol {sym}")
        # Fallback: use computed symbol with id=0 (placeorder accepts name directly)
        self.ctr = {"id": 0, "name": sym}
        log.info(f"Contract (computed): {self.ctr['name']}")

    # ── Market data ──

    def get_price(self) -> float:
        """Current MES last price. Tradovate quote → yfinance ES=F fallback."""
        if self.ctr["id"]:
            try:
                resp = self._get("/quote/quotes", symbols=self.ctr["name"])
                if isinstance(resp, list) and resp and resp[0].get("last"):
                    return float(resp[0]["last"])
            except Exception as e:
                log.warning(f"Tradovate quote failed ({e}) — trying yfinance")
        # yfinance ES=F: 1-min bars, recent enough for a T-2 reference price
        try:
            import yfinance as yf
            import pandas as pd
            df = yf.download("ES=F", period="1d", interval="1m", progress=False)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df.columns = [c.lower() for c in df.columns]
            if not df.empty:
                return float(df["close"].iloc[-1])
        except Exception as e:
            log.warning(f"yfinance ES=F failed: {e}")
        raise RuntimeError("Cannot obtain MES price from any source")

    # ── Risk check ──

    def daily_realized_pnl(self) -> float:
        """Sum of today's realized P&L from Tradovate fill history."""
        try:
            fills = self._get("/fill/list")
            today = datetime.now(ET).date()
            total = 0.0
            for f in fills:
                ts = f.get("timestamp", "")
                if not ts:
                    continue
                dt = datetime.fromisoformat(ts.replace("Z", "+00:00")).astimezone(ET)
                if dt.date() == today:
                    total += float(f.get("realizedPnl", 0))
            return total
        except Exception as e:
            log.warning(f"daily_realized_pnl unavailable: {e}")
            return 0.0

    # ── Orders ──

    def _place(self, action: str, order_type: str, **kwargs) -> int:
        body = {
            "accountSpec": self.acct["name"],
            "accountId":   self.acct["id"],
            "action":      action,
            "symbol":      self.ctr["name"],
            "orderQty":    CONTRACTS,
            "orderType":   order_type,
            "isAutomated": True,
            **kwargs,
        }
        resp = self._post("/order/placeorder", body)
        oid  = resp.get("id") or resp.get("orderId")
        if not oid:
            raise RuntimeError(f"Order rejected by Tradovate: {resp}")
        return int(oid)

    def stop(self, action: str, stop_px: float) -> int:
        px  = tick(stop_px)
        oid = self._place(action, "Stop", stopPrice=px)
        log.info(f"[ORDER] {action} stop @ {px} | id={oid}")
        return oid

    def limit(self, action: str, px: float) -> int:
        px  = tick(px)
        oid = self._place(action, "Limit", price=px)
        log.info(f"[ORDER] {action} limit @ {px} | id={oid}")
        return oid

    def cancel(self, oid: int) -> None:
        try:
            self._delete(f"/order/{oid}")
            log.info(f"[CANCEL] id={oid}")
        except Exception as e:
            log.warning(f"cancel id={oid} failed: {e}")

    def order_status(self, oid: int) -> dict:
        return self._get("/order/item", id=oid)

    def has_position(self) -> bool:
        try:
            for p in self._get("/position/list"):
                if p.get("contractId") == self.ctr["id"] and p.get("netPos", 0) != 0:
                    return True
        except Exception:
            pass
        return False

    def liquidate(self) -> None:
        try:
            self._post("/order/liquidateposition", {
                "accountId":  self.acct["id"],
                "contractId": self.ctr["id"],
                "admin":      False,
            })
            log.info("[LIQUIDATE] emergency close sent")
        except Exception as e:
            log.error(f"liquidate failed: {e}")


# ── News calendar ─────────────────────────────────────────────────────────────

def today_events() -> list[dict]:
    if not EVENTS_FILE.exists():
        log.warning(f"news_events.json not found at {EVENTS_FILE}")
        return []
    events = json.loads(EVENTS_FILE.read_text())
    today  = datetime.now(ET).strftime("%Y-%m-%d")
    return [e for e in events if e["date"] == today]


# ── Straddle ──────────────────────────────────────────────────────────────────

def run_straddle(tv: TV, event: dict) -> None:
    name     = event["event"]
    event_et = datetime.now(ET).replace(
        hour=event["hour"], minute=event["minute"], second=0, microsecond=0,
    )
    arm_et = event_et - timedelta(minutes=2)
    now    = datetime.now(ET)

    if now >= event_et:
        log.info(f"{name} at {event_et.strftime('%H:%M ET')} already passed — skip")
        return

    # ── Wait until T-2 ────────────────────────────────────────────────────────
    wait = (arm_et - now).total_seconds()
    if wait > 0:
        log.info(f"Waiting {wait:.0f}s until T-2 before {name} ({event_et.strftime('%H:%M ET')})")
        time.sleep(max(0, wait))

    # ── Pre-flight ─────────────────────────────────────────────────────────────
    pnl = tv.daily_realized_pnl()
    if pnl <= -(DLL * DLL_BUFFER):
        notify(f"SKIP {name}: daily P&L ${pnl:,.0f} near DLL — standing down")
        return

    if tv.has_position():
        notify(f"SKIP {name}: open position already exists — manual check required")
        return

    # ── Arm ───────────────────────────────────────────────────────────────────
    price    = tv.get_price()
    buy_stop = tick(price + OFFSET_PTS)
    sel_stop = tick(price - OFFSET_PTS)

    notify(
        f"ARMING {name} | {tv.ctr['name']} ref={price:.2f} "
        f"| BUY>{buy_stop} SELL<{sel_stop} | {CONTRACTS}x MES"
    )

    buy_id = tv.stop("Buy",  buy_stop)
    sel_id = tv.stop("Sell", sel_stop)

    # ── Wait for first fill ────────────────────────────────────────────────────
    deadline   = time.monotonic() + FILL_TIMEOUT
    fill_side  = None   # "Buy" or "Sell"
    fill_price = None

    while time.monotonic() < deadline:
        time.sleep(POLL)
        bo = tv.order_status(buy_id)
        so = tv.order_status(sel_id)
        b  = _is_filled(bo)
        s  = _is_filled(so)

        # Whipsaw: both filled before we could cancel the loser
        if b and s:
            tv.liquidate()
            notify(f"WHIPSAW on {name} — both legs filled, position liquidated")
            return

        if b:
            fill_price = float(bo.get("avgPx") or buy_stop)
            fill_side  = "Buy"
            tv.cancel(sel_id)
            break

        if s:
            fill_price = float(so.get("avgPx") or sel_stop)
            fill_side  = "Sell"
            tv.cancel(buy_id)
            break

        # Both dead with no fill — abort
        if _is_dead(bo) and _is_dead(so):
            notify(f"{name}: both orders dead (rejected/expired) — abort")
            return

    if fill_side is None:
        tv.cancel(buy_id)
        tv.cancel(sel_id)
        notify(f"{name}: no fill after {FILL_TIMEOUT//60} min — cancelled (market didn't move)")
        return

    # ── Place TP + SL ──────────────────────────────────────────────────────────
    pts_val = CONTRACTS * 5   # $5 per MES per point

    if fill_side == "Buy":
        tp_px  = tick(fill_price + TP_PTS)
        sl_px  = tick(fill_price - SL_PTS)
        tp_id  = tv.limit("Sell", tp_px)
        sl_id  = tv.stop("Sell",  sl_px)
    else:
        tp_px  = tick(fill_price - TP_PTS)
        sl_px  = tick(fill_price + SL_PTS)
        tp_id  = tv.limit("Buy",  tp_px)
        sl_id  = tv.stop("Buy",   sl_px)

    notify(
        f"{name} FILLED {fill_side.upper()} @ {fill_price:.2f} "
        f"| TP {tp_px:.2f} (+${TP_PTS * pts_val:,.0f}) "
        f"| SL {sl_px:.2f} (-${SL_PTS * pts_val:,.0f})"
    )

    # ── Monitor TP / SL ────────────────────────────────────────────────────────
    deadline = time.monotonic() + POS_TIMEOUT

    while time.monotonic() < deadline:
        time.sleep(POLL)
        tp_o = tv.order_status(tp_id)
        sl_o = tv.order_status(sl_id)

        if _is_filled(tp_o):
            tv.cancel(sl_id)
            notify(
                f"{name} TP HIT ✅ | {fill_side.upper()} +{TP_PTS:.0f}pts "
                f"| +${TP_PTS * pts_val:,.0f}"
            )
            return

        if _is_filled(sl_o):
            tv.cancel(tp_id)
            notify(
                f"{name} SL HIT ❌ | {fill_side.upper()} -{SL_PTS:.0f}pts "
                f"| -${SL_PTS * pts_val:,.0f}"
            )
            return

        # Position closed externally (DLL breach by platform, manual close, etc.)
        if not tv.has_position():
            tv.cancel(tp_id)
            tv.cancel(sl_id)
            notify(f"{name}: position closed externally — TP/SL cancelled")
            return

    # ── Timeout force-close ────────────────────────────────────────────────────
    log.warning(f"{name}: {POS_TIMEOUT//60}-min timeout reached — force-closing")
    tv.cancel(tp_id)
    tv.cancel(sl_id)
    tv.liquidate()
    notify(f"{name}: {POS_TIMEOUT//60}-min timeout — position force-closed")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    events = today_events()
    if not events:
        log.info(f"No news events today ({datetime.now(ET).strftime('%Y-%m-%d')}) — exiting")
        return

    log.info("=" * 60)
    log.info(f"News Straddle | {datetime.now(ET).strftime('%Y-%m-%d %H:%M ET')}")
    log.info(f"Events today: {[e['event'] for e in events]}")
    log.info(f"Sizing: {CONTRACTS}x MES | TP {TP_PTS}pts (+${TP_PTS * CONTRACTS * 5:,.0f}) | SL {SL_PTS}pts (-${SL_PTS * CONTRACTS * 5:,.0f})")

    if not TV_USER or not TV_PASS:
        log.error("TRADOVATE_USER / TRADOVATE_PASS not set — add to .env file")
        sys.exit(1)

    tv = TV()
    try:
        tv.auth()
        tv.load_account()
        tv.load_contract()
    except Exception as e:
        log.exception(f"Tradovate setup failed: {e}")
        notify(f"SETUP FAILED: {e}")
        sys.exit(1)

    for event in events:
        try:
            run_straddle(tv, event)
        except Exception as e:
            log.exception(f"Straddle error on {event['event']}: {e}")
            notify(f"ERROR {event['event']}: {e}")


if __name__ == "__main__":
    main()
