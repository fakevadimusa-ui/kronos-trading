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

import fcntl
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

FILL_TIMEOUT    = 90       # seconds — give up if no fill 90s after news release
POS_TIMEOUT     = 45 * 60  # seconds — force-close if no TP/SL in 45 min
POLL            = 1        # seconds between order-status polls (was 4 — reduced for faster OCO response)
TOKEN_REFRESH_BUFFER = 600 # refresh token this many seconds before expiry

DLL             = 1_800.0  # Lucid daily loss limit
DLL_BUFFER      = 0.90     # skip if daily P&L has used ≥90% of DLL

EVENTS_FILE         = Path(__file__).parent / "news_events.json"
LOG_DIR             = Path("/root/logs/straddle")
_LOCK_PATH          = Path("/tmp/news_straddle.lock")
_STATE_FILE         = Path("/tmp/straddle_state.json")
_DAILY_PNL_PATH     = Path("/tmp/straddle_daily_pnl.json")    # local DLL ledger
_TRADED_TODAY_PATH  = Path("/tmp/straddle_traded_today.json") # one-event-per-day gate

MAX_SPREAD_PTS  = 3.0   # skip event if bid/ask spread > this at T-2

# Event-specific parameters — tuned to each event's historical volatility profile
# offset: stop distance from price at T-2 (lower → tighter, catches fast moves)
# tp: take-profit from fill; sl: stop-loss from fill
EVENT_PARAMS: dict[str, dict] = {
    "NFP":     {"offset": 10.0, "tp": 65.0, "sl": 18.0},  # most volatile
    "CPI":     {"offset":  8.0, "tp": 50.0, "sl": 15.0},
    "PPI":     {"offset":  5.0, "tp": 35.0, "sl": 12.0},
    "FOMC":    {"offset":  6.0, "tp": 45.0, "sl": 14.0},
    "JOLTS":   {"offset":  4.0, "tp": 30.0, "sl": 10.0},
    "CLAIMS":  {"offset":  4.0, "tp": 30.0, "sl": 10.0},
    "RETAIL":  {"offset":  5.0, "tp": 35.0, "sl": 12.0},
    "PCE":     {"offset":  7.0, "tp": 45.0, "sl": 14.0},
    "ISM":     {"offset":  4.0, "tp": 28.0, "sl": 10.0},
}

# Challenge profit protect: reduce contracts when close to target
CHALLENGE_TARGET_PNL = 6_000.0  # 6% on $100K Lucid challenge


# ── Helpers ───────────────────────────────────────────────────────────────────

def tick(price: float) -> float:
    """Round to nearest MES tick (0.25)."""
    return round(round(price / TICK) * TICK, 2)


# ── Local DLL ledger ─ guards against API lag on daily_realized_pnl() ─────────

def _record_local_pnl(pnl: float) -> float:
    """Write realised P&L to disk. Used as conservative fallback for DLL check."""
    today = datetime.now(ET).strftime("%Y-%m-%d")
    data: dict = {}
    if _DAILY_PNL_PATH.exists():
        try:
            data = json.loads(_DAILY_PNL_PATH.read_text())
        except Exception:
            pass
    data[today] = data.get(today, 0.0) + pnl
    # Prune stale days
    cutoff = (datetime.now(ET).date() - timedelta(days=3)).isoformat()
    data = {k: v for k, v in data.items() if k >= cutoff}
    _DAILY_PNL_PATH.write_text(json.dumps(data))
    return data[today]


def _local_daily_pnl() -> float:
    """Read local P&L ledger. Returns 0.0 if no file or parse error."""
    today = datetime.now(ET).strftime("%Y-%m-%d")
    if not _DAILY_PNL_PATH.exists():
        return 0.0
    try:
        return float(json.loads(_DAILY_PNL_PATH.read_text()).get(today, 0.0))
    except Exception:
        return 0.0


def _conservative_daily_pnl(tv: "TV") -> float:
    """Return the most negative (worst-case) daily P&L from API or local ledger."""
    api_pnl   = tv.daily_realized_pnl()
    local_pnl = _local_daily_pnl()
    return min(api_pnl, local_pnl)   # most negative wins


# ── One-event-per-day gate ─────────────────────────────────────────────────────

def _already_traded_today() -> bool:
    today = datetime.now(ET).strftime("%Y-%m-%d")
    if not _TRADED_TODAY_PATH.exists():
        return False
    try:
        return json.loads(_TRADED_TODAY_PATH.read_text()).get("date") == today
    except Exception:
        return False


def _mark_traded_today() -> None:
    today = datetime.now(ET).strftime("%Y-%m-%d")
    _TRADED_TODAY_PATH.write_text(json.dumps({"date": today}))


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
    return f"MES{codes[m]}{str(y)[-2:]}"


# ── Trade state persistence (survives VPS reboot during live trade) ───────────

class StateFile:
    """Atomic JSON state file for crash-recovery of live straddle positions."""

    @staticmethod
    def write(event: str, fill_side: str, fill_price: float,
              tp_id: int, sl_id: int, tp_px: float, sl_px: float,
              contracts: int = CONTRACTS, tp_pts: float = TP_PTS, sl_pts: float = SL_PTS) -> None:
        payload = {
            "event":        event,
            "fill_side":    fill_side,
            "fill_price":   fill_price,
            "tp_order_id":  tp_id,
            "sl_order_id":  sl_id,
            "tp_price":     tp_px,
            "sl_price":     sl_px,
            "contracts":    contracts,
            "tp_pts":       tp_pts,
            "sl_pts":       sl_pts,
            "timestamp":    datetime.now(ET).isoformat(),
            "status":       "monitoring",
        }
        tmp = _STATE_FILE.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, indent=2))
        tmp.replace(_STATE_FILE)   # atomic rename

    @staticmethod
    def mark_closed(result: str) -> None:
        if not _STATE_FILE.exists():
            return
        try:
            s = json.loads(_STATE_FILE.read_text())
            s["status"] = result
            s["closed_at"] = datetime.now(ET).isoformat()
            _STATE_FILE.write_text(json.dumps(s, indent=2))
        except Exception:
            pass

    @staticmethod
    def clear() -> None:
        if _STATE_FILE.exists():
            _STATE_FILE.unlink(missing_ok=True)

    @staticmethod
    def load() -> "dict | None":
        if not _STATE_FILE.exists():
            return None
        try:
            return json.loads(_STATE_FILE.read_text())
        except Exception:
            return None


# ── Tradovate client ──────────────────────────────────────────────────────────

class TV:
    """Hardened Tradovate REST client — token auto-refresh + exponential backoff retry."""

    def __init__(self):
        self.s            = requests.Session()
        self.token        = ""
        self._expires_at  = 0.0   # unix timestamp of token expiry
        self.acct         = {}    # {id, name}
        self.ctr          = {}    # {id, name}  ← front-month MES

    # ── Transport ──

    def _h(self) -> dict:
        h = {"Content-Type": "application/json"}
        if self.token:
            h["Authorization"] = f"Bearer {self.token}"
        return h

    def _maybe_refresh(self) -> None:
        """Proactively renew token TOKEN_REFRESH_BUFFER seconds before expiry."""
        if not self._expires_at or time.time() < self._expires_at - TOKEN_REFRESH_BUFFER:
            return
        try:
            log.info("Token nearing expiry — renewing...")
            r = self.s.post(
                f"{TV_BASE}/auth/renewaccesstoken",
                headers=self._h(), json={}, timeout=10,
            )
            if r.status_code == 200:
                resp = r.json()
                if "accessToken" in resp:
                    self.token = resp["accessToken"]
                    self._expires_at = self._parse_expiry(resp)
                    log.info(f"Token renewed — expires in {(self._expires_at - time.time())/60:.0f}min")
                    return
            # If renewal fails, full re-auth
            log.warning(f"Renewal returned {r.status_code} — doing full re-auth")
            self.auth()
        except Exception as e:
            log.warning(f"Token renewal failed ({e}) — attempting full re-auth")
            try:
                self.auth()
            except Exception as e2:
                log.error(f"Full re-auth also failed: {e2}")

    @staticmethod
    def _parse_expiry(resp: dict) -> float:
        exp_str = resp.get("expirationTime", "")
        if exp_str:
            try:
                return datetime.fromisoformat(exp_str.replace("Z", "+00:00")).timestamp()
            except Exception:
                pass
        return time.time() + 3600  # 1h fallback

    def _request(self, method: str, path: str, **kwargs) -> object:
        """Single-entrypoint transport: token refresh + retry with exponential backoff."""
        self._maybe_refresh()
        last_err: Exception = RuntimeError("no attempt made")
        for attempt in range(4):
            try:
                r = self.s.request(
                    method, f"{TV_BASE}{path}",
                    headers=self._h(), timeout=10, **kwargs,
                )
                if r.status_code == 401:
                    log.warning(f"401 on {method} {path} (attempt {attempt+1}) — re-authenticating")
                    self.auth()
                    continue
                r.raise_for_status()
                return r.json()
            except requests.exceptions.HTTPError:
                raise
            except (requests.exceptions.ConnectionError,
                    requests.exceptions.Timeout) as e:
                last_err = e
                if attempt < 3:
                    wait = 2 ** attempt  # 1s, 2s, 4s
                    log.warning(
                        f"Network error on {method} {path} "
                        f"(attempt {attempt+1}/4) — retry in {wait}s: {type(e).__name__}"
                    )
                    time.sleep(wait)
        raise RuntimeError(f"All 4 attempts failed for {method} {path}") from last_err

    def _get(self, path: str, **params) -> object:
        return self._request("GET", path, params=params)

    def _post(self, path: str, body: dict) -> object:
        return self._request("POST", path, json=body)

    def _delete(self, path: str) -> object:
        return self._request("DELETE", path)

    # ── Auth / setup ──

    def auth(self) -> None:
        # Post directly (can't use self._post — would recurse through _maybe_refresh)
        r = self.s.post(
            f"{TV_BASE}/auth/accesstokenrequest",
            headers={"Content-Type": "application/json"},
            json={
                "name": TV_USER, "password": TV_PASS,
                "appId": TV_APP_ID, "appVersion": TV_APP_VER,
                "cid": TV_CID, "sec": TV_SEC,
            },
            timeout=15,
        )
        r.raise_for_status()
        resp = r.json()
        if "accessToken" not in resp:
            raise RuntimeError(f"Auth failed: {resp}")
        self.token = resp["accessToken"]
        self._expires_at = self._parse_expiry(resp)
        log.info(
            f"Tradovate authenticated | userId={resp.get('userId')} "
            f"| token expires in {(self._expires_at - time.time())/60:.0f}min"
        )

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

    def _place(self, action: str, order_type: str, qty: int = CONTRACTS, **kwargs) -> int:
        body = {
            "accountSpec": self.acct["name"],
            "accountId":   self.acct["id"],
            "action":      action,
            "symbol":      self.ctr["name"],
            "orderQty":    qty,
            "orderType":   order_type,
            "isAutomated": True,
            **kwargs,
        }
        resp = self._post("/order/placeorder", body)
        oid  = resp.get("id") or resp.get("orderId")
        if not oid:
            raise RuntimeError(f"Order rejected by Tradovate: {resp}")
        return int(oid)

    def stop(self, action: str, stop_px: float, qty: int = CONTRACTS) -> int:
        px  = tick(stop_px)
        oid = self._place(action, "Stop", qty=qty, stopPrice=px)
        log.info(f"[ORDER] {action} stop @ {px} qty={qty} | id={oid}")
        return oid

    def limit(self, action: str, px: float, qty: int = CONTRACTS) -> int:
        px  = tick(px)
        oid = self._place(action, "Limit", qty=qty, price=px)
        log.info(f"[ORDER] {action} limit @ {px} qty={qty} | id={oid}")
        return oid

    def cancel(self, oid: int) -> None:
        """
        Cancel an order and verify it is actually dead (V8).
        Polls order status for up to 3 s after the DELETE request.
        Raises RuntimeError (and sends CRITICAL Telegram alert) if the order
        is still live — or fills unexpectedly during the cancel window.
        Never silently ignores a failed cancel.
        """
        try:
            self._delete(f"/order/{oid}")
        except Exception as e:
            log.warning(f"[CANCEL] DELETE failed id={oid}: {e}")
        # V8: poll until confirmed dead
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline:
            try:
                st = self.order_status(oid)
            except Exception:
                time.sleep(0.5)
                continue
            if _is_dead(st):
                log.info(f"[CANCEL] id={oid} confirmed dead (ordStatus={st.get('ordStatus')})")
                return
            if _is_filled(st):
                msg = f"[CANCEL] id={oid} filled during cancel window — HALTING"
                log.critical(msg)
                notify(f"🚨 CRITICAL: {msg} — manual intervention required")
                raise RuntimeError(msg)
            time.sleep(0.5)
        msg = f"[CANCEL FAILED] id={oid} still live after 3s — HALTING"
        log.critical(msg)
        notify(f"🚨 CRITICAL: {msg} — manual intervention required")
        raise RuntimeError(msg)

    def order_status(self, oid: int) -> dict:
        return self._get("/order/item", id=oid)

    def has_position(self) -> bool:
        try:
            for p in self._get("/position/list"):
                # Primary: match by contractId; fallback: any non-zero position
                ctr_match = (self.ctr.get("id") and
                             p.get("contractId") == self.ctr["id"])
                any_pos   = p.get("netPos", 0) != 0
                if ctr_match and any_pos:
                    return True
            # Fallback: if contractId matching found nothing, check for ANY open position
            # Catches contract-roll edge case where ctr["id"] has changed
            for p in self._get("/position/list"):
                if p.get("netPos", 0) != 0:
                    log.warning(
                        f"has_position fallback: found non-zero position "
                        f"contractId={p.get('contractId')} netPos={p.get('netPos')}"
                    )
                    return True
        except Exception:
            pass
        return False

    def get_spread(self) -> float:
        """Return current bid/ask spread in points. Returns 0 if unavailable."""
        if not self.ctr.get("id"):
            return 0.0
        try:
            resp = self._get("/quote/quotes", symbols=self.ctr["name"])
            if isinstance(resp, list) and resp:
                bid = float(resp[0].get("bid", 0) or 0)
                ask = float(resp[0].get("ask", 0) or 0)
                if bid > 0 and ask > 0:
                    return round(ask - bid, 2)
        except Exception:
            pass
        return 0.0

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
    # Only HIGH impact events — medium/low have insufficient directional movement
    # to cover the offset + SL cost. If event has no impact field, include it.
    qualified = []
    for e in events:
        if e["date"] != today:
            continue
        impact = str(e.get("impact", "HIGH")).upper()
        if impact not in ("HIGH", ""):
            log.info(f"Skipping {e['event']} (impact={impact} — only HIGH events traded)")
            continue
        qualified.append(e)
    return qualified


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

    # Event-specific sizing — look up by first word of event name (e.g. "NFP Jobs Report" → "NFP")
    _event_key = name.upper().split()[0]
    _ep        = EVENT_PARAMS.get(_event_key, {})
    offset_pts = _ep.get("offset", OFFSET_PTS)
    tp_pts     = _ep.get("tp",     TP_PTS)
    sl_pts     = _ep.get("sl",     SL_PTS)
    log.info(f"Event params for {_event_key}: offset={offset_pts} tp={tp_pts} sl={sl_pts}")

    # Profit protect mode — at ≥75% of challenge target, halve contracts to lock in gains
    effective_contracts = CONTRACTS
    try:
        from challenge_mode import _load_state as _cg_state
        _cg = _cg_state()
        _total_pnl = _cg.get("total_pnl", 0.0)
        if _total_pnl >= 0.75 * CHALLENGE_TARGET_PNL:
            effective_contracts = max(1, CONTRACTS // 2)
            log.info(f"Profit protect: {_total_pnl:.0f} ≥ 75% of target — "
                     f"reducing to {effective_contracts} contracts")
    except Exception:
        pass  # challenge_mode not configured — use default CONTRACTS

    # ── Wait until T-2 ────────────────────────────────────────────────────────
    wait = (arm_et - now).total_seconds()
    if wait > 0:
        log.info(f"Waiting {wait:.0f}s until T-2 before {name} ({event_et.strftime('%H:%M ET')})")
        time.sleep(max(0, wait))

    # V6: snapshot reference price immediately at T-2 — used to detect pre-event drift
    _drift_ref: Optional[float] = None
    try:
        _drift_ref = tv.get_price()
    except Exception:
        pass  # drift check skipped if initial fetch fails

    # ── Pre-flight ─────────────────────────────────────────────────────────────
    # Use conservative (most negative) of API P&L and local ledger
    pnl = _conservative_daily_pnl(tv)
    if pnl <= -(DLL * DLL_BUFFER):
        notify(f"SKIP {name}: daily P&L ${pnl:,.0f} near DLL — standing down")
        return

    if tv.has_position():
        notify(f"SKIP {name}: open position already exists — manual check required")
        return

    # One-event-per-day gate (challenge stage: never double-dip on DLL)
    try:
        from challenge_mode import detect_stage
        if detect_stage() == "CHALLENGE" and _already_traded_today():
            notify(f"SKIP {name}: already traded a straddle today — one-event-per-day in CHALLENGE stage")
            return
    except Exception:
        pass  # challenge_mode not configured

    # Spread explosion protection
    spread = tv.get_spread()
    if spread > 0 and spread > MAX_SPREAD_PTS:
        notify(f"SKIP {name}: spread {spread:.2f}pts > {MAX_SPREAD_PTS}pts max — market dislocated")
        return

    # ── Challenge guard ────────────────────────────────────────────────────────
    try:
        from challenge_mode import ChallengeGuard
        _guard = ChallengeGuard()
        _allowed, _reason = _guard.can_trade("STRADDLE")
        if not _allowed:
            notify(f"SKIP {name}: challenge guard — {_reason}")
            return
    except Exception:
        pass  # challenge_mode not configured — continue

    # ── Pre-event ATR check — skip dead markets ────────────────────────────────
    # If the 15-min ATR < 3pts, the market is too quiet to cover offset + SL.
    # This filters holiday-thinned sessions, summer Fridays, post-close noise.
    try:
        import yfinance as yf
        import pandas as pd
        _atr_df = yf.download("ES=F", period="2d", interval="15m",
                              auto_adjust=True, progress=False)
        if isinstance(_atr_df.columns, pd.MultiIndex):
            _atr_df.columns = _atr_df.columns.get_level_values(0)
        if not _atr_df.empty and len(_atr_df) >= 14:
            _h = _atr_df["High"].values; _l = _atr_df["Low"].values; _c = _atr_df["Close"].values
            _tr = [max(_h[i]-_l[i], abs(_h[i]-_c[i-1]), abs(_l[i]-_c[i-1]))
                   for i in range(1, len(_h))]
            _atr14 = sum(_tr[-14:]) / 14
            if _atr14 < 3.0:
                notify(f"SKIP {name}: 15-min ATR {_atr14:.2f}pts < 3.0 — dead market, no trade")
                return
            log.info(f"Pre-event ATR check: {_atr14:.2f}pts — OK")
    except Exception as _atr_err:
        log.warning(f"ATR check failed ({_atr_err}) — proceeding without gate")

    # ── V6: pre-event drift check ─────────────────────────────────────────────
    # Measure price drift from T-2 snapshot to now (after pre-flight checks).
    # Pre-flight takes 5-30s; a move of >40% of the offset in that window
    # means the market is already directionally committed — no edge left.
    price = tv.get_price()
    if _drift_ref is not None:
        _drift = abs(price - _drift_ref)
        _drift_threshold = 0.40 * offset_pts
        if _drift > _drift_threshold:
            notify(
                f"SKIP {name}: pre-event drift {_drift:.2f}pts > {_drift_threshold:.2f}pts "
                f"(T-2 ref={_drift_ref:.2f} → now={price:.2f}) — market pre-positioning, no edge"
            )
            return

    # ── Pre-event checklist ───────────────────────────────────────────────────
    _dll_remaining = DLL + pnl          # pnl is negative for a loss day
    _env_label     = "⚠️ DEMO" if "demo" in TV_BASE.lower() else "✅ LIVE"
    _mode_str      = "N/A"
    _stage_str     = "N/A"
    _max_loss_rem  = 3_000.0
    try:
        from challenge_mode import _load_state as _cg_ls
        _cgs          = _cg_ls()
        _mode_str     = _cgs.get("mode", "NORMAL")
        _stage_str    = _cgs.get("stage", "CHALLENGE")
        _max_loss_rem = 3_000.0 + _cgs.get("total_pnl", 0.0)
    except Exception:
        pass
    _sep = "─" * 46
    checklist_lines = [
        _sep,
        f"  PRE-EVENT CHECKLIST: {name}",
        _sep,
        f"  Time       : {event_et.strftime('%H:%M ET')}",
        f"  Environment: {_env_label}",
        f"  Offset     : {offset_pts:.2f} pts",
        f"  TP         : {tp_pts:.2f} pts  (+${tp_pts * effective_contracts * 5:,.0f})",
        f"  SL         : {sl_pts:.2f} pts  (-${sl_pts * effective_contracts * 5:,.0f})",
        f"  Contracts  : {effective_contracts}x MES",
        f"  Mode/Stage : {_mode_str} / {_stage_str}",
        f"  DLL remain : ${_dll_remaining:,.0f} / ${DLL:,.0f}",
        f"  MaxLoss rem: ${max(0.0, _max_loss_rem):,.0f} / $3,000",
        _sep,
    ]
    log.info("\n" + "\n".join(checklist_lines))
    notify(
        f"📋 {name} | {_env_label} | {effective_contracts}x MES | "
        f"offset={offset_pts} TP={tp_pts} SL={sl_pts} | "
        f"DLL rem=${_dll_remaining:,.0f}"
    )

    # ── Arm ───────────────────────────────────────────────────────────────────
    buy_stop = tick(price + offset_pts)
    sel_stop = tick(price - offset_pts)

    notify(
        f"ARMING {name} | {tv.ctr['name']} ref={price:.2f} "
        f"| BUY>{buy_stop} SELL<{sel_stop} | {effective_contracts}x MES"
    )

    buy_id = tv.stop("Buy",  buy_stop, qty=effective_contracts)
    sel_id = tv.stop("Sell", sel_stop, qty=effective_contracts)

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
        notify(f"{name}: no fill after {FILL_TIMEOUT}s — cancelled (market didn't move)")
        return

    # ── Place TP + SL ──────────────────────────────────────────────────────────
    pts_val = effective_contracts * 5   # $5 per MES per point

    if fill_side == "Buy":
        tp_px  = tick(fill_price + tp_pts)
        sl_px  = tick(fill_price - sl_pts)
        tp_id  = tv.limit("Sell", tp_px, qty=effective_contracts)
        sl_id  = tv.stop("Sell",  sl_px, qty=effective_contracts)
    else:
        tp_px  = tick(fill_price - tp_pts)
        sl_px  = tick(fill_price + sl_pts)
        tp_id  = tv.limit("Buy",  tp_px, qty=effective_contracts)
        sl_id  = tv.stop("Buy",   sl_px, qty=effective_contracts)

    # Persist state immediately — survives VPS reboot, process crash, or network drop
    # Post-fill DLL re-check — if fill itself consumed most of DLL headroom, abort
    pnl_postfill = _conservative_daily_pnl(tv)
    if pnl_postfill <= -(DLL * 0.85):
        notify(
            f"{name} POST-FILL DLL CHECK: ${pnl_postfill:,.0f} at 85% of DLL — "
            f"liquidating immediately before TP/SL"
        )
        tv.liquidate()
        return

    # Mark that we traded today (one-event-per-day gate)
    _mark_traded_today()

    StateFile.write(name, fill_side, fill_price, tp_id, sl_id, tp_px, sl_px,
                    contracts=effective_contracts, tp_pts=tp_pts, sl_pts=sl_pts)

    notify(
        f"{name} FILLED {fill_side.upper()} @ {fill_price:.2f} "
        f"| TP {tp_px:.2f} (+${tp_pts * pts_val:,.0f}) "
        f"| SL {sl_px:.2f} (-${sl_pts * pts_val:,.0f})"
    )

    _monitor_position(tv, name, fill_side, fill_price, tp_id, sl_id, tp_px, sl_px,
                      pts_val, tp_pts=tp_pts, sl_pts=sl_pts,
                      contracts=effective_contracts)


def _monitor_position(
    tv: "TV", name: str, fill_side: str, fill_price: float,
    tp_id: int, sl_id: int, tp_px: float, sl_px: float, pts_val: int,
    tp_pts: float = TP_PTS, sl_pts: float = SL_PTS, contracts: int = CONTRACTS,
) -> None:
    """
    Monitoring loop for TP/SL outcome.
    Extracted so crash-recovery can call it directly after reloading state.

    Trailing stop: once unrealized gain ≥ 50% of TP distance, cancel existing SL
    and replace with a breakeven stop (fill_price ± 2 ticks buffer). This converts
    a potential loss into a scratch/small-win and dramatically improves risk profile.
    """
    deadline          = time.monotonic() + POS_TIMEOUT
    be_stop_placed    = False   # True once breakeven stop has been set
    be_trigger_pts    = tp_pts * 0.50   # move SL to BE once 50% of TP distance reached
    be_buffer_pts     = 0.5     # 2 ticks buffer above/below fill so BE stop isn't at exact fill

    while time.monotonic() < deadline:
        time.sleep(POLL)

        try:
            tp_o = tv.order_status(tp_id)
            sl_o = tv.order_status(sl_id)
        except Exception as e:
            log.warning(f"Order status poll failed: {e} — retrying next tick")
            continue

        if _is_filled(tp_o):
            tv.cancel(sl_id)
            realised = tp_pts * pts_val
            _record_local_pnl(+realised)
            StateFile.mark_closed("tp_hit")
            StateFile.clear()
            notify(
                f"{name} TP HIT ✅ | {fill_side.upper()} +{tp_pts:.0f}pts "
                f"| +${realised:,.0f}"
            )
            return

        if _is_filled(sl_o):
            tv.cancel(tp_id)
            label    = "BE STOP" if be_stop_placed else "SL HIT"
            realised = -(sl_pts * pts_val)
            _record_local_pnl(realised)
            StateFile.mark_closed("sl_hit")
            StateFile.clear()
            notify(
                f"{name} {label} ❌ | {fill_side.upper()} -{sl_pts:.0f}pts "
                f"| ${realised:,.0f}"
            )
            return

        # Both dead (platform issue / DLL auto-close)
        if _is_dead(tp_o) and _is_dead(sl_o):
            StateFile.mark_closed("both_dead")
            StateFile.clear()
            notify(f"{name}: both TP/SL orders dead — liquidating position")
            tv.liquidate()
            return

        # Position closed externally (prop firm DLL breach, manual close)
        if not tv.has_position():
            tv.cancel(tp_id)
            tv.cancel(sl_id)
            StateFile.mark_closed("external_close")
            StateFile.clear()
            notify(f"{name}: position closed externally — TP/SL cancelled")
            return

        # ── Trailing stop: move SL to breakeven at 50% of TP ─────────────────
        if not be_stop_placed and not _is_dead(sl_o):
            try:
                current_px = tv.get_price()
                if fill_side == "Buy":
                    unrealized_pts = current_px - fill_price
                    if unrealized_pts >= be_trigger_pts:
                        be_px = tick(fill_price + be_buffer_pts)
                        tv.cancel(sl_id)
                        sl_id = tv.stop("Sell", be_px, qty=contracts)
                        be_stop_placed = True
                        sl_pts = be_buffer_pts   # update for notification accuracy
                        notify(
                            f"{name}: trailing stop moved to breakeven @ {be_px:.2f} "
                            f"(+{unrealized_pts:.1f}pts unrealized)"
                        )
                else:  # Sell
                    unrealized_pts = fill_price - current_px
                    if unrealized_pts >= be_trigger_pts:
                        be_px = tick(fill_price - be_buffer_pts)
                        tv.cancel(sl_id)
                        sl_id = tv.stop("Buy", be_px, qty=contracts)
                        be_stop_placed = True
                        sl_pts = be_buffer_pts
                        notify(
                            f"{name}: trailing stop moved to breakeven @ {be_px:.2f} "
                            f"(+{unrealized_pts:.1f}pts unrealized)"
                        )
            except Exception as _te:
                log.warning(f"Trailing stop check failed: {_te}")

    # ── Timeout force-close ────────────────────────────────────────────────────
    log.warning(f"{name}: {POS_TIMEOUT//60}-min timeout reached — force-closing")
    tv.cancel(tp_id)
    tv.cancel(sl_id)
    tv.liquidate()
    StateFile.mark_closed("timeout")
    StateFile.clear()
    notify(f"{name}: {POS_TIMEOUT//60}-min timeout — position force-closed")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    # Run lock — prevents two cron ticks from arming the same event simultaneously
    lock_fh = open(_LOCK_PATH, "w")
    try:
        fcntl.flock(lock_fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        log.info("Run lock held — another straddle process is active, exiting")
        lock_fh.close()
        return

    try:
        _run()
    finally:
        fcntl.flock(lock_fh, fcntl.LOCK_UN)
        lock_fh.close()


def _run() -> None:
    # ── Crash recovery — resume monitoring if we rebooted during a live trade ──
    saved = StateFile.load()
    if saved and saved.get("status") == "monitoring":
        age_s = (datetime.now(ET) - datetime.fromisoformat(saved["timestamp"])).total_seconds()
        if age_s < POS_TIMEOUT:
            notify(
                f"🔄 RECOVERING straddle state after crash/reboot\n"
                f"Event: {saved['event']} | Side: {saved['fill_side']} "
                f"@ {saved['fill_price']} | State age: {age_s/60:.1f}min"
            )
            log.info(f"Recovered state: {saved}")
            if not TV_USER or not TV_PASS:
                log.error("Cannot recover — no Tradovate credentials")
                StateFile.clear()
            else:
                tv_r = TV()
                try:
                    tv_r.auth()
                    tv_r.load_account()
                    tv_r.load_contract()
                    _saved_contracts = int(saved.get("contracts", CONTRACTS))
                    pts_val = _saved_contracts * 5
                    if tv_r.has_position():
                        _monitor_position(
                            tv_r,
                            saved["event"],
                            saved["fill_side"],
                            float(saved["fill_price"]),
                            int(saved["tp_order_id"]),
                            int(saved["sl_order_id"]),
                            float(saved["tp_price"]),
                            float(saved["sl_price"]),
                            pts_val,
                            tp_pts=float(saved.get("tp_pts", TP_PTS)),
                            sl_pts=float(saved.get("sl_pts", SL_PTS)),
                            contracts=_saved_contracts,
                        )
                    else:
                        notify(f"Recovery: no open position found — state cleared")
                        StateFile.clear()
                except Exception as e:
                    notify(f"⛔ Recovery FAILED: {e} — manual check required")
                    log.exception("Recovery failed")
        else:
            log.info(f"Stale state file ({age_s/60:.0f}min old) — clearing")
            StateFile.clear()

    # Demo environment warning — critical: silent failure if URL not changed
    if "demo" in TV_BASE.lower():
        msg = "⚠️ DEMO MODE ACTIVE — connected to demo.tradovateapi.com. No real orders will execute."
        log.warning(msg)
        notify(msg)

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
