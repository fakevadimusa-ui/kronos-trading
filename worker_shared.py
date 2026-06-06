"""
worker_shared.py — Hardened shared utilities for all ICT workers
=================================================================
Every guard ported from ict_alpaca_trader.py (1682-line production bot).
Workers pass their own paths/names so all state is isolated per-worker.

Guards included:
  [A] News blackout embargo (2-phase hard/soft)
  [B] Magnitude-weighted circuit breaker + physical day-lock padlock
  [C] Z-score spread smoke detector (disk-backed baseline)
  [D] Broker-first loss counting (Alpaca order history — no local drift)
  [E] Orphan reaper (ghost SL cancellation)
  [F] Bouncer + Decoy Detector (bar-dedup + SHA256 signal hash)
  [G] Fail-closed DD checks (daily, total, trailing)
  [H] flock run lock (per-worker, no cross-worker blocking)

Usage in each worker:
    from worker_shared import (
        make_logger, load_credentials, send_telegram, RunLock,
        WorkerGuards,
    )
    guards = WorkerGuards(worker_name="ict_ny", log_dir=LOG_DIR,
                          symbol="SPY", env_file=ENV_FILE,
                          circuit_breaker_losses=2)
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import logging
import os
import statistics
import sys
from datetime import datetime, timedelta, time, timezone
from pathlib import Path
from zoneinfo import ZoneInfo
from typing import Optional

import requests
from dotenv import load_dotenv

ET = ZoneInfo("America/New_York")


# ═══════════════════════════════════════════════════════════
# LOGGING
# ═══════════════════════════════════════════════════════════

def make_logger(worker_name: str, log_dir: str) -> logging.Logger:
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    today    = datetime.now(ET).strftime("%Y%m%d")
    log_path = Path(log_dir) / f"{worker_name}_{today}.log"
    logger   = logging.getLogger(worker_name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.DEBUG)
    fmt = logging.Formatter(
        f"[%(asctime)s] [{worker_name}] %(levelname)-8s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    fh = logging.FileHandler(log_path)
    fh.setFormatter(fmt)
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


# ═══════════════════════════════════════════════════════════
# CREDENTIALS
# ═══════════════════════════════════════════════════════════

def load_credentials(env_file: str | None = None) -> tuple[str, str]:
    if env_file:
        load_dotenv(env_file, override=True)
    key    = os.environ.get("ALPACA_KEY", "")
    secret = os.environ.get("ALPACA_SECRET", "")
    if key and secret:
        return key, secret
    cfg_path = Path.home() / "freqtrade/user_data/config_kronos_nvda.json"
    if cfg_path.exists():
        cfg = json.loads(cfg_path.read_text())
        return cfg["exchange"]["key"], cfg["exchange"]["secret"]
    raise RuntimeError("No Alpaca credentials found.")


# ═══════════════════════════════════════════════════════════
# TELEGRAM
# ═══════════════════════════════════════════════════════════

def send_telegram(msg: str, logger: logging.Logger | None = None) -> None:
    token   = os.environ.get("TELEGRAM_TOKEN", "")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID", "")
    if not token or not chat_id:
        return
    try:
        requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            json={"chat_id": chat_id, "text": msg, "parse_mode": "HTML"},
            timeout=5,
        )
    except Exception as e:
        if logger:
            logger.warning(f"Telegram failed: {e}")


# ═══════════════════════════════════════════════════════════
# RUN LOCK  (per-worker flock — workers never block each other)
# ═══════════════════════════════════════════════════════════

class RunLock:
    def __init__(self, lock_path: str, logger: logging.Logger):
        self.path   = lock_path
        self.logger = logger
        self._fh    = None

    def acquire(self) -> bool:
        try:
            self._fh = open(self.path, "w")
            fcntl.flock(self._fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self._fh.write(str(os.getpid()))
            self._fh.flush()
            return True
        except BlockingIOError:
            self.logger.warning(f"Run lock held — skipping tick. Lock: {self.path}")
            return False
        except Exception as e:
            self.logger.error(f"Lock acquire error: {e}")
            return False

    def release(self) -> None:
        if self._fh:
            try:
                fcntl.flock(self._fh, fcntl.LOCK_UN)
                self._fh.close()
            except Exception:
                pass
            self._fh = None


# ═══════════════════════════════════════════════════════════
# NEWS EMBARGO WINDOWS  [Guard A]
# Ported directly from ict_alpaca_trader.py — do not modify
# ═══════════════════════════════════════════════════════════

_EMBARGO_HARD_ET = [
    (time(8, 15),  time(8, 33)),    # CPI/PPI/NFP:  -15min pre + 3min post
    (time(9, 25),  time(9, 32)),    # Market open:  -5min pre  + 2min post
    (time(13, 45), time(14, 3)),    # FOMC:         -15min pre + 3min post
]
_EMBARGO_SOFT_ET = [
    (time(8, 33),  time(8, 45)),    # CPI/PPI/NFP:  up to 12min post
    (time(9, 32),  time(9, 35)),    # Market open:  up to 3min post
    (time(14, 3),  time(14, 15)),   # FOMC:         up to 12min post
]
NEWS_HARD_POST_MINUTES = 3


# ═══════════════════════════════════════════════════════════
# WORKER GUARDS CLASS
# Instantiated once per worker with worker-specific paths.
# All state files are prefixed by worker_name — zero cross-talk.
# ═══════════════════════════════════════════════════════════

class WorkerGuards:
    """
    All production guards scoped to a single worker.
    Pass worker_name and the paths for this worker's state directory.
    Each guard is independently tunable per worker via constructor args.
    """

    def __init__(
        self,
        worker_name:              str,
        log_dir:                  str,
        symbol:                   str,
        env_file:                 str,
        circuit_breaker_losses:   int   = 2,
        cb_magnitude_ratio:       float = 2.5,
        cb_max_losses:            int   = 6,
        spread_z_threshold:       float = 0.5,
        spread_z_soft:            float = 1.0,
        spread_history_size:      int   = 20,
        daily_dd_limit:           float = -0.04,
        total_dd_limit:           float = -0.08,
        trailing_dd_limit:        float = 0.04,
    ):
        self.worker_name            = worker_name
        self.symbol                 = symbol
        self.env_file               = env_file
        self.circuit_breaker_losses = circuit_breaker_losses
        self.cb_magnitude_ratio     = cb_magnitude_ratio
        self.cb_max_losses          = cb_max_losses
        self.spread_z_threshold     = spread_z_threshold
        self.spread_z_soft          = spread_z_soft
        self.spread_history_size    = spread_history_size
        self.daily_dd_limit         = daily_dd_limit
        self.total_dd_limit         = total_dd_limit
        self.trailing_dd_limit      = trailing_dd_limit

        base = Path(log_dir)
        base.mkdir(parents=True, exist_ok=True)

        # All state files prefixed by worker_name — fully isolated
        self.day_lock_path      = base / f"{worker_name}_day_lock.json"
        self.loss_log_path      = base / f"{worker_name}_loss_log.jsonl"
        self.security_state_path= base / f"{worker_name}_security_state.json"
        self.spread_cache_path  = base / f"{worker_name}_spread_cache.json"

        self.logger = make_logger(worker_name, log_dir)

        # Spread history — loaded from disk so Z-score survives cron gaps
        self._spread_history: list[float] = []
        self._current_spread: Optional[float] = None
        self._max_spread_pct = 0.0005   # cold-start fallback cap (5bp)
        try:
            if self.spread_cache_path.exists():
                data = json.loads(self.spread_cache_path.read_text())
                self._spread_history = data.get("history", [])[-spread_history_size:]
        except Exception:
            pass

    def log(self, msg: str) -> None:
        self.logger.info(msg)

    # ── [A] News Embargo ───────────────────────────────────────────────────────

    def is_embargo_active(self) -> bool:
        """Two-phase embargo. Hard = always blocked. Soft = blocked if spread elevated."""
        t = datetime.now(ET).time()
        if any(s <= t <= e for s, e in _EMBARGO_HARD_ET):
            self.log("[EMBARGO-HARD] News blackout active — no entry")
            return True
        if any(s <= t <= e for s, e in _EMBARGO_SOFT_ET):
            spread_normal = self._check_spread_gate(soft_mode=True)
            if not spread_normal:
                self.log("[EMBARGO-SOFT] Post-news spread still elevated — window held")
                return True
            self.log("[EMBARGO-SOFT] Spread normalized — window cleared early")
        return False

    # ── [B] Day-lock padlock + circuit breaker ─────────────────────────────────

    def is_day_locked(self) -> bool:
        try:
            if not self.day_lock_path.exists():
                return False
            lock = json.loads(self.day_lock_path.read_text())
            if lock.get("locked_date") == datetime.now().date().isoformat():
                self.log(f"[PADLOCK] Circuit breaker active — locked at {lock.get('locked_at')} "
                         f"| Reason: {lock.get('reason')} | Unlock: {lock.get('unlock_after')}")
                return True
        except Exception as e:
            self.log(f"[WARN] Day-lock file unreadable ({e}) — allowing execution")
        return False

    def engage_day_lock(self, reason: str, loss_count: int) -> None:
        today     = datetime.now()
        next_open = (today.date() + timedelta(days=1)).isoformat() + " 09:30:00 ET"
        lock = {
            "locked_date":  today.date().isoformat(),
            "locked_at":    today.isoformat(),
            "reason":       reason,
            "losses_today": loss_count,
            "unlock_after": next_open,
        }
        try:
            self.day_lock_path.write_text(json.dumps(lock, indent=2))
            self.log(f"[PADLOCK] Day-lock engaged — {reason}")
            send_telegram(
                f"🔒 <b>{self.worker_name.upper()} CIRCUIT BREAKER</b>\n"
                f"Reason: {reason}\n"
                f"Losses: {loss_count}\n"
                f"Locked until: {next_open}",
                self.logger,
            )
        except Exception as e:
            self.log(f"[WARN] Could not write day-lock: {e}")

    def record_loss(self, pnl: float, reason: str = "") -> int:
        today = datetime.now().date().isoformat()
        entry = {"date": today, "ts": datetime.now().isoformat(),
                 "pnl": round(pnl, 2), "reason": reason}
        try:
            with open(self.loss_log_path, "a") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception as e:
            self.log(f"[WARN] Loss log write failed: {e}")
        total = self.count_losses_today_local()
        self.log(f"[LOSS LOG] Recorded ${pnl:+.2f} | Today: {total}/{self.circuit_breaker_losses}")
        return total

    def record_win(self, pnl: float) -> None:
        today = datetime.now().date().isoformat()
        entry = {"date": today, "ts": datetime.now().isoformat(),
                 "pnl": round(pnl, 2), "type": "win"}
        try:
            with open(self.loss_log_path, "a") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception:
            pass

    def count_losses_today_local(self) -> int:
        """Local append-log count — used for magnitude gate + audit."""
        today = datetime.now().date().isoformat()
        try:
            if not self.loss_log_path.exists():
                return 0
            count = 0
            with open(self.loss_log_path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        e = json.loads(line)
                        if e.get("date") == today and e.get("pnl", 0) < 0:
                            count += 1
                    except json.JSONDecodeError:
                        continue
            return count
        except Exception:
            return 0

    # ── [D] Broker-first loss count ────────────────────────────────────────────

    def _et_session_start_utc(self) -> datetime:
        et_now      = datetime.now(ET)
        et_midnight = ET.fromutc(
            datetime.combine(et_now.date(), time(0, 0)).replace(tzinfo=ET)
            .astimezone(timezone.utc).replace(tzinfo=None).replace(tzinfo=timezone.utc)
        )
        # Simpler: just subtract hours to midnight ET
        now_utc = datetime.now(timezone.utc)
        et_now2 = now_utc.astimezone(ET)
        midnight_et = ET.fromutc(
            datetime(et_now2.year, et_now2.month, et_now2.day,
                     0, 0, 0, tzinfo=ET).astimezone(timezone.utc)
        )
        return midnight_et

    def count_losses_today_broker(self, client) -> int:
        """
        Authoritative broker count — reads Alpaca order fills directly.
        Fails CLOSED (propagates) so caller halts on API error rather than
        silently assuming 0 losses.
        """
        from alpaca.trading.requests import GetOrdersRequest
        from alpaca.trading.enums    import QueryOrderStatus, OrderSide

        since  = self._et_session_start_utc()
        orders = client.get_orders(filter=GetOrdersRequest(
            status  = QueryOrderStatus.ALL,
            after   = since,
            symbols = [self.symbol],
            limit   = 500,
        ))
        fills = [o for o in orders
                 if o.filled_at and o.filled_qty and float(o.filled_qty) > 0
                 and o.filled_avg_price]
        fills.sort(key=lambda o: o.filled_at)

        trades: list[float] = []
        pos_qty = 0.0
        cash    = 0.0
        for o in fills:
            qty    = float(o.filled_qty)
            px     = float(o.filled_avg_price)
            signed = qty if o.side == OrderSide.BUY else -qty
            cash  -= signed * px
            pos_qty += signed
            if abs(pos_qty) < 1e-9:
                trades.append(cash)
                cash = 0.0

        return sum(1 for pnl in trades if pnl < 0)

    def check_circuit_breaker(self, client) -> bool:
        """
        Returns True if the circuit breaker should engage.
        Checks BOTH broker count (safety gate) and magnitude gate.
        """
        try:
            broker_losses = self.count_losses_today_broker(client)
        except Exception as e:
            self.log(f"[CB] Broker loss count failed ({e}) — using local count")
            broker_losses = self.count_losses_today_local()

        if broker_losses >= self.cb_max_losses:
            self.log(f"[CB] Hard ceiling {broker_losses} >= {self.cb_max_losses} — LOCK")
            return True

        if broker_losses < self.circuit_breaker_losses:
            return False

        # Magnitude gate
        local_losses = []
        try:
            today = datetime.now().date().isoformat()
            if self.loss_log_path.exists():
                with open(self.loss_log_path) as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            e = json.loads(line)
                            if e.get("date") == today and e.get("pnl", 0) < 0:
                                local_losses.append(abs(float(e["pnl"])))
                        except Exception:
                            continue
        except Exception:
            pass

        if not local_losses:
            return True   # can't assess magnitude — lock to be safe

        avg_win = self._get_avg_win()
        total_damage = sum(local_losses[-self.circuit_breaker_losses:])
        threshold    = avg_win * self.cb_magnitude_ratio

        if total_damage > threshold:
            self.log(f"[CB-MAGNITUDE] TRIPS — damage ${total_damage:.2f} > "
                     f"threshold ${threshold:.2f} (avg_win=${avg_win:.2f}×{self.cb_magnitude_ratio})")
            return True

        self.log(f"[CB-MAGNITUDE] Count hit but magnitude OK — "
                 f"${total_damage:.2f} < ${threshold:.2f} — noise, continuing")
        return False

    def _get_avg_win(self) -> float:
        """Average win size from local log. Defaults to $50 if no wins yet."""
        try:
            wins = []
            if self.loss_log_path.exists():
                with open(self.loss_log_path) as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            e = json.loads(line)
                            if e.get("pnl", 0) > 0:
                                wins.append(float(e["pnl"]))
                        except Exception:
                            continue
            return (sum(wins) / len(wins)) if wins else 50.0
        except Exception:
            return 50.0

    # ── [C] Spread smoke detector ──────────────────────────────────────────────

    def refresh_spread_baseline(self, client) -> None:
        """
        Fetches NBBO once per tick. Persists to disk so Z-score baseline
        survives cron gaps. Call this BEFORE is_embargo_active() and
        check_spread_gate() in main().
        """
        try:
            from alpaca.data.historical import StockHistoricalDataClient
            from alpaca.data.requests   import StockLatestQuoteRequest
            key, secret = load_credentials(self.env_file)
            dc = StockHistoricalDataClient(key, secret)
            q  = dc.get_stock_latest_quote(
                StockLatestQuoteRequest(symbol_or_symbols=self.symbol)
            )[self.symbol]
            if q.ask_price <= 0:
                self.log("[BASELINE] Market closed — no NBBO (ask=0)")
                return
            spread_pct = (q.ask_price - q.bid_price) / q.ask_price
            self._current_spread = spread_pct
            self._spread_history.append(spread_pct)
            if len(self._spread_history) > self.spread_history_size:
                self._spread_history.pop(0)
            self.spread_cache_path.write_text(json.dumps(
                {"history": self._spread_history, "ts": datetime.now().isoformat()}
            ))
            self.log(f"[BASELINE] Spread {spread_pct:.5%} | "
                     f"history={len(self._spread_history)}/{self.spread_history_size}")
        except Exception as e:
            self.log(f"[WARN] Spread baseline refresh failed ({e}) — using stale history")

    def _check_spread_gate(self, soft_mode: bool = False) -> bool:
        try:
            if self._current_spread is None:
                return True   # no data yet — fail-open
            threshold = self.spread_z_soft if soft_mode else self.spread_z_threshold
            if len(self._spread_history) >= 5:
                baseline = statistics.median(self._spread_history[:-1])
                spread_z = ((self._current_spread - baseline) / baseline
                            if baseline > 0 else 0.0)
                if spread_z > threshold:
                    self.log(f"[SMOKE DETECTOR] Spread {self._current_spread:.5%} | "
                             f"Z={spread_z:.2f} > {threshold} — aborted")
                    return False
                self.log(f"[SMOKE DETECTOR] Spread {self._current_spread:.5%} | "
                         f"Z={spread_z:.2f} | baseline {baseline:.5%} — cleared")
            else:
                if self._current_spread > self._max_spread_pct:
                    self.log(f"[SMOKE DETECTOR] Cold-start: "
                             f"{self._current_spread:.5%} > {self._max_spread_pct:.5%} — aborted")
                    return False
                self.log(f"[SMOKE DETECTOR] Cold-start: {self._current_spread:.5%} — "
                         f"cleared ({len(self._spread_history)}/5 samples)")
            return True
        except Exception as e:
            self.log(f"[WARN] Spread gate check failed ({e}) — fail-open")
            return True

    def check_spread_gate(self) -> bool:
        return self._check_spread_gate(soft_mode=False)

    # ── [E] Orphan reaper ──────────────────────────────────────────────────────

    def orphan_reaper(self, client) -> None:
        """
        Cancels ghost SL orders whose TP partner already filled.
        Wrapped in try/except — never blocks a live trade.
        """
        try:
            from alpaca.trading.requests import GetOrdersRequest
            from alpaca.trading.enums    import QueryOrderStatus

            open_orders = client.get_orders(GetOrdersRequest(
                status  = QueryOrderStatus.OPEN,
                symbols = [self.symbol],
                limit   = 20,
            ))
            sl_orders = [o for o in open_orders
                         if getattr(o, "type", None)
                         and o.type.value in ("stop", "stop_limit")]
            tp_orders = [o for o in open_orders
                         if getattr(o, "type", None)
                         and o.type.value == "limit"]

            if sl_orders and not tp_orders:
                self.log(f"[ORPHAN REAPER] Ghost SL detected — {len(sl_orders)} orphan(s)")
                for sl in sl_orders:
                    try:
                        client.cancel_order_by_id(sl.id)
                        self.log(f"[ORPHAN REAPER] Cancelled ghost SL id={sl.id}")
                    except Exception as e:
                        self.log(f"[WARN] Could not cancel ghost SL {sl.id}: {e}")
                try:
                    client.close_position(self.symbol)
                    self.log("[ORPHAN REAPER] Position flattened")
                except Exception:
                    self.log("[ORPHAN REAPER] Position already flat")
                send_telegram(
                    f"👻 <b>{self.worker_name.upper()} ORPHAN REAPER</b>\n"
                    f"Ghost SL cancelled — TP already filled.",
                    self.logger,
                )
        except Exception as e:
            self.log(f"[WARN] Orphan Reaper failed ({e}) — continuing")

    # ── [F] Bouncer + Decoy Detector ──────────────────────────────────────────

    def is_signal_valid(self, side: str, bar_ts: str) -> bool:
        """
        Guard 1 — Bouncer: rejects signals from an already-processed bar.
        Guard 2 — Decoy Detector: SHA256 dedup — same signal never fires twice.
        """
        try:
            state = {"last_bar_ts": "", "seen_hashes": []}
            if self.security_state_path.exists():
                state = json.loads(self.security_state_path.read_text())

            if bar_ts and bar_ts <= state["last_bar_ts"]:
                self.log(f"[BOUNCER] Bar {bar_ts} already processed "
                         f"(last: {state['last_bar_ts']}) — dropped")
                return False

            sig_hash = hashlib.sha256(
                f"{self.symbol}|{side}|{bar_ts}".encode()
            ).hexdigest()[:16]
            if sig_hash in state["seen_hashes"]:
                self.log(f"[DECOY DETECTOR] Duplicate hash {sig_hash} — dropped")
                return False

            state["last_bar_ts"] = bar_ts
            state["seen_hashes"].append(sig_hash)
            if len(state["seen_hashes"]) > 100:
                state["seen_hashes"].pop(0)
            self.security_state_path.write_text(json.dumps(state))
            self.log(f"[SIGNAL VERIFIED] {side} bar={bar_ts} hash={sig_hash} — cleared")
            return True
        except Exception as e:
            self.log(f"[WARN] Signal validation failed ({e}) — allowing")
            return True

    # ── [G] DD checks (fail-closed) ────────────────────────────────────────────

    def get_equity(self, client) -> float:
        return float(client.get_account().equity)

    def check_total_dd(self, client) -> float:
        """Fail-CLOSED: propagates on API error so caller halts."""
        from alpaca.trading.requests import GetPortfolioHistoryRequest
        hist    = client.get_portfolio_history(GetPortfolioHistoryRequest(period="1A"))
        eq_list = [e for e in hist.equity if e is not None and e > 0]
        current = self.get_equity(client)
        if not eq_list:
            return 0.0
        peak = max(eq_list + [current])
        return (current - peak) / peak if peak > 0 else 0.0

    def check_trailing_dd(self, client) -> float:
        """Intraday trailing DD from broker portfolio history. Fail-CLOSED."""
        from alpaca.trading.requests import GetPortfolioHistoryRequest
        hist = client.get_portfolio_history(GetPortfolioHistoryRequest(
            period="1D", timeframe="5Min", extended_hours=True))
        eqs     = [float(e) for e in hist.equity if e is not None and float(e) > 0]
        current = self.get_equity(client)
        peak    = max(eqs + [current]) if eqs else current
        return (current - peak) / peak if peak > 0 else 0.0

    def run_risk_preamble(self, client) -> bool:
        """
        Run all DD checks at the top of main(). Returns False if any limit
        is breached — caller should return immediately.
        Fail-CLOSED: if any metric can't be read, halts rather than assuming safe.
        """
        try:
            total_dd    = self.check_total_dd(client)
            trailing_dd = self.check_trailing_dd(client)
        except Exception as e:
            self.log(f"[HALT] Risk metrics unavailable — standing down: {e}")
            send_telegram(f"⛔ {self.worker_name.upper()} HALT: risk check failed — {e}",
                          self.logger)
            return False

        if total_dd <= self.total_dd_limit:
            self.log(f"[HALT] Total DD {total_dd:.2%} breaches {self.total_dd_limit:.0%}")
            send_telegram(f"⛔ {self.worker_name.upper()} Total DD {total_dd:.2%}",
                          self.logger)
            return False

        if trailing_dd <= -self.trailing_dd_limit:
            self.log(f"[HALT] Trailing DD {trailing_dd:.2%} from intraday peak")
            send_telegram(f"⛔ {self.worker_name.upper()} Trailing DD {trailing_dd:.2%}",
                          self.logger)
            return False

        return True
