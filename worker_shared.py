"""
worker_shared.py — Shared utilities for all 3 workers.
=======================================================
STRICT RULES:
  - This file provides ONLY: logging factory, credential loading,
    Telegram notification, and flock-based run lock.
  - It contains ZERO strategy logic, ZERO risk state, ZERO position
    management. Every worker runs in complete isolation.
  - Workers load their OWN .env file via load_dotenv() before importing
    this module so environment variables are set by the time these
    helpers read them.
"""

import fcntl
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import requests
from dotenv import load_dotenv

ET = ZoneInfo("America/New_York")


# ── Logging ───────────────────────────────────────────────────────────────────

def make_logger(worker_name: str, log_dir: str) -> logging.Logger:
    """
    Creates a dedicated logger for `worker_name`.
    Each worker writes to its own daily log file under log_dir.
    No handlers are shared between workers.
    """
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    today    = datetime.now(ET).strftime("%Y%m%d")
    log_path = Path(log_dir) / f"{worker_name}_{today}.log"

    logger = logging.getLogger(worker_name)
    if logger.handlers:
        return logger          # already set up (reused within same process)

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


# ── Credentials ───────────────────────────────────────────────────────────────

def load_credentials(env_file: str | None = None) -> tuple[str, str]:
    """
    Load ALPACA_KEY / ALPACA_SECRET.
    Priority: env_file (dotenv) → os.environ → JSON fallback config.
    Each worker passes its OWN .env path so they can use different accounts.
    """
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

    raise RuntimeError("No Alpaca credentials found. Set ALPACA_KEY / ALPACA_SECRET.")


# ── Telegram ──────────────────────────────────────────────────────────────────

def send_telegram(msg: str, logger: logging.Logger | None = None) -> None:
    """
    Sends a Telegram message. Uses TELEGRAM_TOKEN / TELEGRAM_CHAT_ID from env.
    Silently skips if credentials are absent (no crash).
    """
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


# ── Run lock ──────────────────────────────────────────────────────────────────

class RunLock:
    """
    Per-worker flock-based run lock.
    Prevents two cron ticks of the SAME worker overlapping.
    Workers use separate lock files — they never block each other.

    Usage:
        lock = RunLock("/tmp/worker_ny.lock", logger)
        if not lock.acquire():
            sys.exit(0)          # prior run still active — skip tick
        try:
            ... do work ...
        finally:
            lock.release()
    """

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
            self.logger.warning(
                f"Run lock held by another process — skipping tick. "
                f"Lock file: {self.path}"
            )
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
