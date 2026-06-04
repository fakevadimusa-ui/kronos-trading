#!/usr/bin/env bash
# ict_cron.sh — Cron entry point for ICT bot
# Loads .env, activates virtualenv, runs trader, rotates logs.

set -euo pipefail

APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="$HOME/logs/ict"
LOG_FILE="$LOG_DIR/ict.log"
MAX_LOG_BYTES=5242880  # 5 MB

# Load credentials from .env
if [ -f "$APP_DIR/.env" ]; then
    set -a
    # shellcheck disable=SC1091
    source "$APP_DIR/.env"
    set +a
fi

# Rotate log if over 5 MB
if [ -f "$LOG_FILE" ] && [ "$(stat -c%s "$LOG_FILE" 2>/dev/null || echo 0)" -gt "$MAX_LOG_BYTES" ]; then
    mv "$LOG_FILE" "$LOG_FILE.1"
fi

mkdir -p "$LOG_DIR"

# Activate virtualenv and run
source "$APP_DIR/venv/bin/activate"
cd "$APP_DIR"
python3 ict_alpaca_trader.py
