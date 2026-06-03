#!/usr/bin/env bash
# setup_vps.sh — One-command ICT bot setup on Ubuntu 22.04 LTS
#
# Usage (as root on a fresh DigitalOcean/Linode/Vultr droplet):
#   bash setup_vps.sh
#
# What it does:
#   1. Installs Python 3.11, git
#   2. Clones kronos-trading repo
#   3. Creates virtualenv + installs deps
#   4. Creates .env file (you fill in credentials once)
#   5. Installs cron jobs for London + NY AM kill zones
#   6. Runs a test to confirm everything works

set -euo pipefail

REPO="https://github.com/fakevadimusa-ui/kronos-trading.git"
APP_DIR="$HOME/kronos-trading"
LOG_DIR="$HOME/logs/ict"

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

step() { echo -e "${GREEN}==>${NC} $1"; }
warn() { echo -e "${YELLOW}[!]${NC} $1"; }

# ── 1. System packages ────────────────────────────────────────────────────────
step "Installing system packages"
apt-get update -qq
apt-get install -y -qq python3.11 python3.11-venv python3-pip git curl

# ── 2. Clone or update repo ───────────────────────────────────────────────────
step "Cloning/updating kronos-trading"
if [ -d "$APP_DIR/.git" ]; then
    git -C "$APP_DIR" pull --quiet
    echo "  Updated existing repo"
else
    git clone --quiet "$REPO" "$APP_DIR"
    echo "  Cloned fresh"
fi

# ── 3. Virtualenv + dependencies ─────────────────────────────────────────────
step "Setting up Python virtualenv"
python3.11 -m venv "$APP_DIR/venv"
# shellcheck disable=SC1091
source "$APP_DIR/venv/bin/activate"
pip install -q --upgrade pip
pip install -q -r "$APP_DIR/ict_requirements.txt"
deactivate

# ── 4. Environment file ───────────────────────────────────────────────────────
step "Creating .env file"
ENV_FILE="$APP_DIR/.env"
if [ ! -f "$ENV_FILE" ]; then
    cat > "$ENV_FILE" <<'EOF'
ALPACA_KEY=REPLACE_WITH_YOUR_ALPACA_KEY
ALPACA_SECRET=REPLACE_WITH_YOUR_ALPACA_SECRET
TELEGRAM_TOKEN=REPLACE_WITH_YOUR_TELEGRAM_TOKEN
TELEGRAM_CHAT_ID=REPLACE_WITH_YOUR_TELEGRAM_CHAT_ID
EOF
    chmod 600 "$ENV_FILE"
    warn "Fill in your credentials: nano $ENV_FILE"
else
    echo "  .env already exists — skipping"
fi

# ── 5. Log directory ──────────────────────────────────────────────────────────
step "Creating log directory"
mkdir -p "$LOG_DIR"

# ── 6. Make cron wrapper executable ──────────────────────────────────────────
chmod +x "$APP_DIR/ict_cron.sh"

# ── 7. Install cron jobs ──────────────────────────────────────────────────────
# Cron runs in UTC. Kill zones in ET (UTC-4 EDT / UTC-5 EST).
# Using wide windows so DST transitions never miss a session.
#
#   London kill zone  3–5 AM ET  →  7–10 AM UTC  (7-9 EDT, 8-10 EST)
#   NY AM kill zone   7:30–10 AM ET → 11:30 AM–3 PM UTC (11:30-14 EDT, 12:30-15 EST)
#
# The ICT model's is_kill_zone() check handles exact timing — off-window
# runs exit in <1 second with "Not in kill zone".

step "Installing cron jobs"
CRON_MARKER="# ICT-BOT-MANAGED"
CRON_BLOCK="$CRON_MARKER
# London kill zone — 3-5 AM ET (7-10 AM UTC covers EDT+EST)
*/5 7,8,9 * * 1-5 $APP_DIR/ict_cron.sh >> $LOG_DIR/ict.log 2>&1
0 10 * * 1-5 $APP_DIR/ict_cron.sh >> $LOG_DIR/ict.log 2>&1
# NY AM kill zone — 8:30-11 AM ET (12:30-15 UTC EDT / 13:30-16 UTC EST)
30,35,40,45,50,55 12 * * 1-5 $APP_DIR/ict_cron.sh >> $LOG_DIR/ict.log 2>&1
*/5 13,14 * * 1-5 $APP_DIR/ict_cron.sh >> $LOG_DIR/ict.log 2>&1
0,5,10,15,20,25,30 15 * * 1-5 $APP_DIR/ict_cron.sh >> $LOG_DIR/ict.log 2>&1
# Silver Bullet — 1:30-4 PM ET (17:30-20 UTC EDT / 18:30-21 UTC EST)
30,35,40,45,50,55 17 * * 1-5 $APP_DIR/ict_cron.sh >> $LOG_DIR/ict.log 2>&1
*/5 18,19 * * 1-5 $APP_DIR/ict_cron.sh >> $LOG_DIR/ict.log 2>&1
*/5 20 * * 1-5 $APP_DIR/ict_cron.sh >> $LOG_DIR/ict.log 2>&1
0 21 * * 1-5 $APP_DIR/ict_cron.sh >> $LOG_DIR/ict.log 2>&1"

# Remove any existing ICT block, add fresh
(crontab -l 2>/dev/null | grep -v "$CRON_MARKER" | grep -v "ict_cron"; echo "$CRON_BLOCK") | crontab -
echo "  Cron jobs installed"

# ── 8. Summary ────────────────────────────────────────────────────────────────
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  ICT bot setup complete"
echo ""
echo "  Next steps:"
echo "  1. Add your credentials:  nano $ENV_FILE"
echo "  2. Test a manual run:     $APP_DIR/ict_cron.sh"
echo "  3. Watch live logs:       tail -f $LOG_DIR/ict.log"
echo ""
echo "  Cron fires at:"
echo "    London  — every 15 min, 7–10 AM UTC (3–5 AM ET)"
echo "    NY AM   — every 15 min, 11:30 AM–3 PM UTC (7:30–10 AM ET)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
