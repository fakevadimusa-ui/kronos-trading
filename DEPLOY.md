# DEPLOY.md — ICT Trading System Runbook

**Single source of truth:** Git `main`.
**Sole execution host:** the VPS (`root@144.202.3.195`, crontab).

---

## 🥇 The Golden Rule

**Never hand-edit the VPS.** Every change flows one way only:

```
edit on Mac  →  commit  →  git push origin main  →  git reset --hard on the VPS
```

If the VPS code ever drifts from Git, **Git wins.** Hand-edits on the server are how
this system once ended up with ~1500 lines of uncommitted production code stranded on
a single machine. Never again — the server is a *replica*, not a workspace.

---

## 🚀 Rollout (deploy a change)

```bash
# 1. Publish the truth (from the Mac repo, on main):
git push origin main

# 2. Adopt the truth on the VPS (overwrites TRACKED files only):
ssh root@144.202.3.195 'cd ~/kronos-trading && git fetch origin && git reset --hard origin/main'
```

`git reset --hard` touches only **tracked** files. It does **not** remove untracked
files, so the VPS's `.env` (credentials), `venv/`, and `*.backup` files are preserved.

---

## 🔥 Smoke Test (run after every deploy)

```bash
# A) Compiles + imports cleanly on the VPS:
ssh root@144.202.3.195 'cd ~/kronos-trading && \
  ./venv/bin/python3 -m py_compile ict_alpaca_trader.py && \
  ./venv/bin/python3 -c "import ict_alpaca_trader; print(\"import OK\")"'

# B) No-trade check — run OUTSIDE a kill zone. It MUST log "Not in kill zone" /
#    "standing by" and place NO order:
ssh root@144.202.3.195 'cd ~/kronos-trading && ./venv/bin/python3 ict_alpaca_trader.py 2>&1 | tail -25'
```

Kill zones are London **3–5am ET** and NY **8:30–11am ET**. Outside those windows the
bot cannot enter, so step B is a safe dry run at any other time.

---

## 🏛️ Architecture — one hunter, one truth

| Component | Role |
|---|---|
| **Git `main`** | 🟢 The single source of truth. All production code lives here. |
| **VPS** (`144.202.3.195`, crontab) | 🟢 The **SOLE** ICT hunter. Pulls from Git; never hand-edited. |
| **GitHub Actions `ict.yml`** | ⛔ ICT schedule **DISABLED** (manual `workflow_dispatch` only) — prevents a 2nd bot on the same account. |
| **Alpaca account** | One account, one hunter. `SPY` is ICT-exclusive (Kronos uses NVDA/USO). |

---

## 🛡️ Safety systems in the trader (do not remove)

- **3-Knob Execution Matrix:** News Embargo (A) · Magnitude Circuit Breaker (B) · Z-Score Spread Gate (C)
- **Broker-first risk truth:** daily-loss count + trailing drawdown reconstructed from Alpaca order history (not local files) — survives any host, cannot silently desync
- **Fail-closed risk checks:** an unreadable metric HALTS the run rather than assuming "safe"
- **Idempotent orders:** deterministic `client_order_id` from the signal bar → Alpaca rejects duplicates server-side
- **Padlock + orphan reaper:** day-lock `sys.exit(0)` and ghost-order sweep run before any new entry
