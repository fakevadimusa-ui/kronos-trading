# Liquidity Trap — Paper Test Setup

## Files

| File | Purpose |
|---|---|
| `worker_liquidity_trap_paper.py` | Paper bot — runs every 5min during 9:30–10:30 ET |
| `research/liquidity_trap/forward_tracker.py` | 50-trade validation dashboard |
| `research/liquidity_trap/results_summary.md` | Phase 10 backtest research report |
| `research/liquidity_trap/red_team_report.md` | Skeptical analysis |

## Run manually (any time)

```bash
cd ~/kronos-trading
python3 worker_liquidity_trap_paper.py
```
Outside 9:25–10:35 ET it exits immediately. During that window it processes signals.

## Check the forward test tracker

```bash
cd ~/kronos-trading
python3 research/liquidity_trap/forward_tracker.py
```

## Add paper-only cron on VPS (PAPER ONLY — no live orders)

```
# Liquidity Trap PAPER BOT — runs every 5min during NY open window (UTC 13:25–14:35)
*/5 13,14 * * 1-5  cd /root/kronos-trading && venv/bin/python3 worker_liquidity_trap_paper.py >> /root/logs/liquidity_trap/paper.log 2>&1
```

Add with: `crontab -e` on the VPS

## Log files (VPS: /root/logs/liquidity_trap/, Mac: ~/logs/liquidity_trap/)

| File | Content |
|---|---|
| `paper.log` | Timestamped run-by-run log |
| `signals.jsonl` | Completed paper trades (one JSON per line) |
| `daily_summary.jsonl` | Per-day summary |
| `state.json` | Active trade state (reset each day) |

## 50-trade gate

**Do not make any live trading decision before 50 forward-test trades.**

After 50 trades, run `forward_tracker.py` to get the official verdict.

## Safety

- Zero live orders ever
- No account credentials used
- No changes to News Straddle, ICT NY, or challenge_mode.py
- All logs prefixed with `[PAPER]`
- `paper_only: true` field on every logged trade
