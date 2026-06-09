#!/usr/bin/env python3
"""
challenge_mode.py — Prop Firm Challenge Protection Framework
=============================================================
Centralizes all prop-firm-specific risk management logic.

Three modes:
  SAFE:       0.3% risk/trade, 0.8% daily limit, DLL buffer 50%
  NORMAL:     0.5% risk/trade, 1.2% daily limit, DLL buffer 65%
  AGGRESSIVE: 0.8% risk/trade, 1.6% daily limit, DLL buffer 88%

Auto-mode switching:
  < 33% of target reached  →  current mode (no change)
  ≥ 33% of target reached  →  step down one level (lock in gains)
  ≥ 75% of target reached  →  SAFE mode (protect payout)

Kill switches (all automatic, no human required):
  [K1] Rolling 20-trade WR < 42%      →  suspend strategy for 5 trading days
  [K2] Daily loss ≥ DLL × mode_pct   →  halt all trading today
  [K3] Total DD ≥ challenge max - 20% →  switch to SAFE mode
  [K4] VIX > 25                       →  ICT suspended, straddle allowed
  [K5] Slippage > threshold (10 trades) →  alert + reduce size 50%
  [K6] 3 consecutive days net-negative →  switch to SAFE mode for 5 days

State is persisted to disk — survives VPS reboots.
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import Literal, Optional
from zoneinfo import ZoneInfo

ET = ZoneInfo("America/New_York")

MODE = Literal["SAFE", "NORMAL", "AGGRESSIVE"]

# ── Challenge parameters ───────────────────────────────────────────────────────

CHALLENGE_CONFIGS: dict[str, dict] = {
    "SAFE": {
        "risk_pct":         0.003,   # 0.3% of account per trade
        "daily_loss_pct":   0.008,   # halt at 0.8% daily loss (vs Lucid's 1.8%)
        "dll_buffer_pct":   0.50,    # use only 50% of prop firm's stated DLL
        "max_contracts_es": 2,
        "max_contracts_mes": 5,
        "description":      "Capital protection — 2× safety margin on all limits",
    },
    "NORMAL": {
        "risk_pct":         0.005,   # 0.5% of account per trade
        "daily_loss_pct":   0.012,   # halt at 1.2% daily loss (vs Lucid's 1.8%)
        "dll_buffer_pct":   0.65,
        "max_contracts_es": 3,
        "max_contracts_mes": 10,
        "description":      "Standard operation — balanced risk/return",
    },
    "AGGRESSIVE": {
        "risk_pct":         0.008,   # 0.8% of account per trade
        "daily_loss_pct":   0.016,   # halt at 1.6% daily loss (vs Lucid's 1.8%)
        "dll_buffer_pct":   0.88,
        "max_contracts_es": 4,
        "max_contracts_mes": 20,
        "description":      "Full sizing — only when losing is not a concern",
    },
}

# ── Default challenge parameters (Lucid $100K) ────────────────────────────────

CHALLENGE_TARGET_PCT   = 0.06     # 6% target (+$6,000 on $100K)
CHALLENGE_MAX_LOSS_PCT = 0.03     # 3% max loss (-$3,000 on $100K)
CHALLENGE_DLL_PCT      = 0.018    # 1.8% daily loss limit (-$1,800 on $100K)
CHALLENGE_EQUITY       = 100_000  # starting equity

# ── Kill switch thresholds ────────────────────────────────────────────────────

KS_ROLLING_WR_MIN        = 0.42   # K1: suspend if rolling WR drops below
KS_ROLLING_WR_WINDOW     = 20     # trades to evaluate rolling WR (8 minimum before K1 active)
KS_ROLLING_WR_MIN_TRADES = 8      # K1 inactive below this trade count
KS_SUSPEND_DAYS          = 5      # days to suspend after K1 or K6 trigger
KS_CONSECUTIVE_LOSS_DAYS = 3      # K6: days before forced SAFE mode
KS_SLIPPAGE_THRESHOLD    = 1.5    # K5: pts — alert when avg slippage exceeds this
KS_SLIPPAGE_WINDOW       = 10     # K5: trades to evaluate average slippage

VIX_PAUSE_THRESHOLD    = 25.0    # K4: ICT suspended above this VIX
VIX_RESUME_THRESHOLD   = 20.0    # K4: ICT resumes below this VIX

# K7 — profit factor kill
KS_K7_PF_MIN            = 1.0    # pause if rolling PF drops below this
KS_K7_PF_WINDOW         = 30     # trades evaluated for rolling PF
KS_K7_MIN_TRADES        = 10     # K7 inactive below this trade count
KS_K7_PAUSE_DAYS        = 5      # days to pause after K7 trigger

# K8 — consecutive trade losses kill
KS_K8_CONSEC_LOSSES     = 5      # disable after this many consecutive losing trades
KS_K8_DISABLE_DAYS      = 10     # days to disable after K8 trigger

# K9 — realised volatility regime kill
KS_K9_VOL_MULT          = 2.0    # suspend if rolling daily P&L std doubles vs baseline
KS_K9_VOL_WINDOW        = 10     # trading days for rolling vol calculation
KS_K9_RISK_REDUCTION    = 0.50   # reduce sizing by this fraction when K9 fires

# ── State file ────────────────────────────────────────────────────────────────

_STATE_PATH = Path("/tmp/challenge_mode_state.json")
_TRADE_LOG_PATH = Path("/root/logs/challenge/trades.jsonl")


# ═══════════════════════════════════════════════════════════════════════════════
# STATE MANAGEMENT
# ═══════════════════════════════════════════════════════════════════════════════

def _load_state() -> dict:
    if _STATE_PATH.exists():
        try:
            return json.loads(_STATE_PATH.read_text())
        except Exception:
            pass
    return {
        "mode":               "NORMAL",
        "session_pnl":        0.0,
        "total_pnl":          0.0,
        "suspension_until":   None,
        "safe_mode_until":    None,
        "consecutive_loss_days": 0,
        "last_trading_date":  None,
        "daily_pnl":          {},   # {"2026-06-08": -1200.0, ...}
    }


def _save_state(state: dict) -> None:
    _STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = _STATE_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2, default=str))
    tmp.replace(_STATE_PATH)


def _log_trade(record: dict) -> None:
    _TRADE_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(_TRADE_LOG_PATH, "a") as f:
        f.write(json.dumps(record, default=str) + "\n")


def _load_trades(n: int = 100) -> list[dict]:
    if not _TRADE_LOG_PATH.exists():
        return []
    trades = []
    try:
        with open(_TRADE_LOG_PATH) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        trades.append(json.loads(line))
                    except Exception:
                        pass
    except Exception:
        pass
    return trades[-n:]


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN GUARD CLASS
# ═══════════════════════════════════════════════════════════════════════════════

class ChallengeGuard:
    """
    Single entrypoint for all prop-firm-challenge risk decisions.

    Typical usage in a worker's main():

        from challenge_mode import ChallengeGuard
        guard = ChallengeGuard(equity=100_000)

        if not guard.can_trade("ICT"):
            return

        contracts = guard.position_size("ES", risk_pts=2.5)
        ...
        guard.record_trade("ICT", pnl=+620, strategy="ICT_NY", slippage_pts=0.3)
    """

    def __init__(
        self,
        equity: float = CHALLENGE_EQUITY,
        target_pct: float = CHALLENGE_TARGET_PCT,
        max_loss_pct: float = CHALLENGE_MAX_LOSS_PCT,
        dll_pct: float = CHALLENGE_DLL_PCT,
    ):
        self.equity      = equity
        self.target      = equity * target_pct
        self.max_loss    = equity * max_loss_pct
        self.dll         = equity * dll_pct
        self._state      = _load_state()

    # ── Public API ─────────────────────────────────────────────────────────────

    @property
    def mode(self) -> str:
        return self._auto_mode()

    @property
    def config(self) -> dict:
        return CHALLENGE_CONFIGS[self.mode]

    def can_trade(self, strategy: str) -> tuple[bool, str]:
        """
        Master gate. Returns (allowed: bool, reason: str).
        Call this at the top of every worker tick.

        strategy: "ICT" or "STRADDLE"
        """
        state = self._state
        today = datetime.now(ET).date().isoformat()

        # K1 — rolling WR check
        if self._is_suspended():
            return False, f"[K1] Strategy suspended until {state.get('suspension_until')}"

        # K2 — daily loss limit
        daily_pnl = state.get("daily_pnl", {}).get(today, 0.0)
        dll_limit  = -(self.dll * self.config["dll_buffer_pct"])
        if daily_pnl <= dll_limit:
            return False, (f"[K2] Daily loss ${daily_pnl:,.0f} hit {self.config['dll_buffer_pct']:.0%} "
                           f"of DLL ${dll_limit:,.0f} — no more trades today")

        # K3 — total drawdown
        if state.get("total_pnl", 0) <= -self.max_loss * 0.8:
            return False, f"[K3] Total loss approaching max — trading frozen in SAFE mode only"

        # K4 — VIX gate (ICT only)
        if strategy == "ICT":
            vix = self._get_cached_vix()
            if vix and vix > VIX_PAUSE_THRESHOLD:
                return False, f"[K4] VIX {vix:.1f} > {VIX_PAUSE_THRESHOLD} — ICT suspended"

        # K6 — consecutive loss days: block ALL trading during forced SAFE period
        # (SAFE mode means reduced sizing, but K6 means full halt until recovery)
        if state.get("consecutive_loss_days", 0) >= KS_CONSECUTIVE_LOSS_DAYS:
            safe_until = state.get("safe_mode_until")
            if safe_until and datetime.now(ET).date().isoformat() <= safe_until:
                return False, f"[K6] {KS_CONSECUTIVE_LOSS_DAYS} consecutive loss days — halted until {safe_until}"

        # K7 — rolling profit factor
        k7_until = state.get("k7_pause_until")
        if k7_until and datetime.now(ET).date().isoformat() <= k7_until:
            return False, f"[K7] PF < {KS_K7_PF_MIN} over last {KS_K7_PF_WINDOW} trades — paused until {k7_until}"

        # K8 — consecutive trade losses
        k8_until = state.get("k8_disable_until")
        if k8_until and datetime.now(ET).date().isoformat() <= k8_until:
            return False, f"[K8] {KS_K8_CONSEC_LOSSES} consecutive losses — disabled until {k8_until}"

        return True, "OK"

    def position_size(
        self,
        instrument: str,  # "ES", "MES"
        risk_pts: float,
        account_equity: float | None = None,
    ) -> int:
        """
        Calculate position size based on current mode's risk budget.
        Returns number of contracts (always ≥ 1, never exceeds mode max).
        """
        equity = account_equity or self.equity
        risk_budget = equity * self.config["risk_pct"]
        cfg = self.config

        if instrument == "ES":
            dollars_per_pt = 50.0
            max_c = cfg["max_contracts_es"]
        elif instrument == "MES":
            dollars_per_pt = 5.0
            max_c = cfg["max_contracts_mes"]
        else:
            raise ValueError(f"Unknown instrument: {instrument}")

        raw = risk_budget / (risk_pts * dollars_per_pt)
        contracts = max(1, min(int(raw), max_c))

        # K9 — apply volatility reduction if active
        k9_reduction = self._state.get("k9_vol_reduction", 0.0)
        if k9_reduction > 0:
            contracts = max(1, int(contracts * (1.0 - k9_reduction)))

        return contracts

    def record_trade(
        self,
        strategy: str,
        pnl: float,
        slippage_pts: float = 0.0,
        win: bool | None = None,
        instrument: str = "ES",
    ) -> None:
        """
        Record a completed trade. Updates all internal state for kill switches.
        Call this after every trade exits (TP or SL).
        """
        today = datetime.now(ET).date().isoformat()
        state = self._state

        # Update P&L
        state["total_pnl"] = state.get("total_pnl", 0.0) + pnl
        daily_pnl = state.get("daily_pnl", {})
        daily_pnl[today] = daily_pnl.get(today, 0.0) + pnl
        state["daily_pnl"] = daily_pnl

        # Track consecutive loss days
        if pnl < 0 and today != state.get("last_trading_date"):
            state["consecutive_loss_days"] = state.get("consecutive_loss_days", 0) + 1
        elif pnl > 0:
            state["consecutive_loss_days"] = 0
        state["last_trading_date"] = today

        # Log trade
        record = {
            "ts":           datetime.now(ET).isoformat(),
            "date":         today,
            "strategy":     strategy,
            "pnl":          round(pnl, 2),
            "slippage_pts": round(slippage_pts, 2),
            "win":          (pnl > 0) if win is None else win,
            "mode":         self.mode,
            "instrument":   instrument,
        }
        _log_trade(record)

        # Track consecutive TRADE losses (K8 — different from K6 which counts days)
        if pnl < 0:
            state["consec_trade_losses"] = state.get("consec_trade_losses", 0) + 1
        else:
            state["consec_trade_losses"] = 0

        # K1 — check rolling WR after recording
        self._check_rolling_wr_kill(state)

        # K5 — slippage alert
        self._check_slippage_kill(state, slippage_pts)

        # K6 — check consecutive loss days
        if state.get("consecutive_loss_days", 0) >= KS_CONSECUTIVE_LOSS_DAYS:
            resume_date = (datetime.now(ET).date() + timedelta(days=KS_SUSPEND_DAYS)).isoformat()
            state["safe_mode_until"] = resume_date
            self._notify(
                f"[K6] {KS_CONSECUTIVE_LOSS_DAYS} consecutive loss days — "
                f"switching to SAFE mode until {resume_date}"
            )

        # K7 — rolling profit factor
        self._check_k7_pf_kill(state)

        # K8 — 5 consecutive trade losses
        if state.get("consec_trade_losses", 0) >= KS_K8_CONSEC_LOSSES:
            if not state.get("k8_disable_until"):
                resume = (datetime.now(ET).date() + timedelta(days=KS_K8_DISABLE_DAYS)).isoformat()
                state["k8_disable_until"] = resume
                self._notify(
                    f"[K8] {KS_K8_CONSEC_LOSSES} consecutive losing trades — "
                    f"strategy disabled until {resume}"
                )

        # K9 — realised volatility doubles
        self._check_k9_vol_kill(state)

        _save_state(state)
        self._state = state

    def progress_summary(self) -> dict:
        """Return a dict with challenge progress metrics."""
        state = self._state
        pnl = state.get("total_pnl", 0.0)
        pct_to_target = pnl / self.target if self.target else 0
        pct_of_max_loss = pnl / -self.max_loss if self.max_loss else 0
        today = datetime.now(ET).date().isoformat()
        daily_pnl = state.get("daily_pnl", {}).get(today, 0.0)

        today_s = datetime.now(ET).date().isoformat()
        k7_active = bool(state.get("k7_pause_until")) and today_s <= str(state.get("k7_pause_until", ""))
        k8_active = bool(state.get("k8_disable_until")) and today_s <= str(state.get("k8_disable_until", ""))
        k9_active = state.get("k9_vol_reduction", 0.0) > 0

        return {
            "mode":                  self.mode,
            "total_pnl":             round(pnl, 2),
            "target":                round(self.target, 2),
            "pct_to_target":         round(pct_to_target * 100, 1),
            "daily_pnl":             round(daily_pnl, 2),
            "dll_used_pct":          round(daily_pnl / -self.dll * 100, 1) if daily_pnl < 0 else 0,
            "max_loss_used_pct":     round(pct_of_max_loss * 100, 1),
            "consecutive_loss_days": state.get("consecutive_loss_days", 0),
            "consec_trade_losses":   state.get("consec_trade_losses", 0),
            "suspended":             self._is_suspended(),
            "k7_paused":             k7_active,
            "k7_pause_until":        state.get("k7_pause_until"),
            "k8_disabled":           k8_active,
            "k8_disable_until":      state.get("k8_disable_until"),
            "k9_vol_reduction":      state.get("k9_vol_reduction", 0.0),
        }

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _auto_mode(self) -> str:
        """
        Automatically select the appropriate mode based on challenge progress.
        Rules:
          ≥ 75% of target → SAFE (protect payout)
          ≥ 33% of target → step down from AGGRESSIVE to NORMAL
          Loss > 50% of max → step down one level
          Otherwise       → use persisted mode (default NORMAL)
        """
        state   = self._state
        pnl     = state.get("total_pnl", 0.0)
        pct     = pnl / self.target if self.target else 0
        loss_pct = pnl / -self.max_loss if pnl < 0 and self.max_loss else 0

        # Forced SAFE conditions
        safe_until = state.get("safe_mode_until")
        if safe_until and datetime.now(ET).date().isoformat() <= safe_until:
            return "SAFE"

        if pct >= 0.75:
            return "SAFE"    # 75%+ of target — protect the payout

        if loss_pct >= 0.50:
            return "SAFE"    # halfway to max loss — go conservative

        if pct >= 0.33:
            # Step down: if we were AGGRESSIVE → NORMAL
            persisted = state.get("mode", "NORMAL")
            if persisted == "AGGRESSIVE":
                return "NORMAL"
            return persisted

        return state.get("mode", "NORMAL")

    def _is_suspended(self) -> bool:
        until = self._state.get("suspension_until")
        if not until:
            return False
        return datetime.now(ET).date().isoformat() <= until

    def _check_rolling_wr_kill(self, state: dict) -> None:
        trades = _load_trades(KS_ROLLING_WR_WINDOW)
        if len(trades) < KS_ROLLING_WR_MIN_TRADES:   # inactive until min trades reached
            return
        wins = sum(1 for t in trades if t.get("win", t.get("pnl", 0) > 0))
        wr   = wins / len(trades)
        if wr < KS_ROLLING_WR_MIN:
            resume = (datetime.now(ET).date() + timedelta(days=KS_SUSPEND_DAYS)).isoformat()
            state["suspension_until"] = resume
            self._notify(
                f"[K1] Rolling WR {wr:.1%} < {KS_ROLLING_WR_MIN:.0%} over last {len(trades)} trades. "
                f"All trading suspended until {resume}."
            )

    def _check_k7_pf_kill(self, state: dict) -> None:
        """K7: Pause strategy if rolling profit factor < 1.0 over last 30 trades."""
        trades = _load_trades(KS_K7_PF_WINDOW)
        if len(trades) < KS_K7_MIN_TRADES:
            return
        gross_profit = sum(t.get("pnl", 0) for t in trades if t.get("pnl", 0) > 0)
        gross_loss   = sum(abs(t.get("pnl", 0)) for t in trades if t.get("pnl", 0) < 0)
        if gross_loss <= 0:
            return
        pf = gross_profit / gross_loss
        if pf < KS_K7_PF_MIN:
            resume = (datetime.now(ET).date() + timedelta(days=KS_K7_PAUSE_DAYS)).isoformat()
            if not state.get("k7_pause_until") or state["k7_pause_until"] < resume:
                state["k7_pause_until"] = resume
                self._notify(
                    f"[K7] Profit factor {pf:.2f} < {KS_K7_PF_MIN} over last {len(trades)} trades. "
                    f"Strategy paused until {resume}."
                )

    def _check_k9_vol_kill(self, state: dict) -> None:
        """
        K9: If rolling daily P&L volatility doubles vs baseline, reduce risk 50%.
        Uses std of daily P&L over last KS_K9_VOL_WINDOW active days.
        Stores a baseline_vol in state set on first 10-day window.
        """
        daily_pnl_map: dict = state.get("daily_pnl", {})
        if len(daily_pnl_map) < KS_K9_VOL_WINDOW * 2:
            return   # need enough history to establish baseline

        sorted_days = sorted(daily_pnl_map.keys())
        recent  = [daily_pnl_map[d] for d in sorted_days[-KS_K9_VOL_WINDOW:]]
        earlier = [daily_pnl_map[d] for d in sorted_days[-KS_K9_VOL_WINDOW * 2:-KS_K9_VOL_WINDOW]]

        def _std(vals: list) -> float:
            n = len(vals)
            if n < 2:
                return 0.0
            mean = sum(vals) / n
            return (sum((v - mean) ** 2 for v in vals) / (n - 1)) ** 0.5

        baseline_vol = _std(earlier)
        current_vol  = _std(recent)

        if baseline_vol <= 0:
            return

        vol_ratio = current_vol / baseline_vol
        if vol_ratio >= KS_K9_VOL_MULT:
            state["k9_vol_reduction"] = KS_K9_RISK_REDUCTION
            self._notify(
                f"[K9] Realised daily vol {current_vol:.0f} is {vol_ratio:.1f}× baseline {baseline_vol:.0f}. "
                f"Risk reduced to {(1 - KS_K9_RISK_REDUCTION):.0%} until vol normalises."
            )
        else:
            # Clear reduction if vol has normalised
            state.pop("k9_vol_reduction", None)

    def _check_slippage_kill(self, state: dict, slippage_pts: float) -> None:
        trades = _load_trades(KS_SLIPPAGE_WINDOW)
        if len(trades) < KS_SLIPPAGE_WINDOW:
            return
        avg_slip = sum(t.get("slippage_pts", 0) for t in trades) / len(trades)
        if avg_slip > KS_SLIPPAGE_THRESHOLD:
            self._notify(
                f"[K5] Average slippage {avg_slip:.2f}pts over last {KS_SLIPPAGE_WINDOW} trades "
                f"exceeds {KS_SLIPPAGE_THRESHOLD}pts threshold. "
                f"Recommend manual review — possible order routing issue."
            )

    def _get_cached_vix(self) -> float | None:
        """Fetch VIX. Cache in /tmp for 30 min to avoid redundant downloads."""
        vix_cache = Path("/tmp/vix_cache.json")
        if vix_cache.exists():
            try:
                data = json.loads(vix_cache.read_text())
                if time.time() - data.get("ts", 0) < 1800:  # 30 min TTL
                    return data["vix"]
            except Exception:
                pass
        try:
            import yfinance as yf
            df = yf.download("^VIX", period="2d", interval="1d",
                             auto_adjust=True, progress=False)
            if df.empty:
                return None
            vix = float(df["Close"].iloc[-1])
            vix_cache.write_text(json.dumps({"vix": vix, "ts": time.time()}))
            return vix
        except Exception:
            return None

    def _notify(self, msg: str) -> None:
        import os, requests
        token   = os.environ.get("TELEGRAM_TOKEN", "")
        chat_id = os.environ.get("TELEGRAM_CHAT_ID", "")
        print(f"[CHALLENGE MODE] {msg}", flush=True)
        if token and chat_id:
            try:
                requests.post(
                    f"https://api.telegram.org/bot{token}/sendMessage",
                    json={"chat_id": chat_id, "text": f"⚠️ [CHALLENGE MODE]\n{msg}"},
                    timeout=5,
                )
            except Exception:
                pass


# ═══════════════════════════════════════════════════════════════════════════════
# UTILITY FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

STAGE = Literal["CHALLENGE", "PAYOUT", "SCALING"]

# Stage-specific Lucid $100K parameters
STAGE_CONFIGS: dict[str, dict] = {
    "CHALLENGE": {
        # Evaluation phase — must hit +6% before losing -3%
        "target_pct":       0.06,
        "max_loss_pct":     0.03,
        "dll_pct":          0.018,
        "dll_buffer":       0.65,    # only use 65% of DLL headroom
        "default_mode":     "NORMAL",
        "description":      "Evaluation — growing toward target, preserving drawdown room",
    },
    "PAYOUT": {
        # Funded stage — payout available; goal is to not blow the account
        # Lucid funded: trailing max loss 3%, same DLL. No target needed.
        "target_pct":       0.03,    # aim for 3% per cycle for payout
        "max_loss_pct":     0.03,    # same drawdown limit
        "dll_pct":          0.018,
        "dll_buffer":       0.75,    # more conservative — protecting real funded account
        "default_mode":     "SAFE",
        "description":      "Funded — consistent withdrawals, max capital preservation",
    },
    "SCALING": {
        # Operating multiple funded accounts simultaneously
        "target_pct":       0.03,
        "max_loss_pct":     0.03,
        "dll_pct":          0.018,
        "dll_buffer":       0.80,
        "default_mode":     "SAFE",
        "description":      "Scaling — running 2+ accounts, strict limits per account",
    },
}


def detect_stage() -> str:
    """
    Infer current account stage from state file.
    Heuristic: if total_pnl ≥ challenge target → assume passed, now in PAYOUT.
    Override by setting stage manually: python challenge_mode.py stage PAYOUT
    """
    state = _load_state()
    forced = state.get("stage")
    if forced and forced in STAGE_CONFIGS:
        return forced
    pnl = state.get("total_pnl", 0.0)
    target = CHALLENGE_EQUITY * CHALLENGE_TARGET_PCT
    if pnl >= target:
        return "PAYOUT"
    return "CHALLENGE"


def set_stage(stage: str) -> None:
    """Manually set stage. python challenge_mode.py stage PAYOUT"""
    stage = stage.upper()
    assert stage in STAGE_CONFIGS, f"Invalid stage: {stage}. Valid: {list(STAGE_CONFIGS)}"
    state = _load_state()
    old   = state.get("stage", "CHALLENGE")
    state["stage"] = stage
    # Reset mode to stage default
    state["mode"] = STAGE_CONFIGS[stage]["default_mode"]
    _save_state(state)
    print(f"Stage changed: {old} → {stage}  (mode reset to {state['mode']})")


def get_stage_config() -> dict:
    """Return the active stage's parameters."""
    return STAGE_CONFIGS[detect_stage()]


def get_current_mode() -> str:
    """Quick helper — read current mode without instantiating full guard."""
    state = _load_state()
    return state.get("mode", "NORMAL")


def set_mode(mode: MODE) -> None:
    """Manually override mode. Use from CLI: python challenge_mode.py set SAFE"""
    assert mode in CHALLENGE_CONFIGS, f"Invalid mode: {mode}"
    state = _load_state()
    old = state.get("mode", "NORMAL")
    state["mode"] = mode
    _save_state(state)
    print(f"Mode changed: {old} → {mode}")


def status() -> None:
    """Print current challenge status to stdout."""
    guard = ChallengeGuard()
    summary = guard.progress_summary()
    cfg = guard.config

    print(f"\n{'='*60}")
    print(f"  CHALLENGE MODE: {summary['mode']}")
    print(f"  {cfg['description']}")
    print(f"{'='*60}")
    print(f"  Total P&L:         ${summary['total_pnl']:>10,.2f}")
    print(f"  Progress to target: {summary['pct_to_target']:>7.1f}%  (target ${guard.target:,.0f})")
    print(f"  Daily P&L:         ${summary['daily_pnl']:>10,.2f}  ({summary['dll_used_pct']:.1f}% of DLL used)")
    print(f"  Max loss used:      {summary['max_loss_used_pct']:>7.1f}%")
    print(f"  Consec loss days:   {summary['consecutive_loss_days']}")
    print(f"  Suspended:         {'YES' if summary['suspended'] else 'No'}")
    print(f"{'='*60}")
    print(f"  Risk/trade:        {cfg['risk_pct']:.1%} of equity")
    print(f"  Daily limit:       {cfg['daily_loss_pct']:.1%} of equity  "
          f"(DLL buffer: {cfg['dll_buffer_pct']:.0%})")
    print(f"  Max ES contracts:  {cfg['max_contracts_es']}")
    print(f"  Max MES contracts: {cfg['max_contracts_mes']}")

    vix = guard._get_cached_vix()
    if vix:
        status_str = "🔴 ICT PAUSED" if vix > VIX_PAUSE_THRESHOLD else "🟢 OK"
        print(f"  VIX:               {vix:.1f}  {status_str}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        cmd = sys.argv[1]
        if cmd == "set" and len(sys.argv) > 2:
            set_mode(sys.argv[2].upper())
        elif cmd == "stage" and len(sys.argv) > 2:
            set_stage(sys.argv[2].upper())
        elif cmd == "status":
            status()
        elif cmd == "reset":
            _STATE_PATH.unlink(missing_ok=True)
            print("Challenge state reset.")
        elif cmd == "stage":
            print(f"Current stage: {detect_stage()}")
    else:
        status()
