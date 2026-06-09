#!/usr/bin/env python3
"""
tests/test_challenge_mode.py — Unit tests for challenge_mode.py
Run: python -m pytest tests/ -v
 or: python tests/test_challenge_mode.py
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch

# ── Make project root importable ─────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent.parent))


def _make_guard(state: dict, trades: list[dict], vix: float = 15.0):
    """Build a ChallengeGuard backed by temp files with injected state."""
    import challenge_mode as cm

    state_f = Path(tempfile.mktemp(suffix=".json"))
    log_f   = Path(tempfile.mktemp(suffix=".jsonl"))

    state_f.write_text(json.dumps(state))
    with open(log_f, "w") as f:
        for t in trades:
            f.write(json.dumps(t) + "\n")

    # Patch module-level paths and VIX fetch
    with (
        patch.object(cm, "_STATE_PATH",      state_f),
        patch.object(cm, "_TRADE_LOG_PATH",  log_f),
    ):
        guard = cm.ChallengeGuard()
        guard._state = cm._load_state()   # force reload under patched path

    # Patch VIX and file paths onto the instance directly for test isolation
    guard._test_state_f = state_f
    guard._test_log_f   = log_f
    guard._fixed_vix    = vix

    # Monkey-patch instance method to return fixed VIX
    guard._get_cached_vix = lambda: vix  # type: ignore[method-assign]

    return guard, state_f, log_f


class TestModeAutoSwitch(unittest.TestCase):

    def test_mode_safe_at_75pct_target(self):
        """At 75% of target PnL → mode must be SAFE regardless of stored mode."""
        # 75% of $6,000 target = $4,500
        state = {"mode": "NORMAL", "total_pnl": 4_500.0, "daily_pnl": {}, "safe_mode_until": None}
        guard, sf, lf = _make_guard(state, [])
        try:
            self.assertEqual(guard.mode, "SAFE")
        finally:
            sf.unlink(missing_ok=True); lf.unlink(missing_ok=True)

    def test_mode_normal_below_33pct(self):
        """Below 33% of target → mode stays NORMAL."""
        state = {"mode": "NORMAL", "total_pnl": 1_000.0, "daily_pnl": {}, "safe_mode_until": None}
        guard, sf, lf = _make_guard(state, [])
        try:
            self.assertEqual(guard.mode, "NORMAL")
        finally:
            sf.unlink(missing_ok=True); lf.unlink(missing_ok=True)

    def test_mode_safe_when_50pct_max_loss_used(self):
        """When 50% of max loss used → mode must be SAFE."""
        # $3,000 max loss; 50% = $1,500
        state = {"mode": "NORMAL", "total_pnl": -1_500.0, "daily_pnl": {}, "safe_mode_until": None}
        guard, sf, lf = _make_guard(state, [])
        try:
            self.assertEqual(guard.mode, "SAFE")
        finally:
            sf.unlink(missing_ok=True); lf.unlink(missing_ok=True)


class TestKillSwitches(unittest.TestCase):

    def _today(self):
        from zoneinfo import ZoneInfo
        return datetime.now(ZoneInfo("America/New_York")).date().isoformat()

    def test_k2_daily_loss_blocks_trade(self):
        """K2: Daily loss ≥ 65% of DLL in NORMAL mode → can_trade returns False."""
        today = self._today()
        # NORMAL mode: dll_buffer_pct=0.65, DLL=$1800 → halt at $1170
        # Set daily loss to $1200 (> $1170)
        state = {
            "mode": "NORMAL", "total_pnl": -1_200.0,
            "daily_pnl": {today: -1_200.0},
            "safe_mode_until": None, "suspension_until": None,
        }
        guard, sf, lf = _make_guard(state, [])
        try:
            allowed, reason = guard.can_trade("ICT")
            self.assertFalse(allowed)
            self.assertIn("K2", reason)
        finally:
            sf.unlink(missing_ok=True); lf.unlink(missing_ok=True)

    def test_k3_total_dd_forces_safe(self):
        """K3: total loss ≥ 80% of max_loss → mode returns SAFE."""
        state = {"mode": "NORMAL", "total_pnl": -2_500.0, "daily_pnl": {}, "safe_mode_until": None}
        guard, sf, lf = _make_guard(state, [])
        try:
            # 80% of $3,000 = $2,400 — $2,500 exceeds that
            self.assertEqual(guard.mode, "SAFE")
        finally:
            sf.unlink(missing_ok=True); lf.unlink(missing_ok=True)

    def test_k4_vix_blocks_ict_not_straddle(self):
        """K4: VIX > 25 blocks ICT but NOT straddle."""
        state = {"mode": "NORMAL", "total_pnl": 0.0, "daily_pnl": {},
                 "safe_mode_until": None, "suspension_until": None}
        guard, sf, lf = _make_guard(state, [], vix=30.0)
        try:
            ict_ok, ict_reason = guard.can_trade("ICT")
            straddle_ok, _    = guard.can_trade("STRADDLE")
            self.assertFalse(ict_ok)
            self.assertIn("K4", ict_reason)
            self.assertTrue(straddle_ok)
        finally:
            sf.unlink(missing_ok=True); lf.unlink(missing_ok=True)

    def test_k6_consecutive_loss_days(self):
        """K6: 3 consecutive loss days → can_trade returns False within suspension window."""
        from zoneinfo import ZoneInfo
        today      = datetime.now(ZoneInfo("America/New_York")).date()
        safe_until = (today + timedelta(days=5)).isoformat()
        state = {
            "mode": "NORMAL", "total_pnl": -300.0, "daily_pnl": {},
            "safe_mode_until": safe_until,
            "suspension_until": None,
            "consecutive_loss_days": 3,
        }
        guard, sf, lf = _make_guard(state, [])
        try:
            allowed, reason = guard.can_trade("ICT")
            self.assertFalse(allowed)
            self.assertIn("K6", reason)
        finally:
            sf.unlink(missing_ok=True); lf.unlink(missing_ok=True)

    def test_suspension_expired(self):
        """If suspension_until is in the past → can_trade returns True."""
        yesterday = (datetime.now().date() - timedelta(days=1)).isoformat()
        state = {
            "mode": "NORMAL", "total_pnl": 0.0, "daily_pnl": {},
            "safe_mode_until": None,
            "suspension_until": yesterday,
            "consecutive_loss_days": 0,
        }
        guard, sf, lf = _make_guard(state, [])
        try:
            allowed, reason = guard.can_trade("ICT")
            self.assertTrue(allowed, f"Should be allowed; reason={reason}")
        finally:
            sf.unlink(missing_ok=True); lf.unlink(missing_ok=True)

    def test_k1_rolling_wr_suspension(self):
        """K1: rolling 20-trade WR < 42% → suspension_until gets set."""
        import challenge_mode as cm
        state  = {"mode": "NORMAL", "total_pnl": -200.0, "daily_pnl": {},
                  "safe_mode_until": None, "suspension_until": None, "consecutive_loss_days": 0}
        # 20 trades, all losses → WR = 0%
        trades = [{"win": False, "pnl": -100.0, "date": "2026-06-01",
                   "strategy": "ICT", "slippage_pts": 0.2, "mode": "NORMAL", "instrument": "ES"}
                  for _ in range(20)]
        state_f = Path(tempfile.mktemp(suffix=".json"))
        log_f   = Path(tempfile.mktemp(suffix=".jsonl"))
        state_f.write_text(json.dumps(state))
        with open(log_f, "w") as f:
            for t in trades:
                f.write(json.dumps(t) + "\n")

        with (
            patch.object(cm, "_STATE_PATH",     state_f),
            patch.object(cm, "_TRADE_LOG_PATH", log_f),
        ):
            guard = cm.ChallengeGuard()
            guard._state = cm._load_state()
            guard._get_cached_vix = lambda: 15.0  # type: ignore[method-assign]
            guard._notify = lambda msg: None       # type: ignore[method-assign]
            # Call the kill-switch check — it mutates state dict and calls _save_state
            guard._check_rolling_wr_kill(guard._state)
            cm._save_state(guard._state)   # explicit save under patched path
            updated = json.loads(state_f.read_text())
            self.assertIsNotNone(updated.get("suspension_until"))

        state_f.unlink(missing_ok=True)
        log_f.unlink(missing_ok=True)


class TestPositionSizing(unittest.TestCase):

    def test_position_size_safe_mode_max_mes(self):
        """In SAFE mode, MES position size never exceeds 5 contracts."""
        state = {"mode": "SAFE", "total_pnl": 0.0, "daily_pnl": {}, "safe_mode_until": None}
        guard, sf, lf = _make_guard(state, [])
        try:
            size = guard.position_size("MES", risk_pts=2.0)
            self.assertLessEqual(size, 5)
            self.assertGreaterEqual(size, 1)
        finally:
            sf.unlink(missing_ok=True); lf.unlink(missing_ok=True)

    def test_position_size_es_respects_max(self):
        """ES position size never exceeds mode max regardless of risk_pts."""
        state = {"mode": "NORMAL", "total_pnl": 0.0, "daily_pnl": {}, "safe_mode_until": None}
        guard, sf, lf = _make_guard(state, [])
        try:
            # Very small risk_pts → raw size would be huge → must cap at 3 (NORMAL ES max)
            size = guard.position_size("ES", risk_pts=0.1)
            self.assertLessEqual(size, 3)
        finally:
            sf.unlink(missing_ok=True); lf.unlink(missing_ok=True)

    def test_position_size_minimum_one(self):
        """Position size is always at least 1 contract."""
        state = {"mode": "SAFE", "total_pnl": 0.0, "daily_pnl": {}, "safe_mode_until": None}
        guard, sf, lf = _make_guard(state, [])
        try:
            # Very large risk_pts → raw = 0 → floor to 1
            size = guard.position_size("ES", risk_pts=500.0)
            self.assertEqual(size, 1)
        finally:
            sf.unlink(missing_ok=True); lf.unlink(missing_ok=True)


class TestProgressSummary(unittest.TestCase):

    def test_progress_summary_has_all_fields(self):
        """progress_summary() must return all required keys."""
        required = {
            "mode", "total_pnl", "target", "pct_to_target",
            "daily_pnl", "dll_used_pct", "max_loss_used_pct",
            "consecutive_loss_days", "suspended",
        }
        state = {"mode": "NORMAL", "total_pnl": 500.0, "daily_pnl": {}, "safe_mode_until": None}
        guard, sf, lf = _make_guard(state, [])
        try:
            summary = guard.progress_summary()
            missing = required - set(summary.keys())
            self.assertEqual(missing, set(), f"Missing keys: {missing}")
        finally:
            sf.unlink(missing_ok=True); lf.unlink(missing_ok=True)

    def test_record_trade_updates_pnl(self):
        """record_trade() must update daily_pnl and total_pnl in state."""
        import challenge_mode as cm
        state  = {"mode": "NORMAL", "total_pnl": 0.0, "daily_pnl": {},
                  "safe_mode_until": None, "suspension_until": None,
                  "consecutive_loss_days": 0, "last_trading_date": None}
        state_f = Path(tempfile.mktemp(suffix=".json"))
        log_f   = Path(tempfile.mktemp(suffix=".jsonl"))
        state_f.write_text(json.dumps(state))
        log_f.write_text("")

        with (
            patch.object(cm, "_STATE_PATH",     state_f),
            patch.object(cm, "_TRADE_LOG_PATH", log_f),
        ):
            guard = cm.ChallengeGuard()
            guard._state = cm._load_state()
            guard._get_cached_vix = lambda: 15.0  # type: ignore[method-assign]
            guard._notify = lambda msg: None       # type: ignore[method-assign]
            with patch.object(cm, "_STATE_PATH", state_f), \
                 patch.object(cm, "_TRADE_LOG_PATH", log_f):
                guard.record_trade("ICT", pnl=1_200.0, win=True)
                updated = json.loads(state_f.read_text())
                self.assertAlmostEqual(updated["total_pnl"], 1_200.0, places=1)

        state_f.unlink(missing_ok=True)
        log_f.unlink(missing_ok=True)


def _make_guard_with_state_file(state: dict, trades: list[dict]):
    """Module-level helper for TestNewKillSwitches."""
    import challenge_mode as cm
    state_f = Path(tempfile.mktemp(suffix=".json"))
    log_f   = Path(tempfile.mktemp(suffix=".jsonl"))
    state_f.write_text(json.dumps(state))
    with open(log_f, "w") as f:
        for t in trades:
            f.write(json.dumps(t) + "\n")

    with (
        patch.object(cm, "_STATE_PATH",     state_f),
        patch.object(cm, "_TRADE_LOG_PATH", log_f),
    ):
        guard = cm.ChallengeGuard()
        guard._state = cm._load_state()

    guard._get_cached_vix = lambda: 15.0  # type: ignore[method-assign]
    guard._notify         = lambda msg: None  # type: ignore[method-assign]
    return guard, state_f, log_f


class TestNewKillSwitches(unittest.TestCase):
    """Tests for K7, K8, K9 added in Phase 6."""

    def _make_guard_with_state_file(self, state: dict, trades: list[dict]):
        """Return guard plus live state/log paths for mutation checks."""
        import challenge_mode as cm
        state_f = Path(tempfile.mktemp(suffix=".json"))
        log_f   = Path(tempfile.mktemp(suffix=".jsonl"))
        state_f.write_text(json.dumps(state))
        with open(log_f, "w") as f:
            for t in trades:
                f.write(json.dumps(t) + "\n")

        with (
            patch.object(cm, "_STATE_PATH",     state_f),
            patch.object(cm, "_TRADE_LOG_PATH", log_f),
        ):
            guard = cm.ChallengeGuard()
            guard._state = cm._load_state()

        guard._get_cached_vix = lambda: 15.0  # type: ignore[method-assign]
        guard._notify         = lambda msg: None  # type: ignore[method-assign]
        return guard, state_f, log_f

    def test_k7_blocks_when_paused(self):
        """K7 pause_until in future → can_trade returns False."""
        from zoneinfo import ZoneInfo
        tomorrow = (datetime.now(ZoneInfo("America/New_York")).date()
                    + timedelta(days=3)).isoformat()
        state = {
            "mode": "NORMAL", "total_pnl": 0.0, "daily_pnl": {},
            "safe_mode_until": None, "suspension_until": None,
            "consecutive_loss_days": 0,
            "k7_pause_until": tomorrow,
        }
        guard, sf, lf = _make_guard_with_state_file(state, [])
        try:
            allowed, reason = guard.can_trade("ICT")
            self.assertFalse(allowed)
            self.assertIn("K7", reason)
        finally:
            sf.unlink(missing_ok=True); lf.unlink(missing_ok=True)

    def test_k8_blocks_when_disabled(self):
        """K8 disable_until in future → can_trade returns False."""
        from zoneinfo import ZoneInfo
        future = (datetime.now(ZoneInfo("America/New_York")).date()
                  + timedelta(days=8)).isoformat()
        state = {
            "mode": "NORMAL", "total_pnl": 0.0, "daily_pnl": {},
            "safe_mode_until": None, "suspension_until": None,
            "consecutive_loss_days": 0,
            "k8_disable_until": future,
        }
        guard, sf, lf = _make_guard_with_state_file(state, [])
        try:
            allowed, reason = guard.can_trade("ICT")
            self.assertFalse(allowed)
            self.assertIn("K8", reason)
        finally:
            sf.unlink(missing_ok=True); lf.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main(verbosity=2)
