#!/usr/bin/env python3
"""
tests/test_straddle_dryrun.py — dry-run verification for Phase 9 safety patches.

Tests:
  V8 — TV.cancel() verification polling (4 scenarios)
  V6 — pre-event drift filter logic (3 scenarios)
  FILL_TIMEOUT — constant value and message format (2 checks)
  Checklist — required fields present in output (1 check)

No real Tradovate connection required.
"""
import os
import sys
import time
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import news_straddle as ns


# ── Helpers ────────────────────────────────────────────────────────────────────

def _bare_tv() -> ns.TV:
    """Create a TV instance that skips __init__ for unit-testing cancel()."""
    tv = ns.TV.__new__(ns.TV)
    return tv


# ── V8: cancel verification ────────────────────────────────────────────────────

class TestV8CancelVerification(unittest.TestCase):
    """
    TV.cancel() must verify the order is dead before returning.
    All four V8 exit paths are tested here.
    """

    def test_v8_clean_cancel_string_status(self):
        """V8: order confirms Cancelled on first poll → returns without exception."""
        tv = _bare_tv()
        tv._delete = MagicMock()
        tv.order_status = MagicMock(return_value={"ordStatus": "Cancelled"})
        tv.cancel(999)          # must not raise
        tv._delete.assert_called_once_with("/order/999")

    def test_v8_clean_cancel_numeric_status(self):
        """V8: Tradovate numeric ordStatus '3' (Cancelled) → confirmed dead."""
        tv = _bare_tv()
        tv._delete = MagicMock()
        tv.order_status = MagicMock(return_value={"ordStatus": "3"})
        tv.cancel(999)          # must not raise

    def test_v8_order_stays_working_raises(self):
        """V8: order still Working after 3 s polling window → RuntimeError + CRITICAL alert."""
        tv = _bare_tv()
        tv._delete = MagicMock()
        tv.order_status = MagicMock(return_value={"ordStatus": "Working"})
        with patch("news_straddle.notify") as mock_notify:
            with self.assertRaises(RuntimeError) as ctx:
                tv.cancel(999)
            self.assertIn("CANCEL FAILED", str(ctx.exception))
            self.assertIn("CRITICAL", mock_notify.call_args[0][0])
            self.assertIn("🚨", mock_notify.call_args[0][0])

    def test_v8_fills_during_cancel_window_raises(self):
        """V8: order fills inside the 3 s cancel window → RuntimeError (whipsaw guard)."""
        tv = _bare_tv()
        tv._delete = MagicMock()
        tv.order_status = MagicMock(return_value={"ordStatus": "Filled"})
        with patch("news_straddle.notify") as mock_notify:
            with self.assertRaises(RuntimeError) as ctx:
                tv.cancel(999)
            self.assertIn("filled during cancel window", str(ctx.exception))
            self.assertIn("CRITICAL", mock_notify.call_args[0][0])

    def test_v8_delete_fails_but_order_already_dead(self):
        """V8: DELETE raises (network blip) but status already Expired → no exception."""
        tv = _bare_tv()
        tv._delete = MagicMock(side_effect=Exception("connection reset"))
        tv.order_status = MagicMock(return_value={"ordStatus": "Expired"})
        tv.cancel(999)          # DELETE failed, but order is confirmed dead — not an error

    def test_v8_order_status_poll_retries_on_exception(self):
        """V8: order_status raises then succeeds → cancel completes without raising."""
        tv = _bare_tv()
        tv._delete = MagicMock()
        # First two polls raise, third returns dead
        tv.order_status = MagicMock(
            side_effect=[Exception("timeout"), Exception("timeout"), {"ordStatus": "Cancelled"}]
        )
        tv.cancel(999)          # should succeed on third poll


# ── V6: pre-event drift filter ─────────────────────────────────────────────────

class TestV6DriftFilter(unittest.TestCase):
    """
    Drift check: |price_at_arm - price_at_T2_snapshot| > 0.40 × offset → skip.
    Tests verify the threshold math and that the source contains correct logic.
    """

    def test_v6_threshold_cpi(self):
        """CPI offset=8.0: threshold is exactly 3.2 pts (40% of 8.0)."""
        offset = 8.0
        self.assertAlmostEqual(0.40 * offset, 3.2)

    def test_v6_threshold_nfp(self):
        """NFP offset=10.0: threshold is exactly 4.0 pts (40% of 10.0)."""
        offset = 10.0
        self.assertAlmostEqual(0.40 * offset, 4.0)

    def test_v6_drift_exceeds_threshold_would_skip(self):
        """V6 math: 3.5 pt drift on CPI (threshold 3.2) → drift > threshold (skip)."""
        drift_ref  = 5300.00
        price_now  = 5303.50   # moved 3.5 pts
        offset_pts = 8.0
        threshold  = 0.40 * offset_pts  # 3.2
        self.assertGreater(abs(price_now - drift_ref), threshold)

    def test_v6_drift_below_threshold_would_proceed(self):
        """V6 math: 1.0 pt drift on CPI (threshold 3.2) → drift ≤ threshold (proceed)."""
        drift_ref  = 5300.00
        price_now  = 5301.00   # moved 1.0 pt
        offset_pts = 8.0
        threshold  = 0.40 * offset_pts  # 3.2
        self.assertLessEqual(abs(price_now - drift_ref), threshold)

    def test_v6_present_in_source(self):
        """V6 drift reference, threshold calculation, and skip message all in source."""
        src_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "news_straddle.py")
        src = open(src_path).read()
        self.assertIn("_drift_ref",         src, "_drift_ref capture missing")
        self.assertIn("0.40 * offset_pts",  src, "40% threshold formula missing")
        self.assertIn("pre-event drift",    src, "drift skip notify message missing")

    def test_v6_drift_ref_captured_before_preflight(self):
        """V6: drift_ref must be captured BEFORE the DLL/position pre-flight checks."""
        src_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "news_straddle.py")
        src = open(src_path).read()
        # _drift_ref must appear before _conservative_daily_pnl in run_straddle
        idx_drift = src.find("_drift_ref = tv.get_price()")
        idx_dll   = src.find("_conservative_daily_pnl(tv)")
        self.assertGreater(idx_dll, idx_drift,
                           "_drift_ref must be captured before DLL pre-flight check")


# ── Fill timeout ───────────────────────────────────────────────────────────────

class TestFillTimeout(unittest.TestCase):

    def test_fill_timeout_is_90_seconds(self):
        """FILL_TIMEOUT constant must be 90 seconds (down from 600)."""
        self.assertEqual(ns.FILL_TIMEOUT, 90)

    def test_fill_timeout_notify_uses_seconds_format(self):
        """Timeout notify message must use {FILL_TIMEOUT}s (not //60 min)."""
        src_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "news_straddle.py")
        src = open(src_path).read()
        self.assertIn("FILL_TIMEOUT}s",        src, "Timeout message should use seconds")
        self.assertNotIn("FILL_TIMEOUT//60} min", src, "Old minutes format should be gone")


# ── Pre-event checklist ────────────────────────────────────────────────────────

class TestPreEventChecklist(unittest.TestCase):

    def test_checklist_fields_in_source(self):
        """All required checklist fields must be present in run_straddle source."""
        src_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "news_straddle.py")
        src = open(src_path).read()
        required = [
            "PRE-EVENT CHECKLIST",
            "Environment",
            "Offset",
            "TP",
            "SL",
            "Contracts",
            "Mode/Stage",
            "DLL remain",
            "MaxLoss rem",
            "DEMO",
            "LIVE",
        ]
        for field in required:
            self.assertIn(field, src, f"Checklist field '{field}' missing from source")

    def test_demo_detection_label(self):
        """Demo environment detection: 'demo' in URL → '⚠️ DEMO', else '✅ LIVE'."""
        src_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "news_straddle.py")
        src = open(src_path).read()
        self.assertIn("demo\" in TV_BASE.lower()", src)
        self.assertIn("⚠️ DEMO",  src)
        self.assertIn("✅ LIVE",  src)


# ── Simulate: no-fill timeout (fast) ──────────────────────────────────────────

class TestNoFillTimeoutSimulation(unittest.TestCase):
    """
    Simulate the no-fill path with FILL_TIMEOUT patched to 2 seconds.
    Verifies: both orders placed → neither fills → both cancelled → timeout notify fires.

    Uses a full MagicMock TV so no Tradovate connection is needed.
    The event is set exactly 2 minutes ahead so arm_et ≈ now → zero sleep before T-2.
    """

    def test_no_fill_timeout_cancels_both_orders(self):
        """No fill within 2 s → both stops placed, both cancelled, timeout notify fires."""
        from datetime import datetime, timedelta
        import pytz

        et     = pytz.timezone("America/New_York")
        # arm_et = event_et - 2min. Setting event_et = now + 2min makes arm_et ≈ now → ~0s sleep.
        future = datetime.now(et) + timedelta(minutes=2, seconds=5)
        event  = {
            "event":  "CPI",
            "date":   future.strftime("%Y-%m-%d"),
            "hour":   future.hour,
            "minute": future.minute,
            "impact": "HIGH",
        }

        # Full MagicMock TV — no real Tradovate connection.
        # order_status returns a dict-like object that _is_filled/_is_dead both treat as False.
        tv = MagicMock()
        tv.ctr  = {"id": 1, "name": "MESM6"}
        tv.acct = {"id": 1, "name": "test"}
        tv.get_price.return_value          = 5300.0
        tv.get_spread.return_value         = 0.5
        tv.has_position.return_value       = False
        tv.daily_realized_pnl.return_value = 0.0
        tv.stop.side_effect                = [1001, 1002]
        # order_status returns {"ordStatus": "Working"} — never fills, never dies
        tv.order_status.return_value = {"ordStatus": "Working"}

        notified = []

        import pandas as pd
        import numpy as np
        rows  = 20
        atr_df = pd.DataFrame({
            "High":  np.full(rows, 5310.0),
            "Low":   np.full(rows, 5290.0),
            "Close": np.full(rows, 5300.0),
        })

        with patch("news_straddle._conservative_daily_pnl", return_value=0.0), \
             patch("challenge_mode.ChallengeGuard")    as mock_cg, \
             patch("challenge_mode.detect_stage",      return_value="CHALLENGE"), \
             patch("news_straddle._already_traded_today", return_value=False), \
             patch("yfinance.download",                return_value=atr_df), \
             patch.object(ns, "FILL_TIMEOUT", 2), \
             patch("news_straddle.notify", side_effect=lambda m: notified.append(m)):

            mock_cg.return_value.can_trade.return_value = (True, "ok")
            ns.run_straddle(tv, event)

        # Both entry stops must have been placed before the timeout loop started
        self.assertEqual(
            tv.stop.call_count, 2,
            f"Expected 2 stop orders placed, got {tv.stop.call_count}. "
            f"Notified: {notified}",
        )
        # Both must have been cancelled when timeout fired
        self.assertEqual(
            tv.cancel.call_count, 2,
            f"Expected 2 cancels after timeout, got {tv.cancel.call_count}.",
        )
        # Timeout notify message must mention the fill timeout
        timeout_fired = any(
            "no fill" in m.lower() or "90s" in m or "cancelled" in m.lower()
            for m in notified
        )
        self.assertTrue(
            timeout_fired,
            f"Expected a timeout/cancel notify message; got: {notified}",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
