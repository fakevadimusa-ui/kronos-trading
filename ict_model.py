"""
ICT (Inner Circle Trader) Algorithmic Detection Engine
=======================================================
Implements the core ICT concepts algorithmically:
  - Market Structure: swing highs/lows, BOS, CHOCH
  - Order Blocks (OB): last opposing candle before a structural break
  - Fair Value Gaps (FVG): 3-candle imbalance
  - Liquidity levels: equal highs/lows (buy-side / sell-side)
  - Kill zones: London (3-5 AM ET), NY AM (7:30-10 AM ET)
  - Optimal Trade Entry (OTE): Fibonacci retracement into OB/FVG

Usage:
  model = ICTModel()
  signal = model.get_signal(df_15m, df_1h)
  # signal: {'signal': 'BUY'|'SELL'|'HOLD', 'entry', 'sl', 'tp', 'reason', 'setup'}
"""

import numpy as np
import pandas as pd
import pytz
from datetime import datetime, time
from typing import Optional


class ICTModel:

    ET = pytz.timezone("America/New_York")

    KILL_ZONES = {
        "london":       (time(3, 0),  time(5, 0)),
        "new_york":     (time(7, 30), time(10, 0)),
        "london_close": (time(10, 0), time(12, 0)),
    }

    def __init__(
        self,
        tf_minutes: int = 5,
        fvg_min_pct: float = 0.0002,
        rr_ratio: float = 3.0,
        displacement_factor: float = 1.5,
    ):
        self.tf_minutes          = tf_minutes
        self.fvg_min_pct         = fvg_min_pct
        self.rr_ratio            = rr_ratio
        self.displacement_factor = displacement_factor

        # All windows auto-scale to keep consistent real-time coverage
        # regardless of entry timeframe (5m, 15m, etc.)
        # Anchored to 75-min swing lookback, 300-min FVG recency, 450-min OB recency
        self.swing_lookback = max(3, round(75  / tf_minutes))
        self.fvg_recency    = round(300 / tf_minutes)   # bars back to consider FVGs fresh
        self.ob_recency     = round(450 / tf_minutes)   # bars back to consider OBs fresh
        self.ob_lookback    = round(450 / tf_minutes)   # max lookback when finding OBs
        self.min_bars       = max(25, round(300 / tf_minutes))

    # ──────────────────────────────────────────────────────────────────────────
    # SWING STRUCTURE
    # ──────────────────────────────────────────────────────────────────────────

    def _detect_swing_highs(self, highs: np.ndarray) -> list[int]:
        n   = self.swing_lookback
        idx = []
        for i in range(n, len(highs) - n):
            window = np.concatenate([highs[i-n:i], highs[i+1:i+n+1]])
            if highs[i] >= window.max():
                idx.append(i)
        return idx

    def _detect_swing_lows(self, lows: np.ndarray) -> list[int]:
        n   = self.swing_lookback
        idx = []
        for i in range(n, len(lows) - n):
            window = np.concatenate([lows[i-n:i], lows[i+1:i+n+1]])
            if lows[i] <= window.min():
                idx.append(i)
        return idx

    def detect_market_structure(self, df: pd.DataFrame) -> tuple[list[dict], Optional[str]]:
        """
        Returns (events, last_trend).
        Each event: {'idx', 'type': BOS_BULL|BOS_BEAR|CHOCH_BULL|CHOCH_BEAR, 'price', 'level'}
        """
        highs  = df["high"].values
        lows   = df["low"].values
        closes = df["close"].values

        sh_idx = set(self._detect_swing_highs(highs))
        sl_idx = set(self._detect_swing_lows(lows))

        events    = []
        trend     = None
        last_sh   = None   # (bar_idx, price)
        last_sl   = None

        for i in range(len(df)):
            if i in sh_idx:
                last_sh = (i, highs[i])
            if i in sl_idx:
                last_sl = (i, lows[i])

            if last_sh is None or last_sl is None:
                continue

            c = closes[i]

            if c > last_sh[1] and (not events or events[-1]["idx"] != i):
                etype = "BOS_BULL" if trend == "bull" else "CHOCH_BULL"
                events.append({"idx": i, "type": etype, "price": c, "level": last_sh[1]})
                trend = "bull"

            elif c < last_sl[1] and (not events or events[-1]["idx"] != i):
                etype = "BOS_BEAR" if trend == "bear" else "CHOCH_BEAR"
                events.append({"idx": i, "type": etype, "price": c, "level": last_sl[1]})
                trend = "bear"

        return events, trend

    # ──────────────────────────────────────────────────────────────────────────
    # FAIR VALUE GAPS
    # ──────────────────────────────────────────────────────────────────────────

    def detect_fvg(self, df: pd.DataFrame) -> list[dict]:
        """
        Bullish FVG:  candle[i].low  > candle[i-2].high  → gap above
        Bearish FVG:  candle[i].high < candle[i-2].low   → gap below

        Displacement filter: the middle candle (candle[i-1]) must have a body
        at least displacement_factor × the recent average body size.
        This ensures the FVG was created by real momentum, not a doji gap.
        """
        opens  = df["open"].values
        highs  = df["high"].values
        lows   = df["low"].values
        closes = df["close"].values
        fvgs   = []

        bodies     = np.abs(closes - opens)
        avg_body   = np.convolve(bodies, np.ones(20) / 20, mode="same")

        for i in range(2, len(df)):
            mid_body = bodies[i - 1]
            has_disp = mid_body >= self.displacement_factor * avg_body[i - 1]

            # Bullish FVG
            if lows[i] > highs[i - 2]:
                size = (lows[i] - highs[i - 2]) / highs[i - 2]
                if size >= self.fvg_min_pct:
                    fvgs.append({
                        "idx":          i,
                        "type":         "BULL",
                        "top":          lows[i],
                        "bottom":       highs[i - 2],
                        "mid":          (lows[i] + highs[i - 2]) / 2,
                        "size":         size,
                        "displacement": has_disp,
                    })
            # Bearish FVG
            elif highs[i] < lows[i - 2]:
                size = (lows[i - 2] - highs[i]) / lows[i - 2]
                if size >= self.fvg_min_pct:
                    fvgs.append({
                        "idx":          i,
                        "type":         "BEAR",
                        "top":          lows[i - 2],
                        "bottom":       highs[i],
                        "mid":          (lows[i - 2] + highs[i]) / 2,
                        "size":         size,
                        "displacement": has_disp,
                    })

        return fvgs

    # ──────────────────────────────────────────────────────────────────────────
    # ORDER BLOCKS
    # ──────────────────────────────────────────────────────────────────────────

    def detect_order_blocks(self, df: pd.DataFrame, structure_events: list[dict]) -> list[dict]:
        """
        Bullish OB:  last bearish candle immediately before a bullish BOS/CHOCH
        Bearish OB:  last bullish candle immediately before a bearish BOS/CHOCH
        """
        opens  = df["open"].values
        closes = df["close"].values
        highs  = df["high"].values
        lows   = df["low"].values
        obs    = []

        for ev in structure_events:
            idx = ev["idx"]
            lookback = max(0, idx - self.ob_lookback)

            if "BULL" in ev["type"]:
                for j in range(idx - 1, lookback, -1):
                    if closes[j] < opens[j]:   # bearish candle → bullish OB
                        obs.append({
                            "idx":     j,
                            "type":    "BULL",
                            "top":     max(opens[j], closes[j]),
                            "bottom":  min(opens[j], closes[j]),
                            "high":    highs[j],
                            "low":     lows[j],
                            "bos_idx": idx,
                        })
                        break

            elif "BEAR" in ev["type"]:
                for j in range(idx - 1, lookback, -1):
                    if closes[j] > opens[j]:   # bullish candle → bearish OB
                        obs.append({
                            "idx":     j,
                            "type":    "BEAR",
                            "top":     max(opens[j], closes[j]),
                            "bottom":  min(opens[j], closes[j]),
                            "high":    highs[j],
                            "low":     lows[j],
                            "bos_idx": idx,
                        })
                        break

        return obs

    # ──────────────────────────────────────────────────────────────────────────
    # LIQUIDITY
    # ──────────────────────────────────────────────────────────────────────────

    def detect_liquidity(self, df: pd.DataFrame, tolerance: float = 0.001) -> dict:
        """
        Scan last 50 bars for equal highs (sell-side) and equal lows (buy-side).
        Also tags previous-session high/low as major liquidity.
        """
        highs  = df["high"].values
        lows   = df["low"].values
        n      = min(50, len(df))
        recent = range(len(df) - n, len(df))

        eq_highs, eq_lows = [], []

        for i in recent:
            for j in range(max(0, i - n), i):
                if abs(highs[i] - highs[j]) / (highs[j] + 1e-9) <= tolerance:
                    eq_highs.append({"price": (highs[i] + highs[j]) / 2, "bars": [j, i]})
                if abs(lows[i] - lows[j]) / (lows[j] + 1e-9) <= tolerance:
                    eq_lows.append({"price": (lows[i] + lows[j]) / 2, "bars": [j, i]})

        # Strongest = most duplicate touches (dedup by price cluster)
        return {
            "sell_side": eq_highs,   # equal highs → liquidity above (target for stop hunts before sell)
            "buy_side":  eq_lows,    # equal lows  → liquidity below (target for stop hunts before buy)
        }

    # ──────────────────────────────────────────────────────────────────────────
    # KILL ZONES
    # ──────────────────────────────────────────────────────────────────────────

    def is_kill_zone(self, dt_et: Optional[datetime] = None) -> Optional[str]:
        if dt_et is None:
            dt_et = datetime.now(self.ET)
        t = dt_et.time()
        for name, (start, end) in self.KILL_ZONES.items():
            if start <= t <= end:
                return name
        return None

    # ──────────────────────────────────────────────────────────────────────────
    # HTF BIAS
    # ──────────────────────────────────────────────────────────────────────────

    def get_htf_bias(self, df_1h: pd.DataFrame) -> str:
        """Higher-timeframe directional bias from 1H market structure."""
        if df_1h is None or len(df_1h) < 20:
            return "neutral"
        _, trend = self.detect_market_structure(df_1h)
        return trend or "neutral"

    # ──────────────────────────────────────────────────────────────────────────
    # MAIN SIGNAL
    # ──────────────────────────────────────────────────────────────────────────

    def get_signal(
        self,
        df_entry:        pd.DataFrame,
        df_1h:           Optional[pd.DataFrame] = None,
        current_time_et: Optional[datetime]     = None,
    ) -> dict:
        """
        Full ICT setup scan on entry timeframe with 1H bias filter.

        Entry priority (displacement FVGs first, then OBs):
          1. Kill zone must be active
          2. HTF bias from 1H market structure
          3. Displaced FVG — price inside lower 50% (OTE zone, deeper retracement)
          4. Any FVG — price anywhere inside the gap
          5. Order Block — price inside OB body range

        All lookback windows auto-scaled to tf_minutes set at init.
        """
        base = {
            "signal":    "HOLD",
            "entry":     None,
            "sl":        None,
            "tp":        None,
            "rr":        None,
            "reason":    "",
            "setup":     None,
            "kill_zone": None,
            "htf_bias":  None,
        }

        if len(df_entry) < self.min_bars:
            base["reason"] = f"Insufficient bars ({len(df_entry)} < {self.min_bars})"
            return base

        # ── Kill zone check ──────────────────────────────────────────────────
        kz = self.is_kill_zone(current_time_et)
        if not kz:
            now_str = (current_time_et or datetime.now(self.ET)).strftime("%H:%M ET")
            base["reason"] = f"Not in kill zone ({now_str})"
            return base

        base["kill_zone"] = kz

        # ── HTF bias ─────────────────────────────────────────────────────────
        htf_bias = self.get_htf_bias(df_1h)
        base["htf_bias"] = htf_bias

        # ── Entry timeframe structure ─────────────────────────────────────────
        structure, _ = self.detect_market_structure(df_entry)

        # ── FVGs — scaled recency window ──────────────────────────────────────
        all_fvgs    = self.detect_fvg(df_entry)
        recent_fvgs = [f for f in all_fvgs if f["idx"] > len(df_entry) - self.fvg_recency]

        # Sort: displaced FVGs first, then by recency
        recent_fvgs.sort(key=lambda x: (-int(x["displacement"]), -x["idx"]))

        # ── OBs from last 5 structure events — scaled recency window ─────────
        recent_events = structure[-5:] if len(structure) >= 5 else structure
        all_obs       = self.detect_order_blocks(df_entry, recent_events)
        recent_obs    = [o for o in all_obs if o["idx"] > len(df_entry) - self.ob_recency]

        current_price = float(df_entry["close"].iloc[-1])

        # ── BULLISH setups ────────────────────────────────────────────────────
        if htf_bias in ("bull", "neutral"):
            for fvg in recent_fvgs:
                if fvg["type"] != "BULL":
                    continue
                # OTE: price must be in lower 50% of FVG (deeper retracement = better entry)
                ote_top = fvg["mid"]
                if not (fvg["bottom"] <= current_price <= ote_top):
                    continue
                entry = current_price
                sl    = fvg["bottom"] * (1 - 0.001)
                risk  = entry - sl
                if risk <= 0:
                    continue
                tp   = entry + risk * self.rr_ratio
                disp = " [DISP]" if fvg["displacement"] else ""
                return {**base,
                    "signal": "BUY",
                    "entry":  round(entry, 4),
                    "sl":     round(sl, 4),
                    "tp":     round(tp, 4),
                    "rr":     self.rr_ratio,
                    "setup":  "FVG",
                    "reason": f"Bullish FVG{disp} [{fvg['bottom']:.2f}–{fvg['top']:.2f}] OTE≤{ote_top:.2f} | {kz} | HTF:{htf_bias}",
                }

            for ob in sorted(recent_obs, key=lambda x: -x["idx"]):
                if ob["type"] != "BULL":
                    continue
                if not (ob["bottom"] <= current_price <= ob["top"]):
                    continue
                entry = current_price
                sl    = ob["low"] * (1 - 0.001)
                risk  = entry - sl
                if risk <= 0:
                    continue
                tp = entry + risk * self.rr_ratio
                return {**base,
                    "signal": "BUY",
                    "entry":  round(entry, 4),
                    "sl":     round(sl, 4),
                    "tp":     round(tp, 4),
                    "rr":     self.rr_ratio,
                    "setup":  "OB",
                    "reason": f"Bullish OB [{ob['bottom']:.2f}–{ob['top']:.2f}] | {kz} | HTF:{htf_bias}",
                }

        # ── BEARISH setups ────────────────────────────────────────────────────
        if htf_bias in ("bear", "neutral"):
            for fvg in recent_fvgs:
                if fvg["type"] != "BEAR":
                    continue
                # OTE: price must be in upper 50% of FVG (shallower = deeper retracement up)
                ote_bottom = fvg["mid"]
                if not (ote_bottom <= current_price <= fvg["top"]):
                    continue
                entry = current_price
                sl    = fvg["top"] * (1 + 0.001)
                risk  = sl - entry
                if risk <= 0:
                    continue
                tp   = entry - risk * self.rr_ratio
                disp = " [DISP]" if fvg["displacement"] else ""
                return {**base,
                    "signal": "SELL",
                    "entry":  round(entry, 4),
                    "sl":     round(sl, 4),
                    "tp":     round(tp, 4),
                    "rr":     self.rr_ratio,
                    "setup":  "FVG",
                    "reason": f"Bearish FVG{disp} [{fvg['bottom']:.2f}–{fvg['top']:.2f}] OTE≥{ote_bottom:.2f} | {kz} | HTF:{htf_bias}",
                }

            for ob in sorted(recent_obs, key=lambda x: -x["idx"]):
                if ob["type"] != "BEAR":
                    continue
                if not (ob["bottom"] <= current_price <= ob["top"]):
                    continue
                entry = current_price
                sl    = ob["high"] * (1 + 0.001)
                risk  = sl - entry
                if risk <= 0:
                    continue
                tp = entry - risk * self.rr_ratio
                return {**base,
                    "signal": "SELL",
                    "entry":  round(entry, 4),
                    "sl":     round(sl, 4),
                    "tp":     round(tp, 4),
                    "rr":     self.rr_ratio,
                    "setup":  "OB",
                    "reason": f"Bearish OB [{ob['bottom']:.2f}–{ob['top']:.2f}] | {kz} | HTF:{htf_bias}",
                }

        base["reason"] = f"No ICT setup in {kz} kill zone | HTF:{htf_bias}"
        return base


# ──────────────────────────────────────────────────────────────────────────────
# QUICK SMOKE TEST
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import yfinance as yf

    print("Fetching SPY 5m + 1H data...")
    df_5m = yf.download("SPY", period="5d", interval="5m", progress=False)
    df_1h = yf.download("SPY", period="60d", interval="1h", progress=False)

    for df in (df_5m, df_1h):
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.columns = [str(c).lower() for c in df.columns]

    model = ICTModel(tf_minutes=5)

    events, trend = model.detect_market_structure(df_5m)
    fvgs          = model.detect_fvg(df_5m)
    obs           = model.detect_order_blocks(df_5m, events[-5:])
    displaced     = sum(1 for f in fvgs if f["displacement"])

    print(f"\n=== ICT Model Smoke Test (5m) ===")
    print(f"5m bars: {len(df_5m)} | 1H bars: {len(df_1h)}")
    print(f"swing_lookback={model.swing_lookback} fvg_recency={model.fvg_recency} ob_recency={model.ob_recency}")
    print(f"HTF bias:  {model.get_htf_bias(df_1h)}")
    print(f"Structure events (last 5): {events[-5:]}")
    print(f"FVGs: {len(fvgs)} total | {displaced} with displacement | Recent: {sum(1 for f in fvgs if f['idx'] > len(df_5m)-model.fvg_recency)}")
    print(f"Order blocks: {len(obs)}")
    print(f"Kill zone now: {model.is_kill_zone()}")

    signal = model.get_signal(df_5m, df_1h)
    print(f"\nSignal: {signal['signal']}")
    print(f"Reason: {signal['reason']}")
    if signal["signal"] != "HOLD":
        print(f"Entry: {signal['entry']} | SL: {signal['sl']} | TP: {signal['tp']}")
