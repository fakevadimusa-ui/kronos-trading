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
from datetime import datetime, date, time, timedelta
from typing import Optional


class ICTModel:

    ET = pytz.timezone("America/New_York")

    KILL_ZONES = {
        "london":        (time(3, 0),  time(5, 0)),
        "new_york":      (time(8, 30), time(11, 0)),    # shifted from 7:30 — avoids pre-market chop
        "london_close":  (time(10, 0), time(12, 0)),
        "silver_bullet": (time(13, 30), time(16, 0)),
    }

    # News blackout windows (ET) — tier-1 events get ±15 min, others ±5 min
    # Liquidity doesn't recover within 5 min after NFP/CPI/FOMC — spreads stay wide
    NEWS_BLACKOUTS = [
        (time(8, 15),  time(8, 45)),    # 8:30 ET: NFP/CPI/PPI/Jobless Claims — ±15 min
        (time(9, 25),  time(9, 35)),    # 9:30 ET: equities open — ±5 min
        (time(13, 45), time(14, 15)),   # 2:00 ET: FOMC — ±15 min
    ]

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

    def is_news_blackout(self, dt_et: Optional[datetime] = None) -> bool:
        """True if within ±5 min of a high-impact news event — no new entries."""
        if dt_et is None:
            dt_et = datetime.now(self.ET)
        t = dt_et.time()
        for start, end in self.NEWS_BLACKOUTS:
            if start <= t <= end:
                return True
        return False

    # ──────────────────────────────────────────────────────────────────────────
    # MIDNIGHT OPEN — PREMIUM / DISCOUNT
    # ──────────────────────────────────────────────────────────────────────────

    def get_midnight_open(self, df: pd.DataFrame) -> Optional[float]:
        """
        Return the open price of the bar at/closest to midnight ET today.
        Price above midnight open = premium (algo selling zone).
        Price below midnight open = discount (algo buying zone).
        """
        try:
            idx = df.index
            if idx.tz is None:
                idx = idx.tz_localize("UTC")
            idx_et = idx.tz_convert(self.ET)
            today  = datetime.now(self.ET).date()
            today_bars = df[idx_et.date == today]
            if len(today_bars) == 0:
                return None
            return float(today_bars["open"].iloc[0])
        except Exception:
            return None

    def is_discount(self, price: float, midnight_open: float) -> bool:
        return price < midnight_open

    def is_premium(self, price: float, midnight_open: float) -> bool:
        return price > midnight_open

    # ──────────────────────────────────────────────────────────────────────────
    # SMT DIVERGENCE (NQ vs ES / QQQ vs SPY)
    # ──────────────────────────────────────────────────────────────────────────

    def detect_smt(
        self,
        df_primary:    pd.DataFrame,
        df_correlated: pd.DataFrame,
        lookback: int = 20,
    ) -> Optional[str]:
        """
        SMT Divergence: one instrument makes a new extreme, the other holds.
        Bullish:  correlated (NQ) sweeps a lower low, primary (ES) holds higher low
                  → institutions absorbing sell orders on ES, move up incoming
        Bearish:  correlated (NQ) sweeps a higher high, primary (ES) holds lower high
                  → institutions distributing ES, move down incoming
        Returns: 'bullish', 'bearish', or None
        """
        if df_primary is None or df_correlated is None:
            return None
        n = min(lookback, len(df_primary), len(df_correlated))
        if n < 5:
            return None

        p_lows  = df_primary["low"].values[-n:]
        c_lows  = df_correlated["low"].values[-n:]
        p_highs = df_primary["high"].values[-n:]
        c_highs = df_correlated["high"].values[-n:]

        # Bullish SMT: NQ new low, ES holds (NQ current low < all prior lows, ES current > its min)
        if c_lows[-1] <= c_lows[:-1].min() and p_lows[-1] > p_lows[:-1].min():
            return "bullish"

        # Bearish SMT: NQ new high, ES holds lower
        if c_highs[-1] >= c_highs[:-1].max() and p_highs[-1] < p_highs[:-1].max():
            return "bearish"

        return None

    # ──────────────────────────────────────────────────────────────────────────
    # PREVIOUS DAY HIGH / LOW
    # ──────────────────────────────────────────────────────────────────────────

    def get_pdh_pdl(self, df_1h: pd.DataFrame) -> tuple[Optional[float], Optional[float]]:
        """Return (previous_day_high, previous_day_low) from 1H bars."""
        try:
            idx = df_1h.index
            if idx.tz is None:
                idx = idx.tz_localize("UTC")
            idx_et   = idx.tz_convert(self.ET)
            today    = datetime.now(self.ET).date()
            prev_day = today - timedelta(days=1)
            while prev_day.weekday() >= 5:
                prev_day -= timedelta(days=1)
            prev_bars = df_1h[idx_et.date == prev_day]
            if len(prev_bars) == 0:
                return None, None
            return float(prev_bars["high"].max()), float(prev_bars["low"].min())
        except Exception:
            return None, None

    # ──────────────────────────────────────────────────────────────────────────
    # AMD STATE MACHINE
    # ──────────────────────────────────────────────────────────────────────────

    def get_amd_phase(self, dt_et: Optional[datetime] = None) -> Optional[str]:
        """
        Returns current AMD phase: 'accumulation', 'manipulation', 'distribution', or None.

        London session:
          00:00–02:00 ET → accumulation  (Asian range forms)
          02:00–03:30 ET → manipulation  (Asian range sweep, Judas Swing)
          03:30–05:00 ET → distribution  (true London move)

        NY AM session:
          07:00–08:30 ET → accumulation  (pre-news ranging)
          08:30–09:45 ET → manipulation  (Judas Swing at 8:30/9:30 macro)
          09:45–11:30 ET → distribution  (true NY trend continuation)

        Silver Bullet:
          13:00–13:30 ET → accumulation
          13:30–14:30 ET → manipulation
          14:30–16:00 ET → distribution
        """
        if dt_et is None:
            dt_et = datetime.now(self.ET)
        t = dt_et.time()

        AMD = [
            (time(0,  0),  time(2,  0),  "accumulation"),   # London A
            (time(2,  0),  time(3, 30),  "manipulation"),    # London M
            (time(3, 30),  time(5,  0),  "distribution"),    # London D
            (time(7,  0),  time(8, 30),  "accumulation"),    # NY A
            (time(8, 30),  time(9, 45),  "manipulation"),    # NY M (Judas)
            (time(9, 45),  time(11, 30), "distribution"),    # NY D
            (time(13,  0), time(13, 30), "accumulation"),    # Silver A
            (time(13, 30), time(14, 30), "manipulation"),    # Silver M
            (time(14, 30), time(16,  0), "distribution"),    # Silver D
        ]
        for start, end, phase in AMD:
            if start <= t < end:
                return phase
        return None

    # ──────────────────────────────────────────────────────────────────────────
    # ACCUMULATION BOX  (session range formed before manipulation)
    # ──────────────────────────────────────────────────────────────────────────

    # Accumulation windows to look back into — keyed by session
    ACCUM_WINDOWS = {
        "london":        (time(0,  0), time(2,  0)),
        "new_york":      (time(7, 30), time(8, 30)),
        "silver_bullet": (time(13, 0), time(13, 30)),
    }

    def get_accumulation_box(
        self,
        df: pd.DataFrame,
        dt_et: Optional[datetime] = None,
    ) -> tuple[Optional[float], Optional[float]]:
        """
        Return (accum_high, accum_low) — the range built during the
        accumulation phase of the current session.
        Used by the AMD filter to decide whether today is a reversal or trend day.
        """
        if dt_et is None:
            dt_et = datetime.now(self.ET)

        # Find the most recently completed accumulation window
        t = dt_et.time()
        window = None
        for sess, (start, end) in self.ACCUM_WINDOWS.items():
            if t >= start:
                window = (start, end)   # last applicable window
        if window is None:
            return None, None

        try:
            idx = df.index
            if idx.tz is None:
                idx = idx.tz_localize("UTC")
            idx_et = idx.tz_convert(self.ET)
            today  = dt_et.date()
            mask   = (
                (idx_et.date == today) &
                (pd.Series(idx_et.time, index=df.index) >= window[0]) &
                (pd.Series(idx_et.time, index=df.index) <  window[1])
            )
            bars = df[mask]
            if len(bars) == 0:
                return None, None
            # Use 95th/5th percentile to exclude wick spikes from redefining the box
            return (
                float(np.percentile(bars["high"].values, 95)),
                float(np.percentile(bars["low"].values,  5)),
            )
        except Exception:
            return None, None

    # ──────────────────────────────────────────────────────────────────────────
    # UNICORN SETUP  (FVG inside Order Block — highest-confluence array)
    # ──────────────────────────────────────────────────────────────────────────

    def detect_unicorn(self, fvg: dict, obs: list[dict]) -> bool:
        """
        True if the FVG overlaps with an active Order Block of the same direction.
        FVG inside OB = Unicorn Setup — highest-probability confluence entry.
        Uses >= so touching arrays (FVG edge exactly at OB boundary) are included.
        """
        for ob in obs:
            if ob["type"] != fvg["type"]:
                continue
            overlap_top = min(fvg["top"], ob["top"])
            overlap_bot = max(fvg["bottom"], ob["bottom"])
            if overlap_top >= overlap_bot:   # >= catches exact-touch Unicorns
                return True
        return False

    # ──────────────────────────────────────────────────────────────────────────
    # JUDAS SWING DETECTION
    # ──────────────────────────────────────────────────────────────────────────

    def detect_judas_swing(
        self,
        df_entry:      pd.DataFrame,
        df_correlated: Optional[pd.DataFrame],
        dt_et:         Optional[datetime] = None,
    ) -> Optional[str]:
        """
        Detect an engineered fake move (Judas Swing) at 8:30 or 9:30 ET macro opens.
        Returns: 'bull_trap' (fake up → real move down)
                 'bear_trap' (fake down → real move up)
                 None if no Judas Swing detected.

        Conditions (all three must be true):
          1. Inside manipulation time window (8:30–9:45 ET or 9:30–10:00 ET)
          2. Volume on current bar < 10-bar moving average (low-conviction move)
          3. SMT divergence: NQ diverges from ES (institutional non-confirmation)
          4. Price swept a local swing high or low in the last 20 bars
        """
        if dt_et is None:
            dt_et = datetime.now(self.ET)
        t = dt_et.time()

        judas_windows = [
            (time(8, 30), time(9, 45)),
            (time(9, 30), time(10, 0)),
        ]
        if not any(s <= t <= e for s, e in judas_windows):
            return None

        if len(df_entry) < 15:
            return None

        vols    = df_entry["volume"].values
        vol_ma  = vols[-11:-1].mean()
        low_vol = vols[-1] < vol_ma

        smt     = self.detect_smt(df_entry, df_correlated, lookback=10)

        highs           = df_entry["high"].values[-20:]
        lows            = df_entry["low"].values[-20:]
        swept_buy_side  = highs[-1] > highs[:-1].max()   # fake breakout up
        swept_sell_side = lows[-1]  < lows[:-1].min()    # fake breakdown down

        if low_vol and smt == "bearish" and swept_buy_side:
            return "bull_trap"
        if low_vol and smt == "bullish" and swept_sell_side:
            return "bear_trap"
        return None

    # ──────────────────────────────────────────────────────────────────────────
    # EDGE CONDITIONS (ATR + INSIDE DAY PAUSE)
    # ──────────────────────────────────────────────────────────────────────────

    def check_edge_conditions(
        self,
        df_entry: pd.DataFrame,
        df_1h:    Optional[pd.DataFrame],
    ) -> tuple[bool, str]:
        """
        Returns (trading_ok, reason).
        False = pause strategy — edge is compromised.

        Killer 1: Daily ATR < 70% of its 20-period MA → chop, false breakouts
        Killer 2: Price contained inside PDH/PDL for last 4 hours → no expansion
        """
        try:
            if df_1h is not None and len(df_1h) >= 35:
                closes = df_1h["close"].values
                highs  = df_1h["high"].values
                lows   = df_1h["low"].values
                tr     = np.maximum.reduce([
                    highs[1:] - lows[1:],
                    np.abs(highs[1:] - closes[:-1]),
                    np.abs(lows[1:]  - closes[:-1]),
                ])
                atr14   = tr[-14:].mean()
                atr_ma  = np.array([tr[i:i+14].mean() for i in range(len(tr) - 13)]).mean()
                if atr14 < atr_ma * 0.70:
                    return False, f"Low volatility: ATR {atr14:.2f} < 70% of MA {atr_ma:.2f} — strategy paused"

            pdh, pdl = self.get_pdh_pdl(df_1h) if df_1h is not None else (None, None)
            if pdh and pdl and len(df_entry) >= 48:
                recent_high = df_entry["high"].values[-48:].max()
                recent_low  = df_entry["low"].values[-48:].min()
                if recent_low >= pdl and recent_high <= pdh:
                    return False, f"Inside day: price in PDH/PDL [{pdl:.2f}–{pdh:.2f}] for 4+ hrs — strategy paused"

            return True, "OK"
        except Exception:
            return True, "OK"   # fail open

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
        df_correlated:   Optional[pd.DataFrame] = None,   # NQ1!/QQQ for SMT
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

        # ── News blackout check ──────────────────────────────────────────────
        if self.is_news_blackout(current_time_et):
            now_str = (current_time_et or datetime.now(self.ET)).strftime("%H:%M ET")
            base["reason"] = f"News blackout ({now_str}) — no entries ±5 min of high-impact event"
            return base

        base["kill_zone"] = kz

        # ── Edge conditions (volatility + inside day) ─────────────────────────
        edge_ok, edge_reason = self.check_edge_conditions(df_entry, df_1h)
        if not edge_ok:
            base["reason"] = edge_reason
            return base

        # ── HTF bias ─────────────────────────────────────────────────────────
        htf_bias = self.get_htf_bias(df_1h)
        base["htf_bias"] = htf_bias

        # ── AMD phase filter ──────────────────────────────────────────────────
        amd_phase = self.get_amd_phase(current_time_et)
        if amd_phase == "accumulation":
            base["reason"] = f"AMD accumulation — marking range, no entries"
            return base

        # ── SMT divergence + Judas Swing / Trend Day ─────────────────────────
        # ── Phase transition ±2 min lock-out ─────────────────────────────────
        PHASE_TRANSITIONS = [
            time(2, 0), time(3, 30),     # London M/D boundaries
            time(8, 30), time(9, 45),    # NY M/D boundaries
            time(13, 30), time(14, 30),  # Silver Bullet M/D boundaries
        ]
        now_t = (current_time_et or datetime.now(self.ET)).time()
        for boundary in PHASE_TRANSITIONS:
            boundary_dt = datetime.combine(
                (current_time_et or datetime.now(self.ET)).date(), boundary
            )
            now_dt = (current_time_et or datetime.now(self.ET))
            if abs((now_dt.replace(tzinfo=None) - boundary_dt).total_seconds()) <= 120:
                base["reason"] = f"Phase transition lock-out (±2 min of {boundary}) — no entries"
                return base

        # London low-volume: need wider lookback to detect structural divergence
        smt_lookback = 12 if kz == "london" else 5
        smt       = self.detect_smt(df_entry, df_correlated, lookback=smt_lookback)
        judas     = self.detect_judas_swing(df_entry, df_correlated, current_time_et)
        smt_tag   = f" SMT:{smt}" if smt else ""
        judas_tag = f" JUDAS:{judas}" if judas else ""

        if amd_phase == "manipulation":
            accum_high, accum_low = self.get_accumulation_box(df_entry, current_time_et)
            if accum_high and accum_low:
                # Validate box range — reject if flat (holiday) or gap-inflated
                box_range = accum_high - accum_low
                atr_ref   = float(np.abs(df_entry["high"].values[-14:] - df_entry["low"].values[-14:]).mean())
                if atr_ref > 0:
                    if box_range < 0.10 * atr_ref:
                        base["reason"] = f"Accumulation box too flat ({box_range:.4f} < 10% ATR) — holiday/no-volume"
                        return base
                    if box_range > 2.0 * atr_ref:
                        base["reason"] = f"Accumulation box gap-inflated ({box_range:.4f} > 2× ATR) — skip session"
                        return base
                price_inside_box = accum_low <= current_price <= accum_high
                if price_inside_box and not judas:
                    # Price still inside accumulation range — need sweep + reversal
                    base["reason"] = (
                        f"AMD manipulation — price in accum box [{accum_low:.2f}–{accum_high:.2f}],"
                        f" waiting for Judas Swing{smt_tag}"
                    )
                    return base
                # Price broke OUT of accum box → trend day, bypass Judas requirement
                if not price_inside_box:
                    judas_tag = " [TREND_DAY]"
            elif not judas:
                # No accum data — fall back to standard Judas requirement
                base["reason"] = f"AMD manipulation — no accum data, waiting for Judas Swing{smt_tag}"
                return base

        # ── Midnight open premium/discount filter ─────────────────────────────
        midnight_open = self.get_midnight_open(df_entry)
        current_price = float(df_entry["close"].iloc[-1])
        if midnight_open is not None:
            in_discount = self.is_discount(current_price, midnight_open)
            in_premium  = self.is_premium(current_price, midnight_open)
            if htf_bias == "bull" and not in_discount:
                base["reason"] = f"Price in premium ({current_price:.2f} > midnight {midnight_open:.2f}) — wait for discount"
                return base
            if htf_bias == "bear" and not in_premium:
                base["reason"] = f"Price in discount ({current_price:.2f} < midnight {midnight_open:.2f}) — wait for premium"
                return base

        # ── PDH/PDL R:R validation ────────────────────────────────────────────
        # Don't block by proximity — PDH/PDL IS the target (liquidity magnet).
        # Only block if so close that minimum 1:3 R:R is impossible.
        # Evaluated per-setup (after SL is known) inside the FVG/OB loops below.
        pdh, pdl = self.get_pdh_pdl(df_1h) if df_1h is not None else (None, None)

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

        # ── BULLISH setups ────────────────────────────────────────────────────
        if htf_bias in ("bull", "neutral"):
            for fvg in recent_fvgs:
                if fvg["type"] != "BULL":
                    continue
                ote_top = fvg["mid"]
                if not (fvg["bottom"] <= current_price <= ote_top):
                    continue
                entry    = current_price
                sl       = fvg["bottom"] * (1 - 0.001)
                risk     = entry - sl
                if risk <= 0:
                    continue
                tp       = entry + risk * self.rr_ratio
                # PDH R:R gate: PDH must be beyond TP (target reachable)
                if pdh and htf_bias == "bull" and pdh < tp:
                    tp = min(tp, pdh)   # cap TP at PDH as Draw on Liquidity
                    if (tp - entry) < risk:   # R:R < 1:1 — not worth it
                        continue
                disp     = " [DISP]" if fvg["displacement"] else ""
                unicorn  = " [UNICORN]" if self.detect_unicorn(fvg, recent_obs) else ""
                return {**base,
                    "signal": "BUY",
                    "entry":  round(entry, 4),
                    "sl":     round(sl, 4),
                    "tp":     round(tp, 4),
                    "rr":     round((tp - entry) / risk, 2),
                    "setup":  "FVG" + unicorn.strip(),
                    "reason": f"Bullish FVG{disp}{unicorn} [{fvg['bottom']:.2f}–{fvg['top']:.2f}] OTE≤{ote_top:.2f} | {kz} | HTF:{htf_bias}{smt_tag}{judas_tag}",
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
                    "reason": f"Bullish OB [{ob['bottom']:.2f}–{ob['top']:.2f}] | {kz} | HTF:{htf_bias}{smt_tag}{judas_tag}",
                }

        # ── BEARISH setups ────────────────────────────────────────────────────
        if htf_bias in ("bear", "neutral"):
            for fvg in recent_fvgs:
                if fvg["type"] != "BEAR":
                    continue
                ote_bottom = fvg["mid"]
                if not (ote_bottom <= current_price <= fvg["top"]):
                    continue
                entry   = current_price
                sl      = fvg["top"] * (1 + 0.001)
                risk    = sl - entry
                if risk <= 0:
                    continue
                tp      = entry - risk * self.rr_ratio
                if pdl and htf_bias == "bear" and pdl > tp:
                    tp = max(tp, pdl)
                    if (entry - tp) < risk:
                        continue
                disp    = " [DISP]" if fvg["displacement"] else ""
                unicorn = " [UNICORN]" if self.detect_unicorn(fvg, recent_obs) else ""
                return {**base,
                    "signal": "SELL",
                    "entry":  round(entry, 4),
                    "sl":     round(sl, 4),
                    "tp":     round(tp, 4),
                    "rr":     round((entry - tp) / risk, 2),
                    "setup":  "FVG" + unicorn.strip(),
                    "reason": f"Bearish FVG{disp}{unicorn} [{fvg['bottom']:.2f}–{fvg['top']:.2f}] OTE≥{ote_bottom:.2f} | {kz} | HTF:{htf_bias}{smt_tag}{judas_tag}",
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
                    "reason": f"Bearish OB [{ob['bottom']:.2f}–{ob['top']:.2f}] | {kz} | HTF:{htf_bias}{smt_tag}{judas_tag}",
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
