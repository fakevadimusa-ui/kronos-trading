#!/usr/bin/env python3
"""
ict_quality.py — FVG Setup Quality Scorer
==========================================
Scores a potential ICT FVG entry 0-100. Only setups scoring >= SCORE_THRESHOLD
should be traded. This prevents marginal setups from consuming DLL headroom
during the challenge.

Scoring components (total 100 points):
  1. Displacement size   (0-30): How strong was the displacement candle?
  2. Volume spike        (0-25): Was volume elevated on the displacement bar?
  3. PDH/PDL proximity   (0-20): Did the setup form near a key liquidity level?
  4. HTF alignment       (0-15): Does 1H structure agree with the trade direction?
  5. VIX regime          (0-10): Is the macro environment suitable for FVG entries?

Usage:
    from ict_quality import fvg_quality_score, score_breakdown, SCORE_THRESHOLD

    score = fvg_quality_score(row, df, side="BUY", atr=3.2, vix=18.5)
    if score < SCORE_THRESHOLD:
        return None  # skip setup
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Union

SCORE_THRESHOLD: int = 70   # minimum score to take the trade


# ── Individual scoring components ─────────────────────────────────────────────

def _score_displacement(row: pd.Series, df: pd.DataFrame) -> int:
    """
    Score displacement candle strength (0-30).
    Uses ratio of displacement body to 14-bar rolling average body.
    Displacement candle is assumed to be the bar before current (i-1 of the FVG).
    """
    body    = (df["Close"] - df["Open"]).abs()
    avg_b14 = body.rolling(14, min_periods=1).mean()

    # Current bar displacement — use the most recent bar's body vs avg
    cur_body = float(body.iloc[-1])
    avg      = float(avg_b14.iloc[-1])

    if avg <= 0:
        return 0

    ratio = cur_body / avg
    if ratio >= 2.5:
        return 30
    if ratio >= 2.0:
        return 20
    if ratio >= 1.5:
        return 10
    return 0


def _score_volume(row: pd.Series, df: pd.DataFrame) -> int:
    """
    Score volume spike on the setup bar (0-25).
    Compares current bar volume to 14-bar rolling mean.
    """
    vol     = df["Volume"]
    vol_ma  = vol.rolling(14, min_periods=1).mean()

    cur_vol = float(row.get("Volume", 0))
    avg_vol = float(vol_ma.iloc[-1])

    if avg_vol <= 0 or cur_vol <= 0:
        return 5   # neutral if no volume data

    ratio = cur_vol / avg_vol
    if ratio >= 1.5:
        return 25
    if ratio >= 1.2:
        return 15
    if ratio >= 0.8:
        return 5
    return 0


def _score_pdh_pdl_proximity(
    row: pd.Series,
    side: str,
    atr: float,
) -> int:
    """
    Score proximity to PDH (for shorts) or PDL (for longs) (0-20).
    FVGs that form near liquidity levels have higher probability of
    being institutional, not retail.
    """
    price = float(row.get("Close", 0))
    if atr <= 0 or price <= 0:
        return 0

    if side == "BUY":
        pdl = row.get("pdl", np.nan)
        if pd.isna(pdl) or float(pdl) <= 0:
            return 5   # neutral — no PDL data
        distance = abs(price - float(pdl))
    else:
        pdh = row.get("pdh", np.nan)
        if pd.isna(pdh) or float(pdh) <= 0:
            return 5
        distance = abs(price - float(pdh))

    atr_multiples = distance / atr
    if atr_multiples <= 1.0:
        return 20
    if atr_multiples <= 2.0:
        return 10
    return 0


def _score_htf_alignment(row: pd.Series, side: str) -> int:
    """
    Score 1H HTF bias alignment (0-15).
    Aligned = same direction as trade. Opposing = zero.
    """
    bias = str(row.get("htf_bias", "neutral")).lower()

    if side == "BUY":
        if bias == "bull":
            return 15
        if bias == "neutral":
            return 8
        return 0   # bear bias on long = 0
    else:
        if bias == "bear":
            return 15
        if bias == "neutral":
            return 8
        return 0   # bull bias on short = 0


def _score_vix_regime(vix: float) -> int:
    """
    Score VIX environment (0-10).
    Low VIX = stable institutional flow. High VIX = chaotic, FVGs less reliable.
    """
    if vix <= 0:
        return 5   # unknown — neutral
    if vix < 15:
        return 10
    if vix < 20:
        return 7
    if vix < 25:
        return 4
    return 0   # >= 25 should be blocked by VIX gate, but score 0 as defence-in-depth


# ── Public API ────────────────────────────────────────────────────────────────

def score_breakdown(
    row:  pd.Series,
    df:   pd.DataFrame,
    side: str,
    atr:  float,
    vix:  float = 0.0,
) -> dict[str, int]:
    """
    Return per-component scores as a dict. Use for logging only.
    Keys: displacement, volume, proximity, htf, vix_regime, total
    """
    d = _score_displacement(row, df)
    v = _score_volume(row, df)
    p = _score_pdh_pdl_proximity(row, side, atr)
    h = _score_htf_alignment(row, side)
    x = _score_vix_regime(vix)
    return {
        "displacement": d,
        "volume":       v,
        "proximity":    p,
        "htf":          h,
        "vix_regime":   x,
        "total":        d + v + p + h + x,
    }


def fvg_quality_score(
    row:  pd.Series,
    df:   pd.DataFrame,
    side: str,
    atr:  float,
    vix:  float = 0.0,
) -> int:
    """
    Score an FVG setup 0-100.

    Parameters
    ----------
    row   : Last bar of df (df.iloc[-1]) — contains Close, Volume, pdh, pdl, htf_bias
    df    : Full bar DataFrame used for rolling calculations
    side  : "BUY" or "SELL"
    atr   : Current 14-bar ATR value
    vix   : Current VIX (0 = unknown/unavailable)

    Returns
    -------
    int 0-100. Trade only if >= SCORE_THRESHOLD (70).
    """
    bd = score_breakdown(row, df, side, atr, vix)
    return bd["total"]
