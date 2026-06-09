"""
data_loader.py — Load and preprocess ES 5-min data for liquidity trap research.

Data source: Alpaca ES 5-min CSV (UTC timestamps, ~2 months).
All datetimes stored in UTC; ET column added for filtering.

Session definitions (EDT = UTC-4, valid for all dates in dataset April-June 2026):
  RTH session : 09:30–16:00 ET  = 13:30–20:00 UTC
  Globex open : 17:00 ET D-1    = 21:00 UTC D-1
  Overnight   : 21:00 UTC D-1   → 13:25 UTC D  (pre-RTH)
  NY open window for trading: 09:30–11:30 ET = 13:30–15:30 UTC
"""

from __future__ import annotations
import pandas as pd
import numpy as np
from pathlib import Path
from zoneinfo import ZoneInfo

UTC = ZoneInfo("UTC")
ET  = ZoneInfo("America/New_York")

PRIMARY_CSV = Path.home() / "ict_backtest" / "alpaca_180d_es_5min.csv"
BACKUP_CSV  = Path.home() / "ict_backtest" / "alpaca_180d_5min.csv"


def load_es_data() -> pd.DataFrame:
    """Load ES 5-min data. Falls back to SPY proxy if ES not available."""
    if PRIMARY_CSV.exists():
        df = pd.read_csv(PRIMARY_CSV)
        source = "ES_5min_Alpaca"
    elif BACKUP_CSV.exists():
        df = pd.read_csv(BACKUP_CSV)
        source = "SPY_5min_Alpaca_PROXY"
        print("WARNING: Using SPY data as ES proxy — label all results PROXY")
    else:
        raise FileNotFoundError(f"No data file found. Expected: {PRIMARY_CSV}")

    # Parse timestamps (Alpaca gives UTC timestamps as naive strings)
    df["Datetime"] = pd.to_datetime(df["Datetime"], utc=True)
    df = df.sort_values("Datetime").reset_index(drop=True)

    # Add ET datetime for session filtering
    df["dt_et"] = df["Datetime"].dt.tz_convert(ET)
    df["date_et"] = df["dt_et"].dt.date
    df["hour_et"] = df["dt_et"].dt.hour
    df["minute_et"] = df["dt_et"].dt.minute
    df["time_et"] = df["dt_et"].dt.time

    # Time-of-day as decimal hours ET (9.5 = 09:30)
    df["tod_et"] = df["hour_et"] + df["minute_et"] / 60.0

    # Session flags
    df["is_rth"]         = (df["tod_et"] >= 9.5)  & (df["tod_et"] < 16.0)
    df["is_overnight"]   = (df["tod_et"] >= 17.0) | (df["tod_et"] < 9.5)
    df["is_ny_window"]   = (df["tod_et"] >= 9.5)  & (df["tod_et"] < 11.5)  # 09:30–11:30

    # Typical price and value
    df["typical_price"] = (df["High"] + df["Low"] + df["Close"]) / 3.0

    # Candle characteristics
    df["body"]      = (df["Close"] - df["Open"]).abs()
    df["wick_up"]   = df["High"] - df[["Open", "Close"]].max(axis=1)
    df["wick_down"] = df[["Open", "Close"]].min(axis=1) - df["Low"]
    df["range"]     = df["High"] - df["Low"]

    df["_source"] = source
    return df


def add_atr(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """Add ATR column (true range based, EWM)."""
    prev_close = df["Close"].shift(1)
    tr = pd.concat([
        df["High"] - df["Low"],
        (df["High"] - prev_close).abs(),
        (df["Low"]  - prev_close).abs(),
    ], axis=1).max(axis=1)
    df["atr14"] = tr.ewm(span=period, adjust=False).mean()
    return df


def add_vwap(df: pd.DataFrame) -> pd.DataFrame:
    """Add session VWAP (resets at 09:30 ET each day)."""
    vwap_vals = np.full(len(df), np.nan)

    for date, grp in df.groupby("date_et"):
        rth_mask = grp["is_rth"]
        rth_idx  = grp.index[rth_mask]
        if len(rth_idx) == 0:
            continue
        tp  = grp.loc[rth_idx, "typical_price"].values
        vol = grp.loc[rth_idx, "Volume"].values.astype(float)
        vol = np.where(vol <= 0, 1.0, vol)  # guard zero-volume bars
        cum_tpv = np.cumsum(tp * vol)
        cum_vol = np.cumsum(vol)
        vwap_vals[rth_idx] = cum_tpv / cum_vol

    df["vwap"] = vwap_vals
    return df


def build_daily_levels(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each trading date, compute:
      - overnight_high / overnight_low  : max/min of bars BEFORE 09:30 ET on this date
      - pdh / pdl                       : prior RTH day's high / low
      - or_high / or_low                : first 15-min opening range (09:30–09:45 ET)
      - or_mid                          : midpoint of opening range
      - pd_mid                          : midpoint of prior day RTH range
      - overnight_mid                   : midpoint of overnight range
    """
    dates = sorted(df["date_et"].unique())

    rows = []
    for i, date in enumerate(dates):
        day_df = df[df["date_et"] == date]

        # True overnight session for day D:
        #   - Prior day (D-1) bars from 17:00 ET onward  (Globex evening)
        #   - Current day (D) bars before 09:30 ET        (Globex pre-RTH)
        # This spans the full overnight futures session from 5pm to 9:30am.
        pre_open_today = day_df[day_df["tod_et"] < 9.5]  # midnight → 9:30am on date D
        if i > 0:
            prior_date = dates[i - 1]
            prior_evening = df[(df["date_et"] == prior_date) & (df["tod_et"] >= 17.0)]
            overnight_bars = pd.concat([prior_evening, pre_open_today])
        else:
            overnight_bars = pre_open_today

        if len(overnight_bars) > 0:
            oh = overnight_bars["High"].max()
            ol = overnight_bars["Low"].min()
        else:
            oh, ol = np.nan, np.nan

        # Opening range: first 3 bars of RTH (09:30–09:45, 3 × 5-min bars)
        or_bars = day_df[
            (day_df["tod_et"] >= 9.5) & (day_df["tod_et"] < 9.75)  # 09:30–09:45
        ]
        if len(or_bars) >= 2:
            orh = or_bars["High"].max()
            orl = or_bars["Low"].min()
        else:
            orh, orl = np.nan, np.nan

        # Prior day RTH high/low
        if i > 0:
            prior_date = dates[i - 1]
            prior_rth  = df[(df["date_et"] == prior_date) & df["is_rth"]]
            if len(prior_rth) > 0:
                pdh = prior_rth["High"].max()
                pdl = prior_rth["Low"].min()
            else:
                pdh, pdl = np.nan, np.nan
        else:
            pdh, pdl = np.nan, np.nan

        rows.append({
            "date_et":       date,
            "overnight_high": oh,
            "overnight_low":  ol,
            "overnight_mid":  (oh + ol) / 2.0 if not (np.isnan(oh) or np.isnan(ol)) else np.nan,
            "or_high":        orh,
            "or_low":         orl,
            "or_mid":         (orh + orl) / 2.0 if not (np.isnan(orh) or np.isnan(orl)) else np.nan,
            "pdh":            pdh,
            "pdl":            pdl,
            "pd_mid":         (pdh + pdl) / 2.0 if not (np.isnan(pdh) or np.isnan(pdl)) else np.nan,
        })

    levels = pd.DataFrame(rows)
    df = df.merge(levels, on="date_et", how="left")
    return df


def prepare_data() -> pd.DataFrame:
    """Full pipeline: load → ATR → VWAP → daily levels."""
    df = load_es_data()
    df = add_atr(df)
    df = add_vwap(df)
    df = build_daily_levels(df)
    return df


if __name__ == "__main__":
    df = prepare_data()
    print(f"Loaded {len(df)} bars")
    print(f"Date range: {df['date_et'].min()} to {df['date_et'].max()}")
    print(f"Trading days: {df['date_et'].nunique()}")
    print(f"Source: {df['_source'].iloc[0]}")
    print(f"\nSample daily levels:")
    sample = df[df["date_et"] == df["date_et"].unique()[5]][
        ["dt_et", "Open", "High", "Low", "Close", "vwap",
         "overnight_high", "overnight_low", "or_high", "or_low", "pdh", "pdl"]
    ].head(10)
    print(sample.to_string())
