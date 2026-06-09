# Red Team Report — NY Open Liquidity Trap Strategy
**Role:** Strategy skeptic. Assume everything is wrong until proven otherwise.

---

## Attack 1: Is the edge just overfitting?

**Verdict: HIGH RISK of overfitting. Cannot be dismissed.**

Evidence:
- Best-performing parameter set (Variant A, 1.0pt sweep, 1.5R) used only **14 trades**
- With n=14, the 95% confidence interval on a 64% win rate spans from **~35% to ~87%**
- The "Grade A" rating is based on 14 data points — statistically worthless
- The grid search selected the best combination from 27 parameter sets tested on the same data
- This is textbook overfitting: we tried 27 combinations and picked the winner

**What this means:** The 64% win rate and PF=3.927 are almost certainly inflated by chance. The true performance on new data is unknown and likely worse.

**What would disprove it:** Out-of-sample test with 100+ trades showing similar metrics. We don't have that yet.

---

## Attack 2: Does it survive slippage?

**Verdict: BORDERLINE. The edge is real but thin at the margins.**

- At +0.5 pts extra friction: PF drops from 1.564 → 1.343 (survives)
- At +1.0 pts extra friction: PF = 1.157 (barely survives)
- At +2.0 pts extra friction: PF = 0.870 (loses money)

**The problem:** The entry signal fires at the CLOSE of a 5-minute candle that has already swept and reclaimed the level. In fast markets, the "close" you see in backtesting is not the price you'd execute at. During high-volatility NY opens, getting 1-2 pts of slippage on a market order is common.

**How to partially fix this:** 
- Enter on a limit order at the overnight low (for longs) instead of the reclaim candle close
- This would reduce entry slippage but might miss some signals entirely

---

## Attack 3: Does it work out-of-sample?

**Verdict: INCONCLUSIVE. Sample size too small.**

The 60/40 in-sample/out-of-sample split showed:
- In-sample (30 days): PF=1.152, E=+0.44
- Out-of-sample (21 days): PF=2.465, E=+3.33

Out-of-sample being better than in-sample sounds positive. But with only 7 OOS trades, this is statistically meaningless. OOS is better by luck, not by signal quality.

What's suspicious: PF=2.465 out-of-sample with 7 trades could be 4 wins and 3 losses. Or 5 wins and 2 losses. One different trade changes the PF dramatically.

**Cannot draw any conclusion from this split.**

---

## Attack 4: Does it only work in one market regime?

**Verdict: YES — high risk of regime sensitivity.**

The data covers April–June 2026 exclusively:
- This period began with extreme volatility (tariff shock, April 2026)
- Followed by a strong recovery/trending move higher
- Volatility was well above historical average for most of the period

In this regime, the overnight session likely had:
- Larger than normal overnight ranges (more volatile = bigger swings)
- More dramatic sweep + reversal patterns (bigger moves create cleaner failed auctions)
- Higher overnight volume (global uncertainty = more Asian/London session activity)

**The same strategy in a calm, low-volatility, slow-trending market (like mid-2019 or mid-2017) would likely produce fewer signals and potentially lower quality setups.**

**No test across different regimes is possible with 51 days of data. This is a major blind spot.**

---

## Attack 5: Does it fail on trend days?

**Verdict: YES — very likely fails on strong trend days. Untested.**

The strategy is designed to catch failed breakouts and reversals. On genuine trend days:
- Price sweeps the overnight low and keeps going lower
- The "reclaim" never happens
- Even if a partial reclaim occurs, the trend resumes and hits the stop

The strategy has NO trend filter currently. On trend days, the overnight sweep often becomes the start of a large move — you'd be entering against the trend.

**Example scenario (bear trend day):**
- Overnight low at 5,250
- Price opens at 9:30, falls to 5,248 (sweeps OL by 2 pts)
- Bar closes at 5,252 (appears to reclaim → signal fires)
- Price then trends lower to 5,200 for the rest of the session
- Result: stop hit at 5,246 (loss)

This scenario happened on real trend days in the dataset and contributed to losses.

**Fix required:** Add a trend-day filter. Ideas:
- Skip if prior day range > 2× prior 5-day avg range
- Skip if opening 15-min range > 1.5× prior 20-day avg OR range
- Skip if VIX > 30 (full panic mode — sweeps are fake-outs or real starts)

---

## Attack 6: Does it fail around news?

**Verdict: YES — news events contaminate signals. No filter tested.**

High-impact events (CPI, NFP, FOMC) create:
- Artificial volatility before the announcement (pre-positioning sweeps)
- Massive directional moves immediately after
- Fake reversal patterns that don't hold

The current backtest includes all trading days, including news days. There are likely 8–12 high-impact news events in the 51-day dataset. These could be responsible for some of the worst losses (or some of the best wins by accident).

**No way to know the impact without isolating these days.**

**Fix required:** Add news event filter using the existing `news_events.json` file. Skip signals on CPI, NFP, FOMC, PPI days.

---

## Attack 7: Does it require perfect fills?

**Verdict: NO perfect fills required, but timing matters.**

The entry is at the close of a 5-minute bar. This is realistic — when a 5-minute candle closes, you have about 0.1–0.2 seconds to act before the next candle opens. An automated system can execute this.

However, the entry point (close of reclaim candle) has inherent slippage risk:
- The market knows the level (overnight low is publicly visible)
- High-frequency traders may be at the same level
- A slow market order could slip 1–3 pts in volatile conditions

This is manageable with limit orders, not market orders. But limit orders risk missing the signal entirely if price moves away quickly.

---

## Attack 8: Does it break the daily loss limit?

**Verdict: UNLIKELY on normal days, but possible on extreme days.**

At NORMAL risk mode ($500/trade):
- One losing trade: -$480 (worst case, 1 trade)
- Two losing trades: -$960 (if both directions fire and both lose)
- This is well within the $1,800 daily loss limit

But: on a very bad day where the signal fires, you take a loss, then a second signal fires (opposite direction), and you lose that too:
- Loss 1: ~$480
- Loss 2: ~$480
- Total: $960 — still fine

The strategy limits to one trade per direction per day, so max 2 trades per day. Maximum daily dollar loss = 2 × $500 = $1,000, which is below the $1,800 DLL.

**DLL risk is acceptable if the one-trade-per-direction limit is strictly enforced.**

---

## Attack 9: Does it trade too much or too little?

**Too little (Variant A):** 1.0 trades per day on average. Some days 0 signals.

Impact: Some trading weeks might have only 2-3 trades. This makes challenge completion slow.

**Too much (Variant B):** 1.3 trades per day. On days with both long and short setups, potentially 2 trades.

With a 40% win rate in Variant B, 2 losing trades on the same day from a 1.3/day strategy happens regularly.

**Verdict:** Variant A frequency is acceptable. Variant B is borderline high for the win rate.

---

## Attack 10: Does it rely on hindsight levels?

**Verdict: NO — all levels are computed from already-elapsed data.**

- Overnight high/low: computed from bars that closed BEFORE 9:30 AM (look-behind only)
- Opening range: computed from the first 3 bars (9:30–9:45) before any trade entry
- PDH/PDL: computed from prior RTH session (closed the day before)
- VWAP: computed rolling from 9:30 AM forward (no future data)

**The backtester uses only closed candle data for signals.** Entry is at the close of the signal bar. This is correct.

**One exception to check:** The "same-candle sweep+reclaim" detection means the signal fires at the close of a candle that swept AND reclaimed within a single 5-minute bar. In reality, you can only see this AFTER the bar closes. This is handled correctly in the code (we check the bar's low, high, and close — all known at bar close). No lookahead.

---

## Attack 11: Does it require subjective human judgment?

**Verdict: NO — fully rule-based and automatable.**

Every element is precisely defined:
- Overnight high/low: max/min of identified time window (objective)
- Sweep: bar.low ≤ level − X pts (objective)
- Reclaim: bar.close > level (objective)
- Entry: bar.close (objective)
- Stop: sweep_extreme − 1 tick (objective)
- Target: fixed R ratio (objective)

No phrases like "strong candle," "clean setup," or "institutional move" — all defined numerically.

**This strategy CAN be fully automated as written.** No human judgment needed during execution.

---

## Attack 12: Can it be automated cleanly?

**Verdict: YES, with caveats.**

Integration with existing Tradovate/Alpaca setup:
- Signal detection: runs at 9:35 AM bar close (5-min bar 1)
- Checks each subsequent bar close for sweep+reclaim signal
- Places market/limit order on signal
- Sets stop and target immediately after fill
- Auto-cancels all if 10:30 AM hit without entry
- Logs everything to ~/logs/liquidity_trap/

**Caveats:**
- Need proper overnight H/L computation from live data (not just historical)
- Need ATR filter computed on live bar data
- Need one-trade-per-direction-per-day state tracking
- Need proper OCO orders (Tradovate supports this)

Estimated development time to integrate into existing system: 3–5 hours.

---

## Summary Attack Rating

| Attack | Severity | Status |
|---|---|---|
| Overfitting (n=14) | CRITICAL | Active problem — unfixed |
| Slippage sensitivity | HIGH | Borderline survivable |
| Out-of-sample | INCONCLUSIVE | Need more data |
| Regime sensitivity | HIGH | Untested — unknown risk |
| Trend day failure | HIGH | No filter — real losses expected |
| News contamination | MEDIUM | No filter — some bad trades |
| Perfect fills | LOW | Manageable with limits |
| Daily loss limit | LOW | Acceptable |
| Trade frequency | LOW | Within acceptable range |
| Hindsight levels | NONE | Clean implementation |
| Subjectivity | NONE | Fully rule-based |
| Automation | LOW | Straightforward to build |

---

## Red Team Final Verdict

**The strategy concept is sound. The current implementation is unproven.**

The critical unsolved problems are:
1. **Insufficient data** — 14–19 trades tells us nothing
2. **No regime filter** — could fail badly in different market conditions
3. **No trend-day filter** — will lose on trend days
4. **No news filter** — news events create false signals

Do NOT deploy to a live or challenge account until these are addressed. The risk is not catastrophic (the DLL rules protect the account) but the probability of losing the challenge is high on unvalidated parameters.

**Next step:** Forward-test on paper for 30–60 days. That gives 20–60 more trades in live conditions. If the paper test shows PF > 1.5 after 50+ trades, begin building toward live integration.
