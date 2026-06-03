#!/usr/bin/env node
/**
 * tv_fetch.js — CLI to extract OHLCV bars from TradingView Desktop via CDP.
 * Usage:  node tv_fetch.js --symbol CME_MINI_DL:ES1! --tf 15 --count 200
 * Output: JSON line to stdout { success, bars, symbol, tf, bar_count }
 *
 * Saves and restores chart state so the user's view is not permanently changed.
 */

import { getState, setSymbol, setTimeframe } from './src/core/chart.js';
import { getOhlcv } from './src/core/data.js';
import { disconnect } from './src/connection.js';

// Parse --key value pairs from CLI args
const argv = {};
const rawArgs = process.argv.slice(2);
for (let i = 0; i < rawArgs.length; i++) {
  if (rawArgs[i].startsWith('--')) {
    argv[rawArgs[i].slice(2)] = rawArgs[i + 1] ?? true;
    i++;
  }
}

const reqSymbol = argv.symbol || 'CME_MINI_DL:ES1!';
const reqTf     = String(argv.tf || '15');
const count     = parseInt(argv.count || '200', 10);

async function main() {
  let original = null;

  try {
    original = await getState();
  } catch (err) {
    process.stdout.write(JSON.stringify({ success: false, error: `CDP connect failed: ${err.message}` }) + '\n');
    process.exit(1);
  }

  const needSymbol = original.symbol !== reqSymbol;
  const needTf     = original.resolution !== reqTf;

  try {
    if (needSymbol) await setSymbol({ symbol: reqSymbol });
    if (needTf)     await setTimeframe({ timeframe: reqTf });

    const data = await getOhlcv({ count });

    process.stdout.write(JSON.stringify({
      success:   true,
      symbol:    reqSymbol,
      tf:        reqTf,
      bar_count: data.bar_count,
      bars:      data.bars,
    }) + '\n');

  } catch (err) {
    process.stdout.write(JSON.stringify({ success: false, error: err.message }) + '\n');
  } finally {
    try {
      if (needTf)     await setTimeframe({ timeframe: original.resolution });
      if (needSymbol) await setSymbol({ symbol: original.symbol });
    } catch { /* ignore restore errors */ }
    await disconnect();
  }
}

main().catch(err => {
  process.stdout.write(JSON.stringify({ success: false, error: err.message }) + '\n');
  process.exit(1);
});
