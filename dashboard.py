#!/usr/bin/env python3
"""
dashboard.py — Kronos Trading Terminal
Dark web dashboard for all trading activity.

Start:   uvicorn dashboard:app --host 0.0.0.0 --port 3000
Access:  http://144.202.3.195:3000

Add to VPS cron for auto-start on reboot:
@reboot cd /root/kronos-trading && /root/kronos-trading/venv/bin/uvicorn dashboard:app --host 0.0.0.0 --port 3000 >> /root/logs/dashboard.log 2>&1
"""

import json, os, sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

import pytz
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse

# ── Load .env ──────────────────────────────────────────────────────────────────
_env = Path(__file__).parent / ".env"
if _env.exists():
    for _line in _env.read_text().splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _v = _line.split("=", 1)
            os.environ.setdefault(_k.strip(), _v.strip().strip('"').strip("'"))

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetOrdersRequest

ET          = pytz.timezone("America/New_York")
CONFIG_PATH = Path.home() / "freqtrade/user_data/config_kronos_nvda.json"
SESSION_PATH = Path.home() / "freqtrade/user_data/logs/ict_session.json"
ICT_LOG     = Path("/root/logs/ict/ict.log")
STRADDLE_LOG = Path("/root/logs/straddle/straddle.log")
EVENTS_FILE = Path(__file__).parent / "news_events.json"

app = FastAPI()


def _client() -> TradingClient:
    key    = os.environ.get("ALPACA_KEY")
    secret = os.environ.get("ALPACA_SECRET")
    if not (key and secret):
        cfg    = json.loads(CONFIG_PATH.read_text())
        key    = cfg["exchange"]["key"]
        secret = cfg["exchange"]["secret"]
    return TradingClient(key, secret, paper=True)


# ── Data helpers ───────────────────────────────────────────────────────────────

def _account_data() -> dict:
    try:
        client = _client()
        acct   = client.get_account()
        equity = float(acct.equity)
        start  = float(acct.last_equity)
        pnl    = equity - start
        pnl_pct = (pnl / start * 100) if start else 0

        # Open positions
        positions = [
            {
                "symbol": p.symbol,
                "side": p.side.value,
                "qty": float(p.qty),
                "entry": float(p.avg_entry_price),
                "pnl": float(p.unrealized_pl),
                "pnl_pct": float(p.unrealized_plpc) * 100,
            }
            for p in client.get_all_positions()
        ]

        # Recent fills (last 7 days) — same query as status.py
        since = (datetime.now(timezone.utc) - timedelta(days=7)).isoformat()
        orders = client.get_orders(filter=GetOrdersRequest(status="all", limit=40, after=since))
        filled = [o for o in orders if o.status.value == "filled"]

        fills = [
            {
                "time":   o.created_at.astimezone(ET).strftime("%m/%d %H:%M"),
                "side":   o.side.value.upper(),
                "qty":    int(float(o.qty)),
                "symbol": o.symbol,
                "price":  float(o.filled_avg_price or 0),
            }
            for o in reversed(filled[-12:])
        ]

        spy_fills = [o for o in filled if o.symbol == "SPY"]

        # Session start equity from ict_session.json
        sess_start = None
        sess_date  = None
        if SESSION_PATH.exists():
            s = json.loads(SESSION_PATH.read_text())
            sess_start = float(s.get("ict_start_equity", 0)) or None
            sess_date  = s.get("date")

        return {
            "equity":        equity,
            "start":         start,
            "pnl":           pnl,
            "pnl_pct":       pnl_pct,
            "from_100k":     equity - 100_000,
            "dd_pct":        (equity / 100_000 - 1) * 100,
            "positions":     positions,
            "fills":         fills,
            "spy_count":     len(spy_fills),
            "sess_start":    sess_start,
            "sess_date":     sess_date,
            "error":         None,
        }
    except Exception as e:
        return {"error": str(e), "equity": 0, "fills": [], "positions": [], "spy_count": 0}


def _next_event() -> dict:
    try:
        events  = json.loads(EVENTS_FILE.read_text())
        now_et  = datetime.now(ET)
        today   = now_et.strftime("%Y-%m-%d")
        future  = [e for e in events if e["date"] >= today]
        if not future:
            return {"name": "—"}
        ev = future[0]
        ev_dt = ET.localize(
            datetime.strptime(ev["date"], "%Y-%m-%d").replace(
                hour=ev["hour"], minute=ev["minute"], second=0
            )
        )
        # If today's event already passed, use next
        if ev_dt <= now_et and len(future) > 1:
            ev = future[1]
            ev_dt = ET.localize(
                datetime.strptime(ev["date"], "%Y-%m-%d").replace(
                    hour=ev["hour"], minute=ev["minute"], second=0
                )
            )
        delta = ev_dt - now_et
        secs  = max(0, int(delta.total_seconds()))
        return {
            "name":    ev["event"],
            "date":    ev_dt.strftime("%B %-d, %Y"),
            "time":    ev_dt.strftime("%-I:%M %p ET"),
            "days":    secs // 86400,
            "hours":   (secs % 86400) // 3600,
            "minutes": (secs % 3600) // 60,
            "seconds": secs % 60,
            "total":   secs,
        }
    except Exception as e:
        return {"name": "—", "error": str(e)}


def _log_lines(path: Path, n: int = 35) -> list:
    try:
        if not path.exists():
            return []
        lines = [l for l in path.read_text().splitlines() if l.strip()]
        return lines[-n:]
    except Exception:
        return []


def _last_signal() -> str:
    for line in reversed(_log_lines(ICT_LOG, 60)):
        if "Signal:" in line:
            return line[line.find("Signal:"):].strip()
    return "Signal: —"


# ── API routes ─────────────────────────────────────────────────────────────────

@app.get("/api/data")
def api_data():
    return JSONResponse({
        "account":     _account_data(),
        "event":       _next_event(),
        "last_signal": _last_signal(),
        "ts":          datetime.now(ET).strftime("%Y-%m-%d %H:%M:%S ET"),
    })


@app.get("/api/logs")
def api_logs():
    ict      = _log_lines(ICT_LOG, 35)
    straddle = _log_lines(STRADDLE_LOG, 10)
    return JSONResponse({"ict": ict, "straddle": straddle})


# ── HTML ───────────────────────────────────────────────────────────────────────

HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Kronos</title>
<style>
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
:root{
  --bg:#131722;--surface:#1e222d;--surface2:#2a2e39;
  --border:#2a2e39;--border2:#363a45;
  --text:#d1d4dc;--muted:#787b86;
  --green:#26a69a;--red:#ef5350;--blue:#2962ff;--orange:#ff9800;
  --ui:-apple-system,BlinkMacSystemFont,'Segoe UI','Trebuchet MS',sans-serif;
  --mono:'Courier New','Consolas',monospace;
}
body{background:var(--bg);color:var(--text);font-family:var(--ui);font-size:13px;min-height:100vh;display:flex;flex-direction:column}

/* topbar */
.topbar{height:46px;background:var(--surface);border-bottom:1px solid var(--border);display:flex;align-items:center;justify-content:space-between;padding:0 20px;flex-shrink:0}
.logo{font-size:13px;font-weight:700;letter-spacing:.1em;color:var(--text);display:flex;align-items:center;gap:8px}
.live-dot{width:7px;height:7px;border-radius:50%;background:var(--green);animation:pulse 2s ease-in-out infinite}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.25}}
.topbar-r{display:flex;align-items:center;gap:20px;color:var(--muted);font-size:12px}
.topbar-clock{color:var(--text);font-variant-numeric:tabular-nums;font-family:var(--mono)}

/* main */
.main{padding:16px;flex:1;display:flex;flex-direction:column;gap:12px}

/* metric strip */
.metrics{display:grid;grid-template-columns:repeat(4,1fr);gap:1px;background:var(--border);border:1px solid var(--border);border-radius:6px;overflow:hidden}
.metric{background:var(--surface);padding:14px 18px}
.metric-label{font-size:11px;color:var(--muted);letter-spacing:.04em;margin-bottom:6px}
.metric-value{font-size:21px;font-weight:600;font-variant-numeric:tabular-nums;line-height:1.2}
.metric-sub{font-size:11px;color:var(--muted);margin-top:3px}

.up{color:var(--green)!important}
.dn{color:var(--red)!important}
.dim{color:var(--muted)!important}

/* mid row */
.mid{display:grid;grid-template-columns:1fr 300px;gap:12px}
.left-stack{display:flex;flex-direction:column;gap:12px}
.right-stack{display:flex;flex-direction:column;gap:12px}

/* panel */
.panel{background:var(--surface);border:1px solid var(--border);border-radius:6px;overflow:hidden}
.ph{padding:10px 16px;border-bottom:1px solid var(--border);display:flex;align-items:center;justify-content:space-between}
.pt{font-size:11px;font-weight:600;letter-spacing:.07em;text-transform:uppercase;color:var(--muted)}
.pb-tag{font-size:10px;padding:2px 7px;border-radius:3px;background:var(--surface2);color:var(--muted)}
.pb{padding:14px 16px}
.pb-0{padding:0 16px}

/* table */
.tv-table{width:100%;border-collapse:collapse}
.tv-table th{text-align:left;font-size:11px;font-weight:500;color:var(--muted);padding:0 12px 8px 0;letter-spacing:.03em;border-bottom:1px solid var(--border)}
.tv-table td{padding:8px 12px 8px 0;font-size:12px;border-bottom:1px solid var(--border);font-variant-numeric:tabular-nums}
.tv-table tr:last-child td{border-bottom:none}
.tv-table tbody tr:hover{background:rgba(255,255,255,.02)}

.badge{display:inline-flex;align-items:center;padding:2px 8px;border-radius:3px;font-size:10px;font-weight:600;letter-spacing:.04em}
.badge-long{background:rgba(38,166,154,.15);color:var(--green)}
.badge-short{background:rgba(239,83,80,.15);color:var(--red)}
.c-buy{color:var(--green)}
.c-sell{color:var(--red)}

/* progress */
.prog-row{display:flex;justify-content:space-between;align-items:baseline;margin-bottom:8px}
.prog-big{font-size:28px;font-weight:700;font-variant-numeric:tabular-nums}
.prog-sub{font-size:12px;color:var(--muted)}
.pbar{width:100%;height:4px;background:var(--surface2);border-radius:2px;margin:6px 0 8px}
.pbar-fill{height:4px;background:var(--blue);border-radius:2px;transition:width .6s ease}
.sig-line{margin-top:10px;padding-top:10px;border-top:1px solid var(--border);font-size:11px;color:var(--muted);font-family:var(--mono);white-space:nowrap;overflow:hidden;text-overflow:ellipsis}

/* event */
.ev-name{font-size:22px;font-weight:700;letter-spacing:.03em;margin-bottom:4px}
.ev-cd{font-size:16px;font-variant-numeric:tabular-nums;color:var(--green);font-weight:500;font-family:var(--mono);margin-bottom:8px}
.ev-meta{font-size:12px;color:var(--muted);line-height:1.9}

/* log */
.log-feed{font-family:var(--mono);font-size:11px;line-height:1.7;height:190px;overflow-y:auto;padding:12px 16px}
.ll{color:var(--muted);white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.ll.order{color:var(--green)}
.ll.halt{color:var(--red)}
.ll.warn{color:var(--orange)}
.ll.sig{color:var(--text)}

/* scrollbar */
::-webkit-scrollbar{width:4px;height:4px}
::-webkit-scrollbar-track{background:transparent}
::-webkit-scrollbar-thumb{background:var(--surface2);border-radius:2px}

/* footer */
.footer{padding:8px 20px;border-top:1px solid var(--border);display:flex;justify-content:space-between;font-size:10px;color:var(--muted);letter-spacing:.04em;flex-shrink:0}

.empty{color:var(--muted);font-size:12px;padding:12px 0}
.loading{color:var(--muted);font-size:12px;animation:fade 1s step-end infinite}
@keyframes fade{50%{opacity:0}}

@media(max-width:900px){
  .metrics{grid-template-columns:repeat(2,1fr)}
  .mid{grid-template-columns:1fr}
}
</style>
</head>
<body>

<div class="topbar">
  <div class="logo">
    <div class="live-dot"></div>
    Kronos
  </div>
  <div class="topbar-r">
    <span class="topbar-clock" id="clock">—</span>
    <span id="upd" style="color:var(--muted)">connecting...</span>
  </div>
</div>

<div class="main">

  <div class="metrics" id="metrics">
    <div class="metric"><div class="metric-label">Equity</div><div class="metric-value loading">—</div></div>
    <div class="metric"><div class="metric-label">Today</div><div class="metric-value loading">—</div></div>
    <div class="metric"><div class="metric-label">From $100K</div><div class="metric-value loading">—</div></div>
    <div class="metric"><div class="metric-label">Peak DD</div><div class="metric-value loading">—</div></div>
  </div>

  <div class="mid">
    <div class="left-stack">
      <div class="panel" id="c-positions">
        <div class="ph"><span class="pt">Open Positions</span></div>
        <div class="pb"><div class="loading">loading...</div></div>
      </div>
      <div class="panel" id="c-fills">
        <div class="ph"><span class="pt">Recent Fills</span><span class="pb-tag">7 days</span></div>
        <div class="pb"><div class="loading">loading...</div></div>
      </div>
    </div>
    <div class="right-stack">
      <div class="panel" id="c-ict">
        <div class="ph"><span class="pt">ICT Paper</span></div>
        <div class="pb"><div class="loading">loading...</div></div>
      </div>
      <div class="panel" id="c-ev">
        <div class="ph"><span class="pt">Next Event</span></div>
        <div class="pb"><div class="loading">loading...</div></div>
      </div>
    </div>
  </div>

  <div class="panel">
    <div class="ph"><span class="pt">ICT Log</span><span class="pb-tag" id="log-ts">—</span></div>
    <div class="log-feed" id="log-ict"></div>
  </div>

</div>

<div class="footer">
  <span>Kronos · Alpaca Paper · VPS 144.202.3.195</span>
  <span>auto-refresh 30s</span>
</div>

<script>
(function tick(){
  document.getElementById('clock').textContent=
    new Intl.DateTimeFormat('en-US',{timeZone:'America/New_York',hour:'2-digit',minute:'2-digit',second:'2-digit',hour12:false}).format(new Date())+' ET';
  setTimeout(tick,1000);
})();

let evSec=0,cdTimer=null;
function startCd(s){
  evSec=s;
  if(cdTimer)clearInterval(cdTimer);
  cdTimer=setInterval(()=>{
    evSec=Math.max(0,evSec-1);
    const el=document.getElementById('cd');
    if(el)el.textContent=fmtCd(evSec);
  },1000);
}
function fmtCd(s){const d=Math.floor(s/86400),h=Math.floor((s%86400)/3600),m=Math.floor((s%3600)/60),sc=s%60;return d+'d '+p2(h)+'h '+p2(m)+'m '+p2(sc)+'s';}
function p2(n){return String(n).padStart(2,'0');}
function fmtMoney(n,sign=true){const abs='$'+Math.abs(n).toLocaleString('en-US',{minimumFractionDigits:2,maximumFractionDigits:2});return sign?(n>=0?'+':'-')+abs:abs;}
function fmtPct(n,sign=true){return(sign&&n>=0?'+':'')+n.toFixed(2)+'%';}

async function loadData(){
  try{
    const r=await fetch('/api/data'),d=await r.json();
    document.getElementById('upd').textContent='updated '+d.ts.split(' ')[1]+' ET';
    renderMetrics(d.account);
    renderPositions(d.account.positions||[]);
    renderFills(d.account.fills||[]);
    renderICT(d.account,d.last_signal);
    renderEvent(d.event);
  }catch(e){document.getElementById('upd').textContent='error';}
}

async function loadLogs(){
  try{
    const r=await fetch('/api/logs'),d=await r.json();
    const lines=d.ict||[];
    const feed=document.getElementById('log-ict');
    feed.innerHTML=lines.map(l=>{
      let c='ll';
      if(l.includes('[ORDER]')||l.includes('FILL'))c+=' order';
      else if(l.includes('[HALT]')||l.includes('ERROR')||l.includes('breach'))c+=' halt';
      else if(l.includes('EMBARGO')||l.includes('[WARN]')||l.includes('[NEAR_MISS]'))c+=' warn';
      else if(l.includes('Signal:')&&!l.includes('HOLD'))c+=' sig';
      return`<div class="${c}">${l.replace(/</g,'&lt;').replace(/>/g,'&gt;')}</div>`;
    }).join('');
    feed.scrollTop=feed.scrollHeight;
    const last=lines.length&&lines[lines.length-1].match(/\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}/);
    if(last)document.getElementById('log-ts').textContent=last[0];
  }catch(e){}
}

function renderMetrics(a){
  if(a.error)return;
  const m=document.getElementById('metrics');
  const pos=a.positions&&a.positions.length;
  m.innerHTML=`
    <div class="metric">
      <div class="metric-label">Equity</div>
      <div class="metric-value">$${a.equity.toLocaleString('en-US',{minimumFractionDigits:2,maximumFractionDigits:2})}</div>
      <div class="metric-sub">${pos?pos+' position'+(pos>1?'s':'')+' open':'flat'}</div>
    </div>
    <div class="metric">
      <div class="metric-label">Today</div>
      <div class="metric-value ${a.pnl>=0?'up':'dn'}">${fmtMoney(a.pnl)}</div>
      <div class="metric-sub ${a.pnl>=0?'up':'dn'}">${fmtPct(a.pnl_pct)}</div>
    </div>
    <div class="metric">
      <div class="metric-label">From $100K</div>
      <div class="metric-value ${a.from_100k>=0?'up':'dn'}">${fmtMoney(a.from_100k)}</div>
      <div class="metric-sub dim">all-time</div>
    </div>
    <div class="metric">
      <div class="metric-label">Peak DD</div>
      <div class="metric-value ${a.dd_pct>=0?'dim':'dn'}">${fmtPct(a.dd_pct,false)}</div>
      <div class="metric-sub dim">from $100K</div>
    </div>`;
}

function renderPositions(positions){
  const el=document.getElementById('c-positions');
  if(!positions.length){
    el.innerHTML=`<div class="ph"><span class="pt">Open Positions</span></div><div class="pb"><div class="empty">No open positions</div></div>`;return;
  }
  const rows=positions.map(p=>`<tr>
    <td><span class="badge ${p.side==='long'?'badge-long':'badge-short'}">${p.side.toUpperCase()}</span></td>
    <td style="font-weight:600">${p.symbol}</td>
    <td style="color:var(--muted)">${Math.abs(p.qty).toLocaleString()}</td>
    <td>$${p.entry.toFixed(2)}</td>
    <td class="${p.pnl>=0?'up':'dn'}">${fmtMoney(p.pnl)}</td>
    <td class="${p.pnl_pct>=0?'up':'dn'}" style="font-size:11px;color:var(--muted)">${fmtPct(p.pnl_pct)}</td>
  </tr>`).join('');
  el.innerHTML=`
    <div class="ph"><span class="pt">Open Positions</span><span class="pb-tag">${positions.length}</span></div>
    <div class="pb-0"><table class="tv-table">
      <thead><tr><th>Side</th><th>Symbol</th><th>Qty</th><th>Entry</th><th>P&L</th><th>%</th></tr></thead>
      <tbody>${rows}</tbody>
    </table></div>`;
}

function renderFills(fills){
  const el=document.getElementById('c-fills');
  if(!fills.length){
    el.innerHTML=`<div class="ph"><span class="pt">Recent Fills</span><span class="pb-tag">7 days</span></div><div class="pb"><div class="empty">No fills</div></div>`;return;
  }
  const rows=fills.map(f=>`<tr>
    <td style="color:var(--muted);font-size:11px">${f.time}</td>
    <td class="${f.side==='BUY'?'c-buy':'c-sell'}">${f.side}</td>
    <td style="color:var(--muted)">${f.qty.toLocaleString()}</td>
    <td style="font-weight:600">${f.symbol}</td>
    <td>$${f.price.toFixed(2)}</td>
  </tr>`).join('');
  el.innerHTML=`
    <div class="ph"><span class="pt">Recent Fills</span><span class="pb-tag">7 days</span></div>
    <div class="pb-0"><table class="tv-table">
      <thead><tr><th>Time</th><th>Side</th><th>Qty</th><th>Symbol</th><th>Price</th></tr></thead>
      <tbody>${rows}</tbody>
    </table></div>`;
}

function renderICT(a,sig){
  const n=a.spy_count||0,t=30,pct=Math.min(100,n/t*100);
  document.getElementById('c-ict').innerHTML=`
    <div class="ph"><span class="pt">ICT Paper</span></div>
    <div class="pb">
      <div class="prog-row">
        <div><span class="prog-big">${n}</span><span class="prog-sub"> / ${t}</span></div>
        <span style="font-size:11px;color:var(--muted)">${pct.toFixed(0)}%</span>
      </div>
      <div class="pbar"><div class="pbar-fill" style="width:${pct}%"></div></div>
      <div style="font-size:11px;color:var(--muted)">${t-n} trades to challenge</div>
      <div class="sig-line" title="${sig||'—'}">${sig||'—'}</div>
    </div>`;
}

function renderEvent(e){
  const el=document.getElementById('c-ev');
  if(!e||e.name==='—'){
    el.innerHTML=`<div class="ph"><span class="pt">Next Event</span></div><div class="pb"><div class="empty">No events scheduled</div></div>`;return;
  }
  el.innerHTML=`
    <div class="ph"><span class="pt">Next Event</span></div>
    <div class="pb">
      <div class="ev-name">${e.name}</div>
      <div class="ev-cd" id="cd">${fmtCd(e.total||0)}</div>
      <div class="ev-meta">${e.date}</div>
      <div class="ev-meta">${e.time}</div>
    </div>`;
  if(e.total)startCd(e.total);
}

loadData();loadLogs();
setInterval(loadData,30000);
setInterval(loadLogs,8000);
</script>
</body>
</html>"""


@app.get("/")
def root():
    return HTMLResponse(HTML)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=3000, log_level="error")
