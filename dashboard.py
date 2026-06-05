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
<title>KRONOS TERMINAL</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
:root{
  --bg:#0d1117;--card:#161b22;--border:#30363d;
  --green:#3fb950;--glow:#00ff7f;--red:#f85149;
  --yellow:#e3b341;--blue:#58a6ff;--dim:#8b949e;
  --text:#c9d1d9;--font:'Courier New',monospace;
}
body{background:var(--bg);color:var(--text);font-family:var(--font);font-size:13px;min-height:100vh;display:flex;flex-direction:column}

/* header */
.hdr{background:var(--card);border-bottom:1px solid var(--border);padding:10px 20px;display:flex;justify-content:space-between;align-items:center;flex-shrink:0}
.logo{color:var(--glow);font-size:15px;font-weight:bold;letter-spacing:4px}
.dot{display:inline-block;width:8px;height:8px;background:var(--glow);border-radius:50%;margin-left:8px;animation:blink 2s infinite}
@keyframes blink{0%,100%{opacity:1}50%{opacity:.25}}
.hdr-right{color:var(--dim);font-size:11px;text-align:right}

/* layout */
.main{padding:14px;flex:1;display:flex;flex-direction:column;gap:14px}
.row3{display:grid;grid-template-columns:repeat(3,1fr);gap:14px}
.row1{display:grid;grid-template-columns:1fr;gap:14px}

/* card */
.card{background:var(--card);border:1px solid var(--border);border-radius:8px;padding:15px}
.ct{color:var(--dim);font-size:10px;letter-spacing:2px;text-transform:uppercase;padding-bottom:9px;margin-bottom:11px;border-bottom:1px solid var(--border)}

/* account */
.eq{font-size:26px;font-weight:bold;margin-bottom:4px}
.pnl-up{color:var(--glow)}
.pnl-dn{color:var(--red)}
.rows{margin-top:11px;display:flex;flex-direction:column;gap:5px}
.srow{display:flex;justify-content:space-between}
.sl{color:var(--dim)}
.sv{color:var(--text)}
.sv.up{color:var(--glow)}
.sv.dn{color:var(--red)}

/* progress */
.big{font-size:24px}
.bsub{color:var(--dim);font-size:12px}
.pb-wrap{margin:10px 0 6px}
.pb-labels{display:flex;justify-content:space-between;font-size:10px;color:var(--dim);margin-bottom:3px}
.pb-bg{background:var(--border);border-radius:3px;height:7px}
.pb-fill{background:var(--glow);border-radius:3px;height:7px;transition:width .5s}
.sig{margin-top:9px;padding-top:8px;border-top:1px solid var(--border);color:var(--dim);font-size:11px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}

/* event */
.ev-name{font-size:22px;color:var(--yellow);letter-spacing:3px;margin-bottom:6px}
.cd{font-size:19px;color:var(--glow);margin-bottom:6px;font-variant-numeric:tabular-nums}
.ev-sub{color:var(--dim);font-size:12px;line-height:1.8}

/* table */
table{width:100%;border-collapse:collapse}
th{color:var(--dim);font-size:10px;letter-spacing:1px;text-align:left;padding:3px 7px;border-bottom:1px solid var(--border)}
td{padding:5px 7px;border-bottom:1px solid #1c2128;font-size:12px}
tr:last-child td{border-bottom:none}
.buy{color:var(--glow)}.sell{color:var(--red)}
.pos-l{background:#1a3326;color:var(--glow);padding:1px 6px;border-radius:3px;font-size:10px}
.pos-s{background:#331a1a;color:var(--red);padding:1px 6px;border-radius:3px;font-size:10px}
.empty{color:var(--dim);padding:14px 7px;font-size:12px}

/* log */
.log-box{background:#080c10;border-radius:5px;padding:11px;height:200px;overflow-y:auto;font-size:11px;line-height:1.65}
.ll{color:#3d4f5e;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.ll.order{color:var(--glow)}
.ll.halt{color:var(--red)}
.ll.em{color:var(--yellow)}
.ll.sig{color:var(--text)}
.ll.warn{color:var(--yellow)}

/* footer */
.ftr{background:var(--card);border-top:1px solid var(--border);padding:6px 20px;font-size:10px;color:var(--dim);letter-spacing:1px;text-align:center;flex-shrink:0}

/* loading */
.ld{color:var(--dim);animation:ld 1s step-end infinite}
@keyframes ld{50%{opacity:0}}

@media(max-width:880px){.row3{grid-template-columns:1fr}}
</style>
</head>
<body>

<div class="hdr">
  <div style="display:flex;align-items:center;gap:10px">
    <span class="logo">⬡ KRONOS</span>
    <span class="dot"></span>
    <span style="color:var(--dim);font-size:11px" id="clock">—</span>
  </div>
  <div class="hdr-right">
    <div id="upd">connecting...</div>
  </div>
</div>

<div class="main">

  <div class="row3">
    <div class="card" id="c-acct"><div class="ct">Account</div><div class="ld">fetching...</div></div>
    <div class="card" id="c-ict"><div class="ct">ICT Bot</div><div class="ld">fetching...</div></div>
    <div class="card" id="c-ev"><div class="ct">Next Event</div><div class="ld">fetching...</div></div>
  </div>

  <div class="row1">
    <div class="card" id="c-fills"><div class="ct">Recent Fills · 7 days</div><div class="ld">fetching...</div></div>
  </div>

  <div class="row1">
    <div class="card">
      <div class="ct">ICT Log Feed</div>
      <div class="log-box" id="log-ict"><div class="ll">loading...</div></div>
    </div>
  </div>

</div>

<div class="ftr">KRONOS TERMINAL &nbsp;·&nbsp; ALPACA PAPER &nbsp;·&nbsp; VPS 144.202.3.195 &nbsp;·&nbsp; AUTO-REFRESH 30s</div>

<script>
// clock
(function tick(){
  const s=new Intl.DateTimeFormat('en-US',{timeZone:'America/New_York',hour:'2-digit',minute:'2-digit',second:'2-digit',hour12:false}).format(new Date());
  document.getElementById('clock').textContent=s+' ET';
  setTimeout(tick,1000);
})();

// countdown
let evSec=0, cdTimer=null;
function startCd(s){
  evSec=s;
  if(cdTimer)clearInterval(cdTimer);
  cdTimer=setInterval(()=>{
    evSec=Math.max(0,evSec-1);
    const d=Math.floor(evSec/86400),h=Math.floor((evSec%86400)/3600),m=Math.floor((evSec%3600)/60),s=evSec%60;
    const el=document.getElementById('cd');
    if(el)el.textContent=d+'d '+pad(h)+'h '+pad(m)+'m '+pad(s)+'s';
  },1000);
}
function pad(n){return String(n).padStart(2,'0');}
function f2(n){return n.toFixed(2).replace(/\B(?=(\d{3})+(?!\d))/g,',');}
function dollar(n,always=false){const s=(n>=0?'+':'-');return(n>=0&&always?'+':n<0?'-':'')+'$'+Math.abs(n).toFixed(0).replace(/\B(?=(\d{3})+(?!\d))/g,',');}

async function loadData(){
  try{
    const r=await fetch('/api/data'),d=await r.json();
    document.getElementById('upd').textContent='updated '+d.ts;
    renderAcct(d.account);
    renderICT(d.account,d.last_signal);
    renderEv(d.event);
    renderFills(d.account.fills||[],d.account.positions||[]);
  }catch(e){document.getElementById('upd').textContent='⚠ '+e;}
}

async function loadLogs(){
  try{
    const r=await fetch('/api/logs'),d=await r.json();
    const lines=d.ict||[];
    const feed=document.getElementById('log-ict');
    feed.innerHTML=lines.map(l=>{
      let c='ll';
      if(l.includes('[ORDER]')||l.includes('FILL'))c+=' order';
      else if(l.includes('[HALT]')||l.includes('breach')||l.includes('ERROR'))c+=' halt';
      else if(l.includes('EMBARGO')||l.includes('[WARN]'))c+=' em';
      else if(l.includes('Signal:')&&!l.includes('HOLD'))c+=' sig';
      else if(l.includes('[NEAR_MISS]'))c+=' warn';
      return`<div class="${c}">${l.replace(/</g,'&lt;').replace(/>/g,'&gt;')}</div>`;
    }).join('');
    feed.scrollTop=feed.scrollHeight;
  }catch(e){}
}

function renderAcct(a){
  if(a.error){
    document.getElementById('c-acct').innerHTML=`<div class="ct">Account</div><div style="color:var(--red);font-size:12px">${a.error}</div>`;return;
  }
  const up=a.pnl>=0,dd=a.dd_pct>=0;
  document.getElementById('c-acct').innerHTML=`
    <div class="ct">Account</div>
    <div class="eq">$${f2(a.equity)}</div>
    <div class="${up?'pnl-up':'pnl-dn'}">${dollar(a.pnl,true)} (${a.pnl_pct>=0?'+':''}${a.pnl_pct.toFixed(2)}%)</div>
    <div class="rows">
      <div class="srow"><span class="sl">From $100K</span><span class="sv ${a.from_100k>=0?'up':'dn'}">${dollar(a.from_100k,true)}</span></div>
      <div class="srow"><span class="sl">DD from peak</span><span class="sv ${dd?'up':'dn'}">${a.dd_pct>=0?'+':''}${a.dd_pct.toFixed(2)}%</span></div>
      <div class="srow"><span class="sl">Positions</span><span class="sv">${a.positions&&a.positions.length?a.positions.length+' open':'flat'}</span></div>
    </div>`;
}

function renderICT(a,sig){
  const n=a.spy_count||0,t=30,pct=Math.min(100,n/t*100);
  document.getElementById('c-ict').innerHTML=`
    <div class="ct">ICT Bot · Alpaca Paper</div>
    <div><span class="big">${n}</span><span class="bsub"> / ${t} paper trades</span></div>
    <div class="pb-wrap">
      <div class="pb-labels"><span>PROGRESS TO CHALLENGE</span><span>${pct.toFixed(0)}%</span></div>
      <div class="pb-bg"><div class="pb-fill" style="width:${pct}%"></div></div>
    </div>
    <div class="srow"><span class="sl">Remaining</span><span class="sv">${t-n} trades</span></div>
    <div class="sig" title="${sig||'—'}">${sig||'—'}</div>`;
}

function renderEv(e){
  if(!e||e.name==='—'){
    document.getElementById('c-ev').innerHTML='<div class="ct">Next Event</div><div class="empty">No events scheduled</div>';return;
  }
  document.getElementById('c-ev').innerHTML=`
    <div class="ct">Next Event</div>
    <div class="ev-name">${e.name}</div>
    <div class="cd" id="cd">${e.days}d ${pad(e.hours)}h ${pad(e.minutes)}m ${pad(e.seconds)}s</div>
    <div class="ev-sub">${e.date}</div>
    <div class="ev-sub">${e.time}</div>`;
  if(e.total)startCd(e.total);
}

function renderFills(fills,positions){
  let html='<div class="ct">Recent Fills · 7 days</div>';
  if(positions&&positions.length){
    const pbody=positions.map(p=>`
      <tr>
        <td><span class="${p.side==='long'?'pos-l':'pos-s'}">${p.side.toUpperCase()}</span></td>
        <td>${p.symbol}</td>
        <td>${p.qty}</td>
        <td>$${p.entry.toFixed(2)}</td>
        <td class="${p.pnl>=0?'pnl-up':'pnl-dn'}">${dollar(p.pnl,true)}</td>
      </tr>`).join('');
    html+=`<div style="margin-bottom:10px">
      <div style="color:var(--dim);font-size:10px;letter-spacing:1px;margin-bottom:5px">OPEN POSITIONS</div>
      <table><thead><tr><th>SIDE</th><th>SYM</th><th>QTY</th><th>ENTRY</th><th>PNL</th></tr></thead><tbody>${pbody}</tbody></table>
    </div>`;
  }
  if(!fills||!fills.length){html+='<div class="empty">No fills in last 7 days</div>';}
  else{
    const rows=fills.map(f=>`<tr>
      <td style="color:var(--dim)">${f.time}</td>
      <td class="${f.side==='BUY'?'buy':'sell'}">${f.side}</td>
      <td>${f.qty}</td>
      <td>${f.symbol}</td>
      <td>$${f.price.toFixed(2)}</td>
    </tr>`).join('');
    html+=`<table><thead><tr><th>TIME</th><th>SIDE</th><th>QTY</th><th>SYM</th><th>PRICE</th></tr></thead><tbody>${rows}</tbody></table>`;
  }
  document.getElementById('c-fills').innerHTML=html;
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
