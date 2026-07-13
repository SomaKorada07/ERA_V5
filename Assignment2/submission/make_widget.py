#!/usr/bin/env python3
"""
Build a self-contained index.html widget from tokenizer.json + corpus + metrics.

The widget:
  * lets the grader DOWNLOAD tokenizer.json (embedded as a Blob),
  * shows the fertility table and score,
  * RE-COMPUTES every fertility live in the browser with a faithful JS
    re-implementation of the exact pipeline (NFKC + strip-punctuation
    normalizer -> whitespace pre-tokenizer -> BPE merges), proving the numbers
    are reproducible from tokenizer.json alone,
  * has a live tokenization playground,
  * has a searchable 10,000-token vocabulary browser.

Run:
    python make_widget.py       # writes index.html
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CORPUS = ROOT / "corpus"
LANGS = ["en", "hi", "te", "la"]
NAMES = {"en": "English", "hi": "Hindi", "te": "Telugu", "la": "Latin"}

tokenizer_json = (ROOT / "tokenizer.json").read_text(encoding="utf-8")
metrics = json.loads((ROOT / "metrics.json").read_text(encoding="utf-8"))
corpus = {c: (CORPUS / f"{c}.faithful.txt").read_text(encoding="utf-8") for c in LANGS}

# also write a plain tokens.txt vocab list
model = json.loads(tokenizer_json)["model"]
items = sorted(model["vocab"].items(), key=lambda kv: kv[1])
(ROOT / "tokens.txt").write_text(
    "\n".join(f"{i}\t{t}" for t, i in items), encoding="utf-8"
)

payload = {
    "tokenizer": tokenizer_json,
    "metrics": metrics,
    "corpus": corpus,
    "names": NAMES,
    "langs": LANGS,
}

HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>ERA V5 A2 — India BPE Tokenizer (English · Hindi · Telugu · Latin)</title>
<style>
  :root{
    --bg:#0f1419; --panel:#171d26; --panel2:#1e2632; --ink:#e6edf3; --mut:#8b97a7;
    --line:#2a3543; --accent:#4aa8ff; --good:#3fb950; --warn:#e3b341; --bad:#f85149;
    --chip:#243244;
  }
  *{box-sizing:border-box}
  body{margin:0;background:var(--bg);color:var(--ink);
       font:15px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif}
  .wrap{max-width:1040px;margin:0 auto;padding:28px 20px 80px}
  h1{font-size:22px;margin:0 0 4px}
  h2{font-size:16px;margin:26px 0 10px;color:var(--ink)}
  .sub{color:var(--mut);font-size:14px;margin:0 0 18px}
  .panel{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:18px;margin:14px 0}
  .kpis{display:flex;gap:12px;flex-wrap:wrap}
  .kpi{flex:1;min-width:150px;background:var(--panel2);border:1px solid var(--line);
       border-radius:10px;padding:14px}
  .kpi .v{font-size:26px;font-weight:700}
  .kpi .l{color:var(--mut);font-size:12px;margin-top:2px}
  .ok{color:var(--good)} .bad{color:var(--bad)} .warn{color:var(--warn)}
  table{width:100%;border-collapse:collapse;font-size:14px}
  th,td{padding:9px 8px;border-bottom:1px solid var(--line);text-align:right}
  th:first-child,td:first-child{text-align:left}
  th{color:var(--mut);font-weight:600}
  .bar{height:8px;border-radius:6px;background:var(--accent);opacity:.85}
  .btn{background:var(--accent);color:#00121f;border:0;border-radius:8px;padding:10px 16px;
       font-weight:700;cursor:pointer;font-size:14px}
  .btn.ghost{background:transparent;color:var(--accent);border:1px solid var(--accent)}
  textarea,input{width:100%;background:var(--panel2);color:var(--ink);border:1px solid var(--line);
       border-radius:8px;padding:10px;font:14px/1.5 ui-monospace,Menlo,Consolas,monospace}
  .toks{display:flex;flex-wrap:wrap;gap:4px;margin-top:12px}
  .tok{background:var(--chip);border:1px solid var(--line);border-radius:6px;padding:2px 6px;
       font:12px ui-monospace,Menlo,Consolas,monospace;white-space:pre}
  .tok.unk{background:#3a1414;border-color:var(--bad);color:#ffb4ae}
  .row{display:flex;gap:10px;align-items:center;flex-wrap:wrap}
  .muted{color:var(--mut)} .mono{font-family:ui-monospace,Menlo,Consolas,monospace}
  code{background:var(--panel2);padding:1px 5px;border-radius:5px}
  .pill{display:inline-block;background:var(--chip);border:1px solid var(--line);
        border-radius:999px;padding:2px 10px;font-size:12px;margin-right:6px}
  a{color:var(--accent)}
</style>
</head>
<body>
<div class="wrap">
  <h1>India BPE Tokenizer — English · Hindi · Telugu · Latin</h1>
  <p class="sub">ERA V5 · Assignment 2 (resubmission). One shared <b>10,000-token</b> BPE
  tokenizer, trained and evaluated on the <b>faithful Wikipedia Markdown</b> corpus.
  Every number below is re-computed live in your browser from the embedded
  <code>tokenizer.json</code>.</p>

  <div class="row">
    <button class="btn" id="dl">⬇ Download tokenizer.json</button>
    <button class="btn ghost" id="verify">↻ Re-compute fertilities live</button>
    <span class="pill" id="vsize"></span>
    <span class="pill" id="hpill"></span>
  </div>

  <div class="panel">
    <div class="kpis">
      <div class="kpi"><div class="v" id="k_score">–</div><div class="l">raw score = 1000 / spread</div></div>
      <div class="kpi"><div class="v" id="k_spread">–</div><div class="l">spread (max − min fertility)</div></div>
      <div class="kpi"><div class="v" id="k_hi">–</div><div class="l">Hindi fertility (must ≤ 1.2)</div></div>
      <div class="kpi"><div class="v" id="k_adj">–</div><div class="l">adjusted score (after Hindi penalty)</div></div>
    </div>
  </div>

  <h2>Fertility per language <span class="muted" id="liveflag"></span></h2>
  <div class="panel">
    <table id="ftable">
      <thead><tr><th>Language</th><th>tokens</th><th>word-ish units</th>
        <th>fertility</th><th>UNK</th><th style="width:180px">&nbsp;</th></tr></thead>
      <tbody></tbody>
    </table>
    <p class="muted" style="margin:10px 0 0">
      word-ish unit = one <code>[\p{L}\p{M}\p{N}]+</code> run (the grader's denominator).
      fertility = tokens ÷ word-ish units. Score = 1000 ÷ (max − min).
      Hindi &gt; 1.2 is penalised by <code>exp(hindi/1.2 − 1)</code>.
    </p>
  </div>

  <h2>Tokenization playground</h2>
  <div class="panel">
    <textarea id="pg" rows="3">India భారతదేశం भारत — https://en.wikipedia.org/wiki/India</textarea>
    <div class="row" style="margin-top:8px">
      <span class="pill" id="pg_tok"></span>
      <span class="pill" id="pg_word"></span>
      <span class="pill" id="pg_fert"></span>
      <span class="pill" id="pg_unk"></span>
    </div>
    <div class="toks" id="pg_out"></div>
  </div>

  <h2>Vocabulary browser (10,000 tokens)</h2>
  <div class="panel">
    <input id="q" placeholder="search token or id… (e.g. भारत, wiki, 42)"/>
    <div id="vocab" class="toks" style="margin-top:12px;max-height:320px;overflow:auto"></div>
  </div>

  <p class="muted">Reproduce locally: <code>pip install tokenizers regex</code> →
  <code>python evaluate_tokenizer.py</code>. Rebuild from scratch:
  <code>python build_wiki_faithful_markdown.py &amp;&amp; python train_tokenizer.py</code>.</p>
</div>

<script id="payload" type="application/json">__PAYLOAD__</script>
<script>
const DATA = JSON.parse(document.getElementById('payload').textContent);
const TJ = JSON.parse(DATA.tokenizer);
const MODEL = TJ.model;
const VOCAB = MODEL.vocab;                       // token -> id
const ID2TOK = []; for(const t in VOCAB) ID2TOK[VOCAB[t]] = t;
const RANK = new Map();                           // "a\tb" -> rank
MODEL.merges.forEach((m,i)=>{ const p = Array.isArray(m) ? m : m.split(' '); RANK.set(p[0]+'\t'+p[1], i); });
const UNK = MODEL.unk_token || '[UNK]';

// ---- faithful pipeline: NFKC + strip-punct normalizer -> whitespace -> BPE ----
const WORDISH = /[\p{L}\p{M}\p{N}]+/gu;
function normalize(s){ return s.normalize('NFKC').replace(/[^\p{L}\p{M}\p{N}]+/gu,' '); }
function preTokens(s){ return normalize(s).match(WORDISH) || []; }

function bpe(word){
  let syms = Array.from(word);                    // unicode code points
  if(syms.length===0) return [];
  while(true){
    let best=-1, bestPair=null, bestI=-1;
    for(let i=0;i<syms.length-1;i++){
      const r = RANK.get(syms[i]+'\t'+syms[i+1]);
      if(r!==undefined && (best===-1 || r<best)){ best=r; bestPair=[syms[i],syms[i+1]]; bestI=i; }
    }
    if(bestI===-1) break;
    const merged = bestPair[0]+bestPair[1];
    const out=[];
    for(let i=0;i<syms.length;){
      if(i<syms.length-1 && syms[i]===bestPair[0] && syms[i+1]===bestPair[1]){ out.push(merged); i+=2; }
      else { out.push(syms[i]); i++; }
    }
    syms = out;
  }
  return syms;
}
function encode(text){
  const toks=[];
  for(const w of preTokens(text)){
    for(const s of bpe(w)) toks.push(s in VOCAB ? s : UNK);
  }
  return toks;
}
function wordishCount(text){ const m = normalize(text).match(WORDISH); return m ? m.length : 0; }

// ---- render metrics (from metrics.json, then optionally recompute live) ----
const M = DATA.metrics;
document.getElementById('vsize').textContent = 'vocab = ' + M.vocab_size;
function setHiPill(hi){
  const p=document.getElementById('hpill');
  p.textContent = 'Hindi ' + hi.toFixed(4) + (hi<=1.2?' ✓ ≤ 1.2':' ✗ > 1.2');
  p.className = 'pill ' + (hi<=1.2?'ok':'bad');
}
function renderTable(rows, live){
  const order = Object.keys(rows).sort((a,b)=>rows[a].ratio-rows[b].ratio);
  const max = Math.max(...order.map(c=>rows[c].ratio));
  const tb = document.querySelector('#ftable tbody'); tb.innerHTML='';
  for(const c of order){
    const r=rows[c];
    const tr=document.createElement('tr');
    tr.innerHTML = `<td>${DATA.names[c]}</td><td>${r.tokens.toLocaleString()}</td>
      <td>${r.units.toLocaleString()}</td><td>${r.ratio.toFixed(4)}</td>
      <td class="${r.unk?'bad':'ok'}">${r.unk}</td>
      <td><div class="bar" style="width:${(r.ratio/max*100).toFixed(1)}%"></div></td>`;
    tb.appendChild(tr);
  }
  const rr = order.map(c=>rows[c].ratio);
  const spread = Math.max(...rr)-Math.min(...rr);
  const raw = 1000/spread;
  const hi = rows['hi'].ratio;
  const pen = Math.exp(Math.max(0, hi/1.2-1));
  document.getElementById('k_score').textContent = raw.toFixed(0);
  document.getElementById('k_spread').textContent = spread.toFixed(4);
  document.getElementById('k_hi').textContent = hi.toFixed(4);
  document.getElementById('k_hi').className = 'v ' + (hi<=1.2?'ok':'bad');
  document.getElementById('k_adj').textContent = (raw/pen).toFixed(0);
  setHiPill(hi);
  document.getElementById('liveflag').textContent = live ? '— recomputed live in your browser ✓' : '';
}
// initial: from metrics.json
const initRows={};
for(const c of DATA.langs){
  initRows[c]={tokens:M.token_counts[c], units:M.wordish_units[c],
               ratio:M.ratios[c], unk:(M.unk_counts?M.unk_counts[c]:0)};
}
renderTable(initRows,false);

document.getElementById('verify').onclick = ()=>{
  const rows={};
  for(const c of DATA.langs){
    const text=DATA.corpus[c];
    const toks=encode(text);
    rows[c]={tokens:toks.length, units:wordishCount(text),
             ratio:toks.length/wordishCount(text), unk:toks.filter(t=>t===UNK).length};
  }
  renderTable(rows,true);
};

// ---- download ----
document.getElementById('dl').onclick = ()=>{
  const blob=new Blob([DATA.tokenizer],{type:'application/json'});
  const a=document.createElement('a'); a.href=URL.createObjectURL(blob);
  a.download='tokenizer.json'; a.click();
};

// ---- playground ----
const pg=document.getElementById('pg');
function runPg(){
  const text=pg.value;
  const toks=encode(text);
  const words=wordishCount(text);
  const unk=toks.filter(t=>t===UNK).length;
  document.getElementById('pg_tok').textContent = toks.length+' tokens';
  document.getElementById('pg_word').textContent = words+' word-ish';
  document.getElementById('pg_fert').textContent = 'fertility '+(words?(toks.length/words).toFixed(3):'0');
  document.getElementById('pg_unk').textContent = unk+' UNK';
  const out=document.getElementById('pg_out'); out.innerHTML='';
  toks.forEach(t=>{
    const s=document.createElement('span');
    s.className='tok'+(t===UNK?' unk':''); s.textContent=t;
    out.appendChild(s);
  });
}
pg.addEventListener('input',runPg); runPg();

// ---- vocab browser ----
const q=document.getElementById('q'); const vc=document.getElementById('vocab');
function renderVocab(){
  const term=q.value.trim().toLowerCase(); vc.innerHTML='';
  let n=0;
  for(let id=0; id<ID2TOK.length && n<500; id++){
    const t=ID2TOK[id]; if(t===undefined) continue;
    if(term && !(t.toLowerCase().includes(term) || (''+id)===term)) continue;
    const s=document.createElement('span'); s.className='tok';
    s.textContent = id+'·'+t; vc.appendChild(s); n++;
  }
  if(n===0) vc.innerHTML='<span class="muted">no matches</span>';
}
q.addEventListener('input',renderVocab); renderVocab();
</script>
</body>
</html>
"""

html = HTML.replace("__PAYLOAD__", json.dumps(payload, ensure_ascii=False))
(ROOT / "index.html").write_text(html, encoding="utf-8")
print(f"index.html written ({len(html)/1024:.0f} KB); tokens.txt written ({len(items)} tokens)")
