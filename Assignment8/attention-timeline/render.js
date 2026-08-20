/* =====================================================================
   RENDER — legend filters, era bands, five-beat cards, mini diagrams
   ===================================================================== */
(function(){
  const DIMS = window.__DIMS__, DATA = window.__DATA__, ERAS = window.__ERAS__;
  const active = new Set(Object.keys(DIMS)); // all dims on

  /* ---------- tiny inline-SVG diagram library (theme-aware via tokens) --- */
  const C = {
    ink:"var(--ink)", ink2:"var(--ink-2)", muted:"var(--muted)",
    surf:"var(--surface-2)", hair:"var(--hair-strong)"
  };
  function figFor(kind, color){
    const s = (w,h,inner)=>`<svg viewBox="0 0 ${w} ${h}" width="100%" role="img" preserveAspectRatio="xMidYMid meet" style="max-width:520px">${inner}</svg>`;
    if(kind==="heads"){ // MHA / GQA / MQA head sharing
      function block(x,title,groups){ // groups: array of arrays of q-head indices sharing one kv
        let g=`<text x="${x+70}" y="16" text-anchor="middle" font-family="var(--mono)" font-size="12" fill="${C.ink}">${title}</text>`;
        let qy=34, kvcount=groups.length, qcount=groups.reduce((a,b)=>a+b,0);
        let qi=0;
        groups.forEach((gsz,gi)=>{
          const kvY=34+gi*(84/kvcount)+ (84/kvcount)/2 -8;
          // kv box
          g+=`<rect x="${x+108}" y="${34+gi*(84/kvcount)+4}" width="26" height="${84/kvcount-8}" rx="4" fill="${color}" opacity="0.9"/>`;
          for(let k=0;k<gsz;k++){
            const y=34+qi*(84/qcount);
            g+=`<rect x="${x+8}" y="${y+2}" width="26" height="${84/qcount-4}" rx="4" fill="none" stroke="${C.ink2}" stroke-width="1.4"/>`;
            g+=`<line x1="${x+34}" y1="${y+(84/qcount)/2}" x2="${x+108}" y2="${34+gi*(84/kvcount)+(84/kvcount)/2}" stroke="${C.hair}" stroke-width="1.2"/>`;
            qi++;
          }
        });
        g+=`<text x="${x+21}" y="132" text-anchor="middle" font-family="var(--mono)" font-size="10" fill="${C.muted}">Q ×${qcount}</text>`;
        g+=`<text x="${x+121}" y="132" text-anchor="middle" font-family="var(--mono)" font-size="10" fill="${C.muted}">K/V ×${kvcount}</text>`;
        return g;
      }
      return s(560,150,
        block(0,"MHA",[1,1,1,1])+
        block(190,"GQA",[2,2])+
        block(380,"MQA",[4]));
    }
    if(kind==="window"){ // sliding window vs full
      let g=`<text x="169" y="14" text-anchor="middle" font-family="var(--mono)" font-size="11" fill="${C.muted}">each query reads only nearby keys</text>`;
      const n=9, cell=26, ox=52, oy=26, w=2;
      for(let i=0;i<n;i++)for(let j=0;j<n;j++){
        const near=(j<=i)&&(i-j<=w);
        g+=`<rect x="${ox+j*cell}" y="${oy+i*cell}" width="${cell-3}" height="${cell-3}" rx="3" fill="${near?color:C.surf}" opacity="${near?0.92:1}"/>`;
      }
      g+=`<text x="${ox-8}" y="${oy+9}" text-anchor="end" font-family="var(--mono)" font-size="10" fill="${C.muted}">q0</text>`;
      g+=`<text x="${ox-8}" y="${oy+8*cell+9}" text-anchor="end" font-family="var(--mono)" font-size="10" fill="${C.muted}">q8</text>`;
      return s(300,270,g);
    }
    if(kind==="topk"){ // scores bar, keep top-k
      const vals=[0.2,0.9,0.35,0.7,0.15,0.8,0.28,0.5], k=3;
      const sorted=[...vals].sort((a,b)=>b-a); const thresh=sorted[k-1];
      let g=`<text x="180" y="14" text-anchor="middle" font-family="var(--mono)" font-size="11" fill="${C.muted}">score every key · keep the top ${k} · softmax those</text>`;
      vals.forEach((v,i)=>{
        const x=24+i*44, h=v*90, keep=v>=thresh;
        g+=`<rect x="${x}" y="${120-h}" width="30" height="${h}" rx="4" fill="${keep?color:C.surf}" opacity="${keep?0.92:1}"/>`;
        g+=`<text x="${x+15}" y="136" text-anchor="middle" font-family="var(--mono)" font-size="10" fill="${keep?C.ink:C.muted}">k${i}</text>`;
      });
      g+=`<line x1="16" y1="${120-thresh*90}" x2="360" y2="${120-thresh*90}" stroke="${C.ink2}" stroke-width="1" stroke-dasharray="4 3"/>`;
      return s(380,150,g);
    }
    if(kind==="state"){ // growing cache vs fixed state
      let g=`<text x="86" y="14" text-anchor="middle" font-family="var(--mono)" font-size="11" fill="${C.muted}">exact softmax</text>`;
      for(let i=0;i<6;i++) g+=`<rect x="${14+i*26}" y="26" width="22" height="22" rx="4" fill="${C.surf}" stroke="${C.ink2}" stroke-width="1"/>`;
      g+=`<text x="86" y="66" text-anchor="middle" font-family="var(--mono)" font-size="10" fill="${C.muted}">keep every K,V → grows with T</text>`;
      g+=`<text x="300" y="14" text-anchor="middle" font-family="var(--mono)" font-size="11" fill="${C.muted}">linear state</text>`;
      g+=`<rect x="270" y="22" width="60" height="30" rx="6" fill="${color}" opacity="0.9"/>`;
      g+=`<text x="300" y="42" text-anchor="middle" font-family="var(--mono)" font-size="12" fill="#fff">S</text>`;
      g+=`<text x="300" y="66" text-anchor="middle" font-family="var(--mono)" font-size="10" fill="${C.muted}">one fixed matrix, any T</text>`;
      return s(380,80,g);
    }
    if(kind==="delta"){ // 40 -> +15 -> 55
      let g="";
      const box=(x,lab,val,fill)=>`<rect x="${x}" y="24" width="72" height="40" rx="8" fill="${fill}"/><text x="${x+36}" y="42" text-anchor="middle" font-family="var(--mono)" font-size="12" fill="#fff">${lab}</text><text x="${x+36}" y="58" text-anchor="middle" font-family="var(--mono)" font-size="14" fill="#fff">${val}</text>`;
      g+=box(10,"read",40,C.ink2);
      g+=`<text x="118" y="49" text-anchor="middle" font-family="var(--mono)" font-size="13" fill="${C.muted}">gap</text>`;
      g+=box(140,"want−read","+15",color);
      g+=`<text x="252" y="49" text-anchor="middle" font-family="var(--mono)" font-size="16" fill="${C.muted}">→</text>`;
      g+=box(272,"write",55,"var(--good)");
      g+=`<text x="180" y="84" text-anchor="middle" font-family="var(--mono)" font-size="10" fill="${C.muted}">add-only would give 40+55 = 95 (wrong)</text>`;
      return s(360,96,g);
    }
    if(kind==="rope"){ // two arrows same gap, rotated
      function pair(cx,posI,posJ,lab){
        const th=0.55;
        const ai=posI*th, aj=posJ*th, R=34;
        const ax=cx+R*Math.cos(-ai), ay=64+R*Math.sin(-ai);
        const bx=cx+R*Math.cos(-aj), by=64+R*Math.sin(-aj);
        let g=`<circle cx="${cx}" cy="64" r="${R}" fill="none" stroke="${C.hair}" stroke-width="1"/>`;
        g+=`<line x1="${cx}" y1="64" x2="${ax.toFixed(1)}" y2="${ay.toFixed(1)}" stroke="${C.ink2}" stroke-width="2"/>`;
        g+=`<line x1="${cx}" y1="64" x2="${bx.toFixed(1)}" y2="${by.toFixed(1)}" stroke="${color}" stroke-width="2.4"/>`;
        g+=`<text x="${cx}" y="118" text-anchor="middle" font-family="var(--mono)" font-size="10" fill="${C.muted}">${lab}</text>`;
        return g;
      }
      let g=`<text x="150" y="14" text-anchor="middle" font-family="var(--mono)" font-size="11" fill="${C.muted}">same gap → same relative angle → same score</text>`;
      g+=pair(80,2,8,"pos 2 & 8 · gap 6");
      g+=pair(230,12,18,"pos 12 & 18 · gap 6");
      return s(320,130,g);
    }
    if(kind==="compress"){ // T entries -> T/m blocks -> topk read
      let g=`<text x="200" y="14" text-anchor="middle" font-family="var(--mono)" font-size="11" fill="${C.muted}">compress many tokens into blocks · read only top blocks</text>`;
      for(let i=0;i<12;i++) g+=`<rect x="${14+i*20}" y="26" width="16" height="16" rx="3" fill="${C.surf}" stroke="${C.ink2}" stroke-width="0.8"/>`;
      // arrows to 3 blocks
      const blocks=[{x:60,sel:false},{x:180,sel:true},{x:300,sel:false}];
      blocks.forEach((b,bi)=>{
        g+=`<rect x="${b.x}" y="78" width="44" height="26" rx="6" fill="${b.sel?color:C.surf}" stroke="${b.sel?color:C.ink2}" stroke-width="1.2" opacity="${b.sel?0.95:1}"/>`;
        g+=`<text x="${b.x+22}" y="95" text-anchor="middle" font-family="var(--mono)" font-size="10" fill="${b.sel?'#fff':C.muted}">blk${bi}</text>`;
        for(let k=0;k<4;k++) g+=`<line x1="${22+ (bi*4+k)*20}" y1="42" x2="${b.x+22}" y2="78" stroke="${C.hair}" stroke-width="0.8"/>`;
      });
      g+=`<text x="202" y="120" text-anchor="middle" font-family="var(--mono)" font-size="10" fill="${C.muted}">a cheap indexer ranks blocks; exact attention reads only the chosen one</text>`;
      return s(400,130,g);
    }
    return "";
  }

  /* ---------- LEGEND ---------- */
  const legend = document.getElementById('legend');
  Object.entries(DIMS).forEach(([k,d])=>{
    const b=document.createElement('button');
    b.className='chip'; b.setAttribute('aria-pressed','true'); b.dataset.dim=k;
    b.innerHTML=`<span class="dot" style="background:${d.color}"></span>${d.label}`;
    b.title=d.blurb;
    b.addEventListener('click',()=>{
      if(active.has(k)){ active.delete(k); b.setAttribute('aria-pressed','false'); }
      else { active.add(k); b.setAttribute('aria-pressed','true'); }
      applyFilter();
    });
    legend.appendChild(b);
  });

  /* ---------- TIMELINE ---------- */
  const root = document.getElementById('tlRoot');
  const cards = DATA.slice().sort((a,b)=> a.sort<b.sort?-1:1);
  let eraIdx=0;
  const beat=(cls,lab,txt)=>`<div class="beat ${cls}"><div class="bl">${lab}</div><div class="bt">${txt}</div></div>`;

  cards.forEach((c,i)=>{
    // era band
    while(eraIdx<ERAS.length && c.sort >= ERAS[eraIdx].from){
      const e=ERAS[eraIdx];
      const el=document.createElement('div'); el.className='era';
      el.innerHTML=`<div class="yr">${e.year}</div><div class="band">${e.text}</div>`;
      root.appendChild(el); eraIdx++;
    }
    const d=DIMS[c.dim];
    const row=document.createElement('div'); row.className='card-row'; row.dataset.dim=c.dim;
    row.style.setProperty('--dot', d.color);
    const dateFlag = c.approx?' ≈':'';
    const fig = c.fig?`<div class="minifig">${figFor(c.fig, d.color)}</div>`:'';
    row.innerHTML=`
      <div class="when"><span class="date">${c.date}${dateFlag}</span><span class="idx">#${String(i+1).padStart(2,'0')}</span></div>
      <article class="card">
        <div class="card-top">
          <div class="name"><span class="abbr">${c.abbr}</span><span class="full">${c.full}</span>
            <span class="dimtag" style="background:${d.color}">${d.label}</span></div>
          <div class="src">${c.source.title} · <span class="mono">${c.source.id}</span> · <a href="${c.source.url}" target="_blank" rel="noopener">source ↗</a></div>
        </div>
        <div class="beats">
          ${beat('existed','What existed', c.beats.existed)}
          ${beat('problem','The problem', c.beats.problem)}
          ${beat('mech','New mechanism', c.beats.mechanism)}
          ${beat('fixed','What it fixed', c.beats.fixed)}
          ${beat('cost','The new cost', c.beats.cost)}
        </div>
        ${fig}
        <div class="trade">
          <div class="buys"><div class="tl-h">Buys</div><p>${c.trade.buys}</p></div>
          <div class="gives"><div class="tl-h">Gives up</div><p>${c.trade.gives}</p></div>
          <div class="when-h"><div class="tl-h">Choose when</div><p>${c.trade.when}</p></div>
        </div>
        ${c.note?`<div class="note"><b>Note ·</b> ${c.note}</div>`:''}
      </article>`;
    root.appendChild(row);
  });
  // any trailing eras (none expected)
  while(eraIdx<ERAS.length){ eraIdx++; }

  function applyFilter(){
    document.querySelectorAll('.card-row').forEach(r=>{
      r.style.display = active.has(r.dataset.dim)?'':'none';
    });
  }

  /* ---------- course honesty note ---------- */
  document.getElementById('courseNote').innerHTML =
    `<b>On the course's own model.</b> Session 8 also names techniques from an internal architecture — a "V4" <em>Memory Stream</em>, a <code>DDDGDDDG</code> depth schedule, and "V4 Compressed Sparse Attention". These aren't dated public launches, so they aren't pins on this timeline; the closest <em>published</em> anchors are shown instead — MLA and NSA/DSA for compression &amp; sparsity, Gated DeltaNet for fixed-state layers. The notes themselves say the exact <b>DroPE</b> algorithm "is not established" in their record, so its card here uses the matching public paper and flags the rest as hypothesis. Where a date is a forum/blog post rather than a paper (NTK-aware; the Mistral sliding-window blog), it is marked <b>≈</b> and explained in the card.`;
})();
