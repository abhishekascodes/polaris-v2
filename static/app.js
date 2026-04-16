/* OpenENV Dashboard — Light Terminal JS Engine */
const C={amber:'#d97706',teal:'#0d9488',rose:'#e11d48',sky:'#0284c7',lime:'#65a30d',violet:'#7c3aed',orange:'#ea580c',grid:'#e4e4e7',tick:'#71717a'};
let ws,ch={},D={s:[],sat:[],gdp:[],pol:[],rew:[],tru:[],coal:0,veto:0,bet:0,evt:0,hist:[]};
const MX=250;

function tab(id){document.querySelectorAll('.tab').forEach(t=>t.classList.toggle('on',t.dataset.p===id));document.querySelectorAll('.pane').forEach(p=>p.classList.toggle('on',p.id==='p-'+id));}
document.addEventListener('keydown',e=>{if(e.target.tagName==='INPUT')return;const m={'1':'live','2':'war','3':'causal','4':'metrics','5':'risk','6':'hist'};if(m[e.key])tab(m[e.key]);if(e.key===' '){e.preventDefault();pause();}});

function conn(){
  ws=new WebSocket(`ws://${location.host}/ws`);
  ws.onopen=()=>{document.getElementById('wsd').className='dot up';document.getElementById('wsl').textContent='ONLINE';};
  ws.onclose=()=>{document.getElementById('wsd').className='dot dn';document.getElementById('wsl').textContent='RECONNECTING';setTimeout(conn,2000);};
  ws.onmessage=e=>{const m=JSON.parse(e.data);({init:onInit,step:onStep,episode_start:onStart,episode_end:onEnd,pause:onPause}[m.type]||(_=>{}))(m);};
}
function tx(o){if(ws&&ws.readyState===1)ws.send(JSON.stringify(o));}

function go(){tx({cmd:'start',seed:+document.getElementById('seed').value||42,chaos:parseFloat(document.getElementById('v-chaos').textContent)});}
function pause(){tx({cmd:'pause'});}
function reset(){clearD();tx({cmd:'reset',seed:+document.getElementById('seed').value||42});}
function spd(v){tx({cmd:'speed',value:v});document.querySelectorAll('.controls .btn').forEach(b=>{if(b.textContent.includes('×'))b.classList.toggle('active',b.textContent.trim()===v+'×');});}
function regime(v){tx({cmd:'regime',regime:v});}
document.getElementById('chaos-r').addEventListener('input',e=>{const v=(e.target.value/100).toFixed(2);document.getElementById('v-chaos').textContent=v;tx({cmd:'chaos',value:+v});});
function scrub(i){i=+i;if(D.hist[i])onStep(D.hist[i],1);}
function exp(){const b=new Blob([JSON.stringify(D.hist,null,2)],{type:'application/json'});const a=document.createElement('a');a.href=URL.createObjectURL(b);a.download='openenv_trace.json';a.click();}

function lineOpt(color,yMin,yMax){return{type:'line',data:{labels:[],datasets:[{data:[],borderColor:color,backgroundColor:color+'15',fill:true,tension:.4,pointRadius:0,borderWidth:1.5}]},options:{responsive:true,maintainAspectRatio:false,animation:false,scales:{x:{display:false},y:{min:yMin,max:yMax,grid:{color:C.grid,lineWidth:.5},ticks:{color:C.tick,font:{family:'IBM Plex Mono',size:9}},border:{color:C.grid}}},plugins:{legend:{display:false}}}};}

function initCharts(){
  ch.sat=new Chart(document.getElementById('c-sat'),lineOpt(C.teal,0,100));
  ch.gdp=new Chart(document.getElementById('c-gdp'),lineOpt(C.sky,0,200));
  ch.pol=new Chart(document.getElementById('c-pol'),lineOpt(C.rose,0,500));
  ch.rt=new Chart(document.getElementById('c-rt'),{type:'line',data:{labels:[],datasets:[
    {label:'Reward',data:[],borderColor:C.amber,tension:.4,pointRadius:0,borderWidth:1.5,fill:false},
    {label:'Trust',data:[],borderColor:C.teal,tension:.4,pointRadius:0,borderWidth:1.5,fill:false,borderDash:[3,3]}
  ]},options:{responsive:true,maintainAspectRatio:false,animation:false,scales:{x:{display:false},y:{min:0,max:1,grid:{color:C.grid,lineWidth:.5},ticks:{color:C.tick,font:{family:'IBM Plex Mono',size:9}},border:{color:C.grid}}},plugins:{legend:{position:'top',labels:{color:'#3f3f46',font:{family:'IBM Plex Mono',size:9},boxWidth:12,padding:8}}}}});
  
  ch.inf=new Chart(document.getElementById('c-inf'),{type:'radar',data:{labels:['M0','M1','M2','M3','M4'],datasets:[{data:[.5,.5,.5,.5,.5],borderColor:C.amber,backgroundColor:C.amber+'20',pointBackgroundColor:C.amber,pointRadius:3,borderWidth:1.5}]},options:{responsive:true,maintainAspectRatio:false,scales:{r:{min:0,max:1,grid:{color:C.grid},angleLines:{color:C.grid},pointLabels:{color:'#a1a1aa',font:{family:'IBM Plex Mono',size:10}},ticks:{display:false}}},plugins:{legend:{display:false}},animation:{duration:200}}});
  
  ch.radar=new Chart(document.getElementById('c-radar'),{type:'radar',data:{labels:['Economy','Environment','Social','Health','Cooperation'],datasets:[{label:'Current',data:[.5,.5,.5,.5,.5],borderColor:C.amber,backgroundColor:C.amber+'18',borderWidth:1.5},{label:'Baseline',data:[.3,.3,.3,.3,.3],borderColor:'#3f3f46',backgroundColor:'#3f3f4622',borderDash:[3,3],borderWidth:1}]},options:{responsive:true,maintainAspectRatio:false,scales:{r:{min:0,max:1,grid:{color:C.grid},angleLines:{color:C.grid},pointLabels:{color:'#a1a1aa',font:{family:'IBM Plex Mono',size:9}},ticks:{display:false}}},plugins:{legend:{position:'top',labels:{color:'#71717a',font:{family:'IBM Plex Mono',size:9},boxWidth:10,padding:6}}},animation:{duration:200}}});
  
  ch.rbd=new Chart(document.getElementById('c-rbd'),{type:'bar',data:{labels:['Base','Pareto','Coop','Osc','Total'],datasets:[{data:[0,0,0,0,0],backgroundColor:[C.amber,C.lime,C.teal,C.rose,C.sky],borderRadius:2,barThickness:16}]},options:{responsive:true,maintainAspectRatio:false,indexAxis:'y',scales:{x:{grid:{color:C.grid,lineWidth:.5},ticks:{color:C.tick,font:{family:'IBM Plex Mono',size:9}},border:{color:C.grid}},y:{grid:{display:false},ticks:{color:'#a1a1aa',font:{family:'IBM Plex Mono',size:10}},border:{display:false}}},plugins:{legend:{display:false}},animation:{duration:150}}});
}

function clearD(){D={s:[],sat:[],gdp:[],pol:[],rew:[],tru:[],coal:0,veto:0,bet:0,evt:0,hist:[]};Object.values(ch).forEach(c=>{if(c.data){c.data.labels=[];c.data.datasets.forEach(d=>d.data=[]);c.update();}});document.getElementById('feed').innerHTML='';document.getElementById('agents').innerHTML='';}

function onInit(m){if(m.history)m.history.forEach(s=>onStep(s,1));if(m.episodes)m.episodes.forEach(addRow);}
function onStart(m){clearD();const r=m.task_id.includes('extreme')?'EXTREME':'CALIBRATED';document.getElementById('b-reg').textContent=r;}
function onPause(m){document.getElementById('btn-pause').textContent=m.paused?'▶':'⏸';}

function onStep(s,q){
  D.hist.push(s);
  const push=(a,v)=>{a.push(v);if(a.length>MX)a.shift();};
  const lbl=''+s.step;
  push(D.s,lbl);push(D.sat,s.state.public_satisfaction);push(D.gdp,s.state.gdp_index);push(D.pol,s.state.pollution_index);push(D.rew,s.reward);push(D.tru,s.council.institutional_trust);
  
  const set=(c,d)=>{c.data.labels=D.s.slice();c.data.datasets[0].data=d.slice();c.update();};
  set(ch.sat,D.sat);set(ch.gdp,D.gdp);set(ch.pol,D.pol);
  ch.rt.data.labels=D.s.slice();ch.rt.data.datasets[0].data=D.rew.slice();ch.rt.data.datasets[1].data=D.tru.slice();ch.rt.update();
  
  // top bar
  document.getElementById('v-step').textContent=s.step;
  document.getElementById('v-rew').textContent=s.reward.toFixed(3);
  document.getElementById('v-trust').textContent=s.council.institutional_trust.toFixed(3);
  const st=Math.max(0,Math.min(100,Math.round((s.state.public_satisfaction/100+s.state.gdp_index/200+(1-s.state.pollution_index/500))/3*100)));
  document.getElementById('v-stab').textContent=st+'%';
  const sp=document.getElementById('v-stab').parentElement;
  sp.className='pill '+(st>60?'ok':st>30?'warn':'bad');
  
  // counters
  if(s.council.coalition_formed)D.coal++;
  D.veto+=(s.council.vetoes||[]).length;
  if(s.council.betrayal)D.bet++;
  D.evt+=(s.active_events||[]).length;
  document.getElementById('b-coal').textContent=D.coal;
  document.getElementById('b-veto').textContent=D.veto;
  document.getElementById('b-bet').textContent=D.bet;
  document.getElementById('b-evt').textContent=D.evt;
  
  const iv=s.council.influence_vector||[];
  if(iv.length){document.getElementById('b-inf').textContent=(iv.reduce((a,b)=>a+b,0)/iv.length).toFixed(3);ch.inf.data.datasets[0].data=iv;ch.inf.data.labels=iv.map((_,i)=>'M'+i);ch.inf.update();}
  
  // ministers
  const ms=s.council.ministers||[];
  if(ms.length){document.getElementById('agents').innerHTML=ms.map((m,i)=>`<div class="agent a${i}"><div class="name">${m.name||'Agent '+i}</div><div class="meta">${m.role||'general'}</div><div class="bar"><div class="bar-fill" style="width:${(m.influence*100).toFixed(0)}%"></div></div><div class="stats">INF ${(m.influence||0).toFixed(3)} · ${m.proposal||'—'}</div></div>`).join('');}
  
  // feed
  if(!q){
    const f=document.getElementById('feed');
    const a=s.council_action||s.action;
    let cls='msg',h=`<span class="who">Step ${s.step}</span><span class="ts">${a}</span><br>`;
    if(s.council.coalition_formed){cls+=' coal';h+=`<span class="act">Coalition ▸ str ${s.council.coalition_strength.toFixed(2)}</span>`;}
    if((s.council.vetoes||[]).length){cls+=' veto';h+=`<span class="act">${s.council.vetoes.length} veto(s)</span>`;}
    if(s.council.betrayal){cls+=' veto';h+=`<span class="act">⚡ BETRAYAL</span>`;}
    f.innerHTML=`<div class="${cls}">${h}</div>`+f.innerHTML;
    if(f.children.length>60)f.removeChild(f.lastChild);
  }
  
  // causal
  const cc=s.explanation.causal_chain||[];
  document.getElementById('chain').innerHTML=cc.map((c,i)=>`<div class="node">${typeof c==='string'?c:JSON.stringify(c)}</div>${i<cc.length-1?'<div class="arrow">↓</div>':''}`).join('');
  document.getElementById('narr').textContent=s.explanation.nl_narrative||'—';
  const cfs=s.explanation.counterfactuals||[];
  document.getElementById('cfs').innerHTML=cfs.slice(0,5).map(c=>`<div class="cf"><strong>${c.action||'Alt'}</strong> <span class="delta ${(c.reward_delta||0)>=0?'pos':'neg'}">${(c.reward_delta||0)>=0?'+':''}${(c.reward_delta||0).toFixed(4)}</span> <span style="color:var(--t4)">${c.explanation||''}</span></div>`).join('');
  const al=s.explanation.risk_alerts||[];
  document.getElementById('alerts').innerHTML=al.length?al.map(a=>`<div class="alert-box">${typeof a==='string'?a:a.message||JSON.stringify(a)}</div>`).join(''):'<span style="color:var(--t4)">Clear</span>';
  
  // KPIs
  document.getElementById('k-surv').textContent=st+'%';
  document.getElementById('k-coop').textContent=(D.coal/Math.max(s.step,1)).toFixed(2);
  document.getElementById('k-osc').textContent=D.hist.length>=3?D.hist.slice(-20).filter((_,i,a)=>i>=2&&a[i]&&a[i-2]&&a[i].action===a[i-2].action&&a[i].action!==(a[i-1]||{}).action).length:'0';
  document.getElementById('k-align').textContent=(s.explanation.alignment_score||50).toFixed(0);
  document.getElementById('k-pareto').textContent=s.state.public_satisfaction>20&&s.state.gdp_index>20?'OK':'RISK';
  
  ch.radar.data.datasets[0].data=[Math.min(1,s.state.gdp_index/150),Math.min(1,1-s.state.pollution_index/500),Math.min(1,s.state.public_satisfaction/100),Math.min(1,(s.state.healthcare_index||50)/100),Math.min(1,D.coal/Math.max(s.step,1))];
  ch.radar.update();
  const rb=s.reward_breakdown||{};
  ch.rbd.data.datasets[0].data=[rb.base||0,rb.pareto||0,rb.cooperation||0,rb.oscillation||0,rb.total||0];ch.rbd.update();
  
  // heatmap
  const rk=(v,lo,hi)=>{const n=(v-lo)/(hi-lo);return n<.3?'lo':n<.6?'md':n<.85?'hi':'cr';};
  document.getElementById('hmap').innerHTML=[['SAT',rk(100-s.state.public_satisfaction,0,100),s.state.public_satisfaction.toFixed(0)],['GDP',rk(200-s.state.gdp_index,0,200),s.state.gdp_index.toFixed(0)],['POLL',rk(s.state.pollution_index,0,500),s.state.pollution_index.toFixed(0)],['HEALTH',rk(100-(s.state.healthcare_index||50),0,100),(s.state.healthcare_index||50).toFixed(0)]].map(([l,c,v])=>`<div class="hcell ${c}"><span class="hl">${l}</span>${v}</div>`).join('');
  
  const W=[];
  if(s.state.public_satisfaction<30)W.push('Satisfaction critical ('+s.state.public_satisfaction.toFixed(0)+')');
  if(s.state.gdp_index<40)W.push('GDP declining ('+s.state.gdp_index.toFixed(0)+')');
  if(s.state.pollution_index>300)W.push('Pollution dangerous ('+s.state.pollution_index.toFixed(0)+')');
  if(s.council.institutional_trust<.3)W.push('Trust collapsing');
  document.getElementById('warns').innerHTML=W.length?W.map(w=>`<div class="warn-item">▸ ${w}</div>`).join(''):'<span style="color:var(--t4)">Clear</span>';
  document.getElementById('evts').innerHTML=(s.active_events||[]).length?(s.active_events||[]).map(e=>`<div style="padding:3px 0;border-bottom:1px solid var(--b1);font-family:IBM Plex Mono;font-size:11px">${typeof e==='string'?e:e.name||JSON.stringify(e)}</div>`).join(''):'<span style="color:var(--t4)">None</span>';
  
  document.getElementById('sc-r').max=D.hist.length-1;document.getElementById('sc-r').value=D.hist.length-1;document.getElementById('sc-hi').textContent=D.hist.length;
}

function onEnd(m){
  addRow(m);
  const f=document.getElementById('feed');
  f.innerHTML=`<div class="msg ${m.collapsed?'veto':'coal'}"><span class="who">END</span><span class="ts">${m.steps} steps</span><br><span class="act">Score: ${m.score} · ${m.collapsed?'COLLAPSED':'SURVIVED'}</span></div>`+f.innerHTML;
}
function addRow(e){
  const t=document.getElementById('eptb');
  const r=document.createElement('tr');
  r.innerHTML=`<td>${e.timestamp||'—'}</td><td>${e.seed}</td><td>${(e.task_id||'').includes('extreme')?'EXT':'CAL'}</td><td>${(e.chaos||0).toFixed(1)}</td><td>${e.steps}</td><td>${(e.score||0).toFixed(4)}</td><td style="color:${e.collapsed?'var(--rose)':'var(--lime)'}"> ${e.collapsed?'COLLAPSED':'SURVIVED'}</td>`;
  t.prepend(r);
}

initCharts();conn();
