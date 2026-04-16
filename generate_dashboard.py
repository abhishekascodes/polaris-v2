#!/usr/bin/env python3
"""
AI Policy Engine -- Interactive Dashboard v2
Self-contained HTML with:
  - Light/Dark mode toggle
  - 6 tabs: Overview, Episodes, Explainability, RL Curves, Action Evolution, Comparison
  - Glassmorphism cards, smooth animations, premium typography
  - Chart.js visualisations with theme-aware colours
"""

import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from server.policy_environment import PolicyEnvironment
from server.tasks import grade_trajectory, get_task_ids
from server.config import VALID_ACTIONS, TASK_CONFIGS

def generate_episode_data(task_id, strategy, seed=42):
    env = PolicyEnvironment()
    obs = env.reset(seed=seed, task_id=task_id)
    trace, step = [], 0
    while not obs.done:
        action = strategy(obs.metadata, step) if callable(strategy) else strategy[step % len(strategy)]
        obs = env.step({"action": action})
        step += 1
        expl = obs.metadata.get("explanation", {})
        rb = obs.metadata.get("reward_breakdown", {})
        trace.append({
            "step": step, "action": action,
            "reward": round(obs.reward, 4),
            "pollution": round(obs.metadata.get("pollution_index", 0), 1),
            "gdp": round(obs.metadata.get("gdp_index", 0), 1),
            "satisfaction": round(obs.metadata.get("public_satisfaction", 0), 1),
            "healthcare": round(obs.metadata.get("healthcare_index", 0), 1),
            "unemployment": round(obs.metadata.get("unemployment_rate", 0), 1),
            "renewable_ratio": round(obs.metadata.get("renewable_energy_ratio", 0), 3),
            "economic_score": round(rb.get("economic_score", 0), 4),
            "environmental_score": round(rb.get("environmental_score", 0), 4),
            "social_score": round(rb.get("social_score", 0), 4),
            "summary": expl.get("summary", ""),
            "alerts": expl.get("risk_alerts", []),
            "chain": [{"layer": c["layer"], "severity": c["severity"],
                       "trigger": c["trigger"][:80], "effect": c["effect"][:80]}
                      for c in expl.get("causal_chain", [])],
            "collapsed": obs.metadata.get("collapsed", False),
        })
    return trace, round(grade_trajectory(task_id, env.get_trajectory()), 4)

def smart_policy(meta, step):
    poll = meta.get("pollution_index", 100)
    gdp = meta.get("gdp_index", 100)
    sat = meta.get("public_satisfaction", 50)
    hc = meta.get("healthcare_index", 50)
    if poll > 180: return "enforce_emission_limits"
    if poll > 140: return "restrict_polluting_industries"
    if gdp < 50: return "stimulate_economy"
    if sat < 25: return "increase_welfare"
    if hc < 30: return "invest_in_healthcare"
    if poll > 100: return "subsidize_renewables"
    if sat < 50: return "increase_welfare"
    if gdp < 75: return "stimulate_economy"
    return "invest_in_education"

def greedy_policy(meta, step):
    return ["expand_industry", "stimulate_economy", "decrease_tax", "reduce_interest_rates"][step % 4]

print("Generating episode data...")
traces = {}
for task_id in get_task_ids():
    heuristic = ["subsidize_renewables","invest_in_education","increase_welfare",
                 "stimulate_economy","invest_in_healthcare","incentivize_clean_tech",
                 "enforce_emission_limits","increase_welfare"]
    t_smart, s_smart = generate_episode_data(task_id, smart_policy)
    t_heur,  s_heur  = generate_episode_data(task_id, heuristic)
    t_greedy, s_greedy = generate_episode_data(task_id, greedy_policy)
    traces[task_id] = {
        "smart":     {"trace": t_smart, "score": s_smart, "label": "Smart Policy"},
        "heuristic": {"trace": t_heur,  "score": s_heur,  "label": "Heuristic Cycle"},
        "greedy":    {"trace": t_greedy, "score": s_greedy, "label": "Greedy GDP"},
    }
    print(f"  {task_id}: smart={s_smart:.4f} heuristic={s_heur:.4f} greedy={s_greedy:.4f}")

rl_data = {}
rl_path = os.path.join("outputs", "rl_training_report.json")
if os.path.exists(rl_path):
    with open(rl_path) as f:
        rl_data = json.load(f)
    print("  RL training data loaded")

# =====================================================================
# HTML TEMPLATE
# =====================================================================

html = r"""<!DOCTYPE html>
<html lang="en" data-theme="dark">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>AI Policy Engine — Dashboard</title>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap" rel="stylesheet">
<style>
/* ===== THEME VARIABLES ===== */
:root {
  --accent: #6366f1; --accent2: #818cf8; --accent-glow: rgba(99,102,241,0.15);
  --green: #10b981; --green2: #34d399;
  --red: #ef4444; --red2: #f87171;
  --amber: #f59e0b; --amber2: #fbbf24;
  --purple: #a855f7; --purple2: #c084fc;
  --cyan: #06b6d4;
  --chart-grid: rgba(148,163,184,0.08);
  --transition: 0.3s cubic-bezier(0.4, 0, 0.2, 1);
}
[data-theme="dark"] {
  --bg: #06080f; --bg2: #0c1018;
  --surface: rgba(17,24,39,0.85); --surface2: rgba(30,41,59,0.7); --surface-solid: #111827;
  --border: rgba(51,65,85,0.5); --border2: rgba(71,85,105,0.3);
  --text: #f1f5f9; --text2: #94a3b8; --text3: #64748b;
  --header-bg: linear-gradient(135deg, #0f0a2e 0%, #0c1425 50%, #0a1520 100%);
  --card-shadow: 0 4px 24px rgba(0,0,0,0.4), 0 1px 2px rgba(0,0,0,0.3);
  --glow: rgba(99,102,241,0.06);
}
[data-theme="light"] {
  --bg: #f8fafc; --bg2: #f1f5f9;
  --surface: rgba(255,255,255,0.85); --surface2: rgba(241,245,249,0.8); --surface-solid: #ffffff;
  --border: rgba(203,213,225,0.6); --border2: rgba(226,232,240,0.5);
  --text: #0f172a; --text2: #475569; --text3: #94a3b8;
  --header-bg: linear-gradient(135deg, #312e81 0%, #1e3a5f 50%, #1e3a5f 100%);
  --card-shadow: 0 4px 24px rgba(0,0,0,0.06), 0 1px 3px rgba(0,0,0,0.08);
  --glow: rgba(99,102,241,0.04);
  --chart-grid: rgba(100,116,139,0.1);
}

/* ===== BASE ===== */
* { margin:0; padding:0; box-sizing:border-box; }
body { font-family:'Inter',system-ui,sans-serif; background:var(--bg); color:var(--text); transition:background var(--transition), color var(--transition); min-height:100vh; }

/* ===== HEADER ===== */
.header {
  background:var(--header-bg); padding:28px 40px 24px; position:relative; overflow:hidden;
  border-bottom:1px solid var(--border);
}
.header::before {
  content:''; position:absolute; top:-50%; right:-10%; width:600px; height:600px;
  background:radial-gradient(circle, rgba(99,102,241,0.08) 0%, transparent 70%);
}
.header-inner { display:flex; justify-content:space-between; align-items:center; position:relative; z-index:1; }
.header h1 { font-size:26px; font-weight:800; color:#fff; letter-spacing:-0.5px; }
.header h1 span { background:linear-gradient(135deg,#818cf8,#c084fc); -webkit-background-clip:text; -webkit-text-fill-color:transparent; }
.header p { color:rgba(255,255,255,0.6); font-size:13px; margin-top:4px; font-weight:400; }
.badge { display:inline-block; background:rgba(99,102,241,0.2); color:#a5b4fc; font-size:10px; font-weight:600;
         padding:3px 10px; border-radius:20px; margin-left:12px; letter-spacing:0.5px; border:1px solid rgba(99,102,241,0.25); }

/* Theme toggle */
.theme-toggle {
  background:rgba(255,255,255,0.1); border:1px solid rgba(255,255,255,0.15); border-radius:10px;
  padding:8px 14px; cursor:pointer; display:flex; align-items:center; gap:8px;
  color:#fff; font-size:12px; font-weight:500; transition:all 0.2s;
  backdrop-filter:blur(10px);
}
.theme-toggle:hover { background:rgba(255,255,255,0.18); transform:translateY(-1px); }
.theme-toggle svg { width:16px; height:16px; }
.sun-icon { display:none; }
[data-theme="dark"] .moon-icon { display:block; }
[data-theme="dark"] .sun-icon { display:none; }
[data-theme="light"] .moon-icon { display:none; }
[data-theme="light"] .sun-icon { display:block; }

/* ===== TABS ===== */
.tabs {
  display:flex; background:var(--surface-solid); border-bottom:1px solid var(--border);
  padding:0 40px; gap:2px; transition:background var(--transition);
  overflow-x:auto; -webkit-overflow-scrolling:touch;
}
.tab {
  padding:13px 18px; cursor:pointer; color:var(--text3); font-size:12px; font-weight:600;
  border-bottom:2px solid transparent; transition:all 0.2s; white-space:nowrap;
  text-transform:uppercase; letter-spacing:0.5px;
}
.tab:hover { color:var(--text); }
.tab.active { color:var(--accent); border-bottom-color:var(--accent); }

/* ===== LAYOUT ===== */
.content { padding:24px 40px 60px; max-width:1440px; margin:0 auto; }
.panel { display:none; } .panel.active { display:block; }
.grid2 { display:grid; grid-template-columns:1fr 1fr; gap:20px; margin-bottom:20px; }
.grid3 { display:grid; grid-template-columns:1fr 1fr 1fr; gap:16px; margin-bottom:20px; }
.grid4 { display:grid; grid-template-columns:1fr 1fr 1fr 1fr; gap:14px; margin-bottom:20px; }

/* ===== CARDS ===== */
.card {
  background:var(--surface); backdrop-filter:blur(20px); -webkit-backdrop-filter:blur(20px);
  border:1px solid var(--border); border-radius:14px; padding:22px;
  box-shadow:var(--card-shadow); transition:all var(--transition);
  animation:cardIn 0.4s ease both;
}
.card:hover { border-color:var(--accent); box-shadow:var(--card-shadow), 0 0 30px var(--accent-glow); }
.card h3 {
  font-size:11px; color:var(--text3); font-weight:700; margin-bottom:14px;
  text-transform:uppercase; letter-spacing:1px;
}
@keyframes cardIn { from{opacity:0;transform:translateY(12px)} to{opacity:1;transform:translateY(0)} }

/* ===== METRICS ===== */
.metric { text-align:center; padding:12px 10px; }
.metric .value { font-size:38px; font-weight:800; letter-spacing:-1px; line-height:1; }
.metric .label { font-size:11px; color:var(--text3); margin-top:8px; font-weight:500; }
.metric .sub { font-size:10px; color:var(--text3); margin-top:3px; }

/* ===== CONTROLS ===== */
.controls { display:flex; gap:10px; align-items:center; margin-bottom:20px; flex-wrap:wrap; }
.controls select, .controls button {
  background:var(--surface-solid); border:1px solid var(--border); color:var(--text);
  padding:8px 14px; border-radius:8px; font-size:12px; cursor:pointer; font-weight:500;
  transition:all 0.2s; font-family:'Inter',system-ui,sans-serif;
}
.controls select:hover, .controls button:hover { border-color:var(--accent); }

canvas { width:100%!important; height:280px!important; }

/* ===== CHAIN ITEMS ===== */
.chain-item {
  padding:10px 14px; border-left:3px solid var(--border2); margin-bottom:6px;
  font-size:12px; background:var(--surface2); border-radius:0 8px 8px 0;
  transition:all 0.2s;
}
.chain-item:hover { transform:translateX(4px); }
.chain-item.info { border-left-color:var(--accent); }
.chain-item.warning { border-left-color:var(--amber); }
.chain-item.critical { border-left-color:var(--red); }
.chain-item .layer { color:var(--text3); font-size:9px; text-transform:uppercase; letter-spacing:0.5px; font-weight:700; }
.chain-item .trigger { margin-top:2px; color:var(--text); font-weight:500; }
.chain-item .effect { color:var(--text2); margin-top:2px; font-size:11px; }

.alert { padding:8px 12px; background:rgba(239,68,68,0.08); border:1px solid rgba(239,68,68,0.2); border-radius:8px;
         font-size:12px; color:var(--red); margin-bottom:4px; font-weight:500; }

/* ===== STEP ROWS ===== */
.step-row { display:flex; gap:8px; padding:7px 0; border-bottom:1px solid var(--border2); font-size:12px; align-items:center; transition:background 0.15s; }
.step-row:hover { background:var(--glow); }
.step-row .step-num { width:36px; color:var(--text3); font-weight:600; font-size:11px; }
.step-row .action { width:210px; font-family:'JetBrains Mono',monospace; color:var(--accent); font-size:11px; font-weight:500; }
.step-row .metrics { flex:1; display:flex; gap:14px; font-size:11px; }
.step-row .metrics span { min-width:55px; }

/* ===== TABLES ===== */
table { width:100%; border-collapse:collapse; font-size:13px; }
th { text-align:left; padding:10px 12px; color:var(--text3); font-weight:600; border-bottom:1px solid var(--border); font-size:11px; text-transform:uppercase; letter-spacing:0.5px; }
td { padding:10px 12px; border-bottom:1px solid var(--border2); }

/* ===== ACTION BAR ===== */
.action-bar { display:flex; align-items:center; gap:8px; margin-bottom:6px; font-size:12px; }
.action-bar .name { width:180px; color:var(--text2); font-family:monospace; font-size:11px; overflow:hidden; text-overflow:ellipsis; }
.action-bar .bar-track { flex:1; height:10px; background:var(--surface2); border-radius:5px; overflow:hidden; }
.action-bar .bar-fill { height:100%; border-radius:5px; transition:width 0.6s ease; }
.action-bar .pct { width:48px; text-align:right; color:var(--text3); font-size:11px; font-weight:600; }

/* ===== PRE ===== */
pre { font-size:11px; color:var(--text2); line-height:1.7; font-family:'JetBrains Mono','Fira Code',monospace; }

/* ===== RESPONSIVE ===== */
@media (max-width: 900px) {
  .grid2, .grid3, .grid4 { grid-template-columns:1fr; }
  .content { padding:16px; }
  .header { padding:20px 16px; }
  .tabs { padding:0 16px; }
}
</style>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
</head>
<body>

<div class="header">
  <div class="header-inner">
    <div>
      <h1>AI Policy <span>Engine</span> <span class="badge">RESEARCH v2</span></h1>
      <p>Multi-objective governance simulation · Causal explainability · RL benchmarking</p>
    </div>
    <button class="theme-toggle" onclick="toggleTheme()" title="Toggle light/dark mode">
      <svg class="moon-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"/></svg>
      <svg class="sun-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="5"/><line x1="12" y1="1" x2="12" y2="3"/><line x1="12" y1="21" x2="12" y2="23"/><line x1="4.22" y1="4.22" x2="5.64" y2="5.64"/><line x1="18.36" y1="18.36" x2="19.78" y2="19.78"/><line x1="1" y1="12" x2="3" y2="12"/><line x1="21" y1="12" x2="23" y2="12"/><line x1="4.22" y1="19.78" x2="5.64" y2="18.36"/><line x1="18.36" y1="5.64" x2="19.78" y2="4.22"/></svg>
      <span class="theme-label">Theme</span>
    </button>
  </div>
</div>

<div class="tabs" id="tab-bar">
  <div class="tab active" onclick="showPanel('overview',this)">Overview</div>
  <div class="tab" onclick="showPanel('episodes',this)">Episode Explorer</div>
  <div class="tab" onclick="showPanel('explainability',this)">Explainability</div>
  <div class="tab" onclick="showPanel('learning',this)">RL Curves</div>
  <div class="tab" onclick="showPanel('evolution',this)">Action Evolution</div>
  <div class="tab" onclick="showPanel('comparison',this)">Comparison</div>
</div>

<div class="content">

<!-- ===== OVERVIEW ===== -->
<div id="panel-overview" class="panel active">
  <div class="grid4" id="ov-metrics"></div>
  <div class="grid2">
    <div class="card">
      <h3>Simulation Architecture</h3>
      <pre>
  Agent Action  ─────────►  1. Deterministic Effects
       (1 of 16)                Direct consequences
                            │
                            ▼
                          2. Non-linear Thresholds
                            │  6 tipping points
                            ▼
                          3. Delayed Effects Queue
                            │  Past investments materialise
                            ▼
                          4. Feedback Loops
                            │  6 systemic cascades
                            ▼
                          Event Engine (8 types)
                            │
                            ▼
                     ┌──────────────┐
                     │  Observation  │
                     │  + Reward     │
                     │  + Causal     │
                     │    Chain      │
                     └──────────────┘

  Reward: 30% Economic + 30% Environmental
          25% Social   + 15% Stability
      </pre>
    </div>
    <div class="card">
      <h3>Environment Specifications</h3>
      <table>
        <tr><th>Property</th><th>Value</th></tr>
        <tr><td>State dimensions</td><td><strong>21</strong> continuous variables</td></tr>
        <tr><td>Action space</td><td><strong>16</strong> discrete policy levers</td></tr>
        <tr><td>Tasks</td><td>3 (easy → medium → hard)</td></tr>
        <tr><td>Max horizon</td><td>50 / 100 / 200 steps</td></tr>
        <tr><td>Transition layers</td><td>4 (deterministic, threshold, delayed, feedback)</td></tr>
        <tr><td>Event types</td><td>8 stochastic events</td></tr>
        <tr><td>Explainability</td><td>5-layer causal chain per step</td></tr>
        <tr><td>Collapse conditions</td><td>GDP&lt;15, Pollution&gt;290, Satisfaction&lt;5</td></tr>
        <tr><td>Determinism</td><td>Bit-identical with seed</td></tr>
        <tr><td>Dependencies</td><td>Zero ML frameworks</td></tr>
      </table>
    </div>
  </div>
</div>

<!-- ===== EPISODES ===== -->
<div id="panel-episodes" class="panel">
  <div class="controls">
    <select id="ep-task" onchange="updateEpisode()">
      <option value="environmental_recovery">Environmental Recovery (Easy)</option>
      <option value="balanced_economy">Balanced Economy (Medium)</option>
      <option value="sustainable_governance">Sustainable Governance (Hard)</option>
    </select>
    <select id="ep-policy" onchange="updateEpisode()">
      <option value="smart">Smart Policy</option>
      <option value="heuristic">Heuristic Cycle</option>
      <option value="greedy">Greedy GDP</option>
    </select>
  </div>
  <div class="grid2">
    <div class="card"><h3>State Metrics Over Time</h3><canvas id="ep-chart"></canvas></div>
    <div class="card"><h3>Per-Step Reward Signal</h3><canvas id="ep-reward-chart"></canvas></div>
  </div>
  <div class="grid2">
    <div class="card"><h3>Reward Components</h3><canvas id="ep-components"></canvas></div>
    <div class="card" style="max-height:340px;overflow-y:auto">
      <h3>Step-by-Step Trace</h3>
      <div id="ep-trace"></div>
    </div>
  </div>
</div>

<!-- ===== EXPLAINABILITY ===== -->
<div id="panel-explainability" class="panel">
  <div class="controls">
    <select id="expl-task" onchange="populateSteps();updateExpl()">
      <option value="environmental_recovery">Environmental Recovery</option>
      <option value="balanced_economy">Balanced Economy</option>
      <option value="sustainable_governance">Sustainable Governance</option>
    </select>
    <select id="expl-step" onchange="updateExpl()"></select>
  </div>
  <div class="grid2">
    <div class="card">
      <h3>Causal Reasoning Chain</h3>
      <div id="expl-chain"></div>
    </div>
    <div class="card">
      <h3>Risk Alerts</h3>
      <div id="expl-alerts"></div>
      <h3 style="margin-top:20px">Explanation Summary</h3>
      <p id="expl-summary" style="font-size:13px;color:var(--text2);margin-top:8px;line-height:1.6"></p>
    </div>
  </div>
</div>

<!-- ===== RL CURVES ===== -->
<div id="panel-learning" class="panel">
  <div class="controls">
    <select id="lc-task" onchange="updateLC()">
      <option value="environmental_recovery">Environmental Recovery</option>
      <option value="balanced_economy">Balanced Economy</option>
      <option value="sustainable_governance">Sustainable Governance</option>
    </select>
  </div>
  <div class="grid2">
    <div class="card"><h3>Score Over Training</h3><canvas id="lc-score"></canvas></div>
    <div class="card"><h3>Average Reward</h3><canvas id="lc-reward"></canvas></div>
  </div>
  <div class="grid2">
    <div class="card"><h3>Collapse Rate</h3><canvas id="lc-collapse"></canvas></div>
    <div class="card"><h3>Entropy Annealing Schedule</h3><canvas id="lc-entropy"></canvas></div>
  </div>
  <div class="card"><h3>Evaluation Results</h3><div id="lc-summary"></div></div>
</div>

<!-- ===== ACTION EVOLUTION ===== -->
<div id="panel-evolution" class="panel">
  <div class="controls">
    <select id="ev-task" onchange="updateEvolution()">
      <option value="environmental_recovery">Environmental Recovery</option>
      <option value="balanced_economy">Balanced Economy</option>
      <option value="sustainable_governance">Sustainable Governance</option>
    </select>
  </div>
  <div class="grid2" id="ev-phases"></div>
</div>

<!-- ===== COMPARISON ===== -->
<div id="panel-comparison" class="panel">
  <div class="card"><h3>Policy Comparison Across All Tasks</h3><canvas id="comp-chart" style="height:340px!important"></canvas></div>
  <div class="grid3" id="comp-cards"></div>
</div>

</div>

<script>
const TRACES = TRACE_DATA_PLACEHOLDER;
const RL_DATA = RL_DATA_PLACEHOLDER;
let charts = {};
let currentTheme = 'dark';

/* ===== THEME ===== */
function toggleTheme() {
  currentTheme = currentTheme === 'dark' ? 'light' : 'dark';
  document.documentElement.setAttribute('data-theme', currentTheme);
  localStorage.setItem('theme', currentTheme);
  // Redraw active panel charts
  const active = document.querySelector('.panel.active');
  if (active) {
    const id = active.id.replace('panel-','');
    if (id === 'episodes') updateEpisode();
    if (id === 'learning') updateLC();
    if (id === 'comparison') updateComparison();
    if (id === 'evolution') updateEvolution();
  }
}

function getChartColors() {
  const isDark = currentTheme === 'dark';
  return {
    grid: isDark ? 'rgba(148,163,184,0.06)' : 'rgba(100,116,139,0.1)',
    tick: isDark ? '#64748b' : '#94a3b8',
    legend: isDark ? '#94a3b8' : '#64748b',
  };
}

/* ===== PANELS ===== */
function showPanel(id, el) {
  document.querySelectorAll('.panel').forEach(p => p.classList.remove('active'));
  document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
  document.getElementById('panel-' + id).classList.add('active');
  if (el) el.classList.add('active');
  if (id === 'episodes') updateEpisode();
  if (id === 'explainability') { populateSteps(); updateExpl(); }
  if (id === 'learning') updateLC();
  if (id === 'evolution') updateEvolution();
  if (id === 'comparison') updateComparison();
}

function destroyChart(id) { if (charts[id]) { charts[id].destroy(); delete charts[id]; } }

function chartOpts(extra={}) {
  const c = getChartColors();
  return Object.assign({
    responsive:true,
    scales:{
      x:{ticks:{color:c.tick,maxTicksLimit:15},grid:{color:c.grid}},
      y:{ticks:{color:c.tick},grid:{color:c.grid}}
    },
    plugins:{legend:{labels:{color:c.legend,boxWidth:10,font:{size:11}}}}
  }, extra);
}

/* ===== OVERVIEW ===== */
function initOverview() {
  const metrics = [
    {value:'21',label:'State Dimensions',color:'var(--accent)'},
    {value:'16',label:'Policy Actions',color:'var(--green)'},
    {value:'3',label:'Graded Tasks',color:'var(--amber)'},
    {value:'6500+',label:'Training Episodes',color:'var(--purple)'},
  ];
  let html = '';
  metrics.forEach((m,i) => {
    html += `<div class="card metric" style="animation-delay:${i*0.08}s"><div class="value" style="color:${m.color}">${m.value}</div><div class="label">${m.label}</div></div>`;
  });
  document.getElementById('ov-metrics').innerHTML = html;
}

/* ===== EPISODES ===== */
function updateEpisode() {
  const task = document.getElementById('ep-task').value;
  const pol = document.getElementById('ep-policy').value;
  const data = TRACES[task][pol];
  const trace = data.trace;
  const c = getChartColors();

  destroyChart('ep-chart');
  charts['ep-chart'] = new Chart(document.getElementById('ep-chart'), {
    type:'line', data:{ labels:trace.map(t=>t.step), datasets:[
      {label:'Pollution',data:trace.map(t=>t.pollution),borderColor:'#ef4444',borderWidth:2,tension:0.3,pointRadius:0},
      {label:'GDP',data:trace.map(t=>t.gdp),borderColor:'#3b82f6',borderWidth:2,tension:0.3,pointRadius:0},
      {label:'Satisfaction',data:trace.map(t=>t.satisfaction),borderColor:'#10b981',borderWidth:2,tension:0.3,pointRadius:0},
      {label:'Healthcare',data:trace.map(t=>t.healthcare),borderColor:'#a855f7',borderWidth:2,tension:0.3,pointRadius:0},
    ]}, options:chartOpts()
  });

  destroyChart('ep-reward');
  charts['ep-reward'] = new Chart(document.getElementById('ep-reward-chart'), {
    type:'line', data:{ labels:trace.map(t=>t.step), datasets:[
      {label:'Reward',data:trace.map(t=>t.reward),borderColor:'#f59e0b',backgroundColor:'rgba(245,158,11,0.1)',fill:true,borderWidth:2,tension:0.3,pointRadius:0},
    ]}, options:chartOpts()
  });

  destroyChart('ep-comp');
  charts['ep-comp'] = new Chart(document.getElementById('ep-components'), {
    type:'line', data:{ labels:trace.map(t=>t.step), datasets:[
      {label:'Economic',data:trace.map(t=>t.economic_score),borderColor:'#3b82f6',borderWidth:2,tension:0.3,pointRadius:0},
      {label:'Environmental',data:trace.map(t=>t.environmental_score),borderColor:'#10b981',borderWidth:2,tension:0.3,pointRadius:0},
      {label:'Social',data:trace.map(t=>t.social_score),borderColor:'#a855f7',borderWidth:2,tension:0.3,pointRadius:0},
    ]}, options:chartOpts()
  });

  let html = '';
  trace.forEach(t => {
    const warn = t.alerts.length ? `<span style="color:var(--red)"> ⚠ ${t.alerts.length}</span>` : '';
    const cls = t.collapsed ? ' style="background:rgba(239,68,68,0.06)"' : '';
    html += `<div class="step-row"${cls}><span class="step-num">${t.step}</span><span class="action">${t.action}${warn}</span><span class="metrics"><span style="color:var(--amber)">r=${t.reward.toFixed(3)}</span><span style="color:var(--red)">P:${t.pollution}</span><span style="color:var(--accent)">G:${t.gdp}</span><span style="color:var(--green)">S:${t.satisfaction}</span></span></div>`;
  });
  const last = trace[trace.length-1];
  html += `<div style="padding:14px;color:var(--text2);font-size:13px">Final Score: <strong style="color:var(--accent);font-size:18px">${data.score}</strong>${last.collapsed?' · <span style="color:var(--red);font-weight:600">COLLAPSED</span>':' · <span style="color:var(--green)">Completed</span>'}</div>`;
  document.getElementById('ep-trace').innerHTML = html;
}

/* ===== EXPLAINABILITY ===== */
function populateSteps() {
  const task = document.getElementById('expl-task').value;
  const trace = TRACES[task].smart.trace;
  const sel = document.getElementById('expl-step');
  sel.innerHTML = '';
  trace.forEach(t => {
    const opt = document.createElement('option');
    opt.value = t.step - 1;
    opt.textContent = `Step ${t.step}: ${t.action}`;
    sel.appendChild(opt);
  });
}

function updateExpl() {
  const task = document.getElementById('expl-task').value;
  const idx = parseInt(document.getElementById('expl-step').value || 0);
  const trace = TRACES[task].smart.trace;
  if (!trace[idx]) return;
  const t = trace[idx];

  let chainHtml = '';
  t.chain.forEach(c => {
    chainHtml += `<div class="chain-item ${c.severity}"><div class="layer">${c.layer} · ${c.severity}</div><div class="trigger">${c.trigger}</div><div class="effect">→ ${c.effect}</div></div>`;
  });
  document.getElementById('expl-chain').innerHTML = chainHtml || '<p style="color:var(--text3);padding:12px">No causal links at this step.</p>';

  let alertHtml = '';
  t.alerts.forEach(a => { alertHtml += `<div class="alert">⚠ ${a}</div>`; });
  document.getElementById('expl-alerts').innerHTML = alertHtml || '<p style="color:var(--text3);padding:8px">No risk alerts.</p>';
  document.getElementById('expl-summary').textContent = t.summary;
}

/* ===== RL LEARNING CURVES ===== */
function updateLC() {
  const task = document.getElementById('lc-task').value;
  const data = RL_DATA[task];
  if (!data?.training?.learning_curve) {
    ['lc-score','lc-reward','lc-collapse','lc-entropy'].forEach(id => destroyChart(id));
    document.getElementById('lc-summary').innerHTML = '<p style="color:var(--text3);padding:12px">No RL data. Run: <code>python rl_agent.py</code></p>';
    return;
  }
  const lc = data.training.learning_curve;
  const eps = lc.map(p => p.episode);

  destroyChart('lc-score');
  charts['lc-score'] = new Chart(document.getElementById('lc-score'), {
    type:'line', data:{ labels:eps, datasets:[
      {label:'Avg Score',data:lc.map(p=>p.avg_score),borderColor:'#6366f1',borderWidth:2,tension:0.3,pointRadius:0},
      {label:'Best Score',data:lc.map(p=>p.best_score),borderColor:'#10b981',borderWidth:2,borderDash:[5,5],tension:0.3,pointRadius:0},
    ]}, options:chartOpts()
  });

  destroyChart('lc-reward');
  charts['lc-reward'] = new Chart(document.getElementById('lc-reward'), {
    type:'line', data:{ labels:eps, datasets:[
      {label:'Avg Reward',data:lc.map(p=>p.avg_reward),borderColor:'#f59e0b',backgroundColor:'rgba(245,158,11,0.08)',fill:true,borderWidth:2,tension:0.3,pointRadius:0},
    ]}, options:chartOpts()
  });

  destroyChart('lc-collapse');
  charts['lc-collapse'] = new Chart(document.getElementById('lc-collapse'), {
    type:'line', data:{ labels:eps, datasets:[
      {label:'Collapse Rate',data:lc.map(p=>p.collapse_rate),borderColor:'#ef4444',backgroundColor:'rgba(239,68,68,0.08)',fill:true,borderWidth:2,tension:0.3,pointRadius:0},
    ]}, options:chartOpts({scales:{x:{ticks:{color:getChartColors().tick,maxTicksLimit:15},grid:{color:getChartColors().grid}},y:{min:0,max:1,ticks:{color:getChartColors().tick},grid:{color:getChartColors().grid}}}})
  });

  // Entropy annealing chart
  const entData = lc.filter(p => p.entropy_coeff !== undefined);
  destroyChart('lc-entropy');
  if (entData.length > 0) {
    charts['lc-entropy'] = new Chart(document.getElementById('lc-entropy'), {
      type:'line', data:{ labels:entData.map(p=>p.episode), datasets:[
        {label:'Entropy Coeff',data:entData.map(p=>p.entropy_coeff),borderColor:'#a855f7',backgroundColor:'rgba(168,85,247,0.08)',fill:true,borderWidth:2,tension:0.3,pointRadius:0},
        {label:'LR Multiplier',data:entData.map(p=>p.lr_mult),borderColor:'#06b6d4',borderWidth:2,borderDash:[4,4],tension:0.3,pointRadius:0},
      ]}, options:chartOpts()
    });
  }

  const ev = data.evaluation;
  document.getElementById('lc-summary').innerHTML = `
    <table>
      <tr><th>Method</th><th>Avg Score</th><th>Best</th><th>Collapse Rate</th><th>Top Actions</th></tr>
      <tr><td style="color:var(--accent);font-weight:600">RL Agent</td><td><strong>${ev.trained?.avg_score??'N/A'}</strong></td><td>${ev.trained?.best_score??'-'}</td><td>${ev.trained?.collapse_rate!=null?(ev.trained.collapse_rate*100).toFixed(0)+'%':'N/A'}</td><td style="font-size:11px">${ev.trained?.top_actions?Object.entries(ev.trained.top_actions).slice(0,3).map(([a,v])=>a+'='+Math.round(v*100)+'%').join(', '):'-'}</td></tr>
      <tr><td>Random</td><td>${ev.random?.avg_score??'N/A'}</td><td>—</td><td>${ev.random?.collapse_rate!=null?(ev.random.collapse_rate*100).toFixed(0)+'%':'N/A'}</td><td>—</td></tr>
      <tr><td>Heuristic</td><td>${ev.heuristic?.avg_score??'N/A'}</td><td>—</td><td>${ev.heuristic?.collapse_rate!=null?(ev.heuristic.collapse_rate*100).toFixed(0)+'%':'N/A'}</td><td>—</td></tr>
    </table>
    <p style="margin-top:14px;font-size:12px;color:var(--text3)">
      Best training score: <strong style="color:var(--green);font-size:14px">${data.training.best_score}</strong>
      · Training time: ${data.training.time_seconds}s
      · Episodes: ${data.training.episodes}
    </p>`;
}

/* ===== ACTION EVOLUTION ===== */
const ACTION_COLORS = ['#6366f1','#10b981','#f59e0b','#ef4444','#a855f7','#06b6d4','#ec4899','#84cc16','#f97316','#14b8a6','#8b5cf6','#e879f9','#22d3ee','#facc15','#fb923c','#4ade80'];

function updateEvolution() {
  const task = document.getElementById('ev-task').value;
  const data = RL_DATA[task];
  const container = document.getElementById('ev-phases');

  if (!data?.training?.action_evolution) {
    container.innerHTML = '<div class="card" style="grid-column:span 2"><h3>Action Evolution</h3><p style="color:var(--text3)">No evolution data. Run: <code>python rl_agent.py</code></p></div>';
    return;
  }

  const evolution = data.training.action_evolution;
  const phases = Object.keys(evolution);
  let html = '';

  phases.forEach((phase,pi) => {
    const dist = evolution[phase];
    const actions = Object.entries(dist).sort((a,b) => b[1]-a[1]);
    const maxVal = actions[0]?.[1] || 0.1;
    const phaseLabel = phase.replace(/phase_/,'Phase ').replace(/_ep_/,': Episodes ').replace(/-/,' – ');

    let barsHtml = '';
    actions.forEach(([action, pct],i) => {
      const width = Math.max(2, (pct / maxVal) * 100);
      const color = ACTION_COLORS[i % ACTION_COLORS.length];
      barsHtml += `<div class="action-bar"><span class="name">${action}</span><div class="bar-track"><div class="bar-fill" style="width:${width}%;background:${color}"></div></div><span class="pct">${(pct*100).toFixed(1)}%</span></div>`;
    });

    html += `<div class="card"><h3>${phaseLabel}</h3>${barsHtml}</div>`;
  });

  container.innerHTML = html;
}

/* ===== COMPARISON ===== */
function updateComparison() {
  destroyChart('comp');
  const tasks = Object.keys(TRACES);
  const labels = tasks.map(t => t.replace(/_/g,' '));

  charts['comp'] = new Chart(document.getElementById('comp-chart'), {
    type:'bar', data:{
      labels,
      datasets: [
        {label:'Smart Policy',data:tasks.map(t=>TRACES[t].smart.score),backgroundColor:'rgba(99,102,241,0.8)',borderRadius:4},
        {label:'Heuristic',data:tasks.map(t=>TRACES[t].heuristic.score),backgroundColor:'rgba(168,85,247,0.7)',borderRadius:4},
        {label:'Greedy GDP',data:tasks.map(t=>TRACES[t].greedy.score),backgroundColor:'rgba(239,68,68,0.7)',borderRadius:4},
      ]
    }, options:chartOpts({scales:{x:{ticks:{color:getChartColors().tick},grid:{color:getChartColors().grid}},y:{min:0,max:1,title:{display:true,text:'Score',color:getChartColors().legend},ticks:{color:getChartColors().tick},grid:{color:getChartColors().grid}}}})
  });

  let cardsHtml = '';
  tasks.forEach(task => {
    const s = TRACES[task].smart, h = TRACES[task].heuristic, g = TRACES[task].greedy;
    const lastG = g.trace[g.trace.length-1];
    cardsHtml += `<div class="card"><h3>${task.replace(/_/g,' ')}</h3><table><tr><th>Policy</th><th>Score</th><th>Steps</th></tr><tr><td style="color:var(--accent);font-weight:600">Smart</td><td><strong>${s.score}</strong></td><td>${s.trace.length}</td></tr><tr><td style="color:var(--purple)">Heuristic</td><td>${h.score}</td><td>${h.trace.length}</td></tr><tr><td style="color:var(--red)">Greedy</td><td>${g.score}</td><td>${g.trace.length}${lastG.collapsed?' (💀)':''}</td></tr></table></div>`;
  });
  document.getElementById('comp-cards').innerHTML = cardsHtml;
}

/* ===== INIT ===== */
document.addEventListener('DOMContentLoaded', () => {
  const saved = localStorage.getItem('theme');
  if (saved) { currentTheme = saved; document.documentElement.setAttribute('data-theme', saved); }
  initOverview();
  populateSteps();
});
</script>
</body>
</html>"""

html = html.replace('TRACE_DATA_PLACEHOLDER', json.dumps(traces))
html = html.replace('RL_DATA_PLACEHOLDER', json.dumps(rl_data))

os.makedirs("outputs", exist_ok=True)
with open("dashboard.html", "w", encoding="utf-8") as f:
    f.write(html)

print(f"\nDashboard v2 saved to dashboard.html ({len(html)//1024}KB)")
print("Open in browser to view. Supports light/dark mode toggle.")
