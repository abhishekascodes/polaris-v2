#!/usr/bin/env python3
"""
OpenENV Real-Time Dashboard Server
FastAPI + WebSocket backend for live governance simulation
Launch: python dashboard_server.py
"""
import asyncio, json, os, sys, time, random, copy, threading
from pathlib import Path
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn

from server.policy_environment import PolicyEnvironment
from server.config import VALID_ACTIONS, TASK_CONFIGS, OBS_TOTAL_DIM
from server.tasks import grade_trajectory

app = FastAPI(title="OpenENV Dashboard")
static_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
os.makedirs(static_dir, exist_ok=True)
app.mount("/static", StaticFiles(directory=static_dir), name="static")

# ── State ──
class SimState:
    def __init__(self):
        self.env: Optional[PolicyEnvironment] = None
        self.running = False
        self.paused = False
        self.speed = 1.0  # steps per second
        self.chaos = 0.5
        self.seed = 42
        self.task_id = "sustainable_governance"
        self.step = 0
        self.history = []  # step data
        self.episodes = []  # past episode summaries
        self.ws_clients: list[WebSocket] = []
        self.lock = asyncio.Lock()
        self.obs = None

sim = SimState()

MINISTER_NAMES = [
    "Chancellor Voss", "Dir. Okafor", "Dr. Vasquez",
    "Gen. Tanaka", "Sen. Mwangi",
]

# ── Agent (intelligent heuristic with negotiation reasoning) ──
def agent_policy(obs, step):
    """Smart agent that reads environment state and negotiation context
    to make intelligent decisions with real coalition/veto reasoning."""
    m = obs.metadata
    sat = m.get("public_satisfaction", 50)
    poll = m.get("pollution_index", 100)
    gdp = m.get("gdp_index", 100)
    hc = m.get("healthcare_index", 50)
    edu = m.get("education_index", 50)
    unemp = m.get("unemployment_rate", 10)
    renew = m.get("renewable_energy_ratio", 0.3)

    # Read negotiation context from previous observation
    neg = m.get("negotiation", {})
    proposals = neg.get("minister_proposals", [])
    trust = neg.get("institutional_trust", 0.6)
    briefing = neg.get("diplomatic_briefing", "")

    # --- Intelligent action selection based on state urgency ---
    reasoning = ""
    if sat < 20:
        action = "increase_welfare"
        reasoning = f"Critical satisfaction ({sat:.0f}), emergency welfare needed"
    elif sat < 40:
        action = "increase_welfare"
        reasoning = f"Satisfaction declining ({sat:.0f}), prioritising public support"
    elif poll > 220:
        action = "enforce_emission_limits"
        reasoning = f"Dangerous pollution ({poll:.0f}), enforcing limits"
    elif poll > 160:
        action = "restrict_polluting_industries"
        reasoning = f"High pollution ({poll:.0f}), restricting industry"
    elif gdp < 35:
        action = "stimulate_economy"
        reasoning = f"GDP critical ({gdp:.0f}), economic stimulus required"
    elif gdp < 55:
        action = "decrease_tax"
        reasoning = f"GDP low ({gdp:.0f}), reducing tax to encourage growth"
    elif hc < 30:
        action = "invest_in_healthcare"
        reasoning = f"Healthcare failing ({hc:.0f}), investing urgently"
    elif unemp > 20:
        action = "stimulate_economy"
        reasoning = f"Unemployment high ({unemp:.1f}%), stimulating jobs"
    elif poll > 120:
        action = "subsidize_renewables"
        reasoning = f"Pollution elevated ({poll:.0f}), green investment"
    elif renew < 0.25:
        action = "incentivize_clean_tech"
        reasoning = f"Renewable ratio low ({renew:.2f}), pushing clean tech"
    elif edu < 40:
        action = "invest_in_education"
        reasoning = f"Education low ({edu:.0f}), investing"
    elif sat < 55:
        action = "increase_welfare"
        reasoning = f"Satisfaction below target ({sat:.0f}), boosting welfare"
    else:
        # Balanced state — cycle through beneficial actions
        cycle = [
            "subsidize_renewables", "invest_in_education",
            "increase_welfare", "invest_in_healthcare",
            "incentivize_clean_tech", "stimulate_economy",
        ]
        action = cycle[step % len(cycle)]
        reasoning = f"State balanced, maintaining diverse policy (step {step})"

    # --- Intelligent coalition targeting ---
    # Read proposals to identify allies (who proposed similar actions)
    coalition_target = []
    veto_prediction = []
    for p in proposals:
        minister = p.get("minister", "")
        proposed = p.get("proposed_action", "")
        veto_threat = p.get("veto_threat", False)
        trust_level = p.get("trust_level", 0.5)
        role = p.get("role", "")

        # Target ministers with high trust as coalition partners
        if trust_level > 0.6 and not veto_threat:
            coalition_target.append(minister)
        # Predict vetoes from those with threats or opposing roles
        if veto_threat:
            veto_prediction.append(minister)
        elif trust_level < 0.3:
            veto_prediction.append(minister)

    # Fallback coalition: always try to include at least one minister
    if not coalition_target and MINISTER_NAMES:
        coalition_target = [MINISTER_NAMES[step % len(MINISTER_NAMES)]]

    # Determine stance based on trust level
    if trust < 0.4:
        stance = "assertive"
    elif trust > 0.7:
        stance = "cooperative"
    else:
        stance = "balanced"

    return {
        "action": action,
        "reasoning": reasoning,
        "coalition_target": coalition_target[:3],
        "veto_prediction": veto_prediction[:2],
        "stance": stance,
    }

# ── WebSocket broadcast ──
async def broadcast(data: dict):
    msg = json.dumps(data, default=str)
    dead = []
    for ws in sim.ws_clients:
        try:
            await ws.send_text(msg)
        except:
            dead.append(ws)
    for ws in dead:
        sim.ws_clients.remove(ws)

# ── Step data extractor (reads ALL environment data) ──
def extract_step_data(obs, step, action_data):
    m = obs.metadata
    council = m.get("council", {})
    expl = m.get("explanation", {})
    drift = m.get("drift_vars", {})
    rb = m.get("reward_breakdown", {})

    # Extract minister data from council
    ministers = []
    for i, mi in enumerate(council.get("ministers", [])):
        ministers.append({
            "id": mi.get("id", f"M{i}"),
            "name": mi.get("name", f"Minister {i}"),
            "role": mi.get("role", "general"),
            "influence": round(mi.get("influence", 0.5), 4),
            "utility": mi.get("utility_vector", []),
            "proposal": mi.get("proposal", ""),
            "voted_for": mi.get("voted_for", ""),
        })

    # Extract real negotiation data
    neg = m.get("negotiation", {})
    neg_outcome = m.get("negotiation_outcome", {})

    # Extract action info
    action_str = action_data if isinstance(action_data, str) else action_data.get("action", "")

    result = {
        "type": "step",
        "step": step,
        "action": action_str,
        "council_action": council.get("action", action_str),
        "reward": round(obs.reward, 6),
        "done": obs.done,
        "collapsed": m.get("collapsed", False),
        "state": {
            "public_satisfaction": round(m.get("public_satisfaction", 50), 2),
            "gdp_index": round(m.get("gdp_index", 100), 2),
            "pollution_index": round(m.get("pollution_index", 100), 2),
            "healthcare_index": round(m.get("healthcare_index", 50), 2),
            "education_index": round(m.get("education_index", 50), 2),
            "renewable_energy": round(m.get("renewable_energy_ratio", 0.3), 4),
        },
        "council": {
            "coalition_formed": council.get("coalition_formed", False),
            "coalition_strength": round(council.get("coalition_strength", 0), 4),
            "vetoes": council.get("vetoes", []),
            "betrayal": council.get("betrayal_occurred", False),
            "institutional_trust": round(council.get("institutional_trust", 0.6), 4),
            "influence_vector": [round(v, 4) for v in council.get("influence_vector", [])],
            "ministers": ministers,
            "recommended_action": council.get("recommended_action", ""),
        },
        "explanation": {
            "causal_chain": expl.get("causal_chain", []),
            "nl_narrative": expl.get("nl_narrative", ""),
            "counterfactuals": expl.get("counterfactuals", []),
            "alignment_score": expl.get("alignment_score", 50),
            "risk_alerts": expl.get("risk_alerts", []),
            "credit_attribution": expl.get("credit_attribution", {}),
        },
        "reward_breakdown": {
            "base": round(rb.get("base_reward", 0), 4),
            "pareto": round(rb.get("pareto_bonus", 0), 4),
            "oscillation": round(rb.get("oscillation_penalty", 0), 4),
            "cooperation": round(rb.get("cooperation_bonus", 0), 4),
            "tom_reward": round(rb.get("tom_reward", neg_outcome.get("tom_reward", 0)), 4),
            "total": round(rb.get("total_reward", obs.reward), 4),
        },
        "drift": {
            "institutional_trust": round(drift.get("institutional_trust", 0.6), 4),
            "policy_fatigue": round(drift.get("policy_fatigue", 0.0), 4),
            "regulatory_burden": round(drift.get("regulatory_burden", 0.0), 4),
        },
        "active_events": m.get("active_events", []),
        "chaos": sim.chaos,
        "task_id": sim.task_id,
        "speed": sim.speed,
        # v3: Real negotiation data
        "negotiation": neg,
        "negotiation_outcome": neg_outcome,
        # v3: Agent's actual decision
        "agent_action": action_data if isinstance(action_data, dict) else {"action": action_data},
        # v3: Briefing data
        "new_briefing": m.get("new_briefing", ""),
        "active_briefings": m.get("active_briefings", []),
    }
    return result

# ── Episode runner ──
async def run_episode():
    async with sim.lock:
        orig = copy.deepcopy(TASK_CONFIGS[sim.task_id])
        TASK_CONFIGS[sim.task_id]["chaos_level"] = sim.chaos
        TASK_CONFIGS[sim.task_id]["max_steps"] = 500
        
        sim.env = PolicyEnvironment()
        sim.obs = sim.env.reset(seed=sim.seed, task_id=sim.task_id)
        sim.step = 0
        sim.history = []
        sim.running = True
        sim.paused = False
        
        await broadcast({"type": "episode_start", "seed": sim.seed, "task_id": sim.task_id, "chaos": sim.chaos})
    
    try:
        while sim.running and not sim.obs.done:
            if sim.paused:
                await asyncio.sleep(0.1)
                continue
            
            # Apply live chaos updates
            TASK_CONFIGS[sim.task_id]["chaos_level"] = sim.chaos
            
            action_data = agent_policy(sim.obs, sim.step)
            action_str = action_data.get("action", "subsidize_renewables")
            sim.obs = sim.env.step({
                "action": action_str,
                "reasoning": action_data.get("reasoning", ""),
                "coalition_target": action_data.get("coalition_target", []),
                "veto_prediction": action_data.get("veto_prediction", []),
                "stance": action_data.get("stance", "balanced"),
            })
            sim.step += 1
            
            step_data = extract_step_data(sim.obs, sim.step, action_data)
            sim.history.append(step_data)
            
            await broadcast(step_data)
            
            # Speed control (minimum 0.08s to allow WebSocket to flush)
            delay = max(0.08, 1.0 / max(sim.speed, 0.1))
            await asyncio.sleep(delay)
        
        # Episode ended
        if sim.env:
            traj = sim.env.get_trajectory()
            score = grade_trajectory(sim.task_id, traj)
            summary = {
                "seed": sim.seed, "task_id": sim.task_id, "chaos": sim.chaos,
                "steps": sim.step, "score": round(score, 4),
                "collapsed": sim.obs.metadata.get("collapsed", False) if sim.obs else True,
                "timestamp": time.strftime("%H:%M:%S"),
            }
            sim.episodes.append(summary)
            await broadcast({"type": "episode_end", **summary})
    
    except Exception as e:
        await broadcast({"type": "error", "message": str(e)})
    
    finally:
        TASK_CONFIGS[sim.task_id] = orig
        sim.running = False

# ── API Routes ──

@app.get("/")
async def root():
    return FileResponse(os.path.join(os.path.dirname(__file__), "dashboard.html"))

@app.get("/control")
async def control_panel():
    return FileResponse(os.path.join(os.path.dirname(__file__), "control.html"))

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    sim.ws_clients.append(ws)
    # Send current state
    await ws.send_text(json.dumps({
        "type": "init",
        "task_id": sim.task_id, "chaos": sim.chaos, "speed": sim.speed,
        "seed": sim.seed, "running": sim.running, "step": sim.step,
        "history": sim.history[-50:],  # last 50 steps
        "episodes": sim.episodes[-20:],
        "actions": sorted(VALID_ACTIONS),
        "tasks": list(TASK_CONFIGS.keys()),
    }, default=str))
    try:
        while True:
            data = await ws.receive_text()
            msg = json.loads(data)
            cmd = msg.get("cmd")
            
            if cmd == "start":
                sim.seed = msg.get("seed", random.randint(0, 9999))
                sim.task_id = msg.get("task_id", sim.task_id)
                sim.chaos = msg.get("chaos", sim.chaos)
                if not sim.running:
                    asyncio.create_task(run_episode())
            
            elif cmd == "stop":
                sim.running = False
            
            elif cmd == "pause":
                sim.paused = not sim.paused
                await broadcast({"type": "pause", "paused": sim.paused})
            
            elif cmd == "reset":
                sim.running = False
                await asyncio.sleep(0.2)
                sim.seed = msg.get("seed", random.randint(0, 9999))
                sim.task_id = msg.get("task_id", sim.task_id)
                sim.chaos = msg.get("chaos", sim.chaos)
                sim.history = []
                sim.step = 0
                await broadcast({"type": "reset"})
            
            elif cmd == "chaos":
                sim.chaos = max(0.0, min(1.0, msg.get("value", 0.5)))
                await broadcast({"type": "chaos_update", "chaos": sim.chaos})
            
            elif cmd == "speed":
                sim.speed = max(0.5, min(20.0, msg.get("value", 1.0)))
                await broadcast({"type": "speed_update", "speed": sim.speed})
            
            elif cmd == "regime":
                regime = msg.get("regime", "calibrated")
                sim.task_id = "sustainable_governance_extreme" if regime == "extreme" else "sustainable_governance"
                await broadcast({"type": "regime_update", "task_id": sim.task_id, "regime": regime})
            
            elif cmd == "replay_step":
                idx = msg.get("index", 0)
                if 0 <= idx < len(sim.history):
                    await ws.send_text(json.dumps({"type": "replay", **sim.history[idx]}))
            
            elif cmd == "export":
                await ws.send_text(json.dumps({
                    "type": "export",
                    "data": sim.history,
                    "episodes": sim.episodes,
                }))
    
    except WebSocketDisconnect:
        sim.ws_clients.remove(ws)

@app.get("/api/status")
async def status():
    return {"running": sim.running, "paused": sim.paused, "step": sim.step,
            "chaos": sim.chaos, "speed": sim.speed, "task_id": sim.task_id,
            "seed": sim.seed, "episodes_count": len(sim.episodes)}

if __name__ == "__main__":
    print("\n" + "="*60)
    print("  OpenENV Research Dashboard")
    print("  Open: http://localhost:8765")
    print("="*60 + "\n")
    uvicorn.run(app, host="0.0.0.0", port=8765, log_level="warning")
