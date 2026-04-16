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

# ── Agent (Smart heuristic for demo) ──
def agent_policy(obs, step):
    sat = obs.metadata.get("public_satisfaction", 50)
    poll = obs.metadata.get("pollution_index", 100)
    gdp = obs.metadata.get("gdp_index", 100)
    hc = obs.metadata.get("healthcare_index", 50)
    if sat < 30: return "increase_welfare"
    if poll > 200: return "enforce_emission_limits"
    if gdp < 40: return "stimulate_economy"
    if hc < 30: return "invest_in_healthcare"
    actions = ["subsidize_renewables","invest_in_education","increase_welfare",
               "stimulate_economy","invest_in_healthcare","incentivize_clean_tech"]
    return actions[step % len(actions)]

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

# ── Step data extractor ──
def extract_step_data(obs, step, action):
    m = obs.metadata
    council = m.get("council", {})
    expl = m.get("explanation", {})
    drift = m.get("drift_vars", {})
    rb = m.get("reward_breakdown", {})
    
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
    
    return {
        "type": "step",
        "step": step,
        "action": action,
        "council_action": council.get("action", action),
        "reward": round(obs.reward, 6),
        "done": obs.done,
        "collapsed": m.get("collapsed", False),
        "state": {
            "public_satisfaction": round(m.get("public_satisfaction", 50), 2),
            "gdp_index": round(m.get("gdp_index", 100), 2),
            "pollution_index": round(m.get("pollution_index", 100), 2),
            "healthcare_index": round(m.get("healthcare_index", 50), 2),
            "education_index": round(m.get("education_index", 50), 2),
            "renewable_energy": round(m.get("renewable_energy", 30), 2),
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
    }

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
            
            action = agent_policy(sim.obs, sim.step)
            sim.obs = sim.env.step({"action": action})
            sim.step += 1
            
            step_data = extract_step_data(sim.obs, sim.step, action)
            sim.history.append(step_data)
            
            await broadcast(step_data)
            
            # Speed control
            delay = 1.0 / max(sim.speed, 0.1)
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
