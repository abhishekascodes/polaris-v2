#!/usr/bin/env python3
"""
OpenENV — Live LLM Benchmark (Groq / OpenAI compatible)
=========================================================
Runs a REAL LLM through the governance environment and measures performance.
Shows that frontier models can actually reason about policy decisions.

Usage:
    set GROQ_API_KEY=gsk_...
    python llm_benchmark.py
"""
import sys, os, time, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from openai import OpenAI
from server.policy_environment import PolicyEnvironment
from server.config import VALID_ACTIONS, ACTION_DESCRIPTIONS, TASK_CONFIGS
from server.tasks import grade_trajectory

# ── Configuration ──
GROQ_KEY = os.getenv("GROQ_API_KEY", "")
API_BASE = os.getenv("API_BASE_URL", "https://api.groq.com/openai/v1")
MODEL = os.getenv("MODEL_NAME", "llama-3.3-70b-versatile")

SYSTEM_PROMPT = """You are an expert AI policy advisor governing a simulated nation.
Each turn you MUST choose EXACTLY ONE action from the list below.

AVAILABLE ACTIONS:
{actions}

RULES:
1. Respond with ONLY the action name. No explanation, no quotes, no formatting.
2. Balance GDP, pollution, and public satisfaction simultaneously.
3. React to active events (pandemic → healthcare, recession → economy, etc.)
4. Prevent collapse: keep GDP > 15, pollution < 290, satisfaction > 5.

Respond with EXACTLY one action name. Nothing else."""

ACTION_LIST_STR = "\n".join(f"  - {n}: {d}" for n, d in ACTION_DESCRIPTIONS.items())

def format_obs(meta, step, max_steps):
    events = ", ".join(meta.get("active_events", [])) or "none"
    return (
        f"--- STEP {step}/{max_steps} ---\n"
        f"GDP: {meta.get('gdp_index',0):.0f}/200 | "
        f"Pollution: {meta.get('pollution_index',0):.0f}/500 | "
        f"Satisfaction: {meta.get('public_satisfaction',0):.0f}/100\n"
        f"Healthcare: {meta.get('healthcare_index',0):.0f}/100 | "
        f"Education: {meta.get('education_index',0):.0f}/100 | "
        f"Unemployment: {meta.get('unemployment_rate',0):.1f}%\n"
        f"Renewables: {meta.get('renewable_energy_ratio',0):.0%} | "
        f"Inequality: {meta.get('inequality_index',0):.0f}/100\n"
        f"Events: {events}\n\nChoose your action:"
    )

def get_llm_action(client, obs_text):
    try:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT.format(actions=ACTION_LIST_STR)},
                {"role": "user", "content": obs_text},
            ],
            temperature=0.1,
            max_tokens=30,
        )
        raw = resp.choices[0].message.content.strip().lower().strip("'\"` \n")
        # Match action
        if raw in VALID_ACTIONS: return raw
        for a in VALID_ACTIONS:
            if a in raw: return a
        return "no_action"
    except Exception as e:
        print(f"    [LLM Error: {e}]")
        return "no_action"

def run_benchmark(client, task_id="sustainable_governance", seed=42):
    cfg = TASK_CONFIGS[task_id]
    max_steps = cfg["max_steps"]
    env = PolicyEnvironment()
    obs = env.reset(seed=seed, task_id=task_id)

    print(f"\n  {'='*55}")
    print(f"  Task: {task_id} | Model: {MODEL}")
    print(f"  {'='*55}")

    total_reward, step = 0.0, 0
    action_counts = {}
    t0 = time.time()

    while not obs.done:
        step += 1
        obs_text = format_obs(obs.metadata, step, max_steps)
        
        t_llm = time.time()
        action = get_llm_action(client, obs_text)
        llm_ms = (time.time() - t_llm) * 1000

        obs = env.step({"action": action})
        total_reward += obs.reward
        action_counts[action] = action_counts.get(action, 0) + 1

        # Live output
        gdp = obs.metadata.get("gdp_index", 0)
        poll = obs.metadata.get("pollution_index", 0)
        sat = obs.metadata.get("public_satisfaction", 0)
        events = obs.metadata.get("active_events", [])
        ev_str = f" ⚡{','.join(events)}" if events else ""
        print(f"  Step {step:3d} | {action:30s} | r={obs.reward:.3f} | "
              f"GDP={gdp:.0f} Poll={poll:.0f} Sat={sat:.0f} | "
              f"{llm_ms:.0f}ms{ev_str}")

    elapsed = time.time() - t0
    collapsed = obs.metadata.get("collapsed", False)
    traj = env.get_trajectory()
    score = grade_trajectory(task_id, traj)

    print(f"\n  {'─'*55}")
    print(f"  RESULT: {'COLLAPSED ❌' if collapsed else 'SURVIVED ✅'}")
    print(f"  Score: {score:.4f} | Total Reward: {total_reward:.2f}")
    print(f"  Steps: {step} | Time: {elapsed:.1f}s ({step/elapsed:.1f} steps/sec)")
    print(f"  Action distribution:")
    for a, c in sorted(action_counts.items(), key=lambda x: -x[1]):
        print(f"    {a:30s}: {c:3d} ({c/step*100:4.1f}%)")
    print(f"  {'='*55}")

    return {
        "task": task_id, "model": MODEL, "score": round(score, 4),
        "reward": round(total_reward, 4), "steps": step,
        "collapsed": collapsed, "survived": not collapsed,
        "time_sec": round(elapsed, 1),
        "actions": action_counts,
    }

def main():
    if not GROQ_KEY:
        print("\n  ⚠️  Set GROQ_API_KEY first:")
        print("     set GROQ_API_KEY=gsk_your_key_here")
        print("     python llm_benchmark.py")
        print("\n  Get free key at: https://console.groq.com/keys")
        sys.exit(1)

    client = OpenAI(api_key=GROQ_KEY, base_url=API_BASE)
    print(f"\n  🚀 OpenENV LLM Benchmark — {MODEL} via Groq")

    # Test connection
    try:
        test = client.chat.completions.create(
            model=MODEL, messages=[{"role": "user", "content": "say ok"}],
            max_tokens=5)
        print(f"  ✓ Connection verified: {test.choices[0].message.content.strip()}")
    except Exception as e:
        print(f"  ✗ Connection failed: {e}")
        sys.exit(1)

    # Run all tasks
    results = []
    tasks = [
        ("environmental_recovery", 42),
        ("balanced_economy", 42),
        ("sustainable_governance", 42),
        ("sustainable_governance", 100),
        ("sustainable_governance", 200),
    ]

    for task_id, seed in tasks:
        r = run_benchmark(client, task_id=task_id, seed=seed)
        results.append(r)

    # Summary
    print(f"\n\n  {'='*60}")
    print(f"  LLM BENCHMARK SUMMARY — {MODEL}")
    print(f"  {'='*60}")
    print(f"  {'Task':<30s} {'Score':>7s} {'Reward':>8s} {'Steps':>6s} {'Status':>10s}")
    print(f"  {'─'*65}")
    for r in results:
        status = "SURVIVED ✅" if r["survived"] else "COLLAPSED ❌"
        print(f"  {r['task']:<30s} {r['score']:7.4f} {r['reward']:8.2f} "
              f"{r['steps']:6d} {status:>10s}")
    
    survived = sum(1 for r in results if r["survived"])
    avg_score = sum(r["score"] for r in results) / len(results)
    print(f"\n  Survival: {survived}/{len(results)} | Avg Score: {avg_score:.4f}")
    print(f"  {'='*60}")

    # Save
    out_path = "outputs/llm_benchmark.json"
    os.makedirs("outputs", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"model": MODEL, "results": results}, f, indent=2)
    print(f"\n  Results saved to: {out_path}")

if __name__ == "__main__":
    main()
