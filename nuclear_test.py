#!/usr/bin/env python3
"""
OpenENV — NUCLEAR STRESS TEST
==============================
Pushes every subsystem to absolute limits. Proves research-grade quality.
"""
import sys, os, time, json, random, math, traceback
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from server.policy_environment import PolicyEnvironment
from server.config import VALID_ACTIONS, CORE_ACTIONS, META_ACTIONS, TASK_CONFIGS
from server.tasks import grade_trajectory
from server.reward_engine import RewardEngine
from server.transition_engine import TransitionEngine
from server.event_engine import EventEngine
from server.explainability import ExplainabilityEngine
from server.multi_agent_council import MultiAgentCouncil

PASS = 0
FAIL = 0
TOTAL = 0

def test(name, condition, detail=""):
    global PASS, FAIL, TOTAL
    TOTAL += 1
    if condition:
        PASS += 1
        print(f"  [PASS] {name}")
    else:
        FAIL += 1
        print(f"  [FAIL] {name} — {detail}")

def section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")

# ═══════════════════════════════════════════════════════════
section("1. ENVIRONMENT INTEGRITY (all tasks, all seeds)")
# ═══════════════════════════════════════════════════════════

tasks = list(TASK_CONFIGS.keys())
GRADEABLE = ["environmental_recovery", "balanced_economy", "sustainable_governance", "sustainable_governance_extreme"]
test("Task count >= 4", len(tasks) >= 4, f"got {len(tasks)}")
test("Action space = 19", len(VALID_ACTIONS) == 19, f"got {len(VALID_ACTIONS)}")
test("Core actions = 16", len(CORE_ACTIONS) == 16)
test("Meta actions = 3", len(META_ACTIONS) == 3)

# Run every gradeable task with multiple seeds
for task_id in GRADEABLE:
    cfg = TASK_CONFIGS[task_id]
    max_steps = cfg["max_steps"]
    for seed in [0, 42, 999, 12345]:
        env = PolicyEnvironment()
        obs = env.reset(seed=seed, task_id=task_id)
        test(f"{task_id}/seed={seed}: reset ok", obs is not None and not obs.done)
        test(f"{task_id}/seed={seed}: metadata has keys",
             "gdp_index" in obs.metadata and "pollution_index" in obs.metadata)

        steps, rewards = 0, []
        while not obs.done and steps < max_steps + 10:
            action = CORE_ACTIONS[steps % len(CORE_ACTIONS)]
            obs = env.step({"action": action})
            rewards.append(obs.reward)
            steps += 1

        test(f"{task_id}/seed={seed}: completed ({steps} steps)", steps > 0)
        test(f"{task_id}/seed={seed}: rewards bounded",
             all(-5 <= r <= 5 for r in rewards),
             f"min={min(rewards):.3f} max={max(rewards):.3f}")

        traj = env.get_trajectory()
        score = grade_trajectory(task_id, traj)
        test(f"{task_id}/seed={seed}: score in [0,1]",
             0.0 <= score <= 1.0, f"got {score}")

# ═══════════════════════════════════════════════════════════
section("2. DETERMINISM — same seed = same trajectory")
# ═══════════════════════════════════════════════════════════

for task_id in ["sustainable_governance", "environmental_recovery"]:
    trajectories = []
    for trial in range(3):
        env = PolicyEnvironment()
        obs = env.reset(seed=42, task_id=task_id)
        rewards = []
        for s in range(30):
            if obs.done: break
            obs = env.step({"action": "subsidize_renewables"})
            rewards.append(round(obs.reward, 6))
        trajectories.append(rewards)

    test(f"{task_id}: deterministic (3 trials identical)",
         trajectories[0] == trajectories[1] == trajectories[2],
         f"lengths: {[len(t) for t in trajectories]}")

# ═══════════════════════════════════════════════════════════
section("3. REWARD ENGINE — mathematical properties")
# ═══════════════════════════════════════════════════════════

engine = RewardEngine()
# Rewards should be bounded
env = PolicyEnvironment()
obs = env.reset(seed=42, task_id="sustainable_governance")
all_rewards = []
for s in range(200):
    if obs.done: break
    obs = env.step({"action": CORE_ACTIONS[s % len(CORE_ACTIONS)]})
    all_rewards.append(obs.reward)

test("Rewards non-NaN", all(not math.isnan(r) for r in all_rewards))
test("Rewards non-Inf", all(not math.isinf(r) for r in all_rewards))
test("Rewards bounded [-5, 5]", all(-5 <= r <= 5 for r in all_rewards),
     f"range: [{min(all_rewards):.3f}, {max(all_rewards):.3f}]")
test("Reward variance > 0 (not constant)", 
     max(all_rewards) - min(all_rewards) > 0.01,
     f"range: {max(all_rewards)-min(all_rewards):.4f}")
test("Mean reward positive", sum(all_rewards)/len(all_rewards) > 0,
     f"mean: {sum(all_rewards)/len(all_rewards):.4f}")

# ═══════════════════════════════════════════════════════════
section("4. MULTI-AGENT COUNCIL — coalition dynamics")
# ═══════════════════════════════════════════════════════════

council = MultiAgentCouncil()
env = PolicyEnvironment()
obs = env.reset(seed=42, task_id="sustainable_governance")
coalitions, vetoes, betrayals = 0, 0, 0

for s in range(100):
    if obs.done: break
    obs = env.step({"action": CORE_ACTIONS[s % len(CORE_ACTIONS)]})
    meta = obs.metadata
    if "council_result" in meta:
        cr = meta["council_result"]
        if cr.get("coalition_formed"): coalitions += 1
        if cr.get("vetoed"): vetoes += 1

test("Coalitions formed > 0", coalitions > 0, f"got {coalitions}")
test("Council ran for 100 steps without crash", True)

# ═══════════════════════════════════════════════════════════
section("5. EVENT ENGINE — stochastic events fire")
# ═══════════════════════════════════════════════════════════

event_counts = {}
for seed in range(20):
    env = PolicyEnvironment()
    obs = env.reset(seed=seed, task_id="sustainable_governance")
    for s in range(50):
        if obs.done: break
        obs = env.step({"action": "no_action"})
        for ev in obs.metadata.get("active_events", []):
            event_counts[ev] = event_counts.get(ev, 0) + 1

test("Events fired across 20 episodes", len(event_counts) > 0,
     f"types: {list(event_counts.keys())}")
test("Multiple event types", len(event_counts) >= 2,
     f"got {len(event_counts)} types")

# ═══════════════════════════════════════════════════════════
section("6. EXPLAINABILITY — causal chains + counterfactuals")
# ═══════════════════════════════════════════════════════════

explain = ExplainabilityEngine()
env = PolicyEnvironment()
obs = env.reset(seed=42, task_id="sustainable_governance")
for s in range(5):
    obs = env.step({"action": "subsidize_renewables"})

meta = obs.metadata
has_explain = ("explanation" in meta or "causal_chain" in meta or 
               "narrative" in meta or "counterfactual" in meta)
test("Explainability data present in metadata", has_explain,
     f"keys: {[k for k in meta.keys() if 'explain' in k.lower() or 'causal' in k.lower() or 'narrative' in k.lower()]}")

# ═══════════════════════════════════════════════════════════
section("7. META-ACTIONS — council-level coordination")
# ═══════════════════════════════════════════════════════════

for meta_action in META_ACTIONS:
    env = PolicyEnvironment()
    obs = env.reset(seed=42, task_id="sustainable_governance")
    # Advance a few steps first
    for s in range(5):
        obs = env.step({"action": "no_action"})
    try:
        obs = env.step({"action": meta_action})
        test(f"Meta-action '{meta_action}' executes", True)
        test(f"Meta-action '{meta_action}' returns valid reward",
             not math.isnan(obs.reward))
    except Exception as e:
        test(f"Meta-action '{meta_action}' executes", False, str(e))

# ═══════════════════════════════════════════════════════════
section("8. COLLAPSE DETECTION — system collapses under extreme stress")
# ═══════════════════════════════════════════════════════════

collapse_count = 0
for seed in range(20):
    env = PolicyEnvironment()
    obs = env.reset(seed=seed, task_id="sustainable_governance_extreme")
    while not obs.done:
        obs = env.step({"action": "expand_industry"})  # worst action for stability
    if obs.metadata.get("collapsed"):
        collapse_count += 1

test("Extreme regime causes collapses", collapse_count > 0,
     f"{collapse_count}/20 collapsed")
test("Not ALL collapse (some resilience)", collapse_count < 20,
     f"{collapse_count}/20")

# ═══════════════════════════════════════════════════════════
section("9. STATISTICAL ROBUSTNESS — 500 episodes")
# ═══════════════════════════════════════════════════════════

scores_by_task = {}
t0 = time.time()
for task_id in ["environmental_recovery", "balanced_economy", "sustainable_governance"]:
    scores, survived = [], 0
    for seed in range(500):
        env = PolicyEnvironment()
        obs = env.reset(seed=seed, task_id=task_id)
        cycle = ["subsidize_renewables", "invest_in_education", "increase_welfare",
                 "stimulate_economy", "invest_in_healthcare", "incentivize_clean_tech"]
        s = 0
        while not obs.done:
            obs = env.step({"action": cycle[s % len(cycle)]})
            s += 1
        traj = env.get_trajectory()
        score = grade_trajectory(task_id, traj)
        scores.append(score)
        if not obs.metadata.get("collapsed"):
            survived += 1

    avg = sum(scores) / len(scores)
    std = (sum((s - avg)**2 for s in scores) / len(scores)) ** 0.5
    scores_by_task[task_id] = {"avg": avg, "std": std, "survival": survived/500}

    test(f"{task_id}: 500 eps completed", len(scores) == 500)
    test(f"{task_id}: avg score > 0", avg > 0, f"avg={avg:.4f}")
    test(f"{task_id}: std > 0 (not degenerate)", std > 0, f"std={std:.4f}")
    print(f"    avg={avg:.4f} std={std:.4f} survival={survived/500:.1%}")

elapsed = time.time() - t0
test(f"1500 episodes completed in <120s", elapsed < 120, f"took {elapsed:.1f}s")
print(f"    Throughput: {1500/elapsed:.0f} episodes/sec")

# ═══════════════════════════════════════════════════════════
section("10. DIFFICULTY SCALING — easy < medium < hard")
# ═══════════════════════════════════════════════════════════

easy_avg = scores_by_task["environmental_recovery"]["avg"]
med_avg = scores_by_task["balanced_economy"]["avg"]
hard_avg = scores_by_task["sustainable_governance"]["avg"]

test("Easy > Medium score", easy_avg >= med_avg,
     f"easy={easy_avg:.4f} med={med_avg:.4f}")
test("Medium > Hard score", med_avg >= hard_avg,
     f"med={med_avg:.4f} hard={hard_avg:.4f}")

easy_surv = scores_by_task["environmental_recovery"]["survival"]
hard_surv = scores_by_task["sustainable_governance"]["survival"]
test("Easy survival > Hard survival", easy_surv >= hard_surv,
     f"easy={easy_surv:.1%} hard={hard_surv:.1%}")

# ═══════════════════════════════════════════════════════════
section("11. API ENDPOINT VERIFICATION")
# ═══════════════════════════════════════════════════════════

try:
    import requests
    base = "http://localhost:7860"
    # Try the dashboard server instead
    r = requests.get("http://localhost:8765", timeout=3)
    test("Dashboard server responds", r.status_code == 200)
except Exception as e:
    test("Dashboard server responds", False, str(e))

# ═══════════════════════════════════════════════════════════
section("12. OPENENV SPEC COMPLIANCE")
# ═══════════════════════════════════════════════════════════

import yaml
with open("openenv.yaml") as f:
    spec = yaml.safe_load(f)

test("spec_version = 1", spec.get("spec_version") == 1)
test("name defined", "name" in spec)
test("entrypoint defined", "entrypoint" in spec or "app" in spec)
test("tasks defined", "tasks" in spec and len(spec["tasks"]) >= 3,
     f"got {len(spec.get('tasks', []))}")
test("action_space defined", "action_space" in spec)
test("observation_schema defined", "observation_schema" in spec)

for task in spec["tasks"]:
    test(f"Task '{task['id']}' has description", "description" in task)
    test(f"Task '{task['id']}' has max_steps", "max_steps" in task)

# ═══════════════════════════════════════════════════════════
# FINAL REPORT
# ═══════════════════════════════════════════════════════════

print(f"\n{'='*60}")
print(f"  NUCLEAR STRESS TEST — FINAL REPORT")
print(f"{'='*60}")
print(f"  PASSED: {PASS}/{TOTAL}")
print(f"  FAILED: {FAIL}/{TOTAL}")
print(f"  PASS RATE: {PASS/TOTAL*100:.1f}%")
print()
print(f"  STATISTICAL SUMMARY (500 episodes each):")
for task_id, stats in scores_by_task.items():
    print(f"    {task_id:>30s}: avg={stats['avg']:.4f} std={stats['std']:.4f} surv={stats['survival']:.1%}")
print()
if FAIL == 0:
    print(f"  VERDICT: RESEARCH-GRADE QUALITY CONFIRMED")
    print(f"  Zero failures across {TOTAL} tests, 1500+ episodes")
else:
    print(f"  VERDICT: {FAIL} issues to address")
print(f"{'='*60}")
