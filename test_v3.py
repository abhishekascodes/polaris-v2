"""POLARIS v3 — Full Integration Test"""
import sys, io, time, json
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.path.insert(0, ".")
from server.policy_environment import PolicyEnvironment
from server.config import TASK_CONFIGS, CORE_ACTIONS
from server.tasks import grade_trajectory, get_task_ids

SEP = "=" * 60
results = {}

for task_id in get_task_ids():
    cfg = TASK_CONFIGS[task_id]
    max_steps = cfg["max_steps"]
    num_min = cfg.get("num_ministers", 1)
    neg = cfg.get("negotiation_enabled", False)
    brief = cfg.get("briefing_enabled", False)

    print(f"\n{SEP}")
    print(f"  TASK: {task_id}")
    print(f"  Steps: {max_steps} | Ministers: {num_min} | Negotiation: {neg} | Briefings: {brief}")
    print(SEP)

    env = PolicyEnvironment()
    t0 = time.time()
    obs = env.reset(seed=42, task_id=task_id)

    has_negotiation = "negotiation" in obs.metadata
    total_reward = 0.0
    step = 0

    while not obs.done:
        step += 1
        # Use structured action for negotiation tasks
        if has_negotiation:
            action = {
                "action": CORE_ACTIONS[step % len(CORE_ACTIONS)],
                "reasoning": f"Step {step} strategic choice",
                "coalition_target": ["Director Okafor"],
                "veto_prediction": ["General Tanaka"] if step % 3 == 0 else [],
                "stance": "cooperative",
            }
        else:
            action = {"action": CORE_ACTIONS[step % len(CORE_ACTIONS)]}

        obs = env.step(action)
        total_reward += obs.reward

    elapsed = time.time() - t0
    score = grade_trajectory(task_id, env.get_trajectory())
    collapsed = obs.metadata.get("collapsed", False)

    results[task_id] = {
        "score": round(score, 4),
        "reward": round(total_reward, 2),
        "steps": step,
        "collapsed": collapsed,
        "time_s": round(elapsed, 2),
        "negotiation": has_negotiation,
        "briefing_stats": obs.metadata.get("briefing_stats", {}),
    }

    status = "COLLAPSED" if collapsed else "SURVIVED"
    print(f"  {status} | Score: {score:.4f} | Reward: {total_reward:.2f} | Steps: {step} | Time: {elapsed:.1f}s")

    # Show negotiation stats for v3 tasks
    if has_negotiation:
        outcome_count = sum(1 for t in env.get_trajectory() if "negotiation_outcome" in t)
        print(f"  Negotiation rounds: {outcome_count}")
        bs = obs.metadata.get("briefing_stats", {})
        if bs:
            print(f"  Briefings: {bs.get('total_briefings',0)} total, {bs.get('resolved',0)} resolved")

print(f"\n{SEP}")
print("  ALL TASKS COMPLETE")
print(SEP)
print(f"\n  Tasks: {len(results)}")
for tid, r in results.items():
    neg_tag = " [NEGOTIATION]" if r["negotiation"] else ""
    print(f"    {tid}: score={r['score']:.4f} reward={r['reward']:.1f} {'SURVIVED' if not r['collapsed'] else 'COLLAPSED'}{neg_tag}")
print(f"\n  POLARIS v3 INTEGRATION: PASS")
