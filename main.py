#!/usr/bin/env python3
"""
AI Policy Engine — Single Command Launcher

Usage:
    python main.py --demo           # 3-episode showcase with full narrative output
    python main.py --full-eval      # complete 6-phase validation suite
    python main.py --astronomical   # 1000-ep stress + ablation + scaling report
"""

import sys, os, argparse, time, json, random

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from server.policy_environment import PolicyEnvironment
from server.config import VALID_ACTIONS, TASK_CONFIGS
from server.tasks import grade_trajectory
from episode_logger import EpisodeLogger

SEP = "=" * 72
AL = sorted(VALID_ACTIONS)

# ──────────────────────────────────────────────────────────────────────
# Agents
# ──────────────────────────────────────────────────────────────────────

def agent_heuristic(obs, step, rng):
    cycle = ["subsidize_renewables","invest_in_education","increase_welfare",
             "stimulate_economy","invest_in_healthcare","incentivize_clean_tech",
             "enforce_emission_limits","increase_welfare"]
    return cycle[step % len(cycle)]

def agent_smart(obs, step, rng):
    sat = obs.metadata.get("public_satisfaction", 50)
    poll = obs.metadata.get("pollution_index", 100)
    gdp = obs.metadata.get("gdp_index", 100)
    if sat < 35: return "increase_welfare"
    if poll > 200: return "enforce_emission_limits"
    if gdp < 50: return "stimulate_economy"
    return ["subsidize_renewables","invest_in_education","increase_welfare",
            "stimulate_economy","invest_in_healthcare"][step % 5]

# ──────────────────────────────────────────────────────────────────────
# Demo Mode
# ──────────────────────────────────────────────────────────────────────

def run_demo():
    print(f"\n{SEP}")
    print("  AI POLICY ENGINE — DEMO MODE (3 episodes)")
    print(f"{SEP}")

    logger = EpisodeLogger("outputs/demo_episodes.jsonl", enabled=True)
    rng = random.Random(42)
    tasks = ["environmental_recovery", "balanced_economy", "sustainable_governance"]

    for i, task_id in enumerate(tasks):
        print(f"\n  Episode {i+1}: {task_id}")
        print(f"  {'-'*50}")

        env = PolicyEnvironment()
        obs = env.reset(seed=42 + i, task_id=task_id)
        episode_id = f"demo_{i+1}_{task_id}"
        logger.begin_episode(episode_id, task_id, seed=42 + i)

        step = 0
        while not obs.done:
            action = agent_smart(obs, step, rng)
            obs = env.step({"action": action})
            logger.log_step(step, action, obs.metadata)

            # Print narrative every 10 steps
            if step % 10 == 0 or obs.done:
                expl = obs.metadata.get("explanation", {})
                narrative = expl.get("nl_narrative", "")
                alerts = expl.get("risk_alerts", [])
                council = obs.metadata.get("council", {})
                print(f"  Step {step:3d} | Action: {action}")
                if narrative:
                    print(f"    -> {narrative[:120]}{'...' if len(narrative)>120 else ''}")
                if alerts:
                    print(f"    [!] {alerts[0]}")
                if council.get("coalition_formed"):
                    print(f"    [+] Coalition formed (strength={council.get('coalition_strength',0):.2f})")
                if council.get("vetoes"):
                    print(f"    [X] Vetoes: {council['vetoes']}")
            step += 1

        traj = env.get_trajectory()
        score = grade_trajectory(task_id, traj)
        collapsed = obs.metadata.get("collapsed", False)

        # Log episode end
        end_meta = dict(obs.metadata)
        end_meta.setdefault("final_score", score)
        end_meta.setdefault("collapsed", collapsed)
        end_meta.setdefault("total_steps", step)
        logger.end_episode(end_meta)

        status = "COLLAPSED" if collapsed else "SURVIVED"
        print(f"\n  [{status}] Steps: {step} | Score: {score:.4f}")
        expl_final = obs.metadata.get("explanation", {})
        cfs = expl_final.get("counterfactuals", [])
        if cfs:
            cf = cfs[-1]
            print(f"  Counterfactual: {cf.get('interpretation','')}")

    logger.write_summary_report("outputs/demo_summary.json")
    print(f"\n{SEP}")
    print(f"  Demo complete. Episodes logged to outputs/demo_episodes.jsonl")
    print(f"{SEP}")


# ──────────────────────────────────────────────────────────────────────
# Full Eval Mode
# ──────────────────────────────────────────────────────────────────────

def run_full_eval():
    print(f"\n{SEP}")
    print("  AI POLICY ENGINE — FULL EVALUATION MODE")
    print(f"{SEP}")
    import subprocess
    result = subprocess.run(
        [sys.executable, "validation_suite.py"],
        cwd=os.path.dirname(os.path.abspath(__file__))
    )
    return result.returncode


# ──────────────────────────────────────────────────────────────────────
# Astronomical Mode
# ──────────────────────────────────────────────────────────────────────

def run_astronomical():
    print(f"\n{SEP}")
    print("  AI POLICY ENGINE — ASTRONOMICAL MODE")
    print("  1000-episode stress + ablation + scaling report")
    print(f"{SEP}")

    os.makedirs("outputs", exist_ok=True)
    logger = EpisodeLogger("outputs/astronomical_episodes.jsonl", enabled=True)

    # 1. Run 1000-episode stress test
    print("\n  [1/3] 1000-Episode Stress Test (max chaos, sustainable_governance_extreme)...")
    task_id = "sustainable_governance_extreme"
    rng = random.Random(42)
    scores, steps_all, collapses = [], [], 0

    for i in range(1000):
        env = PolicyEnvironment()
        obs = env.reset(seed=i, task_id=task_id)
        logger.begin_episode(f"stress_{i}", task_id, seed=i)
        step = 0
        while not obs.done:
            action = agent_smart(obs, step, rng)
            obs = env.step({"action": action})
            logger.log_step(step, action, obs.metadata)
            step += 1
        traj = env.get_trajectory()
        score = grade_trajectory(task_id, traj)
        scores.append(score)
        steps_all.append(step)
        if obs.metadata.get("collapsed"): collapses += 1

        end_meta = dict(obs.metadata)
        end_meta.setdefault("final_score", score)
        end_meta.setdefault("collapsed", obs.metadata.get("collapsed", True))
        end_meta.setdefault("total_steps", step)
        logger.end_episode(end_meta)

        if (i + 1) % 100 == 0:
            surv = 1.0 - collapses / (i + 1)
            avg_s = sum(scores) / len(scores)
            print(f"    [{i+1:4d}/1000] surv={surv:.1%}  avg_score={avg_s:.4f}")

    surv_rate = 1.0 - collapses / 1000
    avg_score = sum(scores) / 1000
    avg_steps = sum(steps_all) / 1000
    print(f"\n  Stress Test Complete: surv={surv_rate:.1%} avg_score={avg_score:.4f} avg_steps={avg_steps:.1f}")

    # 2. Run ablation study
    print(f"\n  [2/3] Ablation Study (4 configs x 100 episodes)...")
    ablation_results = _run_ablation(n_eps=100)

    # 3. Run scaling report
    print(f"\n  [3/3] Scaling Report (multi-agent vs baselines)...")
    from server.curriculum_engine import AutomatedBaselineRunner
    runner = AutomatedBaselineRunner()
    scale_results = runner.run_eval_round("sustainable_governance", n_episodes=50)
    runner.print_scaling_report("sustainable_governance", scale_results)

    # Compute Robustness Score
    pareto_q = avg_score
    coop_idx = scale_results.get("multi_council", {}).get("survival_rate", 0.0)
    repro_factor = 1.0  # strict seeding throughout
    robustness = surv_rate * pareto_q * max(coop_idx, 0.1) * repro_factor

    print(f"\n{SEP}")
    print(f"  ROBUSTNESS SCORE: {robustness:.4f}")
    print(f"    = extreme_survival({surv_rate:.3f}) x pareto_quality({pareto_q:.3f})")
    print(f"    x cooperation_index({coop_idx:.3f}) x reproducibility(1.0)")
    print(f"{SEP}")

    # Save results
    summary = {
        "stress_test": {"episodes": 1000, "task": task_id, "survival_rate": surv_rate,
                        "avg_score": avg_score, "avg_steps": avg_steps},
        "ablation": ablation_results,
        "scaling": {k: v for k, v in scale_results.items()},
        "robustness_score": round(robustness, 4),
    }
    with open("outputs/astronomical_results.json", "w") as f:
        json.dump(summary, f, indent=2)

    logger.write_summary_report("outputs/astronomical_summary.json")
    print(f"\n  Results saved to outputs/astronomical_results.json")


def _run_ablation(n_eps: int = 100) -> dict:
    """Run 4 ablation configs and return comparison table."""
    from server.config import TASK_CONFIGS
    import copy as _copy

    task_id = "sustainable_governance"
    rng = random.Random(42)
    results = {}

    ablations = {
        "full_system": {},
        "no_multi_agent": {"num_ministers": 1},
        "no_chaos": {"chaos_level": 0.0, "event_frequency_multiplier": 0.0, "events_enabled": False},
        "no_drift": {"drift_enabled": False},
        "no_events": {"events_enabled": False, "event_frequency_multiplier": 0.0},
    }

    original_cfg = _copy.deepcopy(TASK_CONFIGS[task_id])

    for ablation_name, overrides in ablations.items():
        # Apply overrides
        TASK_CONFIGS[task_id] = _copy.deepcopy(original_cfg)
        TASK_CONFIGS[task_id].update(overrides)

        scores, steps_list, collapses = [], [], 0
        for i in range(n_eps):
            env = PolicyEnvironment()
            obs = env.reset(seed=10000 + i, task_id=task_id)
            step = 0
            while not obs.done:
                action = agent_heuristic(obs, step, rng)
                obs = env.step({"action": action})
                step += 1
            traj = env.get_trajectory()
            scores.append(grade_trajectory(task_id, traj))
            steps_list.append(step)
            if obs.metadata.get("collapsed"): collapses += 1

        results[ablation_name] = {
            "avg_score": round(sum(scores)/n_eps, 4),
            "survival_rate": round(1.0 - collapses/n_eps, 4),
            "avg_steps": round(sum(steps_list)/n_eps, 1),
        }
        TASK_CONFIGS[task_id] = _copy.deepcopy(original_cfg)

    print(f"\n  {'Ablation Config':<25s} {'Score':>7s} {'Surv%':>7s} {'Steps':>7s}")
    print(f"  {'-'*50}")
    full = results.get("full_system", {})
    for name, r in results.items():
        score_delta = r["avg_score"] - full.get("avg_score", r["avg_score"])
        marker = f"  ({score_delta:+.4f})" if name != "full_system" else ""
        print(f"  {name:<25s} {r['avg_score']:7.4f} {r['survival_rate']*100:6.1f}% {r['avg_steps']:6.1f}{marker}")

    return results


# ──────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AI Policy Engine Launcher")
    parser.add_argument("--demo", action="store_true", help="Run 3-episode showcase")
    parser.add_argument("--full-eval", action="store_true", help="Run full validation suite")
    parser.add_argument("--astronomical", action="store_true", help="1000-ep stress + ablation")
    args = parser.parse_args()

    if args.demo:
        run_demo()
    elif args.full_eval:
        sys.exit(run_full_eval())
    elif args.astronomical:
        run_astronomical()
    else:
        print("Usage: python main.py [--demo | --full-eval | --astronomical]")
        parser.print_help()
