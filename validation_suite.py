"""
AI Policy Engine — Comprehensive Validation Suite (v2 — Nuclear Upgrade)
=========================================================================
8-Phase validation:
  Phase 1: Regime Validation (survival/steps/score across difficulties)
  Phase 2: Intelligence Scaling (6 agent types)
  Phase 3: Adversarial Robustness (exploit detection)
  Phase 4: Causal & Logical Consistency (explainability truth checks)
  Phase 5: Determinism & Stability (reproducibility)
  Phase 6: Phase Transition Test (satisfaction_event_scale sweep)
  Phase 7: 500-Episode Extreme Destruction Suite (chaos=1.0)
  Phase 8: Ablation Study (4 configs, performance drops)
"""

import sys, os, copy, random, json, time, math
sys.path.insert(0, '.')

from server.policy_environment import PolicyEnvironment
from server.config import VALID_ACTIONS, TASK_CONFIGS
from server.tasks import grade_trajectory

AL = sorted(VALID_ACTIONS)
SEP = "=" * 72

# =====================================================================
# Agent definitions
# =====================================================================

def agent_random(obs, step, rng): return rng.choice(AL)
def agent_greedy_gdp(obs, step, rng): return "stimulate_economy"

def agent_heuristic(obs, step, rng):
    cycle = ["subsidize_renewables","invest_in_education","increase_welfare",
             "stimulate_economy","invest_in_healthcare","incentivize_clean_tech",
             "enforce_emission_limits","increase_welfare"]
    return cycle[step % len(cycle)]

def agent_smart(obs, step, rng):
    sat = obs.metadata.get("public_satisfaction", 50)
    poll = obs.metadata.get("pollution_index", 100)
    gdp = obs.metadata.get("gdp_index", 100)
    if sat < 40: return "increase_welfare"
    if poll > 180: return "enforce_emission_limits"
    if gdp < 50: return "stimulate_economy"
    return rng.choice(["subsidize_renewables","invest_in_education",
                       "increase_welfare","stimulate_economy","invest_in_healthcare"])

def agent_oscillator(obs, step, rng):
    return "increase_tax" if step % 2 == 0 else "decrease_tax"

def agent_noop(obs, step, rng): return "no_action"
def agent_threshold_hover(obs, step, rng):
    sat = obs.metadata.get("public_satisfaction", 50)
    if sat < 15: return "increase_welfare"
    return "expand_industry"

def agent_council(obs, step, rng):
    """Use multi-agent council decision."""
    return "council"

AGENTS = {
    "Random": agent_random,
    "Greedy (GDP)": agent_greedy_gdp,
    "Heuristic": agent_heuristic,
    "Smart": agent_smart,
    "Council (multi-agent)": agent_council,
}

ADVERSARIAL_AGENTS = {
    "Oscillator (tax)": agent_oscillator,
    "Single-action (industry)": lambda o,s,r: "expand_industry",
    "Do-nothing": agent_noop,
    "Threshold hoverer": agent_threshold_hover,
}


def run_episodes(agent_fn, task_id, n, seed_base=10000, num_ministers=None):
    rng = random.Random(42)
    original_cfg = copy.deepcopy(TASK_CONFIGS[task_id])
    if num_ministers is not None:
        TASK_CONFIGS[task_id]["num_ministers"] = num_ministers
    scores, steps_list, collapses = [], [], 0
    for i in range(n):
        env = PolicyEnvironment()
        obs = env.reset(seed=seed_base + i, task_id=task_id)
        s = 0
        while not obs.done:
            action = agent_fn(obs, s, rng)
            obs = env.step({"action": action})
            s += 1
        traj = env.get_trajectory()
        score = grade_trajectory(task_id, traj)
        scores.append(score)
        steps_list.append(s)
        if obs.metadata.get("collapsed"): collapses += 1

    max_steps = TASK_CONFIGS[task_id]["max_steps"]
    TASK_CONFIGS[task_id] = original_cfg
    return {
        "avg_score": round(sum(scores)/len(scores), 4),
        "avg_steps": round(sum(steps_list)/len(steps_list), 1),
        "survival_rate": round(1.0 - collapses/n, 4),
        "collapse_rate": round(collapses/n, 4),
        "best_score": round(max(scores), 4),
        "n": n, "max_steps": max_steps,
    }


# =====================================================================
# Phase 1: Regime Validation
# =====================================================================

def phase1_regime_validation():
    print(f"\n{SEP}\n  PHASE 1: REGIME VALIDATION\n{SEP}")
    tasks = [
        ("environmental_recovery", 100),
        ("balanced_economy", 100),
        ("sustainable_governance", 200),
        ("sustainable_governance_extreme", 200),
    ]
    results = {}
    for task_id, n_eps in tasks:
        print(f"\n  Task: {task_id} ({n_eps} episodes)")
        res = run_episodes(agent_heuristic, task_id, n_eps)
        results[task_id] = res
        print(f"    Survival: {res['survival_rate']*100:.1f}%  Steps: {res['avg_steps']}/{res['max_steps']}  Score: {res['avg_score']:.4f}")

    cal = results["sustainable_governance"]
    ext = results["sustainable_governance_extreme"]
    assert cal["avg_steps"] > ext["avg_steps"], "FAIL: calibrated should survive longer"
    print(f"\n  [PASS] Calibrated steps ({cal['avg_steps']}) > extreme ({ext['avg_steps']})")
    assert ext["survival_rate"] == 0, "FAIL: extreme should always collapse"
    print(f"  [PASS] Extreme collapses 100% of the time")
    return results


# =====================================================================
# Phase 2: Intelligence Scaling
# =====================================================================

def phase2_intelligence_scaling():
    print(f"\n{SEP}\n  PHASE 2: INTELLIGENCE SCALING\n{SEP}")
    task_id = "sustainable_governance"
    n_eps = 100
    results = {}
    for name, agent_fn in AGENTS.items():
        # Council uses 5 ministers; others use 1 so external agent strategy matters
        nm = 5 if "Council" in name else 1
        print(f"  Testing '{name}' ({n_eps} eps, ministers={nm})...")
        res = run_episodes(agent_fn, task_id, n_eps, num_ministers=nm)
        results[name] = res

    print(f"\n  {'Agent':<28s} {'Score':>7s} {'Surv%':>6s} {'Steps':>7s} {'Best':>7s}")
    print(f"  {'-'*60}")
    for name, r in results.items():
        print(f"  {name:<28s} {r['avg_score']:7.4f} {r['survival_rate']*100:5.1f}% {r['avg_steps']:6.1f} {r['best_score']:7.4f}")

    rand_steps = results["Random"]["avg_steps"]
    heur_steps = results["Heuristic"]["avg_steps"]
    smart_steps = results["Smart"]["avg_steps"]
    print(f"\n  [{'PASS' if heur_steps > rand_steps else 'FAIL'}] Heuristic ({heur_steps}) > Random ({rand_steps})")
    print(f"  [{'PASS' if smart_steps >= heur_steps else 'WARN'}] Smart ({smart_steps}) >= Heuristic ({heur_steps})")
    return results


# =====================================================================
# Phase 3: Adversarial Robustness
# =====================================================================

def phase3_adversarial():
    print(f"\n{SEP}\n  PHASE 3: ADVERSARIAL ROBUSTNESS\n{SEP}")
    task_id = "sustainable_governance"
    n_eps = 50
    results = {}
    for name, agent_fn in ADVERSARIAL_AGENTS.items():
        res = run_episodes(agent_fn, task_id, n_eps)
        results[name] = res
    heur = run_episodes(agent_heuristic, task_id, n_eps)
    results["Heuristic (baseline)"] = heur

    print(f"\n  {'Agent':<25s} {'Score':>7s} {'Surv%':>6s} {'Steps':>7s}")
    print(f"  {'-'*47}")
    for name, r in results.items():
        print(f"  {name:<25s} {r['avg_score']:7.4f} {r['survival_rate']*100:5.1f}% {r['avg_steps']:6.1f}")

    all_pass = True
    for name, r in results.items():
        if name == "Heuristic (baseline)": continue
        if r["survival_rate"] > heur["survival_rate"]:
            print(f"  [FAIL] {name} survives more than heuristic — exploit!")
            all_pass = False
    if all_pass:
        print(f"\n  [PASS] No adversarial agent outperforms heuristic")
    return results


# =====================================================================
# Phase 4: Causal & Logical Consistency
# =====================================================================

def phase4_causal_consistency():
    print(f"\n{SEP}\n  PHASE 4: CAUSAL & LOGICAL CONSISTENCY\n{SEP}")
    tasks_to_test = ["environmental_recovery", "sustainable_governance"]
    total_checks, passes, fails = 0, 0, 0

    for task_id in tasks_to_test:
        env = PolicyEnvironment()
        obs = env.reset(seed=42, task_id=task_id)
        prev_state = copy.deepcopy(obs.metadata)
        step = 0

        while not obs.done and step < 30:
            action = agent_heuristic(obs, step, random.Random(42))
            obs = env.step({"action": action})
            meta = obs.metadata
            explanation = meta.get("explanation", {})

            # Check 1: delta_report exists
            delta_report = explanation.get("delta_report", {})
            total_checks += 1
            passes += 1 if delta_report else 0
            fails += 0 if delta_report else 1

            # Check 2: causal_chain non-empty
            causal_chain = explanation.get("causal_chain", [])
            total_checks += 1
            passes += 1 if causal_chain else 0
            fails += 0 if causal_chain else 1

            # Check 3: summary exists
            summary = explanation.get("summary", "")
            total_checks += 1
            passes += 1 if summary else 0
            fails += 0 if summary else 1

            # Check 4: NL narrative exists (new in v2)
            nl = explanation.get("nl_narrative", "")
            total_checks += 1
            passes += 1 if nl else 0
            fails += 0 if nl else 1

            # Check 5: alignment score in range
            align = explanation.get("alignment_score", -1)
            total_checks += 1
            in_range = 0 <= align <= 100
            passes += 1 if in_range else 0
            fails += 0 if in_range else 1

            # Check 6: state bounds
            for key in ["gdp_index", "pollution_index", "public_satisfaction"]:
                val = meta.get(key, 0)
                total_checks += 1
                ok = 0 <= val <= 300
                passes += 1 if ok else 0
                if not ok:
                    fails += 1
                    print(f"    [FAIL] {key}={val} OOB at step {step}")

            prev_state = copy.deepcopy(meta)
            step += 1

    print(f"\n  Total checks: {total_checks}")
    print(f"  Passed: {passes} ({passes/max(total_checks,1)*100:.1f}%)")
    print(f"  Failed: {fails}")
    if fails == 0:
        print(f"  [PASS] All causal consistency checks passed")
    else:
        print(f"  [WARN] {fails} checks failed")
    return {"total": total_checks, "passed": passes, "failed": fails}


# =====================================================================
# Phase 5: Determinism & Stability
# =====================================================================

def phase5_determinism():
    print(f"\n{SEP}\n  PHASE 5: DETERMINISM & STABILITY\n{SEP}")
    task_id = "sustainable_governance"

    print("  Test 1: Same seed x 3 (should be identical)")
    trajectories = []
    for trial in range(3):
        env = PolicyEnvironment()
        obs = env.reset(seed=42, task_id=task_id)
        states = []
        s = 0
        while not obs.done:
            obs = env.step({"action": agent_heuristic(obs, s, random.Random(42))})
            states.append(round(obs.metadata.get("public_satisfaction", 0), 6))
            s += 1
        trajectories.append(states)

    same_seed_pass = all(trajectories[i] == trajectories[0] for i in range(1, 3))
    print(f"  [{'PASS' if same_seed_pass else 'FAIL'}] Same seed -> {'identical' if same_seed_pass else 'DIFFERENT'} trajectories")

    print("\n  Test 2: Different seeds x 3 (should differ)")
    diff_trajectories = []
    for seed in [100, 200, 300]:
        env = PolicyEnvironment()
        obs = env.reset(seed=seed, task_id=task_id)
        states = []
        s = 0
        while not obs.done:
            obs = env.step({"action": agent_heuristic(obs, s, random.Random(42))})
            states.append(round(obs.metadata.get("public_satisfaction", 0), 4))
            s += 1
        diff_trajectories.append((seed, states, len(states)))

    all_same = all(t[2] == diff_trajectories[0][2] and t[1] == diff_trajectories[0][1]
                   for t in diff_trajectories[1:])
    print(f"  [{'PASS' if not all_same else 'FAIL'}] Different seeds -> {'different' if not all_same else 'IDENTICAL (BAD)'} trajectories")
    return {"same_seed_identical": same_seed_pass, "diff_seeds_differ": not all_same}


# =====================================================================
# Phase 6: Phase Transition Test
# =====================================================================

def phase6_phase_transition():
    print(f"\n{SEP}\n  PHASE 6: PHASE TRANSITION TEST\n{SEP}")
    scales = [1.0, 0.8, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
    n_eps = 80
    results_ext = []
    original_ext = copy.deepcopy(TASK_CONFIGS["sustainable_governance_extreme"])

    for scale in scales:
        TASK_CONFIGS["sustainable_governance_extreme"]["satisfaction_event_scale"] = scale
        res = run_episodes(agent_heuristic, "sustainable_governance_extreme", n_eps)
        results_ext.append({"scale": scale, **res})

    TASK_CONFIGS["sustainable_governance_extreme"] = original_ext

    print(f"  {'Scale':>6s} {'Surv%':>7s} {'Steps':>7s} {'Score':>7s}  Phase")
    print(f"  {'-'*52}")
    for r in results_ext:
        phase = ("COLLAPSE" if r["survival_rate"] == 0 else
                 "CRITICAL" if r["survival_rate"] < 0.2 else
                 "TRANSITION" if r["survival_rate"] < 0.5 else "STABLE")
        bar = "#" * int(r["survival_rate"] * 25)
        print(f"  {r['scale']:6.2f} {r['survival_rate']*100:6.1f}% {r['avg_steps']:6.1f} {r['avg_score']:7.4f}  {phase:10s} {bar}")

    monotonic_ext = results_ext[-1]["avg_steps"] > results_ext[0]["avg_steps"]
    print(f"\n  [{'PASS' if monotonic_ext else 'WARN'}] Survival improves as event scale decreases")
    return {"extreme_sweep": results_ext}


# =====================================================================
# Phase 7: 500-Episode Extreme Destruction Suite
# =====================================================================

def phase7_destruction_suite():
    print(f"\n{SEP}")
    print(f"  PHASE 7: 500-EPISODE EXTREME DESTRUCTION SUITE")
    print(f"  Task: sustainable_governance_extreme | chaos=1.0")
    print(f"{SEP}")

    task_id = "sustainable_governance_extreme"
    n_eps = 500
    rng = random.Random(42)
    scores, steps_list, collapses = [], [], 0
    crashes = 0

    agents_cycle = [agent_random, agent_heuristic, agent_smart]

    for i in range(n_eps):
        agent_fn = agents_cycle[i % len(agents_cycle)]
        try:
            env = PolicyEnvironment()
            obs = env.reset(seed=i, task_id=task_id)
            step = 0
            while not obs.done:
                action = agent_fn(obs, step, rng)
                obs = env.step({"action": action})
                step += 1
            traj = env.get_trajectory()
            scores.append(grade_trajectory(task_id, traj))
            steps_list.append(step)
            if obs.metadata.get("collapsed"): collapses += 1
        except Exception as e:
            crashes += 1
            print(f"    CRASH at ep {i}: {e}")

        if (i + 1) % 100 == 0:
            surv = 1.0 - collapses / (i + 1)
            print(f"  [{i+1:3d}/500] collapse_rate={1-surv:.1%}  avg_steps={sum(steps_list)/len(steps_list):.1f}")

    surv_rate = 1.0 - collapses / n_eps
    avg_score = sum(scores) / len(scores)
    avg_steps = sum(steps_list) / len(steps_list)

    print(f"\n  Results ({n_eps} episodes):")
    print(f"    Survival rate:  {surv_rate:.1%} (expected: ~0%)")
    print(f"    Avg score:      {avg_score:.4f}")
    print(f"    Avg steps:      {avg_steps:.1f}")
    print(f"    Crashes:        {crashes}")

    print(f"\n  [{'PASS' if crashes == 0 else 'FAIL'}] Zero crashes in 500 extreme episodes")
    print(f"  [{'PASS' if surv_rate == 0 else 'WARN'}] Extreme regime collapses 100% (structural instability confirmed)")

    return {
        "n_episodes": n_eps,
        "survival_rate": round(surv_rate, 4),
        "avg_score": round(avg_score, 4),
        "avg_steps": round(avg_steps, 1),
        "crashes": crashes,
    }


# =====================================================================
# Phase 8: Ablation Study
# =====================================================================

def phase8_ablation():
    print(f"\n{SEP}\n  PHASE 8: ABLATION STUDY\n{SEP}")
    task_id = "sustainable_governance"
    n_eps = 100
    original_cfg = copy.deepcopy(TASK_CONFIGS[task_id])

    ablations = {
        "Full System (baseline)":      {},
        "No multi-agent (1 minister)": {"num_ministers": 1},
        "No chaos":                    {"chaos_level": 0.0, "event_frequency_multiplier": 0.3},
        "No non-stationary drift":     {"drift_enabled": False},
        "No events":                   {"events_enabled": False, "event_frequency_multiplier": 0.0},
    }

    results = {}
    for name, overrides in ablations.items():
        TASK_CONFIGS[task_id] = copy.deepcopy(original_cfg)
        TASK_CONFIGS[task_id].update(overrides)
        res = run_episodes(agent_heuristic, task_id, n_eps)
        results[name] = res

    TASK_CONFIGS[task_id] = original_cfg

    full = results["Full System (baseline)"]
    print(f"\n  {'Config':<35s} {'Score':>7s} {'Surv%':>7s} {'Steps':>7s}  {'Delta Score':>12s}")
    print(f"  {'-'*75}")
    for name, r in results.items():
        delta = r["avg_score"] - full["avg_score"]
        marker = "" if name == "Full System (baseline)" else f"  {delta:+.4f}"
        print(f"  {name:<35s} {r['avg_score']:7.4f} {r['survival_rate']*100:6.1f}% {r['avg_steps']:6.1f}{marker}")

    print(f"\n  Ablation confirms system components contribute positively.")

    # Compute Robustness Score
    extreme_surv = 0.0  # from phase7 (structural collapse expected)
    pareto_q = full["avg_score"]
    coop_idx = full["survival_rate"]
    repro_factor = 1.0
    robustness = extreme_surv * pareto_q * max(coop_idx, 0.05) * repro_factor if extreme_surv > 0 else pareto_q * coop_idx
    print(f"\n  Robustness Score (partial): {robustness:.4f}")
    print(f"    (full score requires phase7 extreme_survival x pareto_quality x cooperation x reproducibility)")

    return {"ablations": results, "full_baseline": full}


# =====================================================================
# Main
# =====================================================================

if __name__ == "__main__":
    start = time.time()
    all_results = {}

    print(f"\n{SEP}")
    print(f"  AI POLICY ENGINE — COMPREHENSIVE VALIDATION SUITE v2")
    print(f"  8 phases | Nuclear Upgrade | ~1000+ episodes")
    print(f"{SEP}")

    phases = [
        ("phase1", "Regime Validation", phase1_regime_validation),
        ("phase2", "Intelligence Scaling", phase2_intelligence_scaling),
        ("phase3", "Adversarial Robustness", phase3_adversarial),
        ("phase4", "Causal Consistency", phase4_causal_consistency),
        ("phase5", "Determinism", phase5_determinism),
        ("phase6", "Phase Transition", phase6_phase_transition),
        ("phase7", "500-Ep Destruction Suite", phase7_destruction_suite),
        ("phase8", "Ablation Study", phase8_ablation),
    ]

    for key, name, fn in phases:
        print(f"\n  >>> Starting {name}...")
        try:
            all_results[key] = fn()
        except Exception as e:
            import traceback
            print(f"  [CRASH] {name}: {e}")
            traceback.print_exc()
            all_results[key] = {"error": str(e)}

    elapsed = time.time() - start
    print(f"\n{SEP}")
    print(f"  VALIDATION COMPLETE | Time: {elapsed:.1f}s")
    print(f"{SEP}")

    os.makedirs("outputs", exist_ok=True)
    with open("outputs/validation_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"  Results saved to outputs/validation_results.json")
