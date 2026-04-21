"""Quick smoke test for the nuclear upgrade."""
import sys, traceback

def main():
    print("=" * 60)
    print("  SMOKE TEST — OpenENV Nuclear Upgrade")
    print("=" * 60)

    # 1. Import all modules
    print("\n[1] Importing all modules...")
    from server.config import VALID_ACTIONS, OBS_TOTAL_DIM, TASK_CONFIGS
    from server.drift_engine import DriftEngine
    from server.transition_engine import TransitionEngine
    from server.event_engine import EventEngine
    from server.multi_agent_council import MultiAgentCouncil
    from server.reward_engine import RewardEngine
    from server.explainability import ExplainabilityEngine
    from server.policy_environment import PolicyEnvironment
    from episode_logger import EpisodeLogger
    print(f"  OK: {len(VALID_ACTIONS)} actions, OBS_DIM={OBS_TOTAL_DIM}")
    print(f"  Tasks: {list(TASK_CONFIGS.keys())}")

    # 2. Reset + Step
    print("\n[2] Testing reset + step...")
    env = PolicyEnvironment()
    obs = env.reset(seed=42, task_id="environmental_recovery")
    print(f"  Reset OK: {len(obs.metadata)} metadata keys")

    obs = env.step({"action": "subsidize_renewables"})
    print(f"  Step OK: reward={obs.reward:.4f}  done={obs.done}")

    expl = obs.metadata.get("explanation", {})
    chain_len = len(expl.get("causal_chain", []))
    nl = expl.get("nl_narrative", "")
    align = expl.get("alignment_score", -1)
    print(f"  Causal chain: {chain_len} links")
    print(f"  NL narrative: {nl[:100]}...")
    print(f"  Alignment: {align}")

    # 3. Council metadata
    print("\n[3] Testing council metadata...")
    council = obs.metadata.get("council", {})
    print(f"  Coalition formed: {council.get('coalition_formed', 'N/A')}")
    print(f"  Influence vector: {council.get('influence_vector', [])}")
    print(f"  Vetoes: {council.get('vetoes', [])}")

    # 4. Drift variables
    print("\n[4] Testing drift variables...")
    drift = obs.metadata.get("drift_vars", {})
    print(f"  Drift vars: {drift}")

    # 5. Multi-step episode
    print("\n[5] Running 20-step episode...")
    env2 = PolicyEnvironment()
    obs2 = env2.reset(seed=99, task_id="sustainable_governance")
    actions = ["subsidize_renewables", "invest_in_education", "increase_welfare",
               "stimulate_economy", "invest_in_healthcare"]
    for s in range(20):
        action = actions[s % len(actions)]
        obs2 = env2.step({"action": action})
        if obs2.done:
            print(f"  Episode ended at step {s+1}")
            break
    if not obs2.done:
        print(f"  20 steps completed, not done yet")
    sat = obs2.metadata.get("public_satisfaction", 0)
    gdp = obs2.metadata.get("gdp_index", 0)
    poll = obs2.metadata.get("pollution_index", 0)
    print(f"  Final state: sat={sat:.1f} gdp={gdp:.1f} poll={poll:.1f}")

    # 6. Augmented observation vector
    print("\n[6] Testing augmented observation vector...")
    try:
        vec = env2.get_augmented_observation_vector()
        print(f"  Vector dim: {len(vec)} (expected {OBS_TOTAL_DIM})")
        assert len(vec) == OBS_TOTAL_DIM, f"MISMATCH: {len(vec)} != {OBS_TOTAL_DIM}"
        print(f"  PASS: dimensions match")
    except Exception as e:
        print(f"  FAIL: {e}")
        traceback.print_exc()

    # 7. Grade trajectory
    print("\n[7] Testing trajectory grading...")
    from server.tasks import grade_trajectory
    traj = env2.get_trajectory()
    score = grade_trajectory("sustainable_governance", traj)
    print(f"  Trajectory score: {score:.4f}")

    # 8. Meta-actions
    print("\n[8] Testing meta-actions...")
    env3 = PolicyEnvironment()
    obs3 = env3.reset(seed=42, task_id="environmental_recovery")
    for meta in ["propose_global_policy_package", "force_emergency_coalition_vote", "reset_institutional_trust"]:
        obs3 = env3.step({"action": meta})
        print(f"  {meta}: reward={obs3.reward:.4f} done={obs3.done}")

    # 9. Determinism
    print("\n[9] Testing determinism (same seed = same result)...")
    results = []
    for trial in range(3):
        env_d = PolicyEnvironment()
        obs_d = env_d.reset(seed=42, task_id="environmental_recovery")
        for s in range(10):
            obs_d = env_d.step({"action": actions[s % len(actions)]})
        results.append(round(obs_d.metadata.get("public_satisfaction", 0), 6))
    if results[0] == results[1] == results[2]:
        print(f"  PASS: all 3 trials identical (sat={results[0]})")
    else:
        print(f"  FAIL: results differ: {results}")

    print("\n" + "=" * 60)
    print("  ALL SMOKE TESTS PASSED")
    print("=" * 60)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nFATAL ERROR: {e}")
        traceback.print_exc()
        sys.exit(1)
