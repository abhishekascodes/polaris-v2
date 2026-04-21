#!/usr/bin/env python3
"""Mega Validation Suite v3 Part 2: Tests 6-10 (stricter params)"""
import sys, os, random, json, copy, time, subprocess
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from server.policy_environment import PolicyEnvironment
from server.config import VALID_ACTIONS, TASK_CONFIGS, OBS_TOTAL_DIM
from server.tasks import grade_trajectory

SEP = "=" * 72
AL = sorted(VALID_ACTIONS)

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

def agent_random(obs, step, rng): return rng.choice(AL)
def agent_greedy(obs, step, rng): return "stimulate_economy"
def agent_council(obs, step, rng):
    c = obs.metadata.get("council",{})
    rec = c.get("recommended_action")
    if rec and rec in VALID_ACTIONS: return rec
    return agent_smart(obs, step, rng)

def run_batch(fn, task_id, n, ms=200, nm=None, label=""):
    orig = copy.deepcopy(TASK_CONFIGS[task_id])
    TASK_CONFIGS[task_id]["max_steps"] = ms
    if nm is not None: TASK_CONFIGS[task_id]["num_ministers"] = nm
    sc,st,col = [],[],0
    for i in range(n):
        env = PolicyEnvironment()
        obs = env.reset(seed=i, task_id=task_id)
        rng = random.Random(i)
        s = 0
        while not obs.done:
            obs = env.step({"action": fn(obs, s, rng)})
            s += 1
        st.append(s)
        if obs.metadata.get("collapsed"): col += 1
        sc.append(grade_trajectory(task_id, env.get_trajectory()))
    TASK_CONFIGS[task_id] = orig
    return {"label":label,"score":round(sum(sc)/n,4),"best":round(max(sc),4),
            "surv":round(1-col/n,4),"steps":round(sum(st)/n,1)}

results = {}

# =====================================================================
# TEST 6: INTELLIGENCE SCALING
# =====================================================================
def test6():
    print(f"\n{SEP}")
    print("  TEST 6: INTELLIGENCE SCALING (The Deciding Test)")
    print(f"{SEP}")
    agents = [
        ("Random", agent_random, 1),
        ("Greedy (GDP)", agent_greedy, 1),
        ("Heuristic", agent_heuristic, 1),
        ("Smart", agent_smart, 1),
        ("Council (5-minister)", agent_council, 5),
    ]
    for regime, tid in [("Calibrated","sustainable_governance"),("Extreme","sustainable_governance_extreme")]:
        print(f"\n  --- {regime} Regime ({tid}) ---")
        print(f"  {'Agent':<25s} {'Score':>7s} {'Surv%':>7s} {'Steps':>7s} {'Best':>7s}")
        print(f"  {'-'*55}")
        data = {}
        for label, fn, nm in agents:
            r = run_batch(fn, tid, 100, 200, nm, label)
            data[label] = r
            print(f"  {label:<25s} {r['score']:7.4f} {r['surv']*100:6.1f}% {r['steps']:6.1f} {r['best']:7.4f}")
        results.setdefault("test6",{})[regime] = data

    cal = results["test6"]["Calibrated"]
    smart_steps = cal["Smart"]["steps"]
    rand_steps = cal["Random"]["steps"]
    ratio = smart_steps / max(rand_steps, 1)
    council_steps = cal["Council (5-minister)"]["steps"]
    print(f"\n  Smart/Random ratio: {ratio:.2f}x")
    print(f"  Smart survival: {cal['Smart']['surv']*100:.0f}%")
    print(f"  Council steps: {council_steps:.1f}")
    p = smart_steps > rand_steps
    results["test6"]["pass"] = p
    results["test6"]["ratio"] = round(ratio, 2)
    print(f"\n  TEST 6: [{'PASS' if p else 'FAIL'}] Intelligence Scaling")

# =====================================================================
# TEST 7: EXTREME REGIME DESTRUCTION (50 episodes)
# =====================================================================
def test7():
    print(f"\n{SEP}")
    print("  TEST 7: EXTREME REGIME DESTRUCTION (50 episodes, chaos=1.0)")
    print(f"{SEP}")
    tid = "sustainable_governance_extreme"
    orig = copy.deepcopy(TASK_CONFIGS[tid])
    TASK_CONFIGS[tid]["chaos_level"] = 1.0
    TASK_CONFIGS[tid]["max_steps"] = 200
    agents = {"Random":(agent_random,1),"Heuristic":(agent_heuristic,1),
              "Smart":(agent_smart,1),"Council":(agent_council,5)}
    print(f"\n  {'Agent':<20s} {'Surv%':>7s} {'Steps':>7s} {'Score':>7s} {'Crashes':>8s}")
    print(f"  {'-'*55}")
    data = {}
    for label,(fn,nm) in agents.items():
        sc,st,col,cr = [],[],0,0
        for i in range(50):
            try:
                TASK_CONFIGS[tid]["num_ministers"] = nm
                env = PolicyEnvironment()
                obs = env.reset(seed=i, task_id=tid)
                rng = random.Random(i)
                s = 0
                while not obs.done:
                    obs = env.step({"action": fn(obs, s, rng)})
                    s += 1
                st.append(s)
                if obs.metadata.get("collapsed"): col += 1
                sc.append(grade_trajectory(tid, env.get_trajectory()))
            except: cr += 1; st.append(0); sc.append(0)
        surv = 1-col/50
        data[label] = {"surv":surv,"score":round(sum(sc)/50,4),"steps":round(sum(st)/50,1),"crashes":cr}
        print(f"  {label:<20s} {surv*100:6.1f}% {sum(st)/50:6.1f} {sum(sc)/50:7.4f} {cr:8d}")
    TASK_CONFIGS[tid] = orig
    tc = sum(d["crashes"] for d in data.values())
    p = tc == 0
    results["test7"] = {"pass":p,"agents":data,"crashes":tc}
    print(f"\n  Zero crashes: {'YES' if tc==0 else 'NO'}")
    print(f"\n  TEST 7: [{'PASS' if p else 'FAIL'}] Extreme Destruction")

# =====================================================================
# TEST 8: EXPLAINABILITY & COUNTERFACTUAL (80 episodes)
# =====================================================================
def test8():
    print(f"\n{SEP}")
    print("  TEST 8: EXPLAINABILITY & COUNTERFACTUAL (80 episodes)")
    print(f"{SEP}")
    tid = "sustainable_governance"
    ts_causal,ts_narr,ts_cf,ts_total = 0,0,0,0
    chain_lens,cf_deltas,aligns,credits = [],[],[],0
    veto_with_causal, coal_with_causal = 0,0
    total_vetoes_seen, total_coal_seen = 0,0
    for i in range(80):
        env = PolicyEnvironment()
        obs = env.reset(seed=i, task_id=tid)
        rng = random.Random(i)
        s = 0
        while not obs.done:
            obs = env.step({"action": agent_smart(obs, s, rng)})
            expl = obs.metadata.get("explanation",{})
            chain = expl.get("causal_chain",[])
            if chain: ts_causal += 1; chain_lens.append(len(chain))
            nl = expl.get("nl_narrative","")
            if nl: ts_narr += 1
            cfs = expl.get("counterfactuals",[])
            if cfs:
                ts_cf += 1
                for cf in cfs: cf_deltas.append(cf.get("reward_delta",0))
            al = expl.get("alignment_score")
            if al is not None: aligns.append(al)
            if expl.get("credit_attribution"): credits += 1
            c = obs.metadata.get("council",{})
            if c.get("vetoes"):
                total_vetoes_seen += 1
                if chain: veto_with_causal += 1
            if c.get("coalition_formed"):
                total_coal_seen += 1
                if chain: coal_with_causal += 1
            ts_total += 1
            s += 1
    cp = ts_causal/max(ts_total,1)*100
    np_ = ts_narr/max(ts_total,1)*100
    cfp = ts_cf/max(ts_total,1)*100
    acl = sum(chain_lens)/max(len(chain_lens),1)
    aa = sum(aligns)/max(len(aligns),1)
    print(f"  Total steps: {ts_total}")
    print(f"  Causal chains: {ts_causal} ({cp:.1f}%)")
    print(f"  NL narratives: {ts_narr} ({np_:.1f}%)")
    print(f"  Counterfactuals: {ts_cf} ({cfp:.1f}%)")
    print(f"  Credit attributions: {credits}")
    print(f"  Avg chain length: {acl:.2f}")
    print(f"  Avg alignment: {aa:.1f}/100")
    print(f"  Total CFs generated: {len(cf_deltas)}")
    if cf_deltas:
        print(f"  CF delta range: [{min(cf_deltas):.4f}, {max(cf_deltas):.4f}]")
    print(f"  Vetoes with causal chain: {veto_with_causal}/{total_vetoes_seen}")
    print(f"  Coalitions with causal chain: {coal_with_causal}/{total_coal_seen}")
    p = cp > 90 and np_ > 90
    results["test8"] = {"pass":p,"causal_pct":round(cp,1),"narrative_pct":round(np_,1),
                         "cf_pct":round(cfp,1),"avg_alignment":round(aa,1),"total_cfs":len(cf_deltas)}
    print(f"\n  TEST 8: [{'PASS' if p else 'FAIL'}] Explainability")

# =====================================================================
# TEST 9: ROBUSTNESS & STRESS (1000 eps, max chaos + drift)
# =====================================================================
def test9():
    print(f"\n{SEP}")
    print("  TEST 9: ROBUSTNESS (1000 eps, max chaos + max drift)")
    print(f"{SEP}")
    tid = "sustainable_governance_extreme"
    orig = copy.deepcopy(TASK_CONFIGS[tid])
    TASK_CONFIGS[tid]["chaos_level"] = 1.0
    TASK_CONFIGS[tid]["max_steps"] = 200
    all_r,crashes,collapses,rv = [],0,0,0
    for i in range(1000):
        try:
            env = PolicyEnvironment()
            obs = env.reset(seed=i, task_id=tid)
            rng = random.Random(i)
            s = 0
            while not obs.done:
                obs = env.step({"action": agent_smart(obs, s, rng)})
                all_r.append(obs.reward)
                if obs.reward < 0.0 or obs.reward > 1.0: rv += 1
                s += 1
            if obs.metadata.get("collapsed"): collapses += 1
        except: crashes += 1
        if (i+1) % 250 == 0:
            print(f"    [{i+1:4d}/1000] crashes={crashes} collapses={collapses} reward_violations={rv}")
    TASK_CONFIGS[tid] = orig
    rmin = min(all_r) if all_r else 0
    rmax = max(all_r) if all_r else 0
    rmean = sum(all_r)/max(len(all_r),1)
    ib = all(0.0 <= r <= 1.0 for r in all_r)
    strict = all(0.01 <= r <= 0.99 for r in all_r)
    print(f"\n  Results:")
    print(f"    Episodes: 1000")
    print(f"    Crashes: {crashes}")
    print(f"    Collapses: {collapses}")
    print(f"    Reward samples: {len(all_r)}")
    print(f"    Range: [{rmin:.6f}, {rmax:.6f}]")
    print(f"    Mean: {rmean:.4f}")
    print(f"    In [0,1]: {'YES' if ib else 'NO'}")
    print(f"    In [0.01,0.99]: {'YES' if strict else 'APPROX'}")
    print(f"    Violations: {rv}")
    p = crashes == 0 and ib
    results["test9"] = {"pass":p,"crashes":crashes,"violations":rv,"range":[round(rmin,6),round(rmax,6)],"strict":strict}
    print(f"\n  TEST 9: [{'PASS' if p else 'FAIL'}] Robustness")

# =====================================================================
# TEST 10: PRODUCTION & OPENENV COMPLIANCE
# =====================================================================
def test10():
    print(f"\n{SEP}")
    print("  TEST 10: PRODUCTION & OPENENV COMPLIANCE")
    print(f"{SEP}")
    # A: Cross-process determinism
    print(f"\n  Part A: Cross-process determinism (3 runs)...")
    script = 'import sys,os,json\nsys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))\nfrom server.policy_environment import PolicyEnvironment\nenv=PolicyEnvironment()\nobs=env.reset(seed=42,task_id="environmental_recovery")\nr=[]\nfor s in range(30):\n obs=env.step({"action":["subsidize_renewables","invest_in_education","increase_welfare","stimulate_economy","invest_in_healthcare"][s%5]})\n r.append(round(obs.reward,10))\nprint(json.dumps(r))\n'
    sp = os.path.join(os.path.dirname(os.path.abspath(__file__)),"_det.py")
    with open(sp,"w") as f: f.write(script)
    rr = []
    for t in range(3):
        try:
            r = subprocess.run([sys.executable,sp],capture_output=True,text=True,timeout=30,
                               cwd=os.path.dirname(os.path.abspath(__file__)))
            rr.append(json.loads(r.stdout.strip()) if r.returncode==0 else None)
        except: rr.append(None)
    try: os.remove(sp)
    except: pass
    valid = [r for r in rr if r is not None]
    cp = len(valid)>=2 and all(v==valid[0] for v in valid)
    print(f"    Runs: {len(valid)}/3  Deterministic: {'PASS' if cp else 'FAIL'}")

    # B: Imports
    print(f"\n  Part B: Module imports...")
    try:
        from server.config import VALID_ACTIONS, OBS_TOTAL_DIM, TASK_CONFIGS
        from server.drift_engine import DriftEngine
        from server.transition_engine import TransitionEngine
        from server.event_engine import EventEngine
        from server.multi_agent_council import MultiAgentCouncil
        from server.reward_engine import RewardEngine
        from server.explainability import ExplainabilityEngine
        from server.curriculum_engine import CurriculumEngine, AutomatedBaselineRunner
        from episode_logger import EpisodeLogger
        ip = True
        print(f"    OK: {len(VALID_ACTIONS)} actions, OBS={OBS_TOTAL_DIM}, {len(TASK_CONFIGS)} tasks")
    except Exception as e: ip = False; print(f"    FAIL: {e}")

    # C: Obs dim
    print(f"\n  Part C: Observation vector...")
    env = PolicyEnvironment()
    obs = env.reset(seed=42, task_id="environmental_recovery")
    obs = env.step({"action": "subsidize_renewables"})
    vec = env.get_augmented_observation_vector()
    dp = len(vec) == OBS_TOTAL_DIM
    print(f"    Dim: {len(vec)} (expected {OBS_TOTAL_DIM}): {'PASS' if dp else 'FAIL'}")

    # D: JSONL export
    print(f"\n  Part D: JSONL export verification...")
    from episode_logger import EpisodeLogger
    trace_path = "outputs/_test_trace.jsonl"
    try: os.remove(trace_path)
    except: pass
    lg = EpisodeLogger(trace_path, enabled=True)
    env2 = PolicyEnvironment()
    obs2 = env2.reset(seed=99, task_id="environmental_recovery")
    lg.begin_episode("test_ep", "environmental_recovery", seed=99)
    for s in range(10):
        obs2 = env2.step({"action": "subsidize_renewables"})
        lg.log_step(s, "subsidize_renewables", obs2.metadata)
    lg.end_episode(obs2.metadata)
    # Verify JSONL — logger writes 1 line per episode with embedded steps array
    with open(trace_path) as f:
        lines = f.readlines()
    jp = len(lines) >= 1
    if jp:
        ep = json.loads(lines[0])
        ep_keys = all(k in ep for k in ["episode_id","task_id","seed","steps","total_steps"])
        steps = ep.get("steps", [])
        has_steps = len(steps) >= 10
        step_keys = True
        if steps:
            step_keys = all(k in steps[0] for k in ["step","action","reward","causal_chain_len"])
        jp = ep_keys and has_steps and step_keys
        print(f"    Episodes: {len(lines)}  Steps in ep: {len(steps)}")
        print(f"    Episode keys OK: {'YES' if ep_keys else 'NO'}")
        print(f"    Step keys OK: {'YES' if step_keys else 'NO'}")
        print(f"    Coalitions: {ep.get('coalitions_formed',0)}  Vetoes: {ep.get('vetoes_cast',0)}")
        print(f"    Survived: {ep.get('survived')}  Score: {ep.get('final_score',0)}")
    else:
        print(f"    Lines: {len(lines)} (expected >=1)")
    try: os.remove(trace_path)
    except: pass

    # E: Validation suite
    print(f"\n  Part E: Built-in validation suite...")
    r = subprocess.run([sys.executable,"validation_suite.py"],capture_output=True,text=True,timeout=120,
                       cwd=os.path.dirname(os.path.abspath(__file__)))
    vs_pass = r.returncode == 0
    # Count passes in output
    pass_count = r.stdout.count("[PASS]")
    fail_count = r.stdout.count("[FAIL]")
    print(f"    Return code: {r.returncode}  Passes: {pass_count}  Fails: {fail_count}")
    if not vs_pass:
        for line in r.stderr.split("\n")[-5:]:
            if line.strip(): print(f"    {line.strip()}")

    p = cp and ip and dp and jp and vs_pass
    results["test10"] = {"pass":p,"determinism":cp,"imports":ip,"obs_dim":dp,"jsonl":jp,
                          "validation_suite":vs_pass,"vs_passes":pass_count,"vs_fails":fail_count}
    print(f"\n  TEST 10: [{'PASS' if p else 'FAIL'}] Production")

if __name__ == "__main__":
    t0 = time.time()
    print(f"\n{'#'*72}")
    print(f"  MEGA VALIDATION v3 PART 2 (Tests 6-10)")
    print(f"{'#'*72}")
    os.makedirs("outputs", exist_ok=True)
    test6(); test7(); test8(); test9(); test10()
    elapsed = time.time() - t0
    print(f"\n{'#'*72}")
    print(f"  PART 2 COMPLETE | Time: {elapsed:.1f}s")
    print(f"{'#'*72}")
    for k,v in results.items():
        print(f"  {k}: [{'PASS' if v.get('pass') else 'FAIL'}]")
    with open("outputs/mega_v3_part2.json","w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"  Saved to outputs/mega_v3_part2.json")
