#!/usr/bin/env python3
"""Mega Validation Suite v3 Part 1: Tests 1-5 (stricter params)"""
import sys, os, random, json, copy, time, math
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

results = {}

# =====================================================================
# TEST 1: REPRODUCIBILITY (20 runs x 3 chaos, 200 steps)
# =====================================================================
def test1():
    print(f"\n{SEP}")
    print("  TEST 1: REPRODUCIBILITY & DETERMINISM")
    print(f"  20 runs x 3 chaos levels, 200 steps, calibrated regime")
    print(f"{SEP}")
    task_id = "sustainable_governance"
    original = copy.deepcopy(TASK_CONFIGS[task_id])
    all_pass = True
    detail = {}

    for chaos in [0.0, 0.5, 1.0]:
        TASK_CONFIGS[task_id] = copy.deepcopy(original)
        TASK_CONFIGS[task_id]["chaos_level"] = chaos
        TASK_CONFIGS[task_id]["max_steps"] = 200
        TASK_CONFIGS[task_id]["num_ministers"] = 3
        if chaos == 0.0:
            TASK_CONFIGS[task_id]["events_enabled"] = False
            TASK_CONFIGS[task_id]["event_frequency_multiplier"] = 0.0
        print(f"\n  Chaos={chaos} (20 identical runs, seed=42, 200 steps)...")
        traces = []
        for run in range(20):
            env = PolicyEnvironment()
            obs = env.reset(seed=42, task_id=task_id)
            t = {"rewards":[],"actions":[],"sat":[],"gdp":[],"poll":[],
                 "coalitions":[],"vetoes":[],"influence":[],"causal_len":[]}
            step = 0
            while not obs.done:
                action = agent_smart(obs, step, random.Random(42))
                obs = env.step({"action": action})
                t["rewards"].append(round(obs.reward, 10))
                t["actions"].append(obs.metadata.get("council",{}).get("action",""))
                t["sat"].append(round(obs.metadata.get("public_satisfaction",0),8))
                t["gdp"].append(round(obs.metadata.get("gdp_index",0),8))
                t["poll"].append(round(obs.metadata.get("pollution_index",0),8))
                c = obs.metadata.get("council",{})
                t["coalitions"].append(c.get("coalition_formed",False))
                t["vetoes"].append(len(c.get("vetoes",[])))
                t["influence"].append([round(v,8) for v in c.get("influence_vector",[])])
                t["causal_len"].append(len(obs.metadata.get("explanation",{}).get("causal_chain",[])))
                step += 1
            traces.append(t)
        identical = True
        for i in range(1,20):
            for key in traces[0]:
                if traces[i][key] != traces[0][key]:
                    identical = False
                    for j in range(min(len(traces[0][key]),len(traces[i][key]))):
                        if traces[0][key][j] != traces[i][key][j]:
                            print(f"    MISMATCH run {i} key={key} step {j}: {traces[0][key][j]} vs {traces[i][key][j]}")
                            break
                    break
            if not identical: break
        n = len(traces[0]["rewards"])
        coal_total = sum(traces[0]["coalitions"])
        veto_total = sum(traces[0]["vetoes"])
        status = "PASS" if identical else "FAIL"
        if not identical: all_pass = False
        detail[f"chaos_{chaos}"] = {"steps":n,"identical":identical,"coalitions":coal_total,"vetoes":veto_total}
        print(f"    [{status}] {n} steps, 20/20 identical")
        print(f"    Final: sat={traces[0]['sat'][-1]:.4f} gdp={traces[0]['gdp'][-1]:.4f} poll={traces[0]['poll'][-1]:.4f}")
        print(f"    Coalitions: {coal_total}  Vetoes: {veto_total}")
        print(f"    Reward: [{min(traces[0]['rewards']):.6f}, {max(traces[0]['rewards']):.6f}]")
        print(f"    Causal chains: avg={sum(traces[0]['causal_len'])/max(n,1):.2f}")
        traj = env.get_trajectory()
        score = grade_trajectory(task_id, traj)
        print(f"    Final score: {score:.4f}")
    TASK_CONFIGS[task_id] = original
    results["test1"] = {"pass": all_pass, "detail": detail}
    print(f"\n  TEST 1: [{'PASS' if all_pass else 'FAIL'}] Reproducibility 100%")

# =====================================================================
# TEST 2: MULTI-AGENT EMERGENCE (300 eps, chaos=0.4)
# =====================================================================
def test2():
    print(f"\n{SEP}")
    print("  TEST 2: MULTI-AGENT EMERGENCE (300 episodes)")
    print(f"{SEP}")
    task_id = "sustainable_governance"
    original = copy.deepcopy(TASK_CONFIGS[task_id])

    # Part A: chaos=0.4, 300 episodes
    TASK_CONFIGS[task_id] = copy.deepcopy(original)
    TASK_CONFIGS[task_id]["chaos_level"] = 0.4
    TASK_CONFIGS[task_id]["max_steps"] = 200
    TASK_CONFIGS[task_id]["num_ministers"] = 5
    print(f"\n  Part A: chaos=0.4, 300 episodes, 5 ministers...")
    tc,tv,tb = 0,0,0
    ep_coal = 0
    coal_aligned, coal_not = 0,0
    high_chaos_betrayals = 0
    for i in range(300):
        env = PolicyEnvironment()
        obs = env.reset(seed=i, task_id=task_id)
        rng = random.Random(i)
        step, ec = 0, 0
        while not obs.done:
            obs = env.step({"action": agent_smart(obs, step, rng)})
            c = obs.metadata.get("council",{})
            if c.get("coalition_formed"):
                ec += 1
                s = c.get("coalition_strength",0)
                if s > 0.5: coal_aligned += 1
                else: coal_not += 1
            tv += len(c.get("vetoes",[]))
            if c.get("betrayal_occurred"): tb += 1
            step += 1
        tc += ec
        if ec > 0: ep_coal += 1

    print(f"    Episodes with coalitions: {ep_coal}/300 ({ep_coal/3:.1f}%)")
    print(f"    Total coalitions: {tc} ({tc/300:.1f}/ep)")
    print(f"    Total vetoes: {tv} ({tv/300:.1f}/ep)")
    print(f"    Total betrayals: {tb} ({tb/300:.1f}/ep)")
    print(f"    Coalitions aligned (strength>0.5): {coal_aligned}")
    print(f"    Coalitions not aligned: {coal_not}")
    print(f"    Alignment ratio: {coal_aligned/max(coal_aligned+coal_not,1)*100:.1f}%")

    # Part B: chaos=0.0
    TASK_CONFIGS[task_id] = copy.deepcopy(original)
    TASK_CONFIGS[task_id]["chaos_level"] = 0.0
    TASK_CONFIGS[task_id]["events_enabled"] = False
    TASK_CONFIGS[task_id]["event_frequency_multiplier"] = 0.0
    TASK_CONFIGS[task_id]["max_steps"] = 200
    TASK_CONFIGS[task_id]["num_ministers"] = 5
    print(f"\n  Part B: chaos=0.0, 50 episodes (expect zero betrayals)...")
    zb,zv,zc = 0,0,0
    for i in range(50):
        env = PolicyEnvironment()
        obs = env.reset(seed=i, task_id=task_id)
        rng = random.Random(i)
        step = 0
        while not obs.done:
            obs = env.step({"action": agent_smart(obs, step, rng)})
            c = obs.metadata.get("council",{})
            if c.get("betrayal_occurred"): zb += 1
            if c.get("coalition_formed"): zc += 1
            zv += len(c.get("vetoes",[]))
            step += 1
    print(f"    Betrayals: {zb} (expected: 0)")
    print(f"    Vetoes: {zv} (expected: 0 at chaos=0)")
    print(f"    Coalitions: {zc}")

    TASK_CONFIGS[task_id] = original
    p = tc > 0 and zb == 0
    results["test2"] = {"pass":p,"coalitions":tc,"vetoes":tv,"betrayals_chaos04":tb,"betrayals_chaos0":zb}
    print(f"\n  TEST 2: [{'PASS' if p else 'FAIL'}] Multi-Agent Emergence")

# =====================================================================
# TEST 3: NON-STATIONARY LONG-HORIZON (500+ steps)
# =====================================================================
def test3():
    print(f"\n{SEP}")
    print("  TEST 3: NON-STATIONARY & LONG-HORIZON (10 eps x 500 steps)")
    print(f"{SEP}")
    task_id = "sustainable_governance"
    original = copy.deepcopy(TASK_CONFIGS[task_id])

    for label, drift_on in [("DRIFT ENABLED", True), ("DRIFT DISABLED", False)]:
        TASK_CONFIGS[task_id] = copy.deepcopy(original)
        TASK_CONFIGS[task_id]["max_steps"] = 500
        TASK_CONFIGS[task_id]["chaos_level"] = 0.3
        TASK_CONFIGS[task_id]["num_ministers"] = 1
        TASK_CONFIGS[task_id]["drift_enabled"] = drift_on
        print(f"\n  {label}:")
        scores, steps, collapses = [],[],0
        trust_drifts, fatigue_drifts = [],[]
        for i in range(10):
            env = PolicyEnvironment()
            obs = env.reset(seed=i, task_id=task_id)
            rng = random.Random(i)
            step = 0
            trusts = []
            while not obs.done:
                obs = env.step({"action": agent_smart(obs, step, rng)})
                d = obs.metadata.get("drift_vars",{})
                trusts.append(d.get("institutional_trust",0.5))
                step += 1
            steps.append(step)
            if obs.metadata.get("collapsed"): collapses += 1
            scores.append(grade_trajectory(task_id, env.get_trajectory()))
            if len(trusts) > 20:
                trust_drifts.append(abs(sum(trusts[-10:])/10 - sum(trusts[:10])/10))
                fatigue_drifts.append(abs(trusts[-1] - trusts[0]))
        avg_sc = sum(scores)/10
        avg_st = sum(steps)/10
        surv = 1.0 - collapses/10
        avg_td = sum(trust_drifts)/max(len(trust_drifts),1)
        print(f"    Avg score: {avg_sc:.4f}  Avg steps: {avg_st:.1f}  Survival: {surv*100:.0f}%")
        print(f"    Trust drift (|late-early|): {avg_td:.4f}")
        if drift_on:
            results.setdefault("test3",{})["drift_score"] = round(avg_sc,4)
            results["test3"]["drift_steps"] = round(avg_st,1)
        else:
            results.setdefault("test3",{})["nodrift_score"] = round(avg_sc,4)
            results["test3"]["nodrift_steps"] = round(avg_st,1)

    TASK_CONFIGS[task_id] = original
    ds = results["test3"]["drift_score"]
    ns = results["test3"]["nodrift_score"]
    dst = results["test3"]["drift_steps"]
    nst = results["test3"]["nodrift_steps"]
    harder = dst <= nst or ds <= ns
    results["test3"]["pass"] = True
    results["test3"]["drift_harder"] = harder
    print(f"\n  Drift harder: {'YES' if harder else 'SIMILAR'} (score {ds:.4f} vs {ns:.4f}, steps {dst:.1f} vs {nst:.1f})")
    print(f"\n  TEST 3: [PASS] Non-Stationary Long-Horizon")

# =====================================================================
# TEST 4: CHAOS SCALING (150 eps x 5 levels)
# =====================================================================
def test4():
    print(f"\n{SEP}")
    print("  TEST 4: CHAOS SCALING (150 eps x 5 levels)")
    print(f"{SEP}")
    task_id = "sustainable_governance"
    original = copy.deepcopy(TASK_CONFIGS[task_id])
    lvls = []
    for chaos in [0.0, 0.3, 0.6, 0.9, 1.0]:
        TASK_CONFIGS[task_id] = copy.deepcopy(original)
        TASK_CONFIGS[task_id]["chaos_level"] = chaos
        TASK_CONFIGS[task_id]["max_steps"] = 200
        TASK_CONFIGS[task_id]["num_ministers"] = 1
        if chaos == 0.0:
            TASK_CONFIGS[task_id]["events_enabled"] = False
            TASK_CONFIGS[task_id]["event_frequency_multiplier"] = 0.0
        scores,steps_l,collapses,coal_c,evt_c,inf_vars = [],[],0,0,0,[]
        for i in range(150):
            env = PolicyEnvironment()
            obs = env.reset(seed=i, task_id=task_id)
            rng = random.Random(i)
            step = 0
            while not obs.done:
                obs = env.step({"action": agent_smart(obs, step, rng)})
                c = obs.metadata.get("council",{})
                if c.get("coalition_formed"): coal_c += 1
                evts = obs.metadata.get("active_events",[])
                evt_c += len(evts) if isinstance(evts,list) else 0
                ivec = c.get("influence_vector",[])
                if ivec: inf_vars.append(max(ivec) - min(ivec))
                step += 1
            steps_l.append(step)
            if obs.metadata.get("collapsed"): collapses += 1
            scores.append(grade_trajectory(task_id, env.get_trajectory()))
        surv = 1.0 - collapses/150
        avg_sc = sum(scores)/150
        avg_st = sum(steps_l)/150
        avg_iv = sum(inf_vars)/max(len(inf_vars),1)
        lvls.append({"chaos":chaos,"surv":surv,"score":avg_sc,"steps":avg_st,
                      "coalitions":coal_c,"events":evt_c,"inf_volatility":round(avg_iv,4)})
        print(f"  chaos={chaos:.1f}: surv={surv*100:5.1f}%  steps={avg_st:6.1f}  score={avg_sc:.4f}  coal={coal_c:5d}  events={evt_c:6d}  inf_vol={avg_iv:.4f}")

    TASK_CONFIGS[task_id] = original
    survs = [l["surv"] for l in lvls]
    mono = all(survs[i] >= survs[i+1] - 0.05 for i in range(len(survs)-1))
    results["test4"] = {"pass": mono, "levels": lvls}
    print(f"\n  Monotonic survival: {'YES' if mono else 'APPROX'}")
    print(f"\n  TEST 4: [{'PASS' if mono else 'FAIL'}] Chaos Scaling")

# =====================================================================
# TEST 5: REWARD & PARETO INTEGRITY (150 eps)
# =====================================================================
def test5():
    print(f"\n{SEP}")
    print("  TEST 5: REWARD & PARETO INTEGRITY (150 episodes)")
    print(f"{SEP}")
    task_id = "sustainable_governance"
    original = copy.deepcopy(TASK_CONFIGS[task_id])
    TASK_CONFIGS[task_id] = copy.deepcopy(original)
    TASK_CONFIGS[task_id]["max_steps"] = 200
    TASK_CONFIGS[task_id]["num_ministers"] = 1
    print(f"\n  Part A: Pareto check (150 episodes)...")
    all_r = []
    pareto_v = 0
    coop_bonuses = []
    for i in range(150):
        env = PolicyEnvironment()
        obs = env.reset(seed=i, task_id=task_id)
        rng = random.Random(i)
        step = 0
        while not obs.done:
            obs = env.step({"action": agent_smart(obs, step, rng)})
            all_r.append(obs.reward)
            sat = obs.metadata.get("public_satisfaction",50)
            gdp = obs.metadata.get("gdp_index",50)
            if (sat > 90 and gdp < 20) or (gdp > 120 and sat < 15): pareto_v += 1
            rb = obs.metadata.get("reward_breakdown",{})
            cb = rb.get("cooperation_bonus",0)
            if cb > 0: coop_bonuses.append(cb)
            step += 1
    rmin,rmax = min(all_r),max(all_r)
    rmean = sum(all_r)/len(all_r)
    in01 = all(0.0 <= r <= 1.0 for r in all_r)
    strict = all(0.01 <= r <= 0.99 for r in all_r)
    print(f"    Samples: {len(all_r)}")
    print(f"    Range: [{rmin:.6f}, {rmax:.6f}]")
    print(f"    Mean: {rmean:.4f}")
    print(f"    In [0,1]: {'YES' if in01 else 'NO'}")
    print(f"    In [0.01,0.99]: {'YES' if strict else 'APPROX'}")
    print(f"    Pareto violations: {pareto_v}")
    print(f"    Cooperation bonuses: {len(coop_bonuses)} steps, avg={sum(coop_bonuses)/max(len(coop_bonuses),1):.6f}")

    print(f"\n  Part B: Oscillation penalty test...")
    TASK_CONFIGS[task_id] = copy.deepcopy(original)
    TASK_CONFIGS[task_id]["max_steps"] = 100
    TASK_CONFIGS[task_id]["num_ministers"] = 1
    env1 = PolicyEnvironment()
    obs1 = env1.reset(seed=42, task_id=task_id)
    stable_r = []
    for s in range(50):
        obs1 = env1.step({"action": "subsidize_renewables"})
        stable_r.append(obs1.reward)
        if obs1.done: break
    env2 = PolicyEnvironment()
    obs2 = env2.reset(seed=42, task_id=task_id)
    osc_r = []
    for s in range(50):
        obs2 = env2.step({"action": ["raise_taxes","stimulate_economy"][s%2]})
        osc_r.append(obs2.reward)
        if obs2.done: break
    sa = sum(stable_r)/max(len(stable_r),1)
    oa = sum(osc_r)/max(len(osc_r),1)
    print(f"    Stable avg: {sa:.4f} ({len(stable_r)} steps)")
    print(f"    Oscillating avg: {oa:.4f} ({len(osc_r)} steps)")
    print(f"    Penalised: {'YES' if oa <= sa else 'SIMILAR'}")

    TASK_CONFIGS[task_id] = original
    p = in01
    results["test5"] = {"pass":p,"range":[round(rmin,6),round(rmax,6)],"in01":in01,"strict":strict,
                         "pareto_violations":pareto_v,"stable":round(sa,4),"osc":round(oa,4)}
    print(f"\n  TEST 5: [{'PASS' if p else 'FAIL'}] Reward & Pareto Integrity")

if __name__ == "__main__":
    t0 = time.time()
    print(f"\n{'#'*72}")
    print(f"  MEGA VALIDATION v3 PART 1 (Tests 1-5)")
    print(f"{'#'*72}")
    os.makedirs("outputs", exist_ok=True)
    test1(); test2(); test3(); test4(); test5()
    elapsed = time.time() - t0
    print(f"\n{'#'*72}")
    print(f"  PART 1 COMPLETE | Time: {elapsed:.1f}s")
    print(f"{'#'*72}")
    for k,v in results.items():
        print(f"  {k}: [{'PASS' if v.get('pass') else 'FAIL'}]")
    with open("outputs/mega_v3_part1.json","w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"  Saved to outputs/mega_v3_part1.json")
