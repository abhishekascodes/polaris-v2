#!/usr/bin/env python3
"""
POLARIS v3 — LLM Benchmark Suite
==================================
Runs a real LLM (Llama via Groq, GPT-4o, etc.) against all 6 tasks
and produces publication-quality results with plots.

This generates the "before training" baselines that prove:
1. The environment genuinely challenges frontier LLMs
2. Negotiation tasks are harder than simple governance
3. Theory-of-mind predictions are non-trivial

Usage:
    # Set your API keys
    set API_BASE_URL=https://api.groq.com/openai/v1
    set MODEL_NAME=llama-3.3-70b-versatile
    set HF_TOKEN=gsk_...

    # Run full benchmark
    python benchmark.py

    # Run specific task
    python benchmark.py --task negotiation_arena --episodes 5

    # Just plot existing results
    python benchmark.py --plot
"""

import sys, os, io, json, time, argparse
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from openai import OpenAI

from server.policy_environment import PolicyEnvironment
from server.config import VALID_ACTIONS, ACTION_DESCRIPTIONS, TASK_CONFIGS, CORE_ACTIONS
from server.tasks import grade_trajectory, get_task_ids

# ── Config ──
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.groq.com/openai/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "llama-3.3-70b-versatile")
HF_TOKEN = os.getenv("HF_TOKEN", "")

MINISTERS = ["Chancellor Voss", "Director Okafor", "Dr. Vasquez",
             "General Tanaka", "Senator Mwangi"]

ACTION_LIST_STR = "\n".join(
    f"  - {name}: {desc}" for name, desc in ACTION_DESCRIPTIONS.items()
)

SEP = "=" * 64

# ── System Prompts ──

SYSTEM_NEGOTIATION = """You are the President of a simulated nation. Each turn, your council of ministers presents proposals. You must:
1. Read each minister's proposal, argument, and coalition offer
2. Decide which policy action to take
3. Choose which ministers to form a coalition with
4. Predict which ministers might veto your decision

AVAILABLE ACTIONS:
{actions}

RESPONSE FORMAT — respond with valid JSON only, no markdown:
{{
  "action": "<action_name>",
  "reasoning": "<1-2 sentences>",
  "coalition_target": ["<minister_name>"],
  "veto_prediction": ["<minister_name_who_might_veto>"],
  "stance": "cooperative"
}}

RULES:
- action MUST be one of the valid actions listed above
- Balance GDP, pollution, and satisfaction
- Prevent collapse: GDP > 15, pollution < 290, satisfaction > 5
- React to briefings and events
Respond with ONLY valid JSON."""

SYSTEM_SIMPLE = """You are an expert AI policy advisor governing a simulated nation.
Each turn you must choose EXACTLY ONE policy action.

AVAILABLE ACTIONS:
{actions}

Respond with ONLY the action name. Nothing else."""


# ── Observation Formatting ──

def format_obs_negotiation(meta, step, max_steps):
    lines = [
        f"--- STEP {step}/{max_steps} ---",
        f"GDP: {meta.get('gdp_index',0):.0f}/200 | Pollution: {meta.get('pollution_index',0):.0f}/300 | Satisfaction: {meta.get('public_satisfaction',0):.0f}/100",
        f"Healthcare: {meta.get('healthcare_index',0):.0f} | Education: {meta.get('education_index',0):.0f} | Unemployment: {meta.get('unemployment_rate',0):.1f}%",
    ]
    events = meta.get("active_events", [])
    if events:
        lines.append(f"Events: {', '.join(events)}")
    neg = meta.get("negotiation_narrative", "")
    if neg:
        lines.append(f"\n{neg[:500]}")
    briefings = meta.get("active_briefings", [])
    if briefings:
        lines.append("\nBRIEFINGS:")
        for b in briefings[:2]:
            lines.append(f"  [{b['category']}] ...deadline step {b['deadline_step']} ({b['steps_remaining']} left)")
    new_b = meta.get("new_briefing", "")
    if new_b:
        lines.append(f"\nNEW INTEL: {new_b[:200]}")
    return "\n".join(lines)


def format_obs_simple(meta, step, max_steps):
    events = ", ".join(meta.get("active_events", [])) or "none"
    return (
        f"Step {step}/{max_steps} | GDP: {meta.get('gdp_index',0):.0f} | "
        f"Pollution: {meta.get('pollution_index',0):.0f} | "
        f"Satisfaction: {meta.get('public_satisfaction',0):.0f} | Events: {events}"
    )


# ── LLM Calls ──

def call_llm_negotiation(client, obs_text, model):
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_NEGOTIATION.format(actions=ACTION_LIST_STR)},
                {"role": "user", "content": obs_text},
            ],
            temperature=0.1, max_tokens=200,
        )
        raw = resp.choices[0].message.content.strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1].rsplit("```", 1)[0].strip()
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            for a in VALID_ACTIONS:
                if a in raw.lower():
                    return {"action": a, "reasoning": "parse fallback", "coalition_target": [], "veto_prediction": [], "stance": "cooperative"}
            return {"action": "no_action", "reasoning": "parse error", "coalition_target": [], "veto_prediction": [], "stance": "cooperative"}
        action = data.get("action", "no_action")
        if action not in VALID_ACTIONS:
            for a in VALID_ACTIONS:
                if a in action:
                    action = a; break
            else:
                action = "no_action"
        data["action"] = action
        data.setdefault("reasoning", "")
        data.setdefault("coalition_target", [])
        data.setdefault("veto_prediction", [])
        data.setdefault("stance", "cooperative")
        return data
    except Exception as e:
        return {"action": "no_action", "reasoning": f"error: {e}", "coalition_target": [], "veto_prediction": [], "stance": "cooperative"}


def call_llm_simple(client, obs_text, model):
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_SIMPLE.format(actions=ACTION_LIST_STR)},
                {"role": "user", "content": obs_text},
            ],
            temperature=0.0, max_tokens=30,
        )
        raw = resp.choices[0].message.content.strip().lower().strip("'\"` \n")
        if raw in VALID_ACTIONS:
            return raw
        for a in VALID_ACTIONS:
            if a in raw:
                return a
        return "no_action"
    except Exception:
        return "no_action"


# ── Heuristic Baselines ──

def agent_random(meta, step, rng):
    return {"action": rng.choice(CORE_ACTIONS)}

def agent_smart(meta, step, rng):
    sat = meta.get("public_satisfaction", 50)
    poll = meta.get("pollution_index", 100)
    gdp = meta.get("gdp_index", 100)
    if sat < 30: action = "increase_welfare"
    elif poll > 200: action = "enforce_emission_limits"
    elif gdp < 50: action = "stimulate_economy"
    else: action = rng.choice(["subsidize_renewables", "invest_in_education", "increase_welfare", "stimulate_economy"])
    return {"action": action, "reasoning": "heuristic", "coalition_target": [MINISTERS[1]], "veto_prediction": [], "stance": "cooperative"}


# ── Single Episode Runner ──

def run_episode(client, task_id, seed, agent_type="llm"):
    import random
    rng = random.Random(seed)
    cfg = TASK_CONFIGS[task_id]
    max_steps = cfg["max_steps"]
    use_neg = cfg.get("num_ministers", 1) >= 2

    env = PolicyEnvironment()
    obs = env.reset(seed=seed, task_id=task_id)
    total_reward = 0.0
    step = 0
    tom_correct = 0
    tom_total = 0
    coalitions = 0
    actions_taken = []

    while not obs.done:
        step += 1
        meta = obs.metadata

        if agent_type == "llm":
            if use_neg:
                obs_text = format_obs_negotiation(meta, step, max_steps)
                action_data = call_llm_negotiation(client, obs_text, MODEL_NAME)
            else:
                obs_text = format_obs_simple(meta, step, max_steps)
                action_name = call_llm_simple(client, obs_text, MODEL_NAME)
                action_data = {"action": action_name}
        elif agent_type == "smart":
            action_data = agent_smart(meta, step, rng)
        else:
            action_data = agent_random(meta, step, rng)

        obs = env.step(action_data)
        total_reward += obs.reward
        actions_taken.append(action_data.get("action", "no_action"))

        outcome = obs.metadata.get("negotiation_outcome", {})
        if "veto_prediction_correct" in outcome:
            tom_total += 1
            if outcome["veto_prediction_correct"]:
                tom_correct += 1
        if outcome.get("coalition_formed"):
            coalitions += 1

    score = grade_trajectory(task_id, env.get_trajectory())
    collapsed = obs.metadata.get("collapsed", False)
    briefing_stats = obs.metadata.get("briefing_stats", {})

    return {
        "task_id": task_id, "seed": seed, "agent": agent_type,
        "score": round(score, 4), "reward": round(total_reward, 4),
        "steps": step, "collapsed": collapsed,
        "tom_accuracy": round(tom_correct / max(tom_total, 1), 4) if tom_total > 0 else None,
        "tom_total": tom_total, "coalitions": coalitions,
        "briefing_stats": briefing_stats,
        "unique_actions": len(set(actions_taken)),
        "total_actions": len(actions_taken),
    }


# ── Full Benchmark ──

def run_benchmark(tasks=None, episodes_per_task=3, seeds=None):
    client = OpenAI(api_key=HF_TOKEN, base_url=API_BASE_URL)
    if tasks is None:
        tasks = get_task_ids()
    if seeds is None:
        seeds = [42, 123, 777]

    all_results = {}

    print(f"\n{SEP}")
    print(f"  POLARIS v3 — LLM BENCHMARK")
    print(f"  Model: {MODEL_NAME}")
    print(f"  API: {API_BASE_URL}")
    print(f"  Tasks: {len(tasks)} | Episodes: {episodes_per_task} each")
    print(f"{SEP}\n")

    for task_id in tasks:
        cfg = TASK_CONFIGS.get(task_id)
        if not cfg:
            continue
        num_min = cfg.get("num_ministers", 1)
        neg = cfg.get("negotiation_enabled", num_min >= 2)

        print(f"\n{'─' * 48}")
        print(f"  TASK: {task_id}")
        print(f"  Steps: {cfg['max_steps']} | Ministers: {num_min} | Negotiation: {neg}")
        print(f"{'─' * 48}")

        task_results = {"llm": [], "smart": [], "random": []}

        for ep, seed in enumerate(seeds[:episodes_per_task]):
            # LLM agent
            print(f"\n  [LLM] Episode {ep+1}/{episodes_per_task} (seed={seed})...", end="", flush=True)
            t0 = time.time()
            r = run_episode(client, task_id, seed, "llm")
            elapsed = time.time() - t0
            status = "SURVIVED" if not r["collapsed"] else "COLLAPSED"
            tom_str = f" ToM={r['tom_accuracy']:.0%}" if r["tom_accuracy"] is not None else ""
            print(f" {status} score={r['score']:.4f} reward={r['reward']:.1f}{tom_str} ({elapsed:.1f}s)")
            task_results["llm"].append(r)

            # Smart baseline
            r_smart = run_episode(None, task_id, seed, "smart")
            task_results["smart"].append(r_smart)

            # Random baseline
            r_rand = run_episode(None, task_id, seed, "random")
            task_results["random"].append(r_rand)

        all_results[task_id] = task_results

        # Summary
        llm_avg = sum(r["score"] for r in task_results["llm"]) / len(task_results["llm"])
        smart_avg = sum(r["score"] for r in task_results["smart"]) / len(task_results["smart"])
        rand_avg = sum(r["score"] for r in task_results["random"]) / len(task_results["random"])
        print(f"\n  SUMMARY: LLM={llm_avg:.4f} vs Smart={smart_avg:.4f} vs Random={rand_avg:.4f}")

    # Save results
    os.makedirs("outputs", exist_ok=True)
    results_path = "outputs/benchmark_results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n{SEP}")
    print(f"  BENCHMARK COMPLETE — Results: {results_path}")
    print(f"{SEP}")

    plot_benchmark(results_path)
    return all_results


# ── Plotting ──

def plot_benchmark(results_path="outputs/benchmark_results.json"):
    with open(results_path) as f:
        results = json.load(f)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    tasks = list(results.keys())
    n = len(tasks)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f"POLARIS v3 — LLM Benchmark Results\nModel: {MODEL_NAME}",
                 fontsize=14, fontweight="bold", y=0.98)
    fig.patch.set_facecolor('#fafafa')

    # 1. Score comparison across tasks (top-left)
    ax = axes[0, 0]
    x = np.arange(n)
    w = 0.25
    llm_scores = [np.mean([r["score"] for r in results[t]["llm"]]) for t in tasks]
    smart_scores = [np.mean([r["score"] for r in results[t]["smart"]]) for t in tasks]
    rand_scores = [np.mean([r["score"] for r in results[t]["random"]]) for t in tasks]
    ax.bar(x - w, llm_scores, w, label=f"LLM ({MODEL_NAME.split('/')[-1][:20]})", color="#4f46e5", edgecolor="white")
    ax.bar(x, smart_scores, w, label="Smart Heuristic", color="#d97706", edgecolor="white")
    ax.bar(x + w, rand_scores, w, label="Random", color="#a1a1aa", edgecolor="white")
    ax.set_ylabel("Score (0-1)", fontsize=11)
    ax.set_title("Task Scores: LLM vs Baselines", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([t.replace("_", "\n") for t in tasks], fontsize=8)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    ax.set_facecolor('#fafafa')

    # 2. Reward comparison (top-right)
    ax = axes[0, 1]
    llm_rewards = [np.mean([r["reward"] for r in results[t]["llm"]]) for t in tasks]
    smart_rewards = [np.mean([r["reward"] for r in results[t]["smart"]]) for t in tasks]
    rand_rewards = [np.mean([r["reward"] for r in results[t]["random"]]) for t in tasks]
    ax.bar(x - w, llm_rewards, w, label="LLM", color="#4f46e5", edgecolor="white")
    ax.bar(x, smart_rewards, w, label="Smart", color="#d97706", edgecolor="white")
    ax.bar(x + w, rand_rewards, w, label="Random", color="#a1a1aa", edgecolor="white")
    ax.set_ylabel("Total Reward", fontsize=11)
    ax.set_title("Episode Rewards: LLM vs Baselines", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([t.replace("_", "\n") for t in tasks], fontsize=8)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    ax.set_facecolor('#fafafa')

    # 3. Theory-of-Mind accuracy (bottom-left)
    ax = axes[1, 0]
    neg_tasks = [t for t in tasks if any(r.get("tom_accuracy") is not None for r in results[t]["llm"])]
    if neg_tasks:
        tom_accs = [np.mean([r["tom_accuracy"] for r in results[t]["llm"] if r["tom_accuracy"] is not None]) for t in neg_tasks]
        coalition_rates = [np.mean([r["coalitions"] / max(r["steps"], 1) for r in results[t]["llm"]]) for t in neg_tasks]
        x2 = np.arange(len(neg_tasks))
        ax.bar(x2 - 0.2, [a * 100 for a in tom_accs], 0.35, label="Veto Prediction %", color="#e11d48", edgecolor="white")
        ax.bar(x2 + 0.2, [c * 100 for c in coalition_rates], 0.35, label="Coalition Rate %", color="#059669", edgecolor="white")
        ax.set_ylabel("Percentage", fontsize=11)
        ax.set_title("Theory-of-Mind Metrics (LLM Agent)", fontweight="bold")
        ax.set_xticks(x2)
        ax.set_xticklabels([t.replace("_", "\n") for t in neg_tasks], fontsize=8)
        ax.legend(fontsize=9)
        ax.grid(axis="y", alpha=0.3)
    else:
        ax.text(0.5, 0.5, "No negotiation tasks run", ha="center", va="center", transform=ax.transAxes)
    ax.set_facecolor('#fafafa')

    # 4. Survival rates (bottom-right)
    ax = axes[1, 1]
    llm_surv = [sum(1 for r in results[t]["llm"] if not r["collapsed"]) / len(results[t]["llm"]) * 100 for t in tasks]
    smart_surv = [sum(1 for r in results[t]["smart"] if not r["collapsed"]) / len(results[t]["smart"]) * 100 for t in tasks]
    rand_surv = [sum(1 for r in results[t]["random"] if not r["collapsed"]) / len(results[t]["random"]) * 100 for t in tasks]
    ax.bar(x - w, llm_surv, w, label="LLM", color="#4f46e5", edgecolor="white")
    ax.bar(x, smart_surv, w, label="Smart", color="#d97706", edgecolor="white")
    ax.bar(x + w, rand_surv, w, label="Random", color="#a1a1aa", edgecolor="white")
    ax.set_ylabel("Survival Rate %", fontsize=11)
    ax.set_title("Survival Rate Across Tasks", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([t.replace("_", "\n") for t in tasks], fontsize=8)
    ax.legend(fontsize=9)
    ax.set_ylim(0, 110)
    ax.grid(axis="y", alpha=0.3)
    ax.set_facecolor('#fafafa')

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plot_path = "outputs/benchmark_results.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight", facecolor="#fafafa")
    print(f"  Benchmark plots saved to: {plot_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="POLARIS v3 LLM Benchmark")
    parser.add_argument("--task", type=str, default=None, help="Specific task to benchmark")
    parser.add_argument("--episodes", type=int, default=3, help="Episodes per task")
    parser.add_argument("--plot", action="store_true", help="Just plot existing results")
    args = parser.parse_args()

    if args.plot:
        plot_benchmark()
    else:
        tasks = [args.task] if args.task else None
        run_benchmark(tasks=tasks, episodes_per_task=args.episodes)
