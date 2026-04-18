#!/usr/bin/env python3
"""
OpenENV Policy Engine -- HuggingFace TRL Training Script (GPU-Optimized)
=========================================================================
Trains an LLM using GRPO from TRL to govern a multi-agent policy simulation.
Includes auto-curriculum self-improvement: environment escalates difficulty
as the agent improves.

Covers Themes: #1 Multi-Agent, #2 Long-Horizon, #4 Self-Improvement

Requirements: pip install torch transformers trl accelerate datasets matplotlib
Usage:
    python train_trl.py --steps 200 --model gpt2
    python train_trl.py --plot
"""
import sys, os, json, math, argparse, random, copy, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import GRPOTrainer, GRPOConfig

from server.policy_environment import PolicyEnvironment
from server.config import VALID_ACTIONS, CORE_ACTIONS, TASK_CONFIGS
from server.tasks import grade_trajectory

# ── GPU Optimizations ──
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

ACTION_LIST = sorted(CORE_ACTIONS)
TASK_ID = "sustainable_governance"

def format_state_prompt(obs_meta: dict, step: int) -> str:
    actions_str = ", ".join(ACTION_LIST)
    events = ", ".join(obs_meta.get("active_events", [])) or "none"
    return (
        f"You are an AI governance agent managing a country's policy.\n\n"
        f"STATE at step {step}:\n"
        f"GDP: {obs_meta.get('gdp_index', 100):.0f}/200 | "
        f"Pollution: {obs_meta.get('pollution_index', 100):.0f}/500 | "
        f"Satisfaction: {obs_meta.get('public_satisfaction', 50):.0f}/100 | "
        f"Healthcare: {obs_meta.get('healthcare_index', 50):.0f}/100 | "
        f"Unemployment: {obs_meta.get('unemployment_rate', 10):.1f}% | "
        f"Renewables: {obs_meta.get('renewable_energy_ratio', 0.2):.0%} | "
        f"Events: {events}\n\n"
        f"ACTIONS: {actions_str}\n\nBest action:"
    )

def parse_action(text: str) -> str:
    text_lower = text.lower().strip()
    for action in ACTION_LIST:
        if action in text_lower:
            return action
    for action in ACTION_LIST:
        words = action.split("_")
        if any(w in text_lower for w in words if len(w) > 3):
            return action
    return "no_action"

# ── Dataset Generation ──
def generate_rollout_dataset(num_episodes=20, max_steps=60, seed=42):
    samples = []
    for ep in range(num_episodes):
        env = PolicyEnvironment()
        obs = env.reset(seed=seed + ep, task_id=TASK_ID)
        for step in range(max_steps):
            if obs.done: break
            prompt = format_state_prompt(obs.metadata, step)
            sat = obs.metadata.get("public_satisfaction", 50)
            poll = obs.metadata.get("pollution_index", 100)
            gdp = obs.metadata.get("gdp_index", 100)
            if sat < 35: action = "increase_welfare"
            elif poll > 200: action = "enforce_emission_limits"
            elif gdp < 50: action = "stimulate_economy"
            else: action = ACTION_LIST[step % len(ACTION_LIST)]
            obs = env.step({"action": action})
            samples.append({"prompt": prompt, "action": action,
                          "reward": obs.reward, "episode": ep, "step": step})
    return samples

# ── Reward Function ──
def openenv_reward_func(completions: list[str], **kwargs) -> list[float]:
    prompts = kwargs.get("prompt", [""] * len(completions))
    rewards = []
    for i, completion in enumerate(completions):
        action = parse_action(completion)
        try:
            env = PolicyEnvironment()
            obs = env.reset(seed=hash(prompts[i]) % 10000, task_id=TASK_ID)
            for _ in range(3):
                if obs.done: break
                obs = env.step({"action": "no_action"})
            if not obs.done:
                obs = env.step({"action": action})
                reward = float(obs.reward)
            else: reward = 0.0
        except Exception: reward = 0.0
        if action != "no_action": reward += 0.1
        rewards.append(reward)
    return rewards

# ── Evaluation ──
def evaluate_model(model_path, num_episodes=5, max_steps=60, seed=42,
                   task_overrides=None, label=""):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path)
    if torch.cuda.is_available():
        model = model.cuda().half()
    model.eval()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    episode_rewards, survivals = [], 0
    for ep in range(num_episodes):
        if task_overrides:
            orig = copy.deepcopy(TASK_CONFIGS[TASK_ID])
            TASK_CONFIGS[TASK_ID].update(task_overrides)
        env = PolicyEnvironment()
        obs = env.reset(seed=seed + ep, task_id=TASK_ID)
        ep_reward = 0.0
        for step in range(max_steps):
            if obs.done: break
            prompt = format_state_prompt(obs.metadata, step)
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}
            with torch.no_grad():
                outputs = model.generate(**inputs, max_new_tokens=16, do_sample=True,
                                        temperature=0.7, pad_token_id=tokenizer.eos_token_id)
            response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            action = parse_action(response)
            obs = env.step({"action": action})
            ep_reward += obs.reward
        if task_overrides:
            TASK_CONFIGS[TASK_ID] = orig
        collapsed = obs.metadata.get("collapsed", False)
        if not collapsed: survivals += 1
        episode_rewards.append(round(ep_reward, 4))
        status = "SURVIVED" if not collapsed else "COLLAPSED"
        print(f"    {label}Ep {ep+1}: reward={ep_reward:.2f} steps={step} [{status}]")
    return episode_rewards, survivals

# ── Self-Improvement Curriculum ──
def run_curriculum_evaluation(model_path, seed=5000):
    """Run the trained model through escalating difficulty levels."""
    print("\n  [SELF-IMPROVEMENT] Running curriculum escalation...")
    levels = [
        {"label": "Easy   ", "chaos_level": 0.0, "event_frequency_multiplier": 0.3, "max_steps": 60},
        {"label": "Medium ", "chaos_level": 0.3, "event_frequency_multiplier": 0.6, "max_steps": 80},
        {"label": "Hard   ", "chaos_level": 0.6, "event_frequency_multiplier": 1.0, "max_steps": 100},
        {"label": "Extreme", "chaos_level": 1.0, "event_frequency_multiplier": 1.5, "max_steps": 120},
    ]
    curriculum_results = []
    for i, lvl in enumerate(levels):
        label = lvl.pop("label")
        max_s = lvl.get("max_steps", 60)
        print(f"\n  Level {i+1}/4: {label} (chaos={lvl.get('chaos_level',0):.1f})")
        rewards, survs = evaluate_model(model_path, num_episodes=3, max_steps=max_s,
                                         seed=seed + i*100, task_overrides=lvl, label=f"[{label}] ")
        avg_r = sum(rewards)/len(rewards) if rewards else 0
        curriculum_results.append({
            "level": i+1, "label": label.strip(), "chaos": lvl.get("chaos_level", 0),
            "avg_reward": round(avg_r, 4), "survivals": survs, "total": len(rewards),
            "survival_rate": round(survs/len(rewards), 4) if rewards else 0,
        })
    return curriculum_results

# ── Main Training ──
def train(num_episodes=20, training_steps=50, model_name="gpt2",
          lr=5e-6, save_dir="outputs/trl_training", seed=42):
    os.makedirs(save_dir, exist_ok=True)
    has_gpu = torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name(0) if has_gpu else "CPU"

    print("=" * 60)
    print("  OpenENV -- TRL GRPO Training (GPU-Optimized)")
    print(f"  Model: {model_name}")
    print(f"  Device: {gpu_name}")
    print(f"  Training steps: {training_steps}")
    print(f"  bf16: {has_gpu} | cudnn.benchmark: {has_gpu}")
    print("=" * 60)

    # Step 1: Dataset
    print("\n[1/6] Generating environment rollout dataset...")
    samples = generate_rollout_dataset(num_episodes=num_episodes, seed=seed)
    dataset = Dataset.from_dict({"prompt": [s["prompt"] for s in samples]})
    print(f"  {len(samples)} samples from {num_episodes} episodes")

    # Step 2: Baseline
    print("\n[2/6] Baseline evaluation (before training)...")
    baseline_rewards, baseline_survs = evaluate_model(model_name, num_episodes=5, seed=3000)
    baseline_avg = sum(baseline_rewards) / len(baseline_rewards)
    print(f"  Baseline: avg={baseline_avg:.2f}, survived={baseline_survs}/5")

    # Step 3: Configure GRPO
    print(f"\n[3/6] Configuring GRPO trainer...")
    training_args = GRPOConfig(
        output_dir=os.path.join(save_dir, "checkpoints"),
        num_train_epochs=1, max_steps=training_steps,
        per_device_train_batch_size=4, num_generations=4,
        max_completion_length=32, learning_rate=lr,
        logging_steps=max(1, training_steps // 10),
        save_steps=training_steps, seed=seed,
        report_to="none", log_level="warning",
        bf16=has_gpu, fp16=False, use_cpu=not has_gpu,
    )
    trainer = GRPOTrainer(model=model_name, args=training_args,
                          train_dataset=dataset, reward_funcs=openenv_reward_func)

    # Step 4: Train
    print(f"\n[4/6] Training for {training_steps} steps on {gpu_name}...")
    t0 = time.time()
    train_result = trainer.train()
    train_time = time.time() - t0
    metrics = train_result.metrics if hasattr(train_result, 'metrics') else {}
    print(f"  Done in {train_time:.1f}s ({training_steps/max(train_time,1):.1f} steps/sec)")

    model_path = os.path.join(save_dir, "trained_model")
    trainer.save_model(model_path)

    # Step 5: Post-training evaluation
    print("\n[5/6] Post-training evaluation...")
    post_rewards, post_survs = evaluate_model(model_path, num_episodes=5, seed=4000)
    post_avg = sum(post_rewards) / len(post_rewards)
    improvement = post_avg - baseline_avg

    # Step 6: Self-improvement curriculum
    print("\n[6/6] Self-improvement curriculum evaluation...")
    curriculum = run_curriculum_evaluation(model_path, seed=6000)

    # Save results
    results = {
        "baseline_rewards": baseline_rewards, "post_training_rewards": post_rewards,
        "baseline_avg_reward": round(baseline_avg, 4),
        "post_training_avg_reward": round(post_avg, 4),
        "improvement": round(improvement, 4),
        "improvement_pct": round(improvement / max(abs(baseline_avg), 1) * 100, 1),
        "baseline_survival": f"{baseline_survs}/5",
        "post_survival": f"{post_survs}/5",
        "training_steps": training_steps, "training_time_sec": round(train_time, 1),
        "model": model_name, "device": gpu_name, "task": TASK_ID,
        "curriculum_results": curriculum,
        "training_metrics": {k: str(v) for k, v in metrics.items()},
    }
    results_path = os.path.join(save_dir, "training_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'=' * 60}")
    print(f"  RESULTS")
    print(f"  Baseline:      {baseline_avg:8.2f} reward  ({baseline_survs}/5 survived)")
    print(f"  Post-training: {post_avg:8.2f} reward  ({post_survs}/5 survived)")
    print(f"  Improvement:   {improvement:+8.2f} ({results['improvement_pct']:+.1f}%)")
    print(f"  Training time: {train_time:.1f}s on {gpu_name}")
    print(f"\n  CURRICULUM ESCALATION:")
    for c in curriculum:
        print(f"    {c['label']:>8s}: reward={c['avg_reward']:6.2f} "
              f"survival={c['survivals']}/{c['total']} (chaos={c['chaos']:.1f})")
    print(f"\n  Results: {results_path}")
    print(f"{'=' * 60}")
    return results

# ── Plotting ──
def plot_reward_curves(results_path="outputs/trl_training/training_results.json"):
    with open(results_path) as f:
        results = json.load(f)
    baseline = results.get("baseline_avg_reward", 0)
    post = results.get("post_training_avg_reward", 0)
    baseline_rewards = results.get("baseline_rewards", [])
    post_rewards = results.get("post_training_rewards", [])
    curriculum = results.get("curriculum_results", [])

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("OpenENV -- TRL GRPO Training Results", fontsize=15, fontweight="bold")

    # 1. Before vs After
    bars = axes[0].bar(["Before", "After"], [baseline, post],
                       color=["#a1a1aa", "#0d9488"], width=0.5)
    axes[0].set_ylabel("Avg Episode Reward")
    axes[0].set_title("Before vs After Training")
    for bar, val in zip(bars, [baseline, post]):
        axes[0].text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3,
                     f"{val:.1f}", ha="center", fontweight="bold", fontsize=13)
    axes[0].grid(axis="y", alpha=0.3)

    # 2. Per-episode
    x = list(range(1, max(len(baseline_rewards), len(post_rewards)) + 1))
    axes[1].bar([i-0.2 for i in x[:len(baseline_rewards)]], baseline_rewards, 0.35,
                color="#a1a1aa", label="Before", alpha=0.8)
    axes[1].bar([i+0.2 for i in x[:len(post_rewards)]], post_rewards, 0.35,
                color="#0d9488", label="After", alpha=0.8)
    axes[1].set_xlabel("Episode"); axes[1].set_ylabel("Reward")
    axes[1].set_title("Per-Episode Comparison"); axes[1].legend()
    axes[1].grid(axis="y", alpha=0.3)

    # 3. Curriculum escalation
    if curriculum:
        labels = [c["label"] for c in curriculum]
        rewards = [c["avg_reward"] for c in curriculum]
        colors = ["#10b981", "#f59e0b", "#ef4444", "#7c3aed"]
        axes[2].bar(labels, rewards, color=colors[:len(labels)], width=0.5)
        axes[2].set_ylabel("Avg Reward")
        axes[2].set_title("Self-Improvement Curriculum")
        for i, (lbl, r) in enumerate(zip(labels, rewards)):
            axes[2].text(i, r+0.2, f"{r:.1f}", ha="center", fontweight="bold")
        axes[2].grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(os.path.dirname(results_path), "reward_curves.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"  Chart saved to: {plot_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OpenENV TRL GRPO Training")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--model", type=str, default="gpt2")
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.plot:
        plot_reward_curves()
    else:
        train(num_episodes=args.episodes, training_steps=args.steps,
              model_name=args.model, lr=args.lr, seed=args.seed)
