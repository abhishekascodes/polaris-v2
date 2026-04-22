#!/usr/bin/env python3
"""
POLARIS v3 — Production GRPO Training (RTX 5080 Optimized)
===========================================================
Trains Qwen2.5-3B-Instruct using GRPO with QLoRA on the POLARIS
multi-agent governance environment. Produces before/after evidence.

Hardware: RTX 5080 16GB + Ultra 9 275HX
Model: Qwen/Qwen2.5-3B-Instruct (4-bit QLoRA)

Usage:
    python train_grpo.py                          # Full training
    python train_grpo.py --steps 100 --episodes 30
    python train_grpo.py --eval-only outputs/grpo_training/trained_model
"""
import sys, os, json, argparse, random, copy, time, math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import GRPOTrainer, GRPOConfig

from server.policy_environment import PolicyEnvironment
from server.config import VALID_ACTIONS, CORE_ACTIONS, TASK_CONFIGS

# ── GPU Setup ──
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

ACTION_LIST = sorted(CORE_ACTIONS)
MINISTERS = ["Chancellor Voss", "Director Okafor", "Dr. Vasquez",
             "General Tanaka", "Senator Mwangi"]

# ── Prompt Engineering ──
SYSTEM_PROMPT = """You are the President governing a nation with a council of ministers.
You must choose policy actions, predict which ministers will veto, and form coalitions.
Respond with ONLY a JSON object: {"action": "action_name", "veto_prediction": ["minister_name"], "coalition_target": ["minister_name"]}"""

def format_observation(obs_meta: dict, step: int) -> str:
    """Format environment observation as a chat-style prompt."""
    actions_str = ", ".join(ACTION_LIST)
    events = ", ".join(obs_meta.get("active_events", [])) or "none"
    
    state = (
        f"Step {step}. "
        f"GDP: {obs_meta.get('gdp_index', 100):.0f}/200 | "
        f"Pollution: {obs_meta.get('pollution_index', 100):.0f}/500 | "
        f"Satisfaction: {obs_meta.get('public_satisfaction', 50):.0f}/100 | "
        f"Healthcare: {obs_meta.get('healthcare_index', 50):.0f}/100 | "
        f"Unemployment: {obs_meta.get('unemployment_rate', 10):.1f}% | "
        f"Events: {events}"
    )
    
    # Add negotiation context
    neg = obs_meta.get("negotiation_narrative", "")
    if neg:
        state += f"\n\nCOUNCIL PROPOSALS:\n{neg[:500]}"
    
    # Add briefings
    briefings = obs_meta.get("active_briefings", [])
    if briefings:
        state += "\n\nBRIEFINGS:"
        for b in briefings[:2]:
            state += f"\n  [{b['category'].upper()}] {b['text'][:150]} (deadline: step {b['deadline_step']})"
    
    state += f"\n\nAvailable actions: {actions_str}\nRespond with JSON."
    return state


def parse_action_from_text(text: str) -> dict:
    """Parse structured action from model output."""
    # Try JSON parse first
    try:
        import re
        json_match = re.search(r'\{[^}]+\}', text)
        if json_match:
            data = json.loads(json_match.group())
            action = data.get("action", "no_action")
            if action in ACTION_LIST:
                return {
                    "action": action,
                    "reasoning": "model",
                    "coalition_target": data.get("coalition_target", [])[:2],
                    "veto_prediction": data.get("veto_prediction", [])[:2],
                    "stance": "cooperative",
                }
    except (json.JSONDecodeError, Exception):
        pass
    
    # Fallback: extract action name
    text_lower = text.lower().strip()
    for action in ACTION_LIST:
        if action in text_lower:
            return {"action": action, "reasoning": "parsed", "coalition_target": [],
                    "veto_prediction": [], "stance": "cooperative"}
    return {"action": "no_action", "reasoning": "fallback", "coalition_target": [],
            "veto_prediction": [], "stance": "cooperative"}


# ── Smart Heuristic for Dataset ──
def smart_policy(obs_meta: dict, step: int) -> dict:
    """Heuristic that generates reasonable actions for seeding the dataset."""
    sat = obs_meta.get("public_satisfaction", 50)
    poll = obs_meta.get("pollution_index", 100)
    gdp = obs_meta.get("gdp_index", 100)
    health = obs_meta.get("healthcare_index", 50)
    
    if sat < 25: action = "increase_welfare"
    elif poll > 220: action = "enforce_emission_limits"
    elif gdp < 40: action = "stimulate_economy"
    elif health < 30: action = "invest_in_healthcare"
    elif poll > 150 and gdp > 70: action = "subsidize_renewables"
    elif gdp < 70: action = "decrease_tax"
    elif sat < 50: action = "invest_in_education"
    else: action = ACTION_LIST[step % len(ACTION_LIST)]
    
    proposals = obs_meta.get("negotiation", {}).get("minister_proposals", [])
    coalition, veto_pred = [], []
    for p in proposals:
        if p.get("veto_threat"): veto_pred.append(p["minister"])
        elif p.get("proposed_action") == action: coalition.append(p["minister"])
    if not coalition and proposals: coalition = [proposals[0]["minister"]]
    
    return {"action": action, "reasoning": f"heuristic step {step}",
            "coalition_target": coalition[:2], "veto_prediction": veto_pred[:2],
            "stance": "cooperative"}


# ── Dataset Generation ──
def generate_dataset(tokenizer, num_episodes=25, max_steps=50, seed=42, task_id="negotiation_arena"):
    """Generate prompts from environment rollouts."""
    prompts = []
    for ep in range(num_episodes):
        env = PolicyEnvironment()
        obs = env.reset(seed=seed + ep, task_id=task_id)
        for step in range(max_steps):
            if obs.done: break
            state_text = format_observation(obs.metadata, step)
            # Format as chat
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": state_text},
            ]
            try:
                prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            except Exception:
                prompt = f"{SYSTEM_PROMPT}\n\n{state_text}\n\nResponse:"
            prompts.append(prompt)
            action_data = smart_policy(obs.metadata, step)
            obs = env.step(action_data)
    
    random.shuffle(prompts)
    # Deduplicate similar prompts
    seen = set()
    unique = []
    for p in prompts:
        key = p[:500]
        if key not in seen:
            seen.add(key)
            unique.append(p)
    print(f"  Generated {len(unique)} unique prompts from {num_episodes} episodes")
    return Dataset.from_dict({"prompt": unique})


# ── Reward Function ──
def make_reward_func(task_id="negotiation_arena"):
    """Create a GRPO reward function that evaluates actions in the environment."""
    def reward_func(completions, **kwargs):
        prompts = kwargs.get("prompts", kwargs.get("prompt", [""] * len(completions)))
        if isinstance(prompts, str): prompts = [prompts] * len(completions)
        rewards = []
        for i, completion in enumerate(completions):
            action_data = parse_action_from_text(completion)
            action = action_data["action"]
            try:
                env = PolicyEnvironment()
                seed = abs(hash(prompts[i] if i < len(prompts) else "")) % 10000
                obs = env.reset(seed=seed, task_id=task_id)
                # Advance a few steps
                for _ in range(min(3, random.randint(1, 5))):
                    if obs.done: break
                    obs = env.step({"action": "no_action"})
                if not obs.done:
                    obs = env.step(action_data)
                    r = float(obs.reward)
                    # ToM bonus
                    tom_r = obs.metadata.get("negotiation_outcome", {}).get("tom_reward", 0)
                    r += tom_r * 2.0
                    # Coalition bonus
                    if obs.metadata.get("negotiation_outcome", {}).get("coalition_formed"):
                        r += 0.15
                else:
                    r = -0.1
            except Exception:
                r = -0.2
            
            # Format bonus: reward valid JSON
            if action != "no_action": r += 0.1
            try:
                import re
                if re.search(r'\{[^}]+\}', completion): r += 0.05
            except: pass
            
            rewards.append(float(r))
        return rewards
    return reward_func


# ── Evaluation ──
def evaluate(model, tokenizer, num_episodes=5, max_steps=50, seed=42,
             task_id="negotiation_arena", task_overrides=None, label=""):
    """Evaluate a model on the environment."""
    model.eval()
    device = next(model.parameters()).device
    episode_rewards, survivals = [], 0
    tom_correct, tom_total, coalitions = 0, 0, 0
    
    for ep in range(num_episodes):
        if task_overrides:
            orig = copy.deepcopy(TASK_CONFIGS.get(task_id, {}))
            TASK_CONFIGS.setdefault(task_id, {}).update(task_overrides)
        env = PolicyEnvironment()
        obs = env.reset(seed=seed + ep, task_id=task_id)
        ep_reward = 0.0
        steps_done = 0
        
        for step in range(max_steps):
            if obs.done: break
            steps_done = step
            state_text = format_observation(obs.metadata, step)
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": state_text},
            ]
            try:
                prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            except:
                prompt = f"{SYSTEM_PROMPT}\n\n{state_text}\n\nResponse:"
            
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=80, do_sample=True,
                                     temperature=0.7, top_p=0.9,
                                     pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id)
            response = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            action_data = parse_action_from_text(response)
            obs = env.step(action_data)
            ep_reward += obs.reward
            
            outcome = obs.metadata.get("negotiation_outcome", {})
            if "veto_prediction_correct" in outcome:
                tom_total += 1
                if outcome["veto_prediction_correct"]: tom_correct += 1
            if outcome.get("coalition_formed"): coalitions += 1
        
        if task_overrides:
            TASK_CONFIGS[task_id] = orig
        collapsed = obs.metadata.get("collapsed", False)
        if not collapsed: survivals += 1
        episode_rewards.append(round(ep_reward, 4))
        status = "SURVIVED" if not collapsed else "COLLAPSED"
        print(f"    {label}Ep {ep+1}: reward={ep_reward:.2f} steps={steps_done} [{status}]")
    
    tom_acc = tom_correct / max(tom_total, 1) if tom_total > 0 else 0
    return episode_rewards, survivals, {
        "tom_accuracy": round(tom_acc, 4), "tom_total": tom_total,
        "coalitions": coalitions,
    }


# ── Curriculum ──
def run_curriculum(model, tokenizer, seed=5000, task_id="negotiation_arena"):
    """Escalating difficulty evaluation."""
    print("\n  [CURRICULUM] Running escalation...")
    levels = [
        {"label": "Easy", "chaos_level": 0.0, "event_frequency_multiplier": 0.3, "steps": 50},
        {"label": "Medium", "chaos_level": 0.3, "event_frequency_multiplier": 0.6, "steps": 60},
        {"label": "Hard", "chaos_level": 0.6, "event_frequency_multiplier": 1.0, "steps": 80},
        {"label": "Extreme", "chaos_level": 1.0, "event_frequency_multiplier": 1.5, "steps": 100},
    ]
    results = []
    for i, lvl in enumerate(levels):
        label = lvl.pop("label")
        ms = lvl.pop("steps")
        print(f"\n  Level {i+1}/4: {label} (chaos={lvl.get('chaos_level',0):.1f})")
        rewards, survs, metrics = evaluate(
            model, tokenizer, num_episodes=3, max_steps=ms,
            seed=seed + i * 100, task_id=task_id, task_overrides=lvl, label=f"[{label}] "
        )
        avg = sum(rewards) / len(rewards) if rewards else 0
        results.append({
            "level": i + 1, "label": label, "chaos": lvl.get("chaos_level", 0),
            "avg_reward": round(avg, 4), "survivals": survs, "total": len(rewards),
            "tom_accuracy": metrics["tom_accuracy"], "coalitions": metrics["coalitions"],
        })
    return results


# ── Plotting ──
def plot_results(results, save_dir):
    """Generate publication-quality 4-panel comparison plots."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("POLARIS v3 — GRPO Training Results\nMulti-Agent Governance with Theory-of-Mind",
                 fontsize=14, fontweight="bold", y=0.98)
    fig.patch.set_facecolor('#fafafa')
    
    b_avg = results["baseline_avg_reward"]
    p_avg = results["post_avg_reward"]
    b_rewards = results["baseline_rewards"]
    p_rewards = results["post_rewards"]
    curriculum = results.get("curriculum", [])
    b_tom = results.get("baseline_tom", {})
    p_tom = results.get("post_tom", {})
    
    # 1. Before vs After
    ax = axes[0, 0]
    bars = ax.bar(["Before\nTraining", "After\nGRPO"], [b_avg, p_avg],
                   color=["#a1a1aa", "#4f46e5"], width=0.5, edgecolor='white', linewidth=2)
    for bar, val in zip(bars, [b_avg, p_avg]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f"{val:.1f}", ha="center", fontweight="bold", fontsize=14)
    ax.set_ylabel("Avg Episode Reward")
    ax.set_title("Reward Improvement", fontweight="bold", fontsize=12)
    imp_pct = results.get("improvement_pct", 0)
    ax.text(0.5, 0.95, f"{imp_pct:+.1f}%", transform=ax.transAxes, ha="center",
            fontsize=16, fontweight="bold", color="#059669" if imp_pct > 0 else "#e11d48")
    ax.grid(axis="y", alpha=0.3); ax.set_facecolor('#fafafa')
    
    # 2. Per-episode
    ax = axes[0, 1]
    x = list(range(1, max(len(b_rewards), len(p_rewards)) + 1))
    ax.bar([i - 0.2 for i in x[:len(b_rewards)]], b_rewards, 0.35, color="#a1a1aa", label="Before", alpha=0.8)
    ax.bar([i + 0.2 for i in x[:len(p_rewards)]], p_rewards, 0.35, color="#4f46e5", label="After", alpha=0.8)
    ax.set_xlabel("Episode"); ax.set_ylabel("Total Reward")
    ax.set_title("Per-Episode Comparison", fontweight="bold", fontsize=12)
    ax.legend(framealpha=0.9); ax.grid(axis="y", alpha=0.3); ax.set_facecolor('#fafafa')
    
    # 3. Curriculum
    ax = axes[1, 0]
    if curriculum:
        labels = [c["label"] for c in curriculum]
        rewards = [c["avg_reward"] for c in curriculum]
        colors = ["#059669", "#d97706", "#e11d48", "#7c3aed"]
        bars = ax.bar(labels, rewards, color=colors[:len(labels)], width=0.5, edgecolor='white', linewidth=2)
        for i, r in enumerate(rewards):
            ax.text(i, r + 0.2, f"{r:.1f}", ha="center", fontweight="bold", fontsize=12)
    ax.set_ylabel("Avg Reward"); ax.set_title("Curriculum Escalation", fontweight="bold", fontsize=12)
    ax.grid(axis="y", alpha=0.3); ax.set_facecolor('#fafafa')
    
    # 4. Theory-of-Mind
    ax = axes[1, 1]
    tom_labels = ["Veto Prediction\nAccuracy", "Coalition\nFormation Rate"]
    ep_steps = max(1, 5 * 50)
    tom_b = [b_tom.get("tom_accuracy", 0) * 100, b_tom.get("coalitions", 0) / ep_steps * 100]
    tom_a = [p_tom.get("tom_accuracy", 0) * 100, p_tom.get("coalitions", 0) / ep_steps * 100]
    x = range(len(tom_labels))
    ax.bar([i - 0.2 for i in x], tom_b, 0.35, color="#a1a1aa", label="Before")
    ax.bar([i + 0.2 for i in x], tom_a, 0.35, color="#4f46e5", label="After")
    ax.set_ylabel("Score (%)"); ax.set_title("Theory-of-Mind Metrics", fontweight="bold", fontsize=12)
    ax.legend(framealpha=0.9); ax.set_xticks(list(x)); ax.set_xticklabels(tom_labels)
    ax.grid(axis="y", alpha=0.3); ax.set_facecolor('#fafafa')
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    path = os.path.join(save_dir, "grpo_training_results.png")
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor='#fafafa')
    plt.close()
    print(f"  Plot saved: {path}")


# ── Main ──
def main():
    parser = argparse.ArgumentParser(description="POLARIS v3 GRPO Training")
    parser.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--episodes", type=int, default=25)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-generations", type=int, default=4)
    parser.add_argument("--save-dir", default="outputs/grpo_training")
    parser.add_argument("--eval-only", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--task", default="negotiation_arena")
    args = parser.parse_args()
    
    os.makedirs(args.save_dir, exist_ok=True)
    has_gpu = torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name(0) if has_gpu else "CPU"
    
    print("=" * 64)
    print("  POLARIS v3 — GRPO Training (QLoRA, RTX 5080)")
    print(f"  Model: {args.model}")
    print(f"  Device: {gpu_name}")
    print(f"  Steps: {args.steps} | Episodes: {args.episodes}")
    print(f"  Task: {args.task} (5 ministers, negotiation)")
    print(f"  QLoRA: 4-bit NF4 | LoRA r=16, alpha=32")
    print("=" * 64)
    
    # Load tokenizer
    print("\n[1/7] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model with 4-bit quantization
    print("[2/7] Loading model with 4-bit quantization...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16 if has_gpu else torch.float32,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model, quantization_config=bnb_config,
        device_map="auto", trust_remote_code=True,
        torch_dtype=torch.bfloat16 if has_gpu else torch.float32,
    )
    
    # Apply LoRA
    print("[3/7] Applying LoRA adapters...")
    lora_config = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                         "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM", bias="none",
    )
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, lora_config)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  Trainable: {trainable:,} / {total:,} ({trainable/total*100:.2f}%)")
    
    # Baseline evaluation
    print("\n[4/7] Baseline evaluation (before training)...")
    b_rewards, b_survs, b_metrics = evaluate(
        model, tokenizer, num_episodes=5, max_steps=50,
        seed=3000, task_id=args.task, label="[BASE] "
    )
    b_avg = sum(b_rewards) / len(b_rewards)
    print(f"  Baseline: avg={b_avg:.2f}, survived={b_survs}/5, ToM={b_metrics['tom_accuracy']:.0%}")
    
    if args.eval_only:
        print("\n  Eval-only mode. Done.")
        return
    
    # Generate dataset
    print(f"\n[5/7] Generating dataset ({args.episodes} episodes)...")
    dataset = generate_dataset(tokenizer, num_episodes=args.episodes, seed=args.seed, task_id=args.task)
    
    # GRPO Training
    print(f"\n[6/7] GRPO training ({args.steps} steps)...")
    training_args = GRPOConfig(
        output_dir=os.path.join(args.save_dir, "checkpoints"),
        num_train_epochs=1, max_steps=args.steps,
        per_device_train_batch_size=args.batch_size,
        num_generations=args.num_generations,
        max_completion_length=80,
        learning_rate=args.lr, logging_steps=max(1, args.steps // 10),
        save_steps=args.steps, seed=args.seed,
        report_to="none", log_level="warning",
        bf16=has_gpu, fp16=False,
        gradient_accumulation_steps=4,
        warmup_ratio=0.1,
    )
    
    trainer = GRPOTrainer(
        model=model, args=training_args,
        train_dataset=dataset,
        reward_funcs=make_reward_func(args.task),
        processing_class=tokenizer,
    )
    
    t0 = time.time()
    train_result = trainer.train()
    train_time = time.time() - t0
    t_metrics = train_result.metrics if hasattr(train_result, 'metrics') else {}
    print(f"  Done in {train_time:.1f}s ({args.steps / max(train_time, 1):.2f} steps/sec)")
    
    # Save
    model_path = os.path.join(args.save_dir, "trained_model")
    trainer.save_model(model_path)
    tokenizer.save_pretrained(model_path)
    
    # Post-training evaluation
    print("\n[7/7] Post-training evaluation...")
    p_rewards, p_survs, p_metrics = evaluate(
        model, tokenizer, num_episodes=5, max_steps=50,
        seed=4000, task_id=args.task, label="[TRAINED] "
    )
    p_avg = sum(p_rewards) / len(p_rewards)
    improvement = p_avg - b_avg
    imp_pct = improvement / max(abs(b_avg), 0.01) * 100
    
    # Curriculum
    curriculum = run_curriculum(model, tokenizer, seed=6000, task_id=args.task)
    
    # Results
    results = {
        "baseline_rewards": b_rewards, "post_rewards": p_rewards,
        "baseline_avg_reward": round(b_avg, 4), "post_avg_reward": round(p_avg, 4),
        "improvement": round(improvement, 4), "improvement_pct": round(imp_pct, 1),
        "baseline_survival": f"{b_survs}/5", "post_survival": f"{p_survs}/5",
        "baseline_tom": b_metrics, "post_tom": p_metrics,
        "steps": args.steps, "time_sec": round(train_time, 1),
        "model": args.model, "device": gpu_name, "task": args.task,
        "curriculum": curriculum,
        "training_metrics": {k: str(v) for k, v in t_metrics.items()},
    }
    
    rpath = os.path.join(args.save_dir, "training_results.json")
    with open(rpath, "w") as f: json.dump(results, f, indent=2)
    
    print(f"\n{'=' * 64}")
    print(f"  POLARIS v3 — GRPO TRAINING RESULTS")
    print(f"{'=' * 64}")
    print(f"  Baseline:      {b_avg:8.2f} reward  ({b_survs}/5 survived)")
    print(f"  Post-GRPO:     {p_avg:8.2f} reward  ({p_survs}/5 survived)")
    print(f"  Improvement:   {improvement:+8.2f} ({imp_pct:+.1f}%)")
    print(f"  Training time: {train_time:.1f}s on {gpu_name}")
    print(f"\n  THEORY OF MIND:")
    print(f"    Before: {b_metrics['tom_accuracy']:.0%} veto prediction")
    print(f"    After:  {p_metrics['tom_accuracy']:.0%} veto prediction")
    print(f"\n  CURRICULUM:")
    for c in curriculum:
        print(f"    {c['label']:>8s}: reward={c['avg_reward']:6.2f} "
              f"survive={c['survivals']}/{c['total']} ToM={c['tom_accuracy']:.0%}")
    print(f"\n  Model: {model_path}")
    print(f"  Results: {rpath}")
    print(f"{'=' * 64}")
    
    plot_results(results, args.save_dir)


if __name__ == "__main__":
    main()
