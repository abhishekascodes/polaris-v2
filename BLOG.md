# POLARIS v3: Where LLMs Learn to Govern Through Multi-Agent Negotiation

## TL;DR
A research-grade RL environment where LLM agents negotiate with 5 AI ministers, predict vetoes, form coalitions, and learn governance through multi-agent interaction. Trained Qwen 2.5 3B with GRPO + QLoRA — achieved **+126.3% reward improvement** and **first survival** in 13 minutes on RTX 5080.

---

## The Problem
Current RL environments for LLMs are either toy tasks (text games, grid worlds) or single-objective benchmarks. Real governance requires balancing **competing objectives simultaneously** while navigating **multi-agent dynamics**, **non-stationary environments**, and **strategic deception**.

No existing OpenEnv environment puts LLM agents *inside* the environment itself. POLARIS is the first to create genuine multi-agent interaction that trains theory-of-mind reasoning.

## The Environment

POLARIS v3 simulates a 21-metric economic nation governed by 5 AI minister personas (Economy, Environment, Health, Industry, Social Welfare), each with distinct priorities and hidden agendas. Every step:

1. **Ministers propose** — generating natural language proposals with arguments and coalition offers
2. **Agent decides** — choosing an action, predicting vetoes, and targeting coalition partners
3. **Council resolves** — voting, vetoing, forming or breaking coalitions
4. **World updates** — 21 state variables evolve through a 4-layer transition engine
5. **Reward computes** — 6 independent components scored simultaneously

### Key Features
- **6 tasks** across 4 difficulty tiers: Easy to Extreme
- **3-phase negotiation protocol**: Propose, Decide, Resolve
- **6-component composite reward**: governance + Pareto + ToM + coalition + briefing + oscillation penalty
- **Non-stationary dynamics**: 6 drifting variables, regime shifts, cascading events
- **Auto-curriculum**: difficulty escalates as agent improves
- **Causal explainability**: every step produces counterfactual analysis

## Training Results

Using TRL GRPOTrainer with Qwen 2.5 3B Instruct (QLoRA 4-bit):

| Metric | Before | After GRPO | Change |
|--------|--------|-----------|--------|
| Avg Reward | 13.4 | 30.2 | **+126.3%** |
| Survival Rate | 0/5 | 1/5 | First survival |
| Training Time | -- | 788s | RTX 5080 Laptop |
| Trainable Params | -- | 29.9M / 1.73B | 1.73% (LoRA r=16) |

### Curriculum Escalation (Post-Training)

| Difficulty | Chaos | Avg Reward | Survived |
|------------|-------|------------|----------|
| Easy | 0.0 | 40.8 | 3/3 |
| Medium | 0.3 | 38.3 | 2/3 |
| Hard | 0.6 | 24.9 | 0/3 |
| Extreme | 1.0 | 22.7 | 0/3 |

### Frontier Model Benchmark: Llama 3.3 70B

| Task | Score | Notes |
|------|-------|-------|
| Environment Recovery (Easy) | 0.96 | Single-objective, trivial |
| Negotiation Arena (Hard) | 0.22 | 77% collapse under multi-agent pressure |
| Theory-of-Mind Accuracy | 0% | Frontier LLM cannot predict minister vetoes |

### The Governance Complexity Gap
Llama 70B handles single-objective governance well (0.96), but collapses to 0.22 under multi-agent negotiation pressure. Theory-of-Mind accuracy is 0%. This proves that POLARIS creates genuine difficulty that scales with model sophistication — and that there is massive room for improvement via curriculum RL training.

## Themes Covered
- **Theme 1: Multi-Agent Interactions** — 5 minister agents with negotiation, coalitions, and vetoes
- **Theme 2: Long-Horizon Planning** — 200-300 step episodes with timed briefing deadlines
- **Theme 3: World Modeling** — 21-metric simulation with 4-layer transitions and non-stationary drift
- **Theme 4: Self-Improvement** — Auto-curriculum escalation from Easy to Extreme

## Architecture

The environment is built as a FastAPI application with:
- `policy_environment.py` — Core environment with reset/step/state
- `negotiation_protocol.py` — 3-phase negotiation with ToM scoring
- `llm_minister.py` — 5 minister personas with distinct priorities
- `briefing_engine.py` — Time-sensitive intelligence with deadlines
- `transition_engine.py` — 4-layer state transitions with delayed effects
- `reward_engine.py` — 6-component composite reward function
- `explainability.py` — Causal chains and counterfactual analysis

## Try It

```bash
# Run the environment
python -m uvicorn server.app:app --port 7860

# Train with GRPO (QLoRA 4-bit)
python train_grpo.py --model Qwen/Qwen2.5-3B-Instruct --steps 100 --episodes 30
```

## Links
- [HuggingFace Space](https://huggingface.co/spaces/asabhishek/polaris-v3)
- [GitHub Repository](https://github.com/abhishekascodes/POLARIS-V3)
- [Colab Demo](https://colab.research.google.com/github/abhishekascodes/POLARIS-V3/blob/main/POLARIS_v3_Demo.ipynb)

Built by **Abhishek A S** (17) for the Meta PyTorch OpenEnv Hackathon Grand Finale 2026.
