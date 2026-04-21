# POLARIS v2: Multi-Agent Governance Simulation for LLM Training

## TL;DR
A multi-objective, multi-agent governance simulation where 5 AI ministers negotiate policy decisions under cascading crises. Trained with HuggingFace TRL GRPO — achieved **+19.8% reward improvement** and **0% → 40% survival rate** in 95 seconds on RTX 5080.

---

## The Problem
Current RL environments for LLMs are either toy tasks (text games) or single-objective benchmarks. Real governance requires balancing **competing objectives simultaneously** while managing **multi-agent dynamics** and **non-stationary environments**.

## The Environment

**OpenENV AI Policy Engine** simulates a country governed by 5 AI ministers (Economy, Environment, Social Welfare, Infrastructure, Finance), each with distinct priorities. Every step:

1. **Agent selects** one of 19 policy actions (tax, subsidies, healthcare, emissions, etc.)
2. **Ministers negotiate** — forming coalitions, issuing vetoes, even betraying alliances
3. **Environment responds** — 21 state variables update with cascading effects
4. **Random events trigger** — pandemics, trade wars, climate crises, tech booms
5. **Institutional trust drifts** — affecting future cooperation

### Key Features
- **4 difficulty tiers**: Easy → Extreme (structural instability benchmark)
- **Non-stationary dynamics**: Regime shifts change the rules mid-episode
- **Causal explainability**: Every step produces counterfactual analysis
- **Auto-curriculum**: Environment escalates difficulty as agent improves

## Training Results

Using HuggingFace TRL GRPOTrainer with GPT-2:

| Metric | Before | After | Change |
|--------|--------|-------|--------|
| Avg Reward | 29.4 | 35.3 | **+19.8%** |
| Survival Rate | 0/5 | 2/5 | **+40%** |
| Training Time | — | 95.6s | RTX 5080 |

### Self-Improvement Curriculum
The trained agent was evaluated across escalating difficulty:

| Level | Chaos | Survival | Avg Reward |
|-------|-------|----------|------------|
| Easy | 0.0 | 3/3 | 47.7 |
| Medium | 0.3 | 2/3 | 48.2 |
| Hard | 0.6 | 0/3 | 27.5 |
| Extreme | 1.0 | 0/3 | 33.4 |

## Themes Covered
- **Theme #1**: Multi-Agent Interactions (5-minister council)
- **Theme #2**: Long-Horizon Planning (200-step episodes)
- **Theme #4**: Self-Improvement (auto-curriculum escalation)

## Try It

```bash
# Run the environment
python -m uvicorn server.app:app --port 7860

# Train with TRL
python train_trl.py --steps 200 --model gpt2

# Live dashboard
python dashboard_server.py
```

## Links
- [HuggingFace Space](https://huggingface.co/spaces/asabhishek/polaris-v2)
- [GitHub Repository](https://github.com/abhishekascodes/polaris-v2)
- [Colab Training Notebook](https://github.com/abhishekascodes/polaris-v2/blob/main/POLARIS_v2_Training.ipynb)

Built by **Abhishek A S** for the Meta PyTorch × OpenEnv Hackathon Grand Finale.
