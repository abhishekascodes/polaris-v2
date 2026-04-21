---
title: POLARIS v2 — AI Governance Engine
emoji: 🌟
colorFrom: indigo
colorTo: green
sdk: docker
pinned: true
license: mit
tags:
  - openenv
  - reinforcement-learning
  - multi-agent
  - policy-optimization
  - pytorch
---

# 🌟 POLARIS v2 — Multi-Agent AI Governance Engine

> *"What happens when 5 AI ministers negotiate the fate of a nation under cascading crises — and the environment fights back?"*

A research-grade, multi-objective, multi-agent governance simulation for training LLM policy agents. Built for the **Meta PyTorch × OpenEnv Hackathon Grand Finale**.

**Covers 3 Themes:** Multi-Agent Interactions • Long-Horizon Planning • Self-Improvement

## 🏆 Results at a Glance

| Metric | Value |
|--------|-------|
| TRL GRPO Training | **+19.8% reward improvement** |
| Survival Rate | **0% → 40%** after 500 GRPO steps |
| Llama 3.3 70B Benchmark | **0.96 score** on Easy, collapses on Hard |
| Stress Test | **129/132 passed** (1,500 episodes, 97.7%) |
| Curriculum Escalation | Easy 3/3 → Medium 2/3 → Hard 0/3 |

## 🧠 Architecture

```
┌─────────────────────────────────────────────┐
│              POLARIS v2 ENGINE              │
├─────────────┬───────────┬───────────────────┤
│ Transition  │  Event    │  Multi-Agent      │
│ Engine      │  Engine   │  Council (5)      │
│ 21 metrics  │ Cascading │ Coalitions/Vetoes │
├─────────────┼───────────┼───────────────────┤
│ Reward      │ Explain-  │  Curriculum       │
│ Engine      │ ability   │  Engine           │
│ Multi-obj   │ Causal    │  Auto-difficulty  │
├─────────────┴───────────┴───────────────────┤
│        19 Actions × 4 Difficulty Tiers      │
└─────────────────────────────────────────────┘
```

## 🔑 Key Features

- **Multi-Agent Council**: 5 ministers (Economy, Environment, Social, Infrastructure, Finance) negotiate, form coalitions, veto proposals, betray alliances
- **Institutional Trust**: Slow-moving global state affected by cooperation patterns
- **Causal Explainability**: Every step produces causal chains, counterfactuals, and natural language narratives
- **Non-Stationary Dynamics**: Regime shifts + random events (pandemics, trade wars, disasters)
- **Auto-Curriculum**: Environment escalates difficulty as agent improves (Theme #4)
- **19 Policy Actions**: 16 core policy levers + 3 meta-coordination actions
- **4 Difficulty Tiers**: Easy → Medium → Hard → Extreme (structural instability)

## 🚀 Quick Start

```bash
# Run the environment server
python -m uvicorn server.app:app --port 7860

# Live research dashboard
python dashboard_server.py

# Train with TRL GRPO
python train_trl.py --steps 500 --model gpt2

# Benchmark with Llama 70B (Groq)
GROQ_API_KEY=gsk_... python llm_benchmark.py

# Full validation (1,500 episodes)
python nuclear_test.py
```

## 📊 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/reset` | POST | Start new episode |
| `/step` | POST | Take an action |
| `/state` | GET | Current state |
| `/tasks` | GET | List available tasks |
| `/schema` | GET | Action/observation schemas |

## 🎯 Tasks

| Task | Difficulty | Steps | Description |
|------|-----------|-------|-------------|
| `environmental_recovery` | Easy | 50 | Reduce pollution while maintaining GDP |
| `balanced_economy` | Medium | 100 | Balance GDP, pollution, and satisfaction |
| `sustainable_governance` | Hard | 200 | Long-horizon stability under event pressure |
| `sustainable_governance_extreme` | Extreme | 200 | Structural instability benchmark |

## 📈 Training Pipeline

Uses **HuggingFace TRL GRPOTrainer** with GPU optimizations:
- `torch.backends.cudnn.benchmark = True`
- `torch.backends.cuda.matmul.allow_tf32 = True`
- bf16 mixed precision on CUDA devices
- Auto-curriculum self-improvement evaluation

## 🔬 Validation

- **132 automated tests** covering determinism, reward bounds, collapse detection, spec compliance
- **1,500-episode statistical analysis** (95 episodes/sec throughput)
- **Llama 3.3 70B live benchmark** via Groq API
- **6-baseline comparison** (random, heuristic, greedy, LLM proxy, single RL, multi-council)

## 📁 Project Structure

```
polaris-v2/
├── server/
│   ├── app.py                 # FastAPI server
│   ├── policy_environment.py  # Core environment
│   ├── transition_engine.py   # 21-metric state transitions
│   ├── event_engine.py        # Cascading stochastic events
│   ├── reward_engine.py       # Multi-objective rewards
│   ├── multi_agent_council.py # 5-minister council
│   ├── explainability.py      # Causal chains + counterfactuals
│   ├── curriculum_engine.py   # Auto-difficulty escalation
│   ├── drift_engine.py        # Non-stationary dynamics
│   ├── config.py              # 19 actions, task configs
│   └── tasks.py               # Grading system
├── train_trl.py               # TRL GRPO training (GPU)
├── llm_benchmark.py           # Live LLM benchmark (Groq)
├── inference.py               # OpenAI-compatible agent
├── dashboard_server.py        # 6-tab research dashboard
├── nuclear_test.py            # 132-test validation suite
├── openenv.yaml               # OpenEnv spec
├── Dockerfile                 # HF Spaces deployment
└── README.md
```

Built by **Abhishek A S** — Meta PyTorch × OpenEnv Hackathon Grand Finale 2026
