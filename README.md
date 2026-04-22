---
title: POLARIS v3
emoji: 🌐
colorFrom: indigo
colorTo: purple
sdk: docker
pinned: true
license: mit
short_description: Multi-Agent AI Governance Engine with Theory-of-Mind
tags:
  - openenv
  - reinforcement-learning
  - multi-agent
  - theory-of-mind
  - negotiation
  - governance
---
<div align="center">

# 🌐 POLARIS v3 — Multi-Agent AI Governance Engine

### *An OpenEnv Environment for Training LLMs on Multi-Agent Negotiation, Theory-of-Mind, and Long-Horizon Planning*

[![OpenEnv](https://img.shields.io/badge/OpenEnv-Compatible-4f46e5?style=for-the-badge)](https://github.com/OpenEnv-ai/openenv)
[![HF Space](https://img.shields.io/badge/🤗-HuggingFace_Space-yellow?style=for-the-badge)](https://huggingface.co/spaces/asabhishek/polaris-v3)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

**Built by [Abhishek A S](https://github.com/abhishekascodes) — Solo, Age 17**

---

*POLARIS simulates a 21-metric economic nation where an LLM agent must negotiate with 5 AI minister personas, predict vetoes, form coalitions, and act on time-sensitive intelligence briefings. It is the first OpenEnv environment to put LLM agents inside the environment itself, creating genuine multi-agent interaction that trains theory-of-mind reasoning.*

</div>

---

## 📋 Table of Contents

- [Why POLARIS?](#-why-polaris)
- [Architecture](#-architecture)
- [Themes Covered](#-themes-covered)
- [Tasks](#-tasks)
- [How It Works](#-how-it-works)
- [Negotiation Protocol](#-negotiation-protocol)
- [Reward System](#-reward-system)
- [Training Results](#-training-results)
- [Quick Start](#-quick-start)
- [Project Structure](#-project-structure)
- [Links](#-links)

---

## 🎯 Why POLARIS?

Most RL environments treat the agent as the *only* intelligent entity. The "environment" is just physics or rules. **POLARIS is different.**

In POLARIS, the environment contains **5 AI minister agents** that:
- 🗣️ Generate natural language proposals with arguments
- 🤝 Offer coalitions with conditions
- ⚡ Threaten and execute vetoes based on their priorities
- 🕵️ Have hidden agendas the training agent must discover
- 📋 Deliver time-sensitive intelligence briefings with deadlines

The training agent doesn't just pick actions — it must **read proposals, reason about other agents' motivations, predict their behavior, and negotiate strategically**. This creates genuine theory-of-mind pressure that no grid-world or simple game can match.

### What makes this research-grade?

| Feature | Typical OpenEnv | POLARIS v3 |
|---------|----------------|------------|
| Agents in environment | 0 (just rules) | **5 LLM-powered ministers** |
| Action space | Simple string | **Structured JSON** (action + reasoning + coalition + veto prediction) |
| Observation | Flat metrics | **Natural language negotiation context** + 55-dim vector |
| Reward signal | Single scalar | **Composite**: governance + theory-of-mind + briefing compliance |
| Memory test | None | **Timed briefings with deadlines across 200+ steps** |
| Difficulty | Fixed | **Auto-curriculum escalation** |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    POLARIS v3 Engine                         │
├──────────────┬──────────────┬───────────────┬───────────────┤
│  Transition  │    Event     │     Drift     │  Explainability│
│  Engine (L4) │  Engine (v2) │   Engine      │   Engine       │
│  ─ 4 layers  │  ─ sigmoid   │  ─ 6 vars     │  ─ causal      │
│  ─ delayed   │  ─ chaining  │  ─ non-       │  ─ counter-    │
│    effects   │  ─ memory    │    stationary │    factuals    │
├──────────────┴──────────────┴───────────────┴───────────────┤
│              Multi-Agent Council (5 Ministers)                │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────┐ ┌───────┐ │
│  │Chancellor│ │Director  │ │   Dr.    │ │Gen.  │ │Senator│ │
│  │  Voss    │ │ Okafor   │ │ Vasquez  │ │Tanaka│ │Mwangi │ │
│  │ Finance  │ │Environ.  │ │ Health   │ │Industry│ │Social│ │
│  │  💰      │ │  🌿      │ │  🏥      │ │  🏭   │ │ 🗳️  │ │
│  └──────────┘ └──────────┘ └──────────┘ └──────┘ └───────┘ │
├─────────────────────────────────────────────────────────────┤
│              Negotiation Protocol (3-Phase)                  │
│  PROPOSE → ministers generate proposals with arguments       │
│  NEGOTIATE → agent reads, reasons, forms coalitions          │
│  RESOLVE → council votes, vetoes, outcomes computed          │
├─────────────────────────────────────────────────────────────┤
│              Briefing Engine (Long-Horizon Memory)            │
│  Time-sensitive intelligence with deadlines                  │
│  "GDP must exceed 95 by step 45 or investors withdraw"       │
├─────────────────────────────────────────────────────────────┤
│              Reward Engine                                    │
│  Governance reward + Theory-of-Mind reward + Briefing reward │
│  Pareto optimality + Cooperation bonus + Oscillation penalty │
└─────────────────────────────────────────────────────────────┘
```

---

## 🎪 Themes Covered

### Theme #1: Multi-Agent Interactions ✅
- 5 minister agents with distinct personas, priorities, and hidden agendas
- Natural language negotiation with proposals, arguments, and coalition offers
- Veto mechanics that create real strategic tension
- Theory-of-mind scoring: agent is rewarded for correctly predicting vetoes

### Theme #2: Long-Horizon Planning ✅
- Episodes up to 300 steps with sparse delayed rewards
- Diplomatic briefings with deadlines: "Reduce pollution below 120 by step 45"
- The agent must remember briefings and plan across the full episode
- Briefing compliance is scored and rewarded

### Theme #3: World Modeling ✅
- 21-metric economic simulation with cross-layer feedback
- 4-layer transition engine with delayed policy effects
- Non-stationary drift on 6 variables (the world changes under you)
- Causal reasoning chain in every observation

### Theme #4: Self-Improvement ✅
- Auto-curriculum: difficulty escalates as the agent improves
- Chaos level increases from 0.0 → 1.0 across training
- Event frequency scales up
- The environment *adapts* to the agent's capability

---

## 📝 Tasks

| Task | Steps | Ministers | Difficulty | Negotiation | Briefings |
|------|-------|----------|------------|-------------|-----------|
| `environmental_recovery` | 50 | 1 | Easy | ❌ | ❌ |
| `balanced_economy` | 100 | 1 | Medium | ❌ | ❌ |
| `sustainable_governance` | 200 | 3 | Hard | ✅ | ✅ |
| `sustainable_governance_extreme` | 200 | 5 | Extreme | ✅ | ✅ |
| `multi_agent_council` | 300 | 5 | Extreme+ | ✅ | ✅ |
| `negotiation_arena` | 200 | 5 | Hard | ✅ | ✅ |

---

## 🔬 How It Works

### Step-by-Step Flow

```
1. Agent receives observation:
   - 21 economic metrics (GDP, pollution, satisfaction, etc.)
   - Minister proposals with arguments and veto threats
   - Active briefings with deadlines
   - Coalition offers from ministers

2. Agent outputs structured decision:
   {
     "action": "subsidize_renewables",
     "reasoning": "Pollution at 130 threatens ecological collapse",
     "coalition_target": ["Director Okafor", "Dr. Vasquez"],
     "veto_prediction": ["Chancellor Voss"],
     "stance": "cooperative"
   }

3. Environment resolves:
   - Council votes on the action
   - Coalitions form or fail
   - Vetoes execute (or don't)
   - Theory-of-mind accuracy scored
   - World state updates through 4-layer transition

4. Reward computed:
   = governance_reward      (did metrics improve?)
   + tom_reward             (was veto prediction correct?)
   + briefing_reward        (did agent act on briefings?)
   + cooperation_bonus      (did coalition form?)
   - oscillation_penalty    (did agent flip-flop?)
```

---

## 🤝 Negotiation Protocol

Each step runs a 3-phase negotiation:

### Phase 1: PROPOSE
Ministers generate proposals based on their priorities:
```
💰 Chancellor Voss (Finance):
  Proposes: decrease_tax
  "GDP is at 90. We need economic stimulus now."
  Coalition offer: "I'll support you if Industry joins."
  Trust: 47%
  Intel: Voss reacted strongly to carbon tax.

🌿 Director Okafor (Environment):
  Proposes: subsidize_renewables
  "Pollution at 130 threatens ecological stability."
  Veto threat: NO
  Trust: 69%
```

### Phase 2: NEGOTIATE
The training agent reads all proposals and decides:
- Which action to take
- Which ministers to ally with
- Which ministers might veto
- What argument to make

### Phase 3: RESOLVE
The council votes:
- **Supporters** rally behind the agent's action
- **Opposers** vote against
- **Vetoes** can override the decision entirely
- **Coalition formation** is tracked for cooperation scoring

---

## 💰 Reward System

POLARIS uses a **composite reward function** with 6 components:

| Component | Weight | Signal |
|-----------|--------|--------|
| Base Governance | ~40% | Multi-metric improvement (GDP, pollution, satisfaction) |
| Pareto Optimality | ~15% | Balanced improvement across all dimensions |
| Theory-of-Mind | ~15% | Correct veto predictions (+0.15), wrong (-0.05) |
| Cooperation | ~10% | Coalition formation bonus (+0.08) |
| Briefing Compliance | ~10% | Acting on timed intelligence before deadline |
| Oscillation Penalty | ~10% | Penalizes flip-flopping between opposite actions |

This reward is **dense** (every step), **multi-dimensional** (6 components), and **hard to game** (improving one metric while tanking others gets penalized by Pareto scoring).

---

## 📊 Benchmark Results — Llama 3.3 70B vs POLARIS

We benchmarked **Llama 3.3 70B** (via Groq API) against all 6 tasks to prove the environment genuinely challenges frontier models:

### Task Scores: LLM vs Baselines

| Task | Llama 70B | Smart Heuristic | Random | ToM Accuracy |
|------|-----------|-----------------|--------|-------------|
| environmental_recovery (Easy) | **0.9625** | 0.8841 | 0.7622 | N/A |
| balanced_economy (Medium) | 0.1507 | 0.1579 | 0.1544 | N/A |
| sustainable_governance (Hard) | 0.1734 | 0.1584 | 0.1715 | **4%** |
| sustainable_governance_extreme | 0.2814 | 0.1661 | 0.2932 | **0%** |
| multi_agent_council (300 steps) | 0.2013 | 0.2899 | 0.2181 | **0%** |
| negotiation_arena (5 ministers) | 0.2260 | 0.2914 | 0.2198 | **0%** |

### Key Findings

> **🔥 Llama 70B scores 0.96 on easy governance but COLLAPSES to 0.20 on multi-agent negotiation**
>
> **🧠 Theory-of-Mind accuracy is 0–4% — Llama 70B CANNOT predict minister vetoes**
>
> **📉 On multi-agent tasks, Llama 70B performs AT or BELOW random baseline**

This proves:
1. The environment is **genuinely challenging** for frontier LLMs
2. Multi-agent negotiation creates **real difficulty** that simple governance doesn't
3. Theory-of-mind reasoning is a **non-trivial capability gap** that RL training can target
4. There is **massive room for improvement** via GRPO training

### Training Pipeline (TRL + GRPO)

```
Model: gpt2 → Llama (via HF credits at venue)
Task: negotiation_arena (5 ministers, 200 steps)
Algorithm: GRPO with 6-component reward function
Curriculum: Easy → Medium → Hard → Extreme (auto-escalation)
Hardware: NVIDIA RTX 5080 (local) + HF compute credits (venue)
```

> **Note**: Full GRPO training with HF compute credits will be performed onsite on 25-26 April. The training pipeline (`train_trl.py`) is fully validated and ready. The benchmark data above serves as the "before training" baseline.

---

## 🚀 Quick Start

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run the Environment

```bash
# Start the server
cd openenv && python -m uvicorn server.app:app --host 0.0.0.0 --port 7860

# Run inference
export API_BASE_URL="https://api.groq.com/openai/v1"
export MODEL_NAME="llama-3.3-70b-versatile"
export HF_TOKEN="your_key"
python inference.py
```

### Train with TRL

```bash
python train_trl.py --model gpt2 --steps 200 --episodes 30
python train_trl.py --plot  # Generate result plots
```

### View Dashboard

Open `http://localhost:7860` in your browser to see the real-time negotiation dashboard.

---

## 📁 Project Structure

```
openenv/
├── server/
│   ├── policy_environment.py   # Core environment (v3 POLARIS)
│   ├── llm_minister.py         # 5 LLM-powered minister personas
│   ├── negotiation_protocol.py # 3-phase negotiation with ToM scoring
│   ├── briefing_engine.py      # Time-sensitive intelligence briefings
│   ├── transition_engine.py    # 4-layer state transitions
│   ├── event_engine.py         # Sigmoid probability events with chaining
│   ├── drift_engine.py         # Non-stationary variable drift
│   ├── multi_agent_council.py  # Coalition/voting mechanics
│   ├── reward_engine.py        # Composite reward function
│   ├── explainability.py       # Causal chains + counterfactuals
│   ├── config.py               # 6 task configurations
│   ├── tasks.py                # Deterministic graders
│   └── app.py                  # FastAPI server
├── static/
│   ├── style.css               # Dashboard styling
│   └── app.js                  # Dashboard engine
├── dashboard.html              # Real-time negotiation dashboard
├── inference.py                # LLM inference with structured output
├── train_trl.py                # TRL GRPO training pipeline
├── openenv.yaml                # OpenEnv manifest
├── requirements.txt            # Dependencies
├── Dockerfile                  # Container deployment
└── README.md                   # This file
```

---

## 🔗 Links

| Resource | Link |
|----------|------|
| 🤗 HuggingFace Space | [huggingface.co/spaces/asabhishek/polaris-v3](https://huggingface.co/spaces/asabhishek/polaris-v3) |
| 📦 GitHub Repository | [github.com/abhishekascodes/POLARIS-V3](https://github.com/abhishekascodes/POLARIS-V3) |
| 📝 HuggingFace Blog | *Coming soon* |
| 🎥 Demo Video | *Coming soon* |

---

<div align="center">

**Built with 🧠 for the Meta PyTorch OpenEnv Hackathon × Scaler 2026**

*POLARIS: Where every policy decision is a negotiation, every minister has an agenda, and every veto tests your theory of mind.*

</div>
