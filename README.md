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

I benchmarked **Llama 3.3 70B** (via Groq API) against all 6 tasks to prove the environment genuinely challenges frontier models:

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
Model: Qwen 2.5 3B Instruct (QLoRA 4-bit, LoRA r=16)
Task: negotiation_arena (5 ministers, 200 steps)
Algorithm: GRPO with 6-component composite reward
Curriculum: Easy → Medium → Hard → Extreme (auto-escalation)
Hardware: NVIDIA GeForce RTX 5080 Laptop GPU
Training: 100 steps, 788 seconds, 29.9M trainable params
```

> **Result**: +126.3% reward improvement (13.4 → 30.2), first survival achieved (0/5 → 1/5). Trained agent dominates Easy (3/3) and Medium (2/3) curriculum levels while Hard and Extreme remain unsolved — proving genuine difficulty scaling.

---

## 🚀 Quick Start

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run the Interactive Dashboard

```bash
# Start the full dashboard server (WebSocket + REST API + live simulation)
cd openenv && python dashboard_server.py

# Open http://localhost:8765 → Landing page with interactive simulation
# Open http://localhost:8765/control → Full 7-tab research dashboard
```

### Run LLM Inference

```bash
export API_BASE_URL="https://api.groq.com/openai/v1"
export MODEL_NAME="llama-3.3-70b-versatile"
export HF_TOKEN="your_key"
python inference.py
```

### Train with GRPO (QLoRA)

```bash
# Train Qwen 3B with curriculum ToM reward on RTX 5080
python train_grpo.py --model Qwen/Qwen2.5-3B-Instruct --steps 100 --episodes 30
```

### Interactive Demo

- **Landing Page** (`/`): Scroll down to "Try the Simulation" — select any task, adjust chaos, and watch the agent negotiate in real-time
- **Control Panel** (`/control`): 7-tab research dashboard with live charts, negotiation logs, causal analysis, risk alerts, and episode history
- **HF Space**: [asabhishek-polaris-v3.hf.space](https://asabhishek-polaris-v3.hf.space) — fully deployed, same experience

---

## 🚀 Training Results — GRPO on RTX 5080

### Qwen 2.5 3B Instruct (QLoRA 4-bit, 100 GRPO steps, 13 minutes)

| Metric | Before Training | After GRPO | Change |
|--------|:-:|:-:|:-:|
| **Avg Episode Reward** | 13.4 | **30.2** | **+126.3%** ✅ |
| **Survival Rate** | 0/5 | **1/5** | First survival |
| **Training Time** | — | 788s | RTX 5080 Laptop |
| **Trainable Params** | — | 29.9M / 1.73B | 1.73% (LoRA) |

### Curriculum Escalation (Post-Training)

| Difficulty | Chaos | Avg Reward | Survived | 
|:-:|:-:|:-:|:-:|
| 🟢 Easy | 0.0 | **40.8** | **3/3** |
| 🟡 Medium | 0.3 | **38.3** | **2/3** |
| 🔴 Hard | 0.6 | 24.9 | 0/3 |
| 🟣 Extreme | 1.0 | 22.7 | 0/3 |

### Llama 3.3 70B Benchmark (via Groq API)

| Task | Score | Notes |
|------|:-----:|-------|
| Environment Recovery (Easy) | **0.96** | Single-objective, trivial |
| Negotiation Arena | **0.22** | 77% collapse under multi-agent pressure |
| Theory-of-Mind Accuracy | **0%** | Frontier LLM cannot predict minister vetoes |

> **Key finding:** Llama 70B scores 0.96 on easy tasks but collapses to 0.22 on multi-agent negotiation. POLARIS creates genuine difficulty that scales with agent sophistication.

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
│   └── app.py                  # REST API server
├── static/
│   ├── style.css               # Control panel styling
│   └── app.js                  # Control panel engine (7-tab dashboard)
├── dashboard_server.py         # Full server: WebSocket + REST + simulation
├── dashboard.html              # Landing page with interactive simulation
├── control.html                # 7-tab research control panel
├── inference.py                # LLM inference with structured output
├── train_grpo.py               # GRPO training with curriculum ToM reward
├── POLARIS_v3_Demo.ipynb        # Colab demo notebook
├── openenv.yaml                # OpenEnv manifest
├── requirements.txt            # Dependencies
├── Dockerfile                  # HF Spaces container deployment
└── README.md                   # This file
```

---

## 🔗 Links

| Resource | Link |
|----------|------|
| 🤗 HuggingFace Space | [huggingface.co/spaces/asabhishek/polaris-v3](https://huggingface.co/spaces/asabhishek/polaris-v3) |
| 📦 GitHub Repository | [github.com/abhishekascodes/POLARIS-V3](https://github.com/abhishekascodes/POLARIS-V3) |
| 📓 Colab Demo | [POLARIS_v3_Demo.ipynb](https://colab.research.google.com/github/abhishekascodes/POLARIS-V3/blob/main/POLARIS_v3_Demo.ipynb) |

---

<div align="center">

**Built with 🧠 for the Meta PyTorch OpenEnv Hackathon × Scaler 2026**

*POLARIS: Where every policy decision is a negotiation, every minister has an agenda, and every veto tests your theory of mind.*

</div>
