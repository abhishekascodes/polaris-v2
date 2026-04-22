#!/usr/bin/env python3
"""
POLARIS v3 — Inference Script for OpenEnv Hackathon

Runs an LLM agent against the multi-agent governance environment with
full negotiation protocol. The LLM reads minister proposals, reasons
about coalitions, predicts vetoes, and makes strategic decisions.

Environment Variables Required:
    API_BASE_URL  -- The API endpoint for the LLM
    MODEL_NAME    -- The model identifier
    HF_TOKEN      -- Your API key

Usage:
    export API_BASE_URL="https://api.groq.com/openai/v1"
    export MODEL_NAME="llama-3.3-70b-versatile"
    export HF_TOKEN="gsk_..."
    python inference.py
"""

from __future__ import annotations

import json
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

from openai import OpenAI

# -- Local imports --
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from server.policy_environment import PolicyEnvironment
from server.tasks import grade_trajectory, get_task_ids
from server.config import VALID_ACTIONS, ACTION_DESCRIPTIONS, TASK_CONFIGS


# ====================================================================
# Configuration
# ====================================================================

API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o")
HF_TOKEN = os.getenv("HF_TOKEN")

SEED = 42

ACTION_LIST_STR = "\n".join(
    f"  - {name}: {desc}" for name, desc in ACTION_DESCRIPTIONS.items()
)


# ====================================================================
# v3 System Prompt — Multi-Agent Negotiation
# ====================================================================

SYSTEM_PROMPT_V3 = """You are the President of a simulated nation. Each turn, your council of ministers presents proposals. You must:

1. Read each minister's proposal, argument, and coalition offer
2. Decide which policy action to take
3. Choose which ministers to form a coalition with
4. Predict which ministers might veto your decision
5. Craft an argument to persuade ministers

AVAILABLE ACTIONS:
{actions}

RESPONSE FORMAT — respond with valid JSON only, no markdown:
{{
  "action": "<action_name>",
  "reasoning": "<1-2 sentences explaining your decision>",
  "coalition_target": ["<minister_name_1>", "<minister_name_2>"],
  "negotiation_argument": "<argument to persuade coalition partners>",
  "veto_prediction": ["<minister_name_who_might_veto>"],
  "stance": "cooperative"
}}

RULES:
- action MUST be one of the valid actions listed above
- coalition_target should name 1-3 ministers you want to ally with
- veto_prediction should name ministers who oppose your action
- Balance GDP, pollution, and satisfaction simultaneously
- React to active events and briefings
- Prevent collapse: GDP > 15, pollution < 290, satisfaction > 5

Respond with ONLY valid JSON. No explanation outside the JSON."""


SYSTEM_PROMPT_SIMPLE = """You are an expert AI policy advisor governing a simulated nation.
Each turn you must choose EXACTLY ONE policy action.

AVAILABLE ACTIONS:
{actions}

RULES:
1. Respond with ONLY the action name. No explanation, no formatting, no quotes.
2. Consider delayed effects: education and renewable investments pay off 3-6 steps later.
3. Prevent collapse: GDP > 15, pollution < 290, satisfaction > 5.
4. Balance all three dimensions.

Respond with EXACTLY one action name. Nothing else."""


# ====================================================================
# Policy Reasoner (rule-based pre-filter for critical situations)
# ====================================================================

class PolicyReasoner:
    """Structured policy reasoning — overrides LLM in critical situations."""

    @staticmethod
    def check_critical(meta: Dict) -> Optional[str]:
        """Return override action if state is critical, else None."""
        gdp = meta.get("gdp_index", 100)
        sat = meta.get("public_satisfaction", 50)
        poll = meta.get("pollution_index", 100)

        if gdp < 30:
            return "stimulate_economy"
        if sat < 15:
            return "increase_welfare"
        if poll > 260:
            return "enforce_emission_limits"
        return None


# ====================================================================
# Observation formatting
# ====================================================================

def format_observation_v3(meta: Dict, step: int, max_steps: int) -> str:
    """Format observation with full negotiation context for the LLM."""
    lines = [
        f"--- STEP {step}/{max_steps} ---",
        "",
        "STATE:",
        f"  GDP: {meta.get('gdp_index', 0):.0f}/200",
        f"  Pollution: {meta.get('pollution_index', 0):.0f}/300",
        f"  Satisfaction: {meta.get('public_satisfaction', 0):.0f}/100",
        f"  Healthcare: {meta.get('healthcare_index', 0):.0f}/100",
        f"  Education: {meta.get('education_index', 0):.0f}/100",
        f"  Unemployment: {meta.get('unemployment_rate', 0):.1f}%",
        f"  Renewables: {meta.get('renewable_energy_ratio', 0):.0%}",
        f"  Inequality: {meta.get('inequality_index', 0):.0f}/100",
    ]

    events = meta.get("active_events", [])
    if events:
        lines.append(f"\nACTIVE EVENTS: {', '.join(events)}")

    # v3: Negotiation context
    negotiation = meta.get("negotiation_narrative", "")
    if negotiation:
        lines.append(f"\n{negotiation}")

    # v3: Active briefings
    briefings = meta.get("active_briefings", [])
    if briefings:
        lines.append("\nACTIVE BRIEFINGS:")
        for b in briefings:
            lines.append(f"  [{b['category'].upper()}] {b['text'][:150]}...")
            lines.append(f"    Deadline: step {b['deadline_step']} ({b['steps_remaining']} steps remaining)")

    # v3: New briefing
    new_briefing = meta.get("new_briefing", "")
    if new_briefing:
        lines.append(f"\nNEW INTELLIGENCE:\n{new_briefing}")

    lines.append("\nMake your decision:")
    return "\n".join(lines)


def format_observation_simple(meta: Dict, step: int, max_steps: int) -> str:
    """Simple observation for non-negotiation tasks."""
    events = ", ".join(meta.get("active_events", [])) or "none"
    return (
        f"--- STEP {step}/{max_steps} ---\n"
        f"GDP: {meta.get('gdp_index', 0):.0f}/200 | "
        f"Pollution: {meta.get('pollution_index', 0):.0f}/300 | "
        f"Satisfaction: {meta.get('public_satisfaction', 0):.0f}/100\n"
        f"Healthcare: {meta.get('healthcare_index', 0):.0f}/100 | "
        f"Education: {meta.get('education_index', 0):.0f}/100 | "
        f"Unemployment: {meta.get('unemployment_rate', 0):.1f}%\n"
        f"Renewables: {meta.get('renewable_energy_ratio', 0):.0%} | "
        f"Inequality: {meta.get('inequality_index', 0):.0f}/100\n"
        f"Events: {events}\n\nChoose your action:"
    )


# ====================================================================
# LLM Agent
# ====================================================================

def get_llm_action_v3(client: OpenAI, obs_text: str, model: str) -> Dict:
    """Query LLM for structured v3 action with negotiation."""
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT_V3.format(actions=ACTION_LIST_STR)},
                {"role": "user", "content": obs_text},
            ],
            temperature=0.1,
            max_tokens=200,
        )
        raw = response.choices[0].message.content.strip()

        # Strip markdown fences
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1].rsplit("```", 1)[0].strip()

        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            # Fallback: extract action name
            for action in VALID_ACTIONS:
                if action in raw.lower():
                    return {"action": action, "reasoning": "parse fallback"}
            return {"action": "no_action", "reasoning": "parse error"}

        # Validate action
        action = data.get("action", "no_action")
        if action not in VALID_ACTIONS:
            for a in VALID_ACTIONS:
                if a in action:
                    action = a
                    break
            else:
                action = "no_action"

        data["action"] = action
        # Ensure required fields
        data.setdefault("reasoning", "")
        data.setdefault("coalition_target", [])
        data.setdefault("negotiation_argument", "")
        data.setdefault("veto_prediction", [])
        data.setdefault("stance", "cooperative")
        return data

    except Exception as e:
        return {"action": "no_action", "reasoning": f"error: {e}"}


def get_llm_action_simple(client: OpenAI, obs_text: str, model: str) -> str:
    """Simple action selection for non-negotiation tasks."""
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT_SIMPLE.format(actions=ACTION_LIST_STR)},
                {"role": "user", "content": obs_text},
            ],
            temperature=0.0,
            max_tokens=50,
        )
        raw = response.choices[0].message.content.strip().lower().strip("'\"` \n")
        if raw in VALID_ACTIONS:
            return raw
        for a in VALID_ACTIONS:
            if a in raw:
                return a
        return "no_action"
    except Exception:
        return "no_action"


# ====================================================================
# Task Runner
# ====================================================================

def run_task(client: OpenAI, task_id: str, seed: int = SEED) -> Dict:
    """Run a single task with structured [START]/[STEP]/[END] output."""
    cfg = TASK_CONFIGS[task_id]
    max_steps = cfg["max_steps"]
    num_ministers = cfg.get("num_ministers", 1)
    use_negotiation = num_ministers >= 2
    reasoner = PolicyReasoner()

    print(f"[START] task={task_id}", flush=True)

    env = PolicyEnvironment()
    obs = env.reset(seed=seed, task_id=task_id)

    total_reward = 0.0
    step = 0
    step_log: List[Dict] = []

    while not obs.done:
        step += 1
        meta = obs.metadata

        # Critical override check
        override = reasoner.check_critical(meta)
        if override:
            action_data = {"action": override, "reasoning": "CRITICAL OVERRIDE"}
        elif use_negotiation:
            obs_text = format_observation_v3(meta, step, max_steps)
            action_data = get_llm_action_v3(client, obs_text, MODEL_NAME)
        else:
            obs_text = format_observation_simple(meta, step, max_steps)
            action_name = get_llm_action_simple(client, obs_text, MODEL_NAME)
            action_data = {"action": action_name}

        # Execute step
        obs = env.step(action_data)
        total_reward += obs.reward

        action_name = action_data.get("action", "no_action")
        print(f"[STEP] step={step} action={action_name} reward={obs.reward:.4f}", flush=True)

        # Log
        log_entry = {
            "step": step,
            "action": action_name,
            "reward": obs.reward,
        }
        if "reasoning" in action_data:
            log_entry["reasoning"] = action_data["reasoning"]
        if "coalition_target" in action_data and action_data["coalition_target"]:
            log_entry["coalition"] = action_data["coalition_target"]
        if "veto_prediction" in action_data and action_data["veto_prediction"]:
            log_entry["veto_prediction"] = action_data["veto_prediction"]

        # Log negotiation outcome
        outcome = obs.metadata.get("negotiation_outcome", {})
        if outcome:
            log_entry["coalition_formed"] = outcome.get("coalition_formed", False)
            log_entry["vetoed"] = outcome.get("vetoed", False)
            log_entry["tom_correct"] = outcome.get("veto_prediction_correct", False)
            log_entry["tom_reward"] = outcome.get("tom_reward", 0.0)

        step_log.append(log_entry)

    # Grade
    trajectory = env.get_trajectory()
    score = grade_trajectory(task_id, trajectory)
    collapsed = obs.metadata.get("collapsed", False)

    print(f"[END] task={task_id} score={score:.4f} steps={step} reward={total_reward:.4f} collapsed={collapsed}", flush=True)

    return {
        "task_id": task_id,
        "score": score,
        "total_reward": round(total_reward, 4),
        "steps": step,
        "collapsed": collapsed,
        "step_log": step_log,
        "briefing_stats": obs.metadata.get("briefing_stats", {}),
    }


# ====================================================================
# Main
# ====================================================================

def main() -> None:
    """Entry point — runs all tasks with structured output."""
    client = OpenAI(api_key=HF_TOKEN, base_url=API_BASE_URL)

    results: List[Dict] = []
    for task_id in get_task_ids():
        result = run_task(client, task_id, seed=SEED)
        results.append(result)


if __name__ == "__main__":
    main()
