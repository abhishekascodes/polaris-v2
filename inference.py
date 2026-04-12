#!/usr/bin/env python3
"""
POLARIS -- Inference Script for OpenEnv Hackathon

Runs an LLM agent against all governance tasks using structured policy
reasoning with LLM decision-making.

Environment Variables Required:
    API_BASE_URL  -- The API endpoint for the LLM (default: https://api.openai.com/v1)
    MODEL_NAME    -- The model identifier (default: gpt-4o)
    HF_TOKEN      -- Your Hugging Face / API key (NO default)

Usage:
    export API_BASE_URL="https://api.openai.com/v1"
    export MODEL_NAME="gpt-4o"
    export HF_TOKEN="sk-..."
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
# Configuration  (HF_TOKEN has NO default per hackathon rules)
# ====================================================================

API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o")
HF_TOKEN = os.getenv("HF_TOKEN")

# Optional — if you use from_docker_image():
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")

SEED = 42


# ====================================================================
# Policy Reasoning Engine (rule-based pre-filter)
# ====================================================================

class PolicyReasoner:
    """
    Structured policy reasoning layer that analyses the current state
    and produces a ranked shortlist of recommended actions with
    human-readable justifications.
    """

    @staticmethod
    def analyse(meta: Dict) -> Tuple[Optional[str], str, List[str]]:
        poll = meta.get("pollution_index", 100)
        gdp = meta.get("gdp_index", 100)
        sat = meta.get("public_satisfaction", 50)
        unemp = meta.get("unemployment_rate", 10)
        hc = meta.get("healthcare_index", 50)
        edu = meta.get("education_index", 50)
        rer = meta.get("renewable_energy_ratio", 0.15)
        ineq = meta.get("inequality_index", 40)
        events = meta.get("active_events", [])

        reasons = []
        shortlist = []
        override = None

        # -- CRITICAL: Collapse prevention --
        if gdp < 30:
            override = "stimulate_economy"
            reasons.append(f"GDP critically low at {gdp:.0f}")
            return override, "; ".join(reasons), ["stimulate_economy"]

        if sat < 15:
            override = "increase_welfare"
            reasons.append(f"Satisfaction critically low at {sat:.0f}")
            return override, "; ".join(reasons), ["increase_welfare"]

        if poll > 260:
            override = "enforce_emission_limits"
            reasons.append(f"Pollution at {poll:.0f} near collapse")
            return override, "; ".join(reasons), ["enforce_emission_limits"]

        # -- HIGH PRIORITY --
        if poll > 180:
            reasons.append(f"Pollution high ({poll:.0f})")
            shortlist.extend(["restrict_polluting_industries", "enforce_emission_limits", "implement_carbon_tax"])
        elif poll > 120:
            reasons.append(f"Pollution elevated ({poll:.0f})")
            shortlist.extend(["subsidize_renewables", "enforce_emission_limits"])

        if gdp < 50:
            reasons.append(f"GDP weak ({gdp:.0f})")
            shortlist.extend(["stimulate_economy", "decrease_tax", "reduce_interest_rates"])
        elif gdp < 70:
            reasons.append(f"GDP below target ({gdp:.0f})")
            shortlist.append("stimulate_economy")

        if sat < 30:
            reasons.append(f"Satisfaction low ({sat:.0f})")
            shortlist.extend(["increase_welfare", "invest_in_healthcare"])
        elif sat < 45:
            reasons.append(f"Satisfaction declining ({sat:.0f})")
            shortlist.append("increase_welfare")

        if unemp > 20:
            reasons.append(f"Unemployment high ({unemp:.0f}%)")
            shortlist.extend(["expand_industry", "stimulate_economy"])

        if hc < 35:
            reasons.append(f"Healthcare critically low ({hc:.0f})")
            shortlist.append("invest_in_healthcare")

        # -- STRATEGIC --
        if rer < 0.25 and poll > 100:
            reasons.append(f"Renewables low ({rer:.0%})")
            shortlist.extend(["subsidize_renewables", "incentivize_clean_tech"])

        if edu < 40:
            reasons.append(f"Education low ({edu:.0f})")
            shortlist.append("invest_in_education")

        if ineq > 55:
            reasons.append(f"Inequality high ({ineq:.0f})")
            if "increase_welfare" not in shortlist:
                shortlist.append("increase_welfare")

        # -- EVENT RESPONSE --
        for event in events:
            if event == "pandemic":
                shortlist.insert(0, "invest_in_healthcare")
            elif event == "economic_recession":
                shortlist.insert(0, "stimulate_economy")
            elif event == "climate_crisis":
                shortlist.insert(0, "enforce_emission_limits")
            elif event == "public_protest":
                shortlist.insert(0, "increase_welfare")

        # -- BALANCED --
        if not shortlist:
            reasons.append("Stable conditions")
            if rer < 0.3:
                shortlist.append("subsidize_renewables")
            if edu < 60:
                shortlist.append("invest_in_education")
            shortlist.append("incentivize_clean_tech")

        # Deduplicate
        seen = set()
        unique = []
        for a in shortlist:
            if a not in seen:
                seen.add(a)
                unique.append(a)
        shortlist = unique[:5]

        return override, "; ".join(reasons) if reasons else "Stable conditions", shortlist


# ====================================================================
# System prompt
# ====================================================================

SYSTEM_PROMPT = """You are an expert AI policy advisor governing a simulated nation.
Each turn you must choose EXACTLY ONE policy action.

AVAILABLE ACTIONS:
{actions}

RULES:
1. Respond with ONLY the action name. No explanation, no formatting, no quotes.
2. Consider delayed effects: education and renewable investments pay off 3-6 steps later.
3. Avoid oscillating between opposite actions.
4. Watch for events and adapt your strategy accordingly.
5. Prevent collapse: GDP > 15, pollution < 290, satisfaction > 5.
6. Balance all three dimensions.

Respond with EXACTLY one action name from the list above. Nothing else."""

ACTION_LIST = "\n".join(
    f"  - {name}: {desc}" for name, desc in ACTION_DESCRIPTIONS.items()
)


# ====================================================================
# Observation formatting
# ====================================================================

def format_observation(
    meta: Dict, step: int, max_steps: int,
    reasoning: str, shortlist: List[str],
) -> str:
    lines = [
        f"--- STEP {step}/{max_steps} ---",
        f"Task: {meta.get('task_description', 'N/A')}",
        "",
        "ENVIRONMENTAL",
        f"  Pollution:    {meta.get('pollution_index', 0):6.1f} / 300",
        f"  Carbon Rate:  {meta.get('carbon_emission_rate', 0):6.1f} / 100",
        f"  Renewables:   {meta.get('renewable_energy_ratio', 0):6.1%}",
        f"  Ecology:      {meta.get('ecological_stability', 0):6.1f} / 100",
        "",
        "ECONOMIC",
        f"  GDP:          {meta.get('gdp_index', 0):6.1f} / 200",
        f"  Industry:     {meta.get('industrial_output', 0):6.1f} / 100",
        f"  Unemployment: {meta.get('unemployment_rate', 0):6.1f}%",
        f"  Inflation:    {meta.get('inflation_rate', 0):6.1f}%",
        f"  Trade:        {meta.get('trade_balance', 0):6.1f}",
        f"  Investment:   {meta.get('foreign_investment', 0):6.1f} / 100",
        "",
        "SOCIAL",
        f"  Satisfaction: {meta.get('public_satisfaction', 0):6.1f} / 100",
        f"  Healthcare:   {meta.get('healthcare_index', 0):6.1f} / 100",
        f"  Education:    {meta.get('education_index', 0):6.1f} / 100",
        f"  Inequality:   {meta.get('inequality_index', 0):6.1f} / 100",
        "",
        f"POLICY ANALYSIS: {reasoning}",
        f"RECOMMENDED: {', '.join(shortlist)}",
        "",
        "Choose your action:",
    ]
    return "\n".join(lines)


# ====================================================================
# LLM agent
# ====================================================================

def get_llm_action(client: OpenAI, observation_text: str, model: str) -> str:
    """Query the LLM for a policy action."""
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT.format(actions=ACTION_LIST)},
                {"role": "user", "content": observation_text},
            ],
            temperature=0.0,
            max_tokens=50,
        )
        raw = response.choices[0].message.content.strip().lower()
        raw = raw.strip("'\"` \n")

        # Exact match first
        if raw in VALID_ACTIONS:
            return raw

        # Substring match
        for action in VALID_ACTIONS:
            if action in raw:
                return action

        return "no_action"

    except Exception as e:
        # Fallback to rule-based when LLM fails
        return "no_action"


# ====================================================================
# Task runner — with EXACT [START]/[STEP]/[END] structured output
# ====================================================================

def run_task(client: OpenAI, task_id: str, seed: int = SEED) -> Dict:
    """Run a single task and emit structured [START]/[STEP]/[END] logs."""
    cfg = TASK_CONFIGS[task_id]
    max_steps = cfg["max_steps"]
    reasoner = PolicyReasoner()

    # ═══════════════════════════════════════════════════════
    # [START] — EXACT format required by validator
    # ═══════════════════════════════════════════════════════
    print(f"[START] task={task_id}", flush=True)

    env = PolicyEnvironment()
    obs = env.reset(seed=seed, task_id=task_id)

    total_reward = 0.0
    step = 0
    step_log: List[Dict] = []

    while not obs.done:
        step += 1
        meta = obs.metadata

        # Policy reasoning
        override, reasoning, shortlist = reasoner.analyse(meta)

        if override:
            action = override
        else:
            obs_text = format_observation(meta, step, max_steps, reasoning, shortlist)
            action = get_llm_action(client, obs_text, MODEL_NAME)

        # Execute
        obs = env.step({"action": action})
        total_reward += obs.reward

        # ═══════════════════════════════════════════════════════
        # [STEP] — EXACT format required by validator
        # ═══════════════════════════════════════════════════════
        print(f"[STEP] step={step} action={action} reward={obs.reward:.4f}", flush=True)

        # Internal log
        step_log.append({
            "step": step,
            "action": action,
            "reward": obs.reward,
            "reasoning": reasoning,
        })

    # Grade
    trajectory = env.get_trajectory()
    score = grade_trajectory(task_id, trajectory)
    collapsed = obs.metadata.get("collapsed", False)

    # ═══════════════════════════════════════════════════════
    # [END] — EXACT format required by validator
    # ═══════════════════════════════════════════════════════
    print(f"[END] task={task_id} score={score:.4f} steps={step} reward={total_reward:.4f} collapsed={collapsed}", flush=True)

    return {
        "task_id": task_id,
        "score": score,
        "total_reward": round(total_reward, 4),
        "steps": step,
        "collapsed": collapsed,
        "step_log": step_log,
    }


# ====================================================================
# Main — entry point
# ====================================================================

def main() -> None:
    """Entry point — runs all tasks and emits structured output."""

    # Create OpenAI client using the required env vars
    client = OpenAI(api_key=HF_TOKEN, base_url=API_BASE_URL)

    results: List[Dict] = []

    for task_id in get_task_ids():
        result = run_task(client, task_id, seed=SEED)
        results.append(result)


if __name__ == "__main__":
    main()
