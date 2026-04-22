"""
AI Policy Engine — Task Definitions & Deterministic Graders

Three difficulty-tiered tasks with programmatic graders that
produce a normalised score in [0.0, 1.0].

Tasks:
  1. Environmental Recovery (Easy)  — 50 steps, no events
  2. Balanced Economy (Medium)      — 100 steps, reduced events
  3. Sustainable Governance (Hard)  — 200 steps, calibrated events
  4. Sustainable Governance Extreme — 200 steps, full events (structural collapse)

Graders evaluate the *full trajectory* (list of observation metadata
dicts) and final state to produce a single deterministic score.
"""

from __future__ import annotations

import statistics
from typing import Dict, List

from .config import TASK_CONFIGS, STATE_BOUNDS


# =====================================================================
# Utility helpers
# =====================================================================

def _norm(val: float, lo: float, hi: float) -> float:
    if hi <= lo:
        return 0.0
    return max(0.0, min(1.0, (val - lo) / (hi - lo)))


def _inv(val: float, lo: float, hi: float) -> float:
    return 1.0 - _norm(val, lo, hi)


# =====================================================================
# Per-task graders
# =====================================================================

def grade_environmental_recovery(trajectory: List[Dict]) -> float:
    """
    Task 1 grader — Environmental Recovery (Easy).

    Scoring breakdown:
      50 %  Final pollution reduction (1.0 if ≤ 80)
      25 %  GDP preservation (1.0 if ≥ 60)
      15 %  Trajectory trend (pollution decreasing over time)
      10 %  Ecological stability recovery
    """
    if not trajectory:
        return 0.0

    final = trajectory[-1]

    # -- Pollution target --
    poll = final.get("pollution_index", 300)
    if poll <= 80:
        pollution_score = 1.0
    elif poll <= 120:
        pollution_score = 1.0 - (poll - 80) / 40 * 0.4   # 1.0 -> 0.6
    elif poll <= 200:
        pollution_score = 0.6 - (poll - 120) / 80 * 0.4   # 0.6 -> 0.2
    else:
        pollution_score = max(0.0, 0.2 - (poll - 200) / 100 * 0.2)

    # -- GDP preservation --
    gdp = final.get("gdp_index", 0)
    if gdp >= 60:
        gdp_score = 1.0
    else:
        gdp_score = max(0.0, gdp / 60)

    # -- Trajectory trend (pollution should decline) --
    polls = [t.get("pollution_index", 220) for t in trajectory]
    if len(polls) >= 3:
        first_third = statistics.mean(polls[: len(polls) // 3])
        last_third = statistics.mean(polls[-(len(polls) // 3):])
        improvement = (first_third - last_third) / max(first_third, 1)
        trend_score = max(0.0, min(1.0, improvement / 0.5))
    else:
        trend_score = 0.0

    # -- Ecological stability --
    eco = final.get("ecological_stability", 0)
    eco_score = _norm(eco, 0, 100)

    score = (
        0.50 * pollution_score
        + 0.25 * gdp_score
        + 0.15 * trend_score
        + 0.10 * eco_score
    )
    return round(max(0.0, min(1.0, score)), 4)


def grade_balanced_economy(trajectory: List[Dict]) -> float:
    """
    Task 2 grader — Balanced Economy (Medium).

    Scoring breakdown:
      50 %  Fraction of steps where ALL THREE criteria are met
            (GDP > 80, pollution < 100, satisfaction > 60)
      30 %  Final state composite score
      20 %  Minimum-metric floor (worst metric across the trajectory)
    """
    if not trajectory:
        return 0.0

    # -- Steps balanced --
    balanced_steps = 0
    for t in trajectory:
        gdp_ok = t.get("gdp_index", 0) > 80
        poll_ok = t.get("pollution_index", 300) < 100
        sat_ok = t.get("public_satisfaction", 0) > 60
        if gdp_ok and poll_ok and sat_ok:
            balanced_steps += 1
    steps_score = balanced_steps / len(trajectory)

    # -- Final composite --
    final = trajectory[-1]
    gdp_s = _norm(final.get("gdp_index", 0), 0, 200)
    poll_s = _inv(final.get("pollution_index", 300), 0, 300)
    sat_s = _norm(final.get("public_satisfaction", 0), 0, 100)
    hc_s = _norm(final.get("healthcare_index", 0), 0, 100)
    edu_s = _norm(final.get("education_index", 0), 0, 100)
    final_score = 0.30 * gdp_s + 0.25 * poll_s + 0.25 * sat_s + 0.10 * hc_s + 0.10 * edu_s

    # -- Worst-metric floor --
    min_vals = []
    for t in trajectory:
        min_vals.append(min(
            _norm(t.get("gdp_index", 0), 0, 200),
            _inv(t.get("pollution_index", 300), 0, 300),
            _norm(t.get("public_satisfaction", 0), 0, 100),
        ))
    floor_score = statistics.mean(min_vals) if min_vals else 0.0

    score = 0.50 * steps_score + 0.30 * final_score + 0.20 * floor_score
    return round(max(0.0, min(1.0, score)), 4)


def grade_sustainable_governance(trajectory: List[Dict]) -> float:
    """
    Task 3 grader — Sustainable Governance (Hard).

    Scoring breakdown:
      25 %  No collapse (binary: 1.0 if survived, 0.0 if collapsed)
      30 %  Multi-metric balance across entire trajectory
      25 %  Low volatility (standard deviation of key metrics)
      20 %  Event resilience (recovery speed after shocks)
    """
    if not trajectory:
        return 0.0

    # -- Collapse check --
    collapsed = False
    for t in trajectory:
        if t.get("gdp_index", 100) < 15:
            collapsed = True
        if t.get("pollution_index", 100) > 290:
            collapsed = True
        if t.get("public_satisfaction", 50) < 5:
            collapsed = True
    collapse_score = 0.0 if collapsed else 1.0

    # -- Multi-metric balance (average composite score per step) --
    step_scores = []
    for t in trajectory:
        econ = _norm(t.get("gdp_index", 0), 0, 200)
        env = _inv(t.get("pollution_index", 300), 0, 300)
        soc = _norm(t.get("public_satisfaction", 0), 0, 100)
        hc = _norm(t.get("healthcare_index", 0), 0, 100)
        rer = _norm(t.get("renewable_energy_ratio", 0), 0, 1)
        composite = 0.25 * econ + 0.25 * env + 0.25 * soc + 0.15 * hc + 0.10 * rer
        step_scores.append(composite)
    balance_score = statistics.mean(step_scores) if step_scores else 0.0

    # -- Volatility (lower is better) --
    metric_stds: List[float] = []
    for key in ["gdp_index", "pollution_index", "public_satisfaction",
                "healthcare_index", "unemployment_rate"]:
        lo, hi = STATE_BOUNDS.get(key, (0, 100))
        span = hi - lo if hi > lo else 1.0
        values = [t.get(key, 0) / span for t in trajectory]
        if len(values) >= 2:
            metric_stds.append(statistics.stdev(values))
    avg_std = statistics.mean(metric_stds) if metric_stds else 0.0
    # avg_std of 0.0 -> 1.0;  avg_std ≥ 0.12 -> 0.0
    volatility_score = max(0.0, min(1.0, 1.0 - avg_std / 0.12))

    # -- Event resilience (detect drops and measure recovery) --
    resilience = _compute_resilience(trajectory)

    score = (
        0.25 * collapse_score
        + 0.30 * balance_score
        + 0.25 * volatility_score
        + 0.20 * resilience
    )
    return round(max(0.0, min(1.0, score)), 4)


def _compute_resilience(trajectory: List[Dict]) -> float:
    """
    Measure how quickly metrics recover after event-induced drops.

    Looks at GDP drops > 5 points in a single step, then measures
    how many steps it takes to recover to pre-drop levels.
    """
    if len(trajectory) < 5:
        return 0.5  # neutral

    gdp_vals = [t.get("gdp_index", 100) for t in trajectory]
    drops: List[float] = []

    for i in range(1, len(gdp_vals)):
        drop = gdp_vals[i - 1] - gdp_vals[i]
        if drop > 5:
            # Find recovery: first step where GDP ≥ pre-drop level
            pre = gdp_vals[i - 1]
            recovered = False
            for j in range(i + 1, min(i + 15, len(gdp_vals))):
                if gdp_vals[j] >= pre * 0.95:  # 95 % recovery threshold
                    speed = 1.0 - (j - i) / 15.0
                    drops.append(max(0.0, speed))
                    recovered = True
                    break
            if not recovered:
                drops.append(0.0)

    if not drops:
        return 0.8  # no drops = resilient by default
    return statistics.mean(drops)


def grade_negotiation_arena(trajectory: List[Dict]) -> float:
    """
    Task grader — Negotiation Arena.

    Scoring breakdown:
      25 %  No collapse (binary)
      25 %  Multi-metric balance
      20 %  Negotiation quality (coalition formation + ToM accuracy)
      15 %  Briefing compliance
      15 %  Volatility control
    """
    if not trajectory:
        return 0.0

    # -- Collapse check --
    collapsed = any(
        t.get("gdp_index", 100) < 15
        or t.get("pollution_index", 100) > 290
        or t.get("public_satisfaction", 50) < 5
        for t in trajectory
    )
    collapse_score = 0.0 if collapsed else 1.0

    # -- Multi-metric balance --
    step_scores = []
    for t in trajectory:
        econ = _norm(t.get("gdp_index", 0), 0, 200)
        env = _inv(t.get("pollution_index", 300), 0, 300)
        soc = _norm(t.get("public_satisfaction", 0), 0, 100)
        composite = 0.33 * econ + 0.34 * env + 0.33 * soc
        step_scores.append(composite)
    balance_score = statistics.mean(step_scores) if step_scores else 0.0

    # -- Negotiation quality --
    coalitions_formed = 0
    tom_correct = 0
    tom_total = 0
    for t in trajectory:
        outcome = t.get("negotiation_outcome", {})
        if outcome.get("coalition_formed"):
            coalitions_formed += 1
        if "veto_prediction_correct" in outcome:
            tom_total += 1
            if outcome["veto_prediction_correct"]:
                tom_correct += 1

    coalition_rate = coalitions_formed / max(len(trajectory), 1)
    tom_accuracy = tom_correct / max(tom_total, 1) if tom_total > 0 else 0.5
    negotiation_score = 0.5 * min(1.0, coalition_rate * 2) + 0.5 * tom_accuracy

    # -- Briefing compliance --
    briefing_stats = trajectory[-1].get("briefing_stats", {}) if trajectory else {}
    total_b = briefing_stats.get("total_briefings", 0)
    resolved_b = briefing_stats.get("resolved", 0)
    briefing_score = resolved_b / max(total_b, 1) if total_b > 0 else 0.5

    # -- Volatility --
    metric_stds: List[float] = []
    for key in ["gdp_index", "pollution_index", "public_satisfaction"]:
        lo, hi = STATE_BOUNDS.get(key, (0, 100))
        span = hi - lo if hi > lo else 1.0
        values = [t.get(key, 0) / span for t in trajectory]
        if len(values) >= 2:
            metric_stds.append(statistics.stdev(values))
    avg_std = statistics.mean(metric_stds) if metric_stds else 0.0
    volatility_score = max(0.0, min(1.0, 1.0 - avg_std / 0.12))

    score = (
        0.25 * collapse_score
        + 0.25 * balance_score
        + 0.20 * negotiation_score
        + 0.15 * briefing_score
        + 0.15 * volatility_score
    )
    return round(max(0.0, min(1.0, score)), 4)


# =====================================================================
# Public grading API
# =====================================================================

GRADERS = {
    "environmental_recovery": grade_environmental_recovery,
    "balanced_economy": grade_balanced_economy,
    "sustainable_governance": grade_sustainable_governance,
    "sustainable_governance_extreme": grade_sustainable_governance,
    "multi_agent_council": grade_negotiation_arena,
    "negotiation_arena": grade_negotiation_arena,
}


def grade_trajectory(task_id: str, trajectory: List[Dict]) -> float:
    """
    Score a trajectory for the given task. Returns 0.0–1.0.

    Args:
        task_id: One of the registered task IDs.
        trajectory: List of observation metadata dicts (one per step).

    Raises:
        ValueError: If task_id is unknown.
    """
    grader = GRADERS.get(task_id)
    if grader is None:
        raise ValueError(
            f"Unknown task_id '{task_id}'. Valid: {list(GRADERS.keys())}"
        )
    return grader(trajectory)


def get_task_ids() -> List[str]:
    """Return all available task IDs."""
    return list(TASK_CONFIGS.keys())

