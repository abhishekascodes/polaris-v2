"""
POLARIS v3 — Diplomatic Briefing Engine

Generates time-sensitive intelligence briefings that the LLM agent
must remember and act on across the episode. This directly tests
long-horizon memory and planning (Theme #2).

Briefings contain:
  - Timed threats with deadlines ("Finance plans tax revolt at step 30")
  - Opportunities with windows ("Trade deal if pollution < 100 in 10 steps")
  - Predictive warnings ("Pandemic likely in 15-25 steps")
  - Contradictory intelligence (hard mode — forces reasoning)

Rewards are granted/penalized based on whether the agent acted on
briefings at the right time.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class Briefing:
    """A single intelligence briefing."""
    id: str
    category: str              # "threat", "opportunity", "warning", "contradiction"
    text: str                  # Natural language content
    trigger_step: int          # When the briefing is delivered
    deadline_step: int         # When the condition is checked
    condition_metric: str      # Metric to check
    condition_op: str          # "gt", "lt", "eq"
    condition_value: float     # Threshold
    reward_if_met: float       # Bonus if agent acted on it
    penalty_if_missed: float   # Penalty if agent ignored it
    resolved: bool = False     # Has this been checked?
    delivered: bool = False     # Has this been shown to agent?


# ─────────────────────────────────────────────────────────────
# Briefing Templates
# ─────────────────────────────────────────────────────────────

BRIEFING_TEMPLATES = [
    # === THREATS ===
    {
        "category": "threat",
        "text": "🔴 INTELLIGENCE BRIEFING: {minister} is building a coalition against "
                "your environmental policies. If {metric_label} falls below {value:.0f} "
                "by step {deadline}, they will force a vote of no confidence.",
        "metric": "gdp_index",
        "op": "gt",
        "value_range": (70, 85),
        "window": (15, 30),
        "reward": 0.12,
        "penalty": -0.15,
    },
    {
        "category": "threat",
        "text": "🔴 CLASSIFIED: Social unrest indicators suggest public protests will "
                "escalate to riots if satisfaction drops below {value:.0f} by step {deadline}. "
                "Preventive welfare investment is critical.",
        "metric": "public_satisfaction",
        "op": "gt",
        "value_range": (30, 45),
        "window": (10, 25),
        "reward": 0.10,
        "penalty": -0.12,
    },
    # === OPPORTUNITIES ===
    {
        "category": "opportunity",
        "text": "🟢 DIPLOMATIC CABLE: A trade partner offers renewable technology transfer "
                "if pollution is reduced below {value:.0f} within {window} steps. "
                "This would boost renewable ratio by 8%.",
        "metric": "pollution_index",
        "op": "lt",
        "value_range": (90, 120),
        "window": (10, 20),
        "reward": 0.15,
        "penalty": -0.05,
    },
    {
        "category": "opportunity",
        "text": "🟢 ECONOMIC INTELLIGENCE: Foreign investors are watching. If GDP exceeds "
                "{value:.0f} by step {deadline}, a major investment fund will inject capital. "
                "Foreign investment would rise significantly.",
        "metric": "gdp_index",
        "op": "gt",
        "value_range": (90, 110),
        "window": (15, 30),
        "reward": 0.12,
        "penalty": -0.03,
    },
    # === WARNINGS ===
    {
        "category": "warning",
        "text": "⚠️ PREDICTIVE ANALYSIS: Climate models indicate an 85% probability of "
                "a severe climate crisis within {window} steps. Pre-positioning green "
                "infrastructure now will reduce impact by 60%. Target: pollution below {value:.0f}.",
        "metric": "pollution_index",
        "op": "lt",
        "value_range": (130, 160),
        "window": (15, 30),
        "reward": 0.10,
        "penalty": -0.08,
    },
    {
        "category": "warning",
        "text": "⚠️ HEALTH ADVISORY: Epidemiological data suggests pandemic risk is "
                "elevated. Healthcare index should be above {value:.0f} by step {deadline} "
                "to ensure adequate response capacity.",
        "metric": "healthcare_index",
        "op": "gt",
        "value_range": (45, 60),
        "window": (10, 20),
        "reward": 0.10,
        "penalty": -0.10,
    },
    {
        "category": "warning",
        "text": "⚠️ LABOR REPORT: Automation trends will displace workers. Unemployment "
                "must stay below {value:.0f}% by step {deadline} or industrial output will "
                "cascade downward.",
        "metric": "unemployment_rate",
        "op": "lt",
        "value_range": (12, 18),
        "window": (12, 25),
        "reward": 0.08,
        "penalty": -0.08,
    },
]


class BriefingEngine:
    """
    Generates and tracks diplomatic briefings across an episode.

    Usage:
        engine = BriefingEngine()
        engine.reset(seed=42, difficulty="hard", max_steps=200)
        briefing_text, reward = engine.step(current_step, state)
    """

    def __init__(self) -> None:
        self._briefings: List[Briefing] = []
        self._rng = random.Random(42)
        self._max_steps = 200
        self._generated = False

    def reset(
        self,
        seed: int = 42,
        difficulty: str = "medium",
        max_steps: int = 200,
        minister_names: Optional[List[str]] = None,
    ) -> None:
        """Generate briefings for a new episode."""
        self._rng = random.Random(seed + 55555)
        self._max_steps = max_steps
        self._briefings = []
        self._generated = False

        names = minister_names or [
            "Chancellor Voss", "Director Okafor", "Dr. Vasquez",
            "General Tanaka", "Senator Mwangi",
        ]

        # Number of briefings based on difficulty
        n_briefings = {
            "easy": 2,
            "medium": 4,
            "hard": 6,
            "extreme": 8,
        }.get(difficulty, 4)

        # Generate briefings spread across the episode
        templates = list(BRIEFING_TEMPLATES)
        self._rng.shuffle(templates)

        for i in range(min(n_briefings, len(templates))):
            tmpl = templates[i]
            # Trigger in first 60% of episode
            trigger = self._rng.randint(3, int(max_steps * 0.6))
            window_lo, window_hi = tmpl["window"]
            window = self._rng.randint(window_lo, window_hi)
            deadline = min(trigger + window, max_steps - 5)

            val_lo, val_hi = tmpl["value_range"]
            value = self._rng.uniform(val_lo, val_hi)

            metric_labels = {
                "gdp_index": "GDP",
                "pollution_index": "pollution",
                "public_satisfaction": "public satisfaction",
                "healthcare_index": "healthcare capacity",
                "unemployment_rate": "unemployment",
                "renewable_energy_ratio": "renewable ratio",
            }

            text = tmpl["text"].format(
                minister=self._rng.choice(names),
                metric_label=metric_labels.get(tmpl["metric"], tmpl["metric"]),
                value=value,
                deadline=deadline,
                window=window,
            )

            self._briefings.append(Briefing(
                id=f"briefing_{i}",
                category=tmpl["category"],
                text=text,
                trigger_step=trigger,
                deadline_step=deadline,
                condition_metric=tmpl["metric"],
                condition_op=tmpl["op"],
                condition_value=value,
                reward_if_met=tmpl["reward"],
                penalty_if_missed=tmpl["penalty"],
            ))

        self._generated = True

    def step(
        self, current_step: int, state: Dict[str, float]
    ) -> Tuple[str, float]:
        """
        Process briefings for the current step.

        Returns:
            (briefing_text, reward_delta)
            briefing_text: New briefings delivered this step (empty if none)
            reward_delta: Reward/penalty from briefings resolved this step
        """
        new_briefings: List[str] = []
        reward = 0.0

        for b in self._briefings:
            # Deliver briefing
            if not b.delivered and current_step >= b.trigger_step:
                b.delivered = True
                new_briefings.append(b.text)

            # Check deadline
            if b.delivered and not b.resolved and current_step >= b.deadline_step:
                b.resolved = True
                val = state.get(b.condition_metric, 0.0)
                met = False
                if b.condition_op == "gt":
                    met = val > b.condition_value
                elif b.condition_op == "lt":
                    met = val < b.condition_value

                if met:
                    reward += b.reward_if_met
                else:
                    reward += b.penalty_if_missed

        briefing_text = "\n\n".join(new_briefings) if new_briefings else ""
        return briefing_text, reward

    def get_active_briefings(self, current_step: int) -> List[Dict]:
        """Get all delivered but unresolved briefings."""
        active = []
        for b in self._briefings:
            if b.delivered and not b.resolved and current_step < b.deadline_step:
                active.append({
                    "id": b.id,
                    "category": b.category,
                    "text": b.text,
                    "deadline_step": b.deadline_step,
                    "steps_remaining": b.deadline_step - current_step,
                })
        return active

    def get_stats(self) -> Dict:
        """Get briefing statistics for the episode."""
        delivered = sum(1 for b in self._briefings if b.delivered)
        resolved = sum(1 for b in self._briefings if b.resolved)
        return {
            "total_briefings": len(self._briefings),
            "delivered": delivered,
            "resolved": resolved,
            "pending": delivered - resolved,
        }
