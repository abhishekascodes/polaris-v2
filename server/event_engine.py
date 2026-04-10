"""
AI Policy Engine — Intelligent Stochastic Event Engine (v2 — Nuclear Upgrade)

Upgrades:
  - All 8 events driven by continuous sigmoid/exponential state-dependent
    probability functions (not just linear adjustments).
  - Event chaining: one event probabilistically modifies threshold bias
    vectors for subsequent events during the episode.
  - Event memory: decaying bias vector (exponential decay over 15–25 steps)
    that affects future probability curves.
  - Chaos level also scales agent utility volatility & drift speed.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# ─────────────────────────────────────────────────────────────
# Event Definitions
# ─────────────────────────────────────────────────────────────

@dataclass
class EventType:
    """Template for a category of random event."""

    name: str
    description: str
    base_probability: float          # baseline per-step chance
    duration: int                    # how many steps it lasts
    per_step_deltas: Dict[str, float]  # applied EACH step while active
    onset_deltas: Dict[str, float] = field(default_factory=dict)
    # Chain effects: what biases this event injects when triggered
    chain_biases: Dict[str, float] = field(default_factory=dict)
    # Memory half-life in steps (controls decay speed)
    memory_halflife: int = 20


EVENT_TYPES: List[EventType] = [
    EventType(
        name="pandemic",
        description="A global pandemic strains healthcare and the economy.",
        base_probability=0.025,
        duration=5,
        per_step_deltas={
            "gdp_index": -2.5,
            "public_satisfaction": -3.0,
            "unemployment_rate": 1.5,
            "healthcare_index": -2.0,
        },
        onset_deltas={
            "public_satisfaction": -8.0,
            "foreign_investment": -5.0,
        },
        # Pandemic raises protest probability and weakens supply chain
        chain_biases={"public_protest": 0.6, "economic_recession": 0.4},
        memory_halflife=25,
    ),
    EventType(
        name="industrial_boom",
        description="An industrial boom drives rapid growth — and pollution.",
        base_probability=0.045,
        duration=3,
        per_step_deltas={
            "gdp_index": 2.5,
            "industrial_output": 3.0,
            "pollution_index": 4.0,
            "unemployment_rate": -1.5,
            "carbon_emission_rate": 2.0,
        },
        # Boom raises climate crisis probability
        chain_biases={"climate_crisis": 0.5},
        memory_halflife=15,
    ),
    EventType(
        name="climate_crisis",
        description="Extreme weather destabilises the ecosystem and public mood.",
        base_probability=0.035,
        duration=4,
        per_step_deltas={
            "pollution_index": 6.0,
            "ecological_stability": -5.0,
            "public_satisfaction": -2.0,
        },
        onset_deltas={
            "pollution_index": 10.0,
            "transport_efficiency": -5.0,
        },
        # Climate crisis raises natural disaster probability, spikes protests
        chain_biases={"natural_disaster": 0.7, "public_protest": 0.3},
        memory_halflife=20,
    ),
    EventType(
        name="public_protest",
        description="Widespread protests demand policy change.",
        base_probability=0.055,
        duration=2,
        per_step_deltas={
            "public_satisfaction": -4.0,
            "foreign_investment": -2.5,
            "regulation_strength": 1.5,
        },
        onset_deltas={
            "public_satisfaction": -5.0,
        },
        # Protests can cascade to recession if unresolved
        chain_biases={"economic_recession": 0.3},
        memory_halflife=15,
    ),
    EventType(
        name="tech_breakthrough",
        description="A clean-energy breakthrough accelerates green transition.",
        base_probability=0.035,
        duration=3,
        per_step_deltas={
            "renewable_energy_ratio": 0.025,
            "energy_efficiency": 2.5,
            "gdp_index": 1.0,
        },
        onset_deltas={
            "public_satisfaction": 5.0,
        },
        # Breakthrough reduces climate sensitivity for a while
        chain_biases={"climate_crisis": -0.4},
        memory_halflife=20,
    ),
    EventType(
        name="trade_war",
        description="International trade tensions disrupt commerce.",
        base_probability=0.030,
        duration=4,
        per_step_deltas={
            "trade_balance": -3.5,
            "foreign_investment": -2.5,
            "gdp_index": -1.5,
            "industrial_output": -1.0,
        },
        onset_deltas={
            "trade_balance": -8.0,
        },
        chain_biases={"economic_recession": 0.5, "public_protest": 0.2},
        memory_halflife=20,
    ),
    EventType(
        name="natural_disaster",
        description="A major natural disaster damages infrastructure.",
        base_probability=0.025,
        duration=2,
        per_step_deltas={
            "gdp_index": -3.0,
            "transport_efficiency": -4.0,
            "public_satisfaction": -5.0,
            "pollution_index": 4.0,
        },
        onset_deltas={
            "gdp_index": -5.0,
            "transport_efficiency": -5.0,
        },
        chain_biases={"public_protest": 0.5, "economic_recession": 0.3},
        memory_halflife=15,
    ),
    EventType(
        name="economic_recession",
        description="A recession contracts the economy and labour market.",
        base_probability=0.025,
        duration=5,
        per_step_deltas={
            "gdp_index": -2.0,
            "unemployment_rate": 1.2,
            "industrial_output": -1.5,
            "foreign_investment": -1.5,
            "inflation_rate": -0.5,
        },
        onset_deltas={
            "public_satisfaction": -6.0,
        },
        chain_biases={"public_protest": 0.4, "trade_war": 0.2},
        memory_halflife=25,
    ),
]

EVENT_BY_NAME = {et.name: et for et in EVENT_TYPES}


# ─────────────────────────────────────────────────────────────
# Active Event Instance
# ─────────────────────────────────────────────────────────────

@dataclass
class ActiveEvent:
    """A currently running event instance."""
    event_type: EventType
    remaining_steps: int
    triggered_at_step: int = 0


# ─────────────────────────────────────────────────────────────
# Event Engine
# ─────────────────────────────────────────────────────────────

class EventEngine:
    """
    Manages random event triggering, tracking, and application.

    Args:
        seed: Random seed for reproducibility.
        frequency_multiplier: Scales all event probabilities.
        satisfaction_event_scale: Scales satisfaction deltas from events.
    """

    def __init__(
        self,
        seed: int = 42,
        frequency_multiplier: float = 1.0,
        satisfaction_event_scale: float = 1.0,
    ):
        self._rng = random.Random(seed)
        self._freq_mult = frequency_multiplier
        self._sat_scale = satisfaction_event_scale
        self._active_events: List[ActiveEvent] = []
        # Memory bias vector: event_name -> current bias value (decays each step)
        self._memory_bias: Dict[str, float] = {et.name: 0.0 for et in EVENT_TYPES}
        self._step: int = 0
        # Utility volatility multiplier exposed to environment
        self.utility_volatility: float = 1.0
        # Event history for counterfactuals
        self._event_history: List[Tuple[int, str]] = []  # (step, event_name)

    def reset(
        self,
        seed: int = 42,
        frequency_multiplier: float = 1.0,
        satisfaction_event_scale: float = 1.0,
    ) -> None:
        """Reinitialise the engine for a new episode."""
        self._rng = random.Random(seed)
        self._freq_mult = frequency_multiplier
        self._sat_scale = satisfaction_event_scale
        self._active_events.clear()
        self._memory_bias = {et.name: 0.0 for et in EVENT_TYPES}
        self._step = 0
        self.utility_volatility = 1.0
        self._event_history = []

    # -----------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------

    def step(self, state: Dict[str, float]) -> List[str]:
        """
        Advance one step: trigger new events, apply active events, expire old,
        decay memory bias.

        Returns:
            List of names of currently active events (after this step).
        """
        newly_triggered = self._trigger_new_events(state)
        self._apply_active_events(state)
        self._expire_events()
        self._decay_memory_bias()
        self._update_utility_volatility()
        self._step += 1
        return self.active_event_names

    @property
    def active_event_names(self) -> List[str]:
        return [ae.event_type.name for ae in self._active_events]

    @property
    def event_history(self) -> List[Tuple[int, str]]:
        return list(self._event_history)

    def get_memory_bias(self, event_name: str) -> float:
        return self._memory_bias.get(event_name, 0.0)

    # -----------------------------------------------------------------
    # Internals
    # -----------------------------------------------------------------

    def _trigger_new_events(self, state: Dict[str, float]) -> List[str]:
        """Roll dice using sigmoid/exponential state-dependent probabilities."""
        if self._freq_mult <= 0:
            return []

        active_names = {ae.event_type.name for ae in self._active_events}
        newly_triggered = []

        for et in EVENT_TYPES:
            if et.name in active_names:
                continue

            # Get base probability, scaled by freq multiplier
            base_prob = et.base_probability * self._freq_mult

            # Apply sigmoid state-dependent probability
            prob = self._sigmoid_probability(et, state, base_prob)

            # Apply memory bias (can increase or decrease probability)
            memory_mod = self._memory_bias.get(et.name, 0.0)
            prob = max(0.0, min(0.5, prob + memory_mod * base_prob))

            if self._rng.random() < prob:
                self._active_events.append(
                    ActiveEvent(event_type=et, remaining_steps=et.duration,
                                triggered_at_step=self._step)
                )
                self._event_history.append((self._step, et.name))
                newly_triggered.append(et.name)

                # Apply one-time onset deltas
                for key, delta in et.onset_deltas.items():
                    d = delta * self._sat_scale if key == "public_satisfaction" else delta
                    state[key] = state.get(key, 0.0) + d

                # Apply chain biases to memory bias vector
                for target_name, bias_delta in et.chain_biases.items():
                    if target_name in self._memory_bias:
                        self._memory_bias[target_name] += bias_delta

        return newly_triggered

    def _sigmoid_probability(
        self, et: EventType, state: Dict[str, float], base_prob: float
    ) -> float:
        """
        Compute sigmoid/exponential state-dependent trigger probability.
        Each event has a tailored function of relevant state metrics.
        """
        prob = base_prob

        if et.name == "pandemic":
            # Exponential risk when healthcare is critically low
            hc = state.get("healthcare_index", 50)
            # sigmoid centered at hc=25: very low hc -> high prob
            risk_mod = _sigmoid(25 - hc, sharpness=0.15)
            prob = base_prob * (1.0 + risk_mod * 2.0)

        elif et.name == "industrial_boom":
            # Higher when GDP is moderate + low regulation
            gdp = state.get("gdp_index", 100)
            reg = state.get("regulation_strength", 40)
            econ_heat = _sigmoid(gdp - 80, sharpness=0.05) * _sigmoid(50 - reg, sharpness=0.08)
            prob = base_prob * (0.5 + 1.5 * econ_heat)

        elif et.name == "climate_crisis":
            # Exponential with pollution level
            poll = state.get("pollution_index", 100)
            eco = state.get("ecological_stability", 70)
            risk = _sigmoid(poll - 150, sharpness=0.03) * _sigmoid(50 - eco, sharpness=0.06)
            prob = base_prob * (0.5 + 2.0 * risk)

        elif et.name == "public_protest":
            # Strong sigmoid on satisfaction
            sat = state.get("public_satisfaction", 50)
            ineq = state.get("inequality_index", 40)
            risk = _sigmoid(30 - sat, sharpness=0.12) + 0.3 * _sigmoid(ineq - 55, sharpness=0.08)
            prob = base_prob * (0.5 + 2.0 * min(risk, 1.0))

        elif et.name == "tech_breakthrough":
            # Higher with more R&D investment
            green_sub = state.get("green_subsidies", 10)
            energy_eff = state.get("energy_efficiency", 50)
            boost = _sigmoid(green_sub - 30, sharpness=0.05) * 0.5 + \
                    _sigmoid(energy_eff - 60, sharpness=0.04) * 0.5
            prob = base_prob * (0.7 + 1.3 * boost)

        elif et.name == "trade_war":
            # Rises with trade imbalance and low foreign investment
            tb = state.get("trade_balance", 0)
            fi = state.get("foreign_investment", 50)
            risk = _sigmoid(-tb - 20, sharpness=0.05) * _sigmoid(40 - fi, sharpness=0.06)
            prob = base_prob * (0.7 + 1.5 * risk)

        elif et.name == "natural_disaster":
            # Rises with climate sensitivity (from drift) and low ecological stability
            eco = state.get("ecological_stability", 70)
            risk = _sigmoid(50 - eco, sharpness=0.05)
            prob = base_prob * (0.7 + 1.5 * risk)

        elif et.name == "economic_recession":
            # Rises with high inflation + high unemployment
            infl = state.get("inflation_rate", 3)
            unemp = state.get("unemployment_rate", 8)
            risk = _sigmoid(infl - 10, sharpness=0.10) * 0.6 + \
                   _sigmoid(unemp - 20, sharpness=0.08) * 0.4
            prob = base_prob * (0.5 + 2.0 * risk)

        return min(prob, 0.35)  # cap at 35% per step

    def _apply_active_events(self, state: Dict[str, float]) -> None:
        """Apply per-step deltas of every active event."""
        for ae in self._active_events:
            for key, delta in ae.event_type.per_step_deltas.items():
                d = delta * self._sat_scale if key == "public_satisfaction" else delta
                state[key] = state.get(key, 0.0) + d

    def _expire_events(self) -> None:
        """Decrement remaining steps; remove expired events."""
        for ae in self._active_events:
            ae.remaining_steps -= 1
        self._active_events = [ae for ae in self._active_events if ae.remaining_steps > 0]

    def _decay_memory_bias(self) -> None:
        """Exponentially decay all memory biases each step."""
        for name in self._memory_bias:
            et = EVENT_BY_NAME.get(name)
            if et is None:
                continue
            # Decay rate = ln(2) / halflife
            decay_rate = math.log(2) / max(et.memory_halflife, 1)
            self._memory_bias[name] *= math.exp(-decay_rate)
            # Clamp to avoid floating noise
            if abs(self._memory_bias[name]) < 0.001:
                self._memory_bias[name] = 0.0

    def _update_utility_volatility(self) -> None:
        """Update utility volatility based on number of active events."""
        n = len(self._active_events)
        # More active events -> higher volatility (caps at 2.5x)
        self.utility_volatility = 1.0 + 0.3 * min(n, 5)


# ─────────────────────────────────────────────────────────────
# Sigmoid helper
# ─────────────────────────────────────────────────────────────

def _sigmoid(x: float, sharpness: float = 0.1) -> float:
    """Sigmoid function: 0 when x<<0, 0.5 at x=0, 1 when x>>0."""
    try:
        return 1.0 / (1.0 + math.exp(-sharpness * x))
    except OverflowError:
        return 0.0 if x < 0 else 1.0
