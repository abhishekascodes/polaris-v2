"""
AI Policy Engine — Non-Stationary Drift Engine

Implements slow, cumulative drift on 6 global latent variables that make
the world progressively non-stationary throughout an episode.

Variables drifted:
  climate_sensitivity     — multiplier on climate event impacts
  inequality_tolerance    — public tolerance of inequality (reduces protest threshold)
  public_trust_decay      — per-step satisfaction drain (baseline entropy)
  supply_chain_resilience — dampens trade/industry shocks
  institutional_trust     — slow-moving global trust scalar
  policy_fatigue          — reduces marginal effect of repeatedly used actions

Drift speed scales with cumulative chaos exposure:
  speed = base_speed x (1 + cumulative_chaos / CHAOS_SCALE)
"""

from __future__ import annotations

import math
import random
from typing import Dict, List, Tuple

from .config import DRIFT_BOUNDS, DRIFT_INITIAL

CHAOS_SCALE = 50.0      # chaos exposure units that double drift speed
BASE_DRIFT_SPEED = 0.01 # base per-step Gaussian sigma for each variable


class DriftEngine:
    """
    Manages non-stationary drift of latent world variables.

    Usage:
        engine = DriftEngine()
        engine.reset(seed=42, drift_enabled=True, chaos_level=0.5)
        drift_vars = engine.step(chaos_this_step=1)
    """

    def __init__(self) -> None:
        self._vars: Dict[str, float] = {}
        self._cumulative_chaos: float = 0.0
        self._enabled: bool = False
        self._base_chaos: float = 0.0
        self._rng: random.Random = random.Random(42)
        self._step: int = 0

    def reset(
        self,
        seed: int = 42,
        drift_enabled: bool = True,
        chaos_level: float = 0.5,
    ) -> None:
        """Reinitialise drift state for a new episode."""
        self._rng = random.Random(seed + 31337)
        self._enabled = drift_enabled
        self._base_chaos = chaos_level
        self._cumulative_chaos = 0.0
        self._step = 0
        # Slightly randomise initial drift values around defaults
        self._vars = {}
        for k, v in DRIFT_INITIAL.items():
            lo, hi = DRIFT_BOUNDS[k]
            jitter = (self._rng.random() - 0.5) * 0.05 * (hi - lo)
            self._vars[k] = max(lo, min(hi, v + jitter))

    def step(self, chaos_this_step: int = 0) -> Dict[str, float]:
        """
        Advance drift by one step.

        Args:
            chaos_this_step: number of major events that fired this step

        Returns:
            current drift variable values
        """
        if not self._enabled:
            return dict(self._vars)

        self._cumulative_chaos += chaos_this_step * self._base_chaos
        drift_speed = BASE_DRIFT_SPEED * (1.0 + self._cumulative_chaos / CHAOS_SCALE)
        drift_speed = min(drift_speed, 0.05)  # cap at 5x base speed

        for k in self._vars:
            lo, hi = DRIFT_BOUNDS[k]
            span = hi - lo

            # Gaussian noise with mean-reversion towards initial value
            noise = self._rng.gauss(0, drift_speed * span)

            # Bias: push towards initial value when drifting far away
            init = DRIFT_INITIAL[k]
            reversion = 0.005 * (init - self._vars[k])

            self._vars[k] = max(lo, min(hi, self._vars[k] + noise + reversion))

        self._step += 1
        return dict(self._vars)

    def get(self, key: str, default: float = 1.0) -> float:
        """Get current value of a drift variable."""
        return self._vars.get(key, default)

    def get_all(self) -> Dict[str, float]:
        """Return copy of all current drift values."""
        return dict(self._vars)

    def record_chaos(self, n_events: int) -> None:
        """Record chaos events for compounding drift speed."""
        self._cumulative_chaos += n_events * self._base_chaos

    def compute_risk_heatmap(
        self,
        state: Dict[str, float],
        horizons: Tuple[int, int, int] = (5, 10, 20),
    ) -> List[float]:
        """
        Compute per-objective collapse probability estimates at 3 time horizons.

        Uses exponential decay models to estimate 5 risk dimensions:
          [gdp_risk, eco_risk, social_risk, health_risk, energy_risk]
        at each of the 3 horizons -> 15 floats total.

        This is a lightweight analytical approximation (not simulation).
        """
        result = []
        trust = self._vars.get("institutional_trust", 0.6)
        resilience = self._vars.get("supply_chain_resilience", 0.7)
        fatigue = self._vars.get("policy_fatigue", 0.1)

        for horizon in horizons:
            # GDP risk: exponential approach to collapse threshold
            gdp = state.get("gdp_index", 100)
            gdp_margin = (gdp - 15.0) / max(gdp, 1.0)
            gdp_risk = self._sigmoid_risk(gdp_margin, threshold=0.15, sharpness=10, horizon=horizon)

            # Eco / pollution risk
            poll = state.get("pollution_index", 100)
            eco_margin = (290.0 - poll) / 290.0
            eco_risk = self._sigmoid_risk(eco_margin, threshold=0.10, sharpness=8, horizon=horizon)

            # Social / satisfaction risk
            sat = state.get("public_satisfaction", 50)
            sat_margin = (sat - 5.0) / max(sat, 1.0)
            sat_risk = self._sigmoid_risk(sat_margin, threshold=0.12, sharpness=12, horizon=horizon)

            # Health risk
            hc = state.get("healthcare_index", 50)
            hc_risk = self._sigmoid_risk(hc / 100.0, threshold=0.30, sharpness=6, horizon=horizon)

            # Energy / stability risk (proxy: low energy efficiency + high fatigue)
            ee = state.get("energy_efficiency", 50)
            energy_risk = (1.0 - ee / 100.0) * fatigue * (1.0 - trust)
            energy_risk = max(0.0, min(1.0, energy_risk))

            # Scale risks by resilience factor
            scale = 1.0 + (1.0 - resilience) * 0.5
            result.extend([
                min(1.0, gdp_risk * scale),
                min(1.0, eco_risk * scale),
                min(1.0, sat_risk * scale),
                min(1.0, hc_risk),
                min(1.0, energy_risk),
            ])

        return result  # 15 floats

    @staticmethod
    def _sigmoid_risk(
        margin: float, threshold: float, sharpness: float, horizon: int
    ) -> float:
        """
        Sigmoid-based collapse probability.
        margin: normalised distance from collapse (1.0 = safe, 0.0 = at collapse)
        threshold: margin below which risk accelerates
        """
        # Adjust for horizon: longer horizon = higher cumulative risk
        horizon_factor = 1.0 + math.log(max(horizon, 1)) / 5.0
        x = (threshold - margin) * sharpness * horizon_factor
        try:
            prob = 1.0 / (1.0 + math.exp(-x))
        except OverflowError:
            prob = 0.0 if x < 0 else 1.0
        return max(0.0, min(1.0, prob))
