"""
AI Policy Engine — Multi-Layer Transition Engine (v2 — Nuclear Upgrade)

Implements 4 distinct transition layers that execute sequentially
each step, producing emergent, realistic dynamics:

  Layer 1 — Deterministic: Direct, immediate effects of each action,
             now with multi-metric simultaneous impacts and cross-metric
             side-effects for richer trade-off space.
  Layer 2 — Non-linear:    Threshold-based exponential / quadratic effects,
             plus new cross-layer interactions (pollution->education ROI,
             education amplifier).
  Layer 3 — Delayed:       Queued effects that materialise after N steps,
             with state-dependent health multipliers applied at fire time.
  Layer 4 — Feedback:      Systemic loops including new multi-hop
             pollution->education ROI->GDP drag.

After all layers, metric values are clamped to their configured bounds.
"""

from __future__ import annotations

from collections import deque
from typing import Deque, Dict, List, Optional, Tuple

from .config import STATE_BOUNDS

# A delayed effect: (step_when_it_fires, dict_of_deltas, health_mult_enabled)
DelayedEffect = Tuple[int, Dict[str, float], bool]


class TransitionEngine:
    """
    Applies all four transition layers to advance the world state by one step.

    Usage:
        engine = TransitionEngine()
        engine.reset()
        engine.apply(state, action, step, drift_vars=drift_vars)
    """

    def __init__(self) -> None:
        self._delayed_queue: Deque[DelayedEffect] = deque()
        # Track sustained pollution > 180 for cross-layer feedback
        self._high_pollution_steps: int = 0
        # Education delayed effect reduction flag
        self._edu_roi_reduced: bool = False

    def reset(self) -> None:
        """Clear all delayed effects (called on environment reset)."""
        self._delayed_queue.clear()
        self._high_pollution_steps = 0
        self._edu_roi_reduced = False

    # -----------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------

    def apply(
        self,
        state: Dict[str, float],
        action: str,
        step: int,
        drift_vars: Optional[Dict[str, float]] = None,
        policy_fatigue: float = 0.0,
    ) -> int:
        """
        Apply all four layers in order, then clamp to bounds.

        Returns:
            resolved_delayed_count — number of delayed effects fired this step
        """
        dv = drift_vars or {}
        fatigue_mult = 1.0 - 0.3 * policy_fatigue  # repeated actions lose up to 30% effect

        self._layer1_deterministic(state, action, step, fatigue_mult)
        self._layer2_nonlinear(state, dv)
        resolved = self._layer3_delayed(state, step)
        self._layer4_feedback(state, dv)
        self._clamp(state)
        return resolved

    # =================================================================
    # Layer 1 — Deterministic Effects (enhanced multi-metric impacts)
    # =================================================================

    def _layer1_deterministic(
        self, s: Dict[str, float], action: str, step: int, fatigue_mult: float = 1.0
    ) -> None:
        if action == "no_action":
            return

        f = fatigue_mult  # alias

        if action == "increase_tax":
            s["tax_rate"] += 3.0
            s["gdp_index"] -= 1.5 * f
            s["industrial_output"] -= 2.0 * f
            s["foreign_investment"] -= 3.0 * f
            s["welfare_spending"] += 2.0
            s["public_satisfaction"] -= 2.0 * f
            # NEW: tax hike also raises inequality slightly (capital owners less impacted)
            s["inequality_index"] += 0.5

        elif action == "decrease_tax":
            s["tax_rate"] -= 3.0
            s["gdp_index"] += 1.5 * f
            s["industrial_output"] += 2.0 * f
            s["foreign_investment"] += 3.0 * f
            s["welfare_spending"] -= 2.0
            s["inequality_index"] += 1.5 * f
            # NEW: lower taxes -> slight public satisfaction bump (lower cost of living)
            s["public_satisfaction"] += 0.5 * f

        elif action == "stimulate_economy":
            s["gdp_index"] += 3.0 * f
            s["industrial_output"] += 2.0 * f
            s["inflation_rate"] += 1.5 * f
            s["unemployment_rate"] -= 2.0 * f
            s["pollution_index"] += 5.0
            s["carbon_emission_rate"] += 3.0
            # NEW: stimulus boosts trade balance slightly
            s["trade_balance"] += 1.0 * f

        elif action == "reduce_interest_rates":
            s["interest_rate"] -= 1.0
            s["gdp_index"] += 2.0 * f
            s["inflation_rate"] += 1.0 * f
            s["foreign_investment"] -= 2.0 * f
            s["trade_balance"] -= 2.0 * f
            # NEW: lower rates -> industry expands -> slight pollution uptick
            s["pollution_index"] += 1.5

        elif action == "expand_industry":
            s["industrial_output"] += 5.0 * f
            s["gdp_index"] += 3.0 * f
            s["pollution_index"] += 8.0
            s["carbon_emission_rate"] += 5.0
            s["unemployment_rate"] -= 3.0 * f
            # NEW: industrial expansion also raises trade balance
            s["trade_balance"] += 2.0 * f
            # NEW: satisfaction boost from jobs (short-term)
            s["public_satisfaction"] += 1.0 * f

        elif action == "restrict_polluting_industries":
            s["industrial_output"] -= 4.0 * f
            s["pollution_index"] -= 6.0
            s["carbon_emission_rate"] -= 5.0
            s["unemployment_rate"] += 1.5 * f
            s["gdp_index"] -= 2.0 * f
            s["public_satisfaction"] += 0.5
            # NEW: reduces ecological damage
            s["ecological_stability"] += 1.0

        elif action == "incentivize_clean_tech":
            s["gdp_index"] -= 1.0 * f
            s["green_subsidies"] += 5.0
            # Delayed green tech payoff (state-dep multiplier at fire time)
            self._enqueue_delayed(step + 3, {
                "renewable_energy_ratio": 0.03,
                "pollution_index": -3.0,
                "energy_efficiency": 2.0,
            }, health_mult=True)
            # NEW: small immediate satisfaction from R&D news
            s["public_satisfaction"] += 0.3

        elif action == "enforce_emission_limits":
            s["regulation_strength"] += 5.0
            s["pollution_index"] -= 4.0
            s["carbon_emission_rate"] -= 4.0
            s["industrial_output"] -= 2.0 * f
            s["foreign_investment"] -= 2.0 * f
            s["public_satisfaction"] += 0.5
            # NEW: limits reduce ecological damage
            s["ecological_stability"] += 0.8

        elif action == "subsidize_renewables":
            s["green_subsidies"] += 8.0
            s["gdp_index"] -= 1.5 * f
            # Larger delayed green payoff
            self._enqueue_delayed(step + 4, {
                "renewable_energy_ratio": 0.05,
                "pollution_index": -5.0,
                "energy_efficiency": 3.0,
                "carbon_emission_rate": -3.0,
            }, health_mult=True)
            # NEW: also improves energy security -> small satisfaction
            s["public_satisfaction"] += 0.5

        elif action == "implement_carbon_tax":
            s["pollution_index"] -= 5.0
            s["carbon_emission_rate"] -= 6.0
            s["industrial_output"] -= 3.0 * f
            s["gdp_index"] -= 2.0 * f
            s["foreign_investment"] -= 3.0 * f
            s["tax_rate"] += 2.0
            s["public_satisfaction"] += 0.3
            # NEW: carbon tax revenue -> delayed welfare dividend
            self._enqueue_delayed(step + 2, {
                "welfare_spending": 1.5,
                "public_satisfaction": 0.5,
            }, health_mult=False)

        elif action == "increase_welfare":
            s["welfare_spending"] += 5.0
            s["public_satisfaction"] += 3.0
            s["inequality_index"] -= 2.0
            s["gdp_index"] -= 1.0 * f
            # NEW: welfare also slightly boosts healthcare access
            s["healthcare_index"] += 0.5

        elif action == "invest_in_healthcare":
            s["gdp_index"] -= 1.0 * f
            s["public_satisfaction"] += 1.0
            self._enqueue_delayed(step + 2, {
                "healthcare_index": 5.0,
                "public_satisfaction": 1.5,
                # NEW: healthcare boosts productivity -> slight GDP
                "gdp_index": 0.5,
            }, health_mult=True)

        elif action == "invest_in_education":
            s["gdp_index"] -= 1.0 * f
            # Education returns reduced if pollution is high (cross-layer)
            edu_mult = 0.7 if self._edu_roi_reduced else 1.0
            self._enqueue_delayed(step + 3, {
                "education_index": 4.0 * edu_mult,
                "inequality_index": -1.0 * edu_mult,
            }, health_mult=True)
            # Long-term innovation dividend
            self._enqueue_delayed(step + 6, {
                "gdp_index": 2.0 * edu_mult,
                "industrial_output": 1.0,
            }, health_mult=True)

        elif action == "upgrade_energy_grid":
            s["gdp_index"] -= 1.5 * f
            self._enqueue_delayed(step + 3, {
                "energy_efficiency": 6.0,
                "renewable_energy_ratio": 0.03,
                "pollution_index": -2.0,
                # NEW: grid upgrade reduces carbon
                "carbon_emission_rate": -1.5,
            }, health_mult=True)

        elif action == "invest_in_transport":
            s["gdp_index"] -= 1.0 * f
            self._enqueue_delayed(step + 3, {
                "transport_efficiency": 5.0,
                "gdp_index": 1.0,
                "pollution_index": -2.0,
                # NEW: better transport reduces inequality (access)
                "inequality_index": -0.5,
            }, health_mult=True)

        # Meta-actions have lighter direct effects;
        # their full impact is handled by the council and environment.
        elif action == "propose_global_policy_package":
            # Small immediate negotiation benefit to institutional trust (handled externally)
            # GDP cost of deliberation process
            s["gdp_index"] -= 0.5
            s["public_satisfaction"] += 1.0  # people like collaborative governance

        elif action == "force_emergency_coalition_vote":
            # Forces rapid decision — political cost
            s["public_satisfaction"] -= 1.5
            s["regulation_strength"] += 2.0

        elif action == "reset_institutional_trust":
            # High cost, high potential reward if trust rebuilds
            s["gdp_index"] -= 8.0
            s["public_satisfaction"] -= 5.0
            # Trust reset itself handled by environment (sets institutional_trust via drift_engine)

    # =================================================================
    # Layer 2 — Non-linear / Threshold Effects (enhanced)
    # =================================================================

    def _layer2_nonlinear(
        self, s: Dict[str, float], drift_vars: Dict[str, float]
    ) -> None:
        climate_sens = drift_vars.get("climate_sensitivity", 1.0)
        ineq_tol = drift_vars.get("inequality_tolerance", 0.6)

        # ── Pollution catastrophe ──
        if s["pollution_index"] > 200:
            excess = s["pollution_index"] - 200
            s["healthcare_index"] -= excess * 0.15 * climate_sens
            s["ecological_stability"] -= excess * 0.10 * climate_sens
        if s["pollution_index"] > 250:
            excess = s["pollution_index"] - 250
            s["public_satisfaction"] -= excess * 0.30 * climate_sens

        # ── NEW: pollution -> education ROI reduction ──
        # Sustained high pollution degrades learning environments
        if s["pollution_index"] > 180:
            self._high_pollution_steps += 1
        else:
            self._high_pollution_steps = max(0, self._high_pollution_steps - 1)
        self._edu_roi_reduced = (self._high_pollution_steps >= 3)
        if self._edu_roi_reduced:
            s["education_index"] -= 0.3  # active drain on existing education capital

        # ── Tax over-burden ──
        if s["tax_rate"] > 40:
            excess = s["tax_rate"] - 40
            s["gdp_index"] -= (excess ** 1.5) * 0.10
            s["industrial_output"] -= excess * 0.50
            s["foreign_investment"] -= excess * 0.80

        # ── Mass unemployment crisis ──
        if s["unemployment_rate"] > 25:
            excess = s["unemployment_rate"] - 25
            s["public_satisfaction"] -= excess * 1.50
            s["inequality_index"] += excess * 0.30

        # ── Hyper-inflation ──
        if s["inflation_rate"] > 15:
            excess = s["inflation_rate"] - 15
            s["public_satisfaction"] -= excess * 1.0
            s["foreign_investment"] -= excess * 0.50

        # ── GDP depression spiral ──
        if s["gdp_index"] < 40:
            deficit = 40 - s["gdp_index"]
            s["unemployment_rate"] += deficit * 0.20
            s["public_satisfaction"] -= deficit * 0.15

        # ── Ecological tipping point ──
        if s["ecological_stability"] < 20:
            deficit = 20 - s["ecological_stability"]
            s["pollution_index"] += deficit * 0.25 * climate_sens
            s["public_satisfaction"] -= deficit * 0.20

        # ── NEW: inequality intolerance — dynamic threshold ──
        # ineq_tol drifts; lower tolerance means protests fire sooner
        ineq_threshold = 60.0 * ineq_tol  # threshold between 12–60
        if s["inequality_index"] > ineq_threshold:
            excess = s["inequality_index"] - ineq_threshold
            s["public_satisfaction"] -= excess * (0.15 / max(ineq_tol, 0.2))

        # ── NEW: education amplifier when education > 75 ──
        if s["education_index"] > 75:
            edu_bonus_mult = 1.5  # amplify innovation dividends
            # Small immediate GDP bonus from high-education workforce
            s["gdp_index"] += (s["education_index"] - 75) * 0.02 * edu_bonus_mult

    # =================================================================
    # Layer 3 — Delayed Effects (state-dependent compounding)
    # =================================================================

    def _layer3_delayed(self, s: Dict[str, float], step: int) -> int:
        """Fire any delayed effects whose step has arrived.
        
        Returns number of effects resolved.
        """
        remaining: Deque[DelayedEffect] = deque()
        resolved = 0
        while self._delayed_queue:
            fire_step, deltas, health_mult = self._delayed_queue.popleft()
            if step >= fire_step:
                # Compute health multiplier at fire time
                mult = 1.0
                if health_mult:
                    # System health: 0 (distressed) -> 1 (thriving)
                    gdp_h = max(0.0, min(1.0, (s.get("gdp_index", 100) - 15) / 185))
                    sat_h = max(0.0, min(1.0, (s.get("public_satisfaction", 50) - 5) / 95))
                    sys_health = 0.5 * gdp_h + 0.5 * sat_h
                    # Range: [0.5 if stressed, 1.3 if thriving]
                    mult = 0.5 + 0.8 * sys_health
                for key, delta in deltas.items():
                    s[key] = s.get(key, 0.0) + delta * mult
                resolved += 1
            else:
                remaining.append((fire_step, deltas, health_mult))
        self._delayed_queue = remaining
        return resolved

    def _enqueue_delayed(
        self, fire_step: int, deltas: Dict[str, float], health_mult: bool = False
    ) -> None:
        self._delayed_queue.append((fire_step, deltas, health_mult))

    # =================================================================
    # Layer 4 — Feedback Loops (emergent dynamics + new cross-layer)
    # =================================================================

    def _layer4_feedback(
        self, s: Dict[str, float], drift_vars: Dict[str, float]
    ) -> None:
        trust_decay = drift_vars.get("public_trust_decay", 0.05)
        supply_res = drift_vars.get("supply_chain_resilience", 0.7)
        inst_trust = drift_vars.get("institutional_trust", 0.6)

        # ── Health–Productivity loop ──
        if s["healthcare_index"] < 30:
            deficit = 30 - s["healthcare_index"]
            s["industrial_output"] -= deficit * 0.08
            s["unemployment_rate"] += 0.30

        # ── Education–Innovation loop ──
        if s["education_index"] > 70:
            s["gdp_index"] += 0.35
            s["inequality_index"] -= 0.10

        # ── Inequality–Satisfaction loop (trust-modulated) ──
        ineq_threshold_dyn = 60.0 * drift_vars.get("inequality_tolerance", 0.6)
        if s["inequality_index"] > ineq_threshold_dyn:
            excess = s["inequality_index"] - ineq_threshold_dyn
            sat_drain = excess * 0.10 * (1.0 + (1.0 - inst_trust) * 0.5)
            s["public_satisfaction"] -= sat_drain

        # ── Unemployment–Social loop ──
        if s["unemployment_rate"] > 18:
            excess = s["unemployment_rate"] - 18
            s["public_satisfaction"] -= excess * 0.18
            s["inequality_index"] += 0.15

        # ── Pollution–Health loop ──
        if s["pollution_index"] > 150:
            excess = s["pollution_index"] - 150
            s["healthcare_index"] -= excess * 0.04

        # ── NEW: Multi-hop pollution->education ROI->GDP drag ──
        if self._edu_roi_reduced and s["education_index"] < 60:
            # Sustained pollution kills education pipeline -> GDP drag
            s["gdp_index"] -= 0.2
            s["industrial_output"] -= 0.1

        # ── Renewable energy dividend ──
        rer = s.get("renewable_energy_ratio", 0.0)
        if rer > 0.3:
            green_bonus = (rer - 0.3) * 8.0
            s["pollution_index"] -= green_bonus * 0.40
            s["energy_efficiency"] += green_bonus * 0.10
            s["carbon_emission_rate"] -= green_bonus * 0.30

        # ── Satisfaction–Stability pressure ──
        if s["public_satisfaction"] < 20:
            s["regulation_strength"] += 0.8
            s["foreign_investment"] -= 0.8

        # ── Supply chain resilience modulates trade shocks ──
        if s["trade_balance"] < -30:
            deficit = abs(s["trade_balance"] + 30)
            # Resilient supply chains absorb trade shocks
            s["gdp_index"] -= deficit * 0.02 * (1.0 - supply_res)

        # ── Natural carbon cycle ──
        if s["pollution_index"] > 0:
            s["pollution_index"] -= 0.5
        if s["ecological_stability"] < 100 and s["pollution_index"] < 80:
            s["ecological_stability"] += 0.3

        # ── Natural satisfaction drift (regression toward 45) ──
        sat = s["public_satisfaction"]
        baseline = 45.0
        # trust_decay adds persistent drain
        s["public_satisfaction"] -= trust_decay
        if sat < baseline:
            s["public_satisfaction"] += min(0.6, (baseline - sat) * 0.03)
        elif sat > 80:
            s["public_satisfaction"] -= (sat - 80) * 0.01

        # ── Institutional trust effect ──
        # High trust slightly boosts foreign investment and satisfaction stability
        if inst_trust > 0.7:
            s["foreign_investment"] += (inst_trust - 0.7) * 0.5
        elif inst_trust < 0.3:
            s["foreign_investment"] -= (0.3 - inst_trust) * 1.0
            s["public_satisfaction"] -= (0.3 - inst_trust) * 0.5

    # =================================================================
    # Clamping
    # =================================================================

    @staticmethod
    def _clamp(s: Dict[str, float]) -> None:
        for key, (lo, hi) in STATE_BOUNDS.items():
            if key in s:
                s[key] = max(lo, min(hi, s[key]))
