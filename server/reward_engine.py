"""
AI Policy Engine — Advanced Multi-Objective Reward Engine (v2 — Nuclear Upgrade)

Upgrades:
  - Adaptive Pareto shaping (bonus scales with proximity to historical best Pareto front)
  - Exponential oscillation penalty over rolling 8-step window
  - Long-horizon credit booster (sparse terminal bonus for resolved delayed effects)
  - Cooperation index multiplier based on coalition survival and outcome quality
  - Safety constraint penalties (lightweight threshold heuristic, no shadow sim)
"""

from __future__ import annotations

import math
from collections import deque
from typing import Deque, Dict, List, Optional, Tuple

from .config import (
    COOPERATION_MULTIPLIER_MAX, KEY_METRICS, LONG_HORIZON_BONUS,
    LONG_HORIZON_WINDOW, PARETO_BONUS_MAX, REWARD_WEIGHTS, STATE_BOUNDS,
)


def _normalise(value: float, lo: float, hi: float) -> float:
    if hi <= lo:
        return 0.0
    return max(0.0, min(1.0, (value - lo) / (hi - lo)))


def _inv_normalise(value: float, lo: float, hi: float) -> float:
    return 1.0 - _normalise(value, lo, hi)


class RewardEngine:
    """
    Calculates multi-objective reward each step with Pareto shaping,
    exponential oscillation penalties, long-horizon credit, and
    cooperation multipliers.
    """

    def __init__(self) -> None:
        self._prev_actions: List[str] = []
        # Rolling 8-step window for oscillation detection
        self._action_window: Deque[str] = deque(maxlen=8)
        # Pareto front tracking (list of (econ, env, social) triples)
        self._pareto_front: List[Tuple[float, float, float]] = []
        self._pareto_update_counter: int = 0
        # Long-horizon resolved effects counter
        self._resolved_effects_this_episode: int = 0
        self._step_count: int = 0
        # Coalition survival ratio (updated from council)
        self._coalition_survival: float = 0.0
        # Oscillation severity tracker
        self._oscillation_streak: int = 0

    def reset(self) -> None:
        """Reset for new episode."""
        self._prev_actions.clear()
        self._action_window.clear()
        self._pareto_front.clear()
        self._pareto_update_counter = 0
        self._resolved_effects_this_episode = 0
        self._step_count = 0
        self._coalition_survival = 0.0
        self._oscillation_streak = 0

    def update_coalition_survival(self, ratio: float) -> None:
        """Called by environment with current coalition survival ratio."""
        self._coalition_survival = ratio

    def record_resolved_effects(self, count: int) -> None:
        """Record resolved delayed effects for long-horizon credit."""
        self._resolved_effects_this_episode += count

    def compute(
        self,
        state: Dict[str, float],
        prev_state: Optional[Dict[str, float]],
        action: str,
        is_terminal: bool = False,
        alignment_score: float = 50.0,
    ) -> Dict[str, float]:
        """
        Return a full reward breakdown dict.

        Keys: economic_score, environmental_score, social_score,
              stability_score, pareto_bonus, penalties,
              cooperation_bonus, terminal_bonus, total_reward.
        """
        econ = self._economic_score(state)
        envr = self._environmental_score(state)
        soc = self._social_score(state)
        stab = self._stability_score(state, prev_state)

        weighted = (
            REWARD_WEIGHTS["economic"] * econ
            + REWARD_WEIGHTS["environmental"] * envr
            + REWARD_WEIGHTS["social"] * soc
            + REWARD_WEIGHTS["stability"] * stab
        )

        # ── Adaptive Pareto shaping ──
        pareto_bonus = self._pareto_bonus(econ, envr, soc)

        # ── Safety + oscillation penalties ──
        penalties = self._compute_penalties(state, action)

        # ── Cooperation multiplier ──
        cooperation_bonus = self._cooperation_bonus(alignment_score)

        # ── Long-horizon terminal credit ──
        terminal_bonus = 0.0
        if is_terminal and self._step_count >= LONG_HORIZON_WINDOW:
            terminal_bonus = LONG_HORIZON_BONUS * self._resolved_effects_this_episode
            terminal_bonus = min(terminal_bonus, 0.5)  # cap

        # ── Combine ──
        base = weighted + pareto_bonus - penalties + cooperation_bonus
        total = round(max(0.0, min(1.0, base + terminal_bonus)), 4)

        # Track history
        self._action_window.append(action)
        self._prev_actions.append(action)
        if len(self._prev_actions) > 8:
            self._prev_actions.pop(0)
        self._step_count += 1

        # Update Pareto front periodically
        self._pareto_update_counter += 1

        return {
            "economic_score": round(econ, 4),
            "environmental_score": round(envr, 4),
            "social_score": round(soc, 4),
            "stability_score": round(stab, 4),
            "pareto_bonus": round(pareto_bonus, 4),
            "penalties": round(penalties, 4),
            "cooperation_bonus": round(cooperation_bonus, 4),
            "terminal_bonus": round(terminal_bonus, 4),
            "total_reward": total,
        }

    # -----------------------------------------------------------------
    # Sub-scores (each 0–1)
    # -----------------------------------------------------------------

    @staticmethod
    def _economic_score(s: Dict[str, float]) -> float:
        gdp = _normalise(s["gdp_index"], *STATE_BOUNDS["gdp_index"])
        unemp = _inv_normalise(s["unemployment_rate"], *STATE_BOUNDS["unemployment_rate"])
        ind = _normalise(s["industrial_output"], *STATE_BOUNDS["industrial_output"])
        inv = _normalise(s["foreign_investment"], *STATE_BOUNDS["foreign_investment"])
        trade = _normalise(s["trade_balance"] + 100, 0, 200)
        return 0.35 * gdp + 0.25 * unemp + 0.20 * ind + 0.10 * inv + 0.10 * trade

    @staticmethod
    def _environmental_score(s: Dict[str, float]) -> float:
        poll = _inv_normalise(s["pollution_index"], *STATE_BOUNDS["pollution_index"])
        rer = _normalise(s["renewable_energy_ratio"], *STATE_BOUNDS["renewable_energy_ratio"])
        eco = _normalise(s["ecological_stability"], *STATE_BOUNDS["ecological_stability"])
        carb = _inv_normalise(s["carbon_emission_rate"], *STATE_BOUNDS["carbon_emission_rate"])
        return 0.35 * poll + 0.25 * rer + 0.25 * eco + 0.15 * carb

    @staticmethod
    def _social_score(s: Dict[str, float]) -> float:
        sat = _normalise(s["public_satisfaction"], *STATE_BOUNDS["public_satisfaction"])
        hc = _normalise(s["healthcare_index"], *STATE_BOUNDS["healthcare_index"])
        edu = _normalise(s["education_index"], *STATE_BOUNDS["education_index"])
        ineq = _inv_normalise(s["inequality_index"], *STATE_BOUNDS["inequality_index"])
        return 0.35 * sat + 0.25 * hc + 0.20 * edu + 0.20 * ineq

    @staticmethod
    def _stability_score(
        s: Dict[str, float], prev: Optional[Dict[str, float]]
    ) -> float:
        if prev is None:
            return 0.8
        total_change = 0.0
        for key in KEY_METRICS:
            lo, hi = STATE_BOUNDS.get(key, (0, 100))
            span = hi - lo if hi > lo else 1.0
            delta = abs(s.get(key, 0) - prev.get(key, 0)) / span
            total_change += delta
        avg_change = total_change / len(KEY_METRICS)
        return max(0.0, min(1.0, 1.0 - avg_change / 0.15))

    # -----------------------------------------------------------------
    # Adaptive Pareto Shaping
    # -----------------------------------------------------------------

    def _pareto_bonus(self, econ: float, env: float, soc: float) -> float:
        """
        Compute bonus for being close to or improving the Pareto front.
        The Pareto front tracks the historical best (econ, env, soc) triples
        in objective space. Proximity to it signals near-optimal tradeoffs.
        """
        point = (econ, env, soc)

        # Check if this point dominates or is close to existing front
        if not self._pareto_front:
            self._pareto_front.append(point)
            return 0.0

        # Compute minimum Euclidean distance to current Pareto front
        min_dist = min(
            math.sqrt(sum((point[i] - p[i]) ** 2 for i in range(3)))
            for p in self._pareto_front
        )

        # Update front: add if not dominated
        dominated = any(
            all(p[i] >= point[i] for i in range(3)) for p in self._pareto_front
        )
        if not dominated:
            # Remove dominated points, add new
            self._pareto_front = [
                p for p in self._pareto_front
                if not all(point[i] >= p[i] for i in range(3))
            ]
            self._pareto_front.append(point)
            # Keep front bounded
            if len(self._pareto_front) > 50:
                self._pareto_front = self._pareto_front[-50:]

        # Bonus: higher when close to or on the Pareto front
        # dist=0 -> max bonus; dist=0.5+ -> no bonus
        proximity = max(0.0, 1.0 - min_dist / 0.5)
        return PARETO_BONUS_MAX * proximity

    # -----------------------------------------------------------------
    # Cooperation Bonus
    # -----------------------------------------------------------------

    def _cooperation_bonus(self, alignment_score: float) -> float:
        """
        Scale a small reward bonus based on council alignment and coalition survival.
        Max bonus: (COOPERATION_MULTIPLIER_MAX - 1) applied to a baseline.
        """
        # alignment_score is 0–100
        alignment_norm = alignment_score / 100.0
        # Cooperation index = average of alignment + coalition survival
        coop_index = 0.5 * alignment_norm + 0.5 * self._coalition_survival
        # Bonus: up to COOPERATION_MULTIPLIER_MAX - 1 = 0.30
        bonus = (COOPERATION_MULTIPLIER_MAX - 1.0) * coop_index * 0.1
        return round(max(0.0, min(0.05, bonus)), 4)

    # -----------------------------------------------------------------
    # Penalties (enhanced)
    # -----------------------------------------------------------------

    def _compute_penalties(self, state: Dict[str, float], action: str) -> float:
        p = 0.0

        # ── Exponential oscillation penalty (rolling 8-step window) ──
        p += self._oscillation_penalty(action)

        # ── Rapid flip-flop (undo previous action) ──
        opposite_pairs = {
            "increase_tax": "decrease_tax",
            "decrease_tax": "increase_tax",
            "expand_industry": "restrict_polluting_industries",
            "restrict_polluting_industries": "expand_industry",
        }
        if len(self._prev_actions) >= 1:
            prev = self._prev_actions[-1]
            if opposite_pairs.get(action) == prev:
                p += 0.03

        # ── Safety constraints (lightweight threshold heuristic) ──
        # Penalise actions that are highly dangerous given current state
        p += self._safety_penalty(state, action)

        # ── Collapse proximity ──
        if state["gdp_index"] < 30:
            p += (30 - state["gdp_index"]) * 0.005
        if state["pollution_index"] > 260:
            p += (state["pollution_index"] - 260) * 0.003
        if state["public_satisfaction"] < 15:
            p += (15 - state["public_satisfaction"]) * 0.005

        # ── Extreme inaction under crisis ──
        crisis = (
            state["pollution_index"] > 220
            or state["gdp_index"] < 35
            or state["public_satisfaction"] < 15
        )
        if crisis and action == "no_action":
            p += 0.04

        return p

    def _oscillation_penalty(self, action: str) -> float:
        """
        Exponential penalty for repeated ABAB patterns in rolling 8-step window.
        Streak detection: each additional oscillation cycle multiplies penalty by 1.5.
        """
        if len(self._action_window) < 4:
            return 0.0

        window = list(self._action_window)

        # Check last 4 for ABAB pattern
        if len(window) >= 4:
            a, b, c, d = window[-4], window[-3], window[-2], window[-1]
            if a == c and b == d and a != b:
                self._oscillation_streak += 1
                return 0.05 * (1.5 ** min(self._oscillation_streak - 1, 5))

        # Check last 6 for ABABAB
        if len(window) >= 6:
            w6 = window[-6:]
            if w6[0] == w6[2] == w6[4] and w6[1] == w6[3] == w6[5] and w6[0] != w6[1]:
                self._oscillation_streak = max(self._oscillation_streak, 2)
                return 0.08 * (1.5 ** min(self._oscillation_streak - 1, 5))

        self._oscillation_streak = 0
        return 0.0

    @staticmethod
    def _safety_penalty(state: Dict[str, float], action: str) -> float:
        """
        Lightweight heuristic safety penalty — no shadow simulation.
        Penalises actions that are severely misaligned with current critical state.
        """
        p = 0.0

        gdp = state.get("gdp_index", 100)
        sat = state.get("public_satisfaction", 50)
        poll = state.get("pollution_index", 100)

        # Expanding industry when pollution is at critical levels
        if action == "expand_industry" and poll > 250:
            p += 0.08  # near eco-collapse, this is suicidal

        # Tax cuts when satisfaction is critically low (fiscal capacity needed)
        if action == "decrease_tax" and sat < 15 and gdp < 40:
            p += 0.06

        # Stimulate economy when both GDP and pollution are maxed
        if action == "stimulate_economy" and poll > 240 and gdp > 150:
            p += 0.04

        # No_action when ANY metric is within 5 units of collapse
        if action == "no_action":
            if gdp < 20:
                p += 0.06
            if sat < 10:
                p += 0.06
            if poll > 285:
                p += 0.06

        return p
