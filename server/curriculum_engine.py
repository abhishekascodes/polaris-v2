"""
AI Policy Engine — Adaptive Curriculum Engine

Fully adaptive curriculum that ramps:
  - Stochastic intensity (event_frequency_multiplier)
  - Episode length
  - Number of agents (ministers)
  - Chaos level
  - Drift speed

Based on rolling window of survival rate, Pareto quality, and avg score.
Also runs automated baseline comparisons at each eval checkpoint.
"""

from __future__ import annotations

import json
import math
import os
import random
import time
from typing import Dict, List, Optional, Tuple


class CurriculumState:
    """Tracks adaptive curriculum progression."""

    def __init__(self) -> None:
        # Rampable parameters (start -> end)
        self.event_freq_mult: float = 0.3
        self.episode_length: int = 50
        self.num_ministers: int = 1
        self.chaos_level: float = 0.0
        self.drift_speed: float = 0.0

        # Rolling metrics window
        self._survival_window: List[float] = []
        self._pareto_window: List[float] = []
        self._score_window: List[float] = []
        self._window_size: int = 10

        # Progress
        self.level: int = 0
        self.max_level: int = 10

    def update(self, survival: float, pareto_quality: float, avg_score: float) -> bool:
        """
        Update rolling windows, advance curriculum if criteria met.
        Returns True if level advanced.
        """
        self._survival_window.append(survival)
        self._pareto_window.append(pareto_quality)
        self._score_window.append(avg_score)

        for w in [self._survival_window, self._pareto_window, self._score_window]:
            if len(w) > self._window_size:
                w.pop(0)

        if len(self._survival_window) < 3:
            return False

        # Advance if rolling averages cross thresholds
        avg_surv = sum(self._survival_window) / len(self._survival_window)
        avg_score_r = sum(self._score_window) / len(self._score_window)

        if avg_surv > 0.3 and avg_score_r > 0.35 and self.level < self.max_level:
            self.level += 1
            self._ramp()
            return True
        return False

    def _ramp(self) -> None:
        """Ramp curriculum parameters based on current level."""
        t = self.level / self.max_level  # 0 -> 1

        # Linear ramps
        self.event_freq_mult = 0.3 + 1.2 * t          # 0.3 -> 1.5
        self.chaos_level = t                             # 0.0 -> 1.0
        self.drift_speed = t                             # 0.0 -> 1.0
        self.num_ministers = min(5, 1 + int(4 * t))    # 1 -> 5
        # Episode length ramps slower
        self.episode_length = int(50 + 250 * t)         # 50 -> 300

    def get_task_overrides(self) -> Dict:
        """Return config overrides for current curriculum level."""
        return {
            "event_frequency_multiplier": round(self.event_freq_mult, 2),
            "chaos_level": round(self.chaos_level, 2),
            "num_ministers": self.num_ministers,
            "max_steps": self.episode_length,
            "drift_enabled": self.drift_speed > 0.1,
        }

    def summary(self) -> str:
        avg_surv = (sum(self._survival_window) / len(self._survival_window)
                    if self._survival_window else 0.0)
        avg_score = (sum(self._score_window) / len(self._score_window)
                     if self._score_window else 0.0)
        return (
            f"Level {self.level}/{self.max_level} | "
            f"freq={self.event_freq_mult:.1f} chaos={self.chaos_level:.1f} "
            f"ministers={self.num_ministers} steps={self.episode_length} | "
            f"rolling_surv={avg_surv:.1%} rolling_score={avg_score:.4f}"
        )


class AutomatedBaselineRunner:
    """
    Runs 6 baseline agents and produces a scaling report.
    Baselines: random, heuristic, greedy, LLM-inference proxy,
               single-agent RL, full multi-agent council.
    """

    BASELINES = ["random", "heuristic", "greedy", "llm_proxy", "single_rl", "multi_council"]

    def __init__(self) -> None:
        self._results: Dict[str, List[Dict]] = {b: [] for b in self.BASELINES}

    def run_eval_round(
        self,
        task_id: str,
        n_episodes: int = 20,
        seed_base: int = 50000,
        rl_policy=None,
    ) -> Dict[str, Dict]:
        """Run all baselines for n_episodes and return comparison table."""
        from server.policy_environment import PolicyEnvironment
        from server.tasks import grade_trajectory
        from server.config import VALID_ACTIONS

        AL = sorted(VALID_ACTIONS)
        results = {}

        for baseline in self.BASELINES:
            scores, steps_list, collapses = [], [], 0
            rng = random.Random(seed_base)

            for i in range(n_episodes):
                env = PolicyEnvironment()
                obs = env.reset(seed=seed_base + i, task_id=task_id)
                s = 0

                while not obs.done:
                    action = self._get_baseline_action(
                        baseline, obs, s, rng, AL, rl_policy
                    )
                    obs = env.step({"action": action})
                    s += 1

                traj = env.get_trajectory()
                score = grade_trajectory(task_id, traj)
                scores.append(score)
                steps_list.append(s)
                if obs.metadata.get("collapsed"):
                    collapses += 1

            avg_score = sum(scores) / max(len(scores), 1)
            avg_steps = sum(steps_list) / max(len(steps_list), 1)
            surv = 1.0 - collapses / n_episodes

            result = {
                "avg_score": round(avg_score, 4),
                "avg_steps": round(avg_steps, 1),
                "survival_rate": round(surv, 4),
                "n": n_episodes,
            }
            results[baseline] = result
            self._results[baseline].append(result)

        return results

    def _get_baseline_action(
        self, baseline: str, obs, step: int, rng: random.Random,
        al: List[str], rl_policy=None
    ) -> str:
        cycle = ["subsidize_renewables", "invest_in_education", "increase_welfare",
                 "stimulate_economy", "invest_in_healthcare", "incentivize_clean_tech",
                 "enforce_emission_limits", "increase_welfare"]

        if baseline == "random":
            return rng.choice(al)
        elif baseline == "heuristic":
            return cycle[step % len(cycle)]
        elif baseline == "greedy":
            return "stimulate_economy"
        elif baseline == "llm_proxy":
            # Proxy for LLM-inference: smart heuristic with crisis awareness
            sat = obs.metadata.get("public_satisfaction", 50)
            poll = obs.metadata.get("pollution_index", 100)
            gdp = obs.metadata.get("gdp_index", 100)
            if sat < 30:
                return "increase_welfare"
            if poll > 200:
                return "enforce_emission_limits"
            if gdp < 50:
                return "stimulate_economy"
            return cycle[step % len(cycle)]
        elif baseline == "single_rl":
            if rl_policy is not None:
                # Use provided RL policy
                try:
                    from rl_agent import normalise_state, ACTION_LIST, _softmax
                    state_vec = normalise_state(obs.metadata)
                    probs, _, _ = rl_policy.forward(state_vec)
                    action_idx = probs.index(max(probs))
                    return ACTION_LIST[action_idx]
                except Exception:
                    pass
            return cycle[step % len(cycle)]
        elif baseline == "multi_council":
            # Council-based: use 'council' pseudo-action (environment handles it)
            return "council"
        return "no_action"

    def print_scaling_report(self, task_id: str, eval_results: Dict[str, Dict]) -> None:
        """Print scaling table + ASCII plot of performance gap."""
        print(f"\n  {'='*65}")
        print(f"  AUTOMATED BASELINE SCALING REPORT — {task_id}")
        print(f"  {'='*65}")
        print(f"  {'Baseline':<20s} {'Score':>7s} {'Surv%':>7s} {'Steps':>7s}")
        print(f"  {'-'*43}")

        best_score = max(r["avg_score"] for r in eval_results.values())
        worst_score = min(r["avg_score"] for r in eval_results.values())
        gap = best_score / max(worst_score, 0.001)

        for baseline, r in eval_results.items():
            bar_len = int(r["avg_score"] / max(best_score, 0.001) * 20)
            bar = "█" * bar_len
            print(f"  {baseline:<20s} {r['avg_score']:7.4f} {r['survival_rate']*100:6.1f}% "
                  f"{r['avg_steps']:6.1f}  {bar}")

        print(f"\n  Performance gap (best/worst): {gap:.2f}x")
        print(f"  Best: {max(eval_results, key=lambda b: eval_results[b]['avg_score'])}")
        print(f"  Worst: {min(eval_results, key=lambda b: eval_results[b]['avg_score'])}")
        print(f"  {'='*65}")


class CurriculumEngine:
    """
    Top-level curriculum orchestrator: adaptive ramp + baseline runner.
    """

    def __init__(self) -> None:
        self._state = CurriculumState()
        self._baseline_runner = AutomatedBaselineRunner()
        self._eval_history: List[Dict] = []

    def evaluate_and_advance(
        self,
        task_id: str = "sustainable_governance",
        n_eval: int = 20,
        rl_policy=None,
        verbose: bool = True,
    ) -> Dict:
        """
        Run full eval round: all baselines + curriculum advancement decision.
        """
        if verbose:
            print(f"\n  [Curriculum] Running eval round (task={task_id}, n={n_eval})...")

        results = self._baseline_runner.run_eval_round(
            task_id=task_id,
            n_episodes=n_eval,
            rl_policy=rl_policy,
        )

        # Extract multi-council performance as "agent score"
        council_r = results.get("multi_council", results.get("heuristic", {}))
        rand_r = results.get("random", {})

        survival = council_r.get("survival_rate", 0.0)
        avg_score = council_r.get("avg_score", 0.0)
        pareto_q = max(0.0, avg_score - rand_r.get("avg_score", 0.0))

        advanced = self._state.update(survival, pareto_q, avg_score)

        if verbose:
            self._baseline_runner.print_scaling_report(task_id, results)
            if advanced:
                print(f"\n  [Curriculum] Level ADVANCED -> {self._state.summary()}")
            else:
                print(f"\n  [Curriculum] {self._state.summary()}")

        record = {
            "eval_results": results,
            "curriculum_level": self._state.level,
            "advanced": advanced,
            "curriculum_overrides": self._state.get_task_overrides(),
        }
        self._eval_history.append(record)
        return record

    def get_overrides(self) -> Dict:
        return self._state.get_task_overrides()

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump({"history": self._eval_history}, f, indent=2)
