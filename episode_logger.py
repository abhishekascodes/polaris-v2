"""
AI Policy Engine — JSONL Episode Logger

Exports every episode as a full JSONL trace including:
  - All step actions, utilities, causal chains, influence vectors
  - Coalition events, veto counts
  - Episode metadata summary
"""

from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, List, Optional


class EpisodeLogger:
    """
    Logs every episode to a JSONL file.

    Usage:
        logger = EpisodeLogger("outputs/episodes.jsonl")
        logger.begin_episode(episode_id, task_id)
        for each step:
            logger.log_step(step, action, obs_metadata)
        logger.end_episode(final_metadata)
    """

    def __init__(self, path: str = "outputs/episodes.jsonl", enabled: bool = True) -> None:
        self._path = path
        self._enabled = enabled
        self._current: Optional[Dict] = None
        os.makedirs(os.path.dirname(path), exist_ok=True)

    def begin_episode(
        self,
        episode_id: str,
        task_id: str,
        seed: int = 0,
    ) -> None:
        """Start recording a new episode."""
        if not self._enabled:
            return
        self._current = {
            "episode_id": episode_id,
            "task_id": task_id,
            "seed": seed,
            "start_time": time.time(),
            "steps": [],
            # Summary fields (populated at end)
            "coalitions_formed": 0,
            "vetoes_cast": 0,
            "avg_influence_stability": 0.0,
            "alignment_score": 0.0,
            "black_swan_events": [],
            "final_score": 0.0,
            "survived": False,
            "total_steps": 0,
            "total_resolved_delayed": 0,
        }

    def log_step(self, step: int, action: str, obs_metadata: Dict[str, Any]) -> None:
        """Log a single step."""
        if not self._enabled or self._current is None:
            return

        explanation = obs_metadata.get("explanation", {})
        council = obs_metadata.get("council", {})

        step_record = {
            "step": step,
            "action": action,
            "reward": obs_metadata.get("reward_breakdown", {}).get("total_reward", 0.0),
            "public_satisfaction": obs_metadata.get("public_satisfaction", 0),
            "gdp_index": obs_metadata.get("gdp_index", 0),
            "pollution_index": obs_metadata.get("pollution_index", 0),
            "active_events": obs_metadata.get("active_events", []),
            "causal_chain_len": len(explanation.get("causal_chain", [])),
            "alignment_score": explanation.get("alignment_score", 50.0),
            "coalition_formed": council.get("coalition_formed", False),
            "coalition_strength": council.get("coalition_strength", 0.0),
            "vetoes": council.get("vetoes", []),
            "influence_vector": council.get("influence_vector", []),
            "credit_deltas": council.get("credit_deltas", {}),
            "nl_narrative": explanation.get("nl_narrative", ""),
            "risk_alerts": explanation.get("risk_alerts", []),
            "counterfactuals_count": len(explanation.get("counterfactuals", [])),
        }
        self._current["steps"].append(step_record)

    def end_episode(self, final_metadata: Dict[str, Any]) -> None:
        """Finalise and flush episode to JSONL."""
        if not self._enabled or self._current is None:
            return

        ep = self._current

        # Extract summary from final metadata
        council_summary = final_metadata.get("council_summary", {})
        ep["coalitions_formed"] = council_summary.get("coalitions_formed", 0)
        ep["vetoes_cast"] = council_summary.get("total_vetoes", 0)
        ep["avg_influence_stability"] = council_summary.get("avg_influence_stability", 0.0)
        ep["alignment_score"] = (
            sum(s.get("alignment_score", 50.0) for s in ep["steps"])
            / max(len(ep["steps"]), 1)
        )
        ep["black_swan_events"] = final_metadata.get("black_swan_events", [])
        ep["final_score"] = final_metadata.get("final_score", 0.0)
        ep["survived"] = not final_metadata.get("collapsed", True)
        ep["total_steps"] = final_metadata.get("total_steps", len(ep["steps"]))
        ep["total_resolved_delayed"] = final_metadata.get("total_resolved_delayed", 0)
        ep["duration_s"] = round(time.time() - ep.pop("start_time", time.time()), 2)

        # Write to JSONL
        with open(self._path, "a", encoding="utf-8") as f:
            f.write(json.dumps(ep, default=str) + "\n")

        self._current = None

    def write_summary_report(self, output_path: str = "outputs/episode_summary.json") -> None:
        """
        Read all logged episodes and write an aggregate summary.
        """
        if not os.path.exists(self._path):
            return

        episodes = []
        with open(self._path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        episodes.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass

        if not episodes:
            return

        n = len(episodes)
        survived = sum(1 for ep in episodes if ep.get("survived"))
        avg_score = sum(ep.get("final_score", 0) for ep in episodes) / n
        avg_steps = sum(ep.get("total_steps", 0) for ep in episodes) / n
        avg_coalitions = sum(ep.get("coalitions_formed", 0) for ep in episodes) / n
        avg_vetoes = sum(ep.get("vetoes_cast", 0) for ep in episodes) / n
        avg_alignment = sum(ep.get("alignment_score", 50) for ep in episodes) / n
        black_swans = {}
        for ep in episodes:
            for evt in ep.get("black_swan_events", []):
                black_swans[evt] = black_swans.get(evt, 0) + 1

        summary = {
            "total_episodes": n,
            "survival_rate": round(survived / n, 4),
            "avg_score": round(avg_score, 4),
            "avg_steps": round(avg_steps, 1),
            "avg_coalitions_per_episode": round(avg_coalitions, 2),
            "avg_vetoes_per_episode": round(avg_vetoes, 2),
            "avg_alignment_score": round(avg_alignment, 2),
            "black_swan_event_counts": black_swans,
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        print(f"\n  Episode Summary ({n} episodes):")
        print(f"    Survival rate: {survived/n:.1%}")
        print(f"    Avg score:     {avg_score:.4f}")
        print(f"    Avg steps:     {avg_steps:.1f}")
        print(f"    Avg coalitions:{avg_coalitions:.2f}/ep")
        print(f"    Avg vetoes:    {avg_vetoes:.2f}/ep")
        print(f"    Avg alignment: {avg_alignment:.1f}/100")
        if black_swans:
            top_bsw = sorted(black_swans.items(), key=lambda x: -x[1])
            print(f"    Black-swan events: {dict(top_bsw)}")
