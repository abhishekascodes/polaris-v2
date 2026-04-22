"""
AI Policy Engine — Core Environment Implementation (v3 — POLARIS)

Orchestrates all sub-engines:
  - TransitionEngine (v2): cross-layer feedback, state-dep delayed effects
  - EventEngine (v2): sigmoid probs, chaining, memory bias
  - DriftEngine: non-stationary drift on 6 variables
  - MultiAgentCouncil: 5 ministers, voting, coalition, credit
  - RewardEngine (v2): Pareto, long-horizon, cooperation
  - ExplainabilityEngine (v2): counterfactuals, alignment, NL narrative
  - NegotiationProtocol (v3 NEW): LLM-powered minister negotiation
  - BriefingEngine (v3 NEW): long-horizon memory challenges

Rich 55-dimensional observation + negotiation context:
  21 core state dims
  + 5 agent influence
  + 15 risk heatmap (5 metrics x 3 horizons)
  + 8 action history
  + 1 institutional trust
  + 5 coalition status
  + negotiation context (minister proposals, briefings)
"""

from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional
from uuid import uuid4

try:
    from openenv.core.env_server.interfaces import Environment
    from openenv.core.env_server.types import Action, Observation, State
except ImportError:
    from pydantic import BaseModel, Field

    class Action(BaseModel):  # type: ignore[no-redef]
        metadata: dict = Field(default_factory=dict)

    class Observation(BaseModel):  # type: ignore[no-redef]
        done: bool = False
        reward: float = 0.0
        metadata: dict = Field(default_factory=dict)

    class State(BaseModel):  # type: ignore[no-redef]
        episode_id: Optional[str] = None
        step_count: int = 0

    class Environment:  # type: ignore[no-redef]
        def reset(self, **kw: Any) -> Observation:
            raise NotImplementedError
        def step(self, action: Action, **kw: Any) -> Observation:
            raise NotImplementedError
        @property
        def state(self) -> State:
            raise NotImplementedError
        def close(self) -> None:
            pass
        def reset_async(self, **kw: Any) -> Observation:
            return self.reset(**kw)
        def step_async(self, action: Action, **kw: Any) -> Observation:
            return self.step(action, **kw)

from .config import (
    COLLAPSE_CONDITIONS,
    DEFAULT_STATE,
    DRIFT_INITIAL,
    TASK_CONFIGS,
    VALID_ACTIONS,
    ACTION_DESCRIPTIONS,
    OBS_TOTAL_DIM,
)
from .drift_engine import DriftEngine
from .event_engine import EventEngine
from .explainability import ExplainabilityEngine
from .multi_agent_council import MultiAgentCouncil
from .reward_engine import RewardEngine
from .tasks import grade_trajectory
from .transition_engine import TransitionEngine
from .negotiation_protocol import NegotiationProtocol, AgentResponse
from .briefing_engine import BriefingEngine


class PolicyEnvironment(Environment):
    """
    AI Policy Engine — multi-objective governance simulation (v3 POLARIS).

    55-dimensional observation space. 5-minister multi-agent council.
    Non-stationary drift. Intelligent event chaining. Advanced reward.
    Research-grade explainability with counterfactuals and alignment scoring.

    v3 additions:
      - LLM-powered negotiation protocol (ministers propose, agent negotiates)
      - Diplomatic briefing system (long-horizon memory challenges)
      - Theory-of-mind reward (veto prediction scoring)
    """

    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self) -> None:
        self._state_obj = State(episode_id=str(uuid4()), step_count=0)
        self._world: Dict[str, float] = {}
        self._prev_world: Optional[Dict[str, float]] = None
        self._last_actions: List[str] = []
        self._trajectory: List[Dict] = []

        self._task_id: str = "environmental_recovery"
        self._task_cfg: dict = {}
        self._max_steps: int = 50
        self._done: bool = False

        # Sub-engines (v2)
        self._transition = TransitionEngine()
        self._events = EventEngine()
        self._drift = DriftEngine()
        self._council = MultiAgentCouncil()
        self._reward_eng = RewardEngine()
        self._explainer = ExplainabilityEngine()

        # v3 engines
        self._negotiation = NegotiationProtocol(mode="scripted")
        self._briefing = BriefingEngine()
        self._negotiation_enabled = False
        self._briefing_enabled = False
        self._last_negotiation_context = None
        self._last_negotiation_outcome = None

        # Episode-level counters
        self._total_resolved_delayed: int = 0
        self._black_swan_events: List[str] = []

    # =================================================================
    # reset()
    # =================================================================

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        task_id: Optional[str] = None,
        **kwargs: Any,
    ) -> Observation:
        self._task_id = task_id or kwargs.get("task_id", "environmental_recovery")
        if self._task_id not in TASK_CONFIGS:
            self._task_id = "environmental_recovery"

        cfg = TASK_CONFIGS[self._task_id]
        self._task_cfg = cfg
        self._max_steps = cfg["max_steps"]

        # Build initial world state
        self._world = copy.deepcopy(DEFAULT_STATE)
        for key, val in cfg.get("initial_state_overrides", {}).items():
            self._world[key] = val

        self._prev_world = None
        self._last_actions = []
        self._trajectory = []
        self._done = False
        self._total_resolved_delayed = 0
        self._black_swan_events = []

        real_seed = seed if seed is not None else 42
        chaos_level = cfg.get("chaos_level", 0.5)
        drift_enabled = cfg.get("drift_enabled", False)
        num_ministers = cfg.get("num_ministers", 1)

        # Reset sub-engines
        self._transition.reset()
        self._events.reset(
            seed=real_seed,
            frequency_multiplier=cfg.get("event_frequency_multiplier", 1.0),
            satisfaction_event_scale=cfg.get("satisfaction_event_scale", 1.0),
        )
        self._drift.reset(
            seed=real_seed,
            drift_enabled=drift_enabled,
            chaos_level=chaos_level,
        )
        self._council.reset(
            seed=real_seed,
            num_ministers=num_ministers,
            chaos_level=chaos_level,
            institutional_trust=DRIFT_INITIAL["institutional_trust"],
        )
        self._reward_eng.reset()
        self._explainer.reset()

        # v3: Negotiation protocol
        self._negotiation_enabled = cfg.get("negotiation_enabled", num_ministers >= 2)
        self._briefing_enabled = cfg.get("briefing_enabled", cfg.get("events_enabled", False))
        minister_mode = cfg.get("minister_mode", "scripted")
        self._negotiation = NegotiationProtocol(mode=minister_mode)
        self._negotiation.reset(seed=real_seed, num_ministers=num_ministers)
        self._last_negotiation_context = None
        self._last_negotiation_outcome = None

        # v3: Briefing engine
        difficulty_map = {"environmental_recovery": "easy", "balanced_economy": "medium",
                          "sustainable_governance": "hard", "sustainable_governance_extreme": "extreme",
                          "multi_agent_council": "extreme"}
        if self._briefing_enabled:
            self._briefing.reset(
                seed=real_seed,
                difficulty=difficulty_map.get(self._task_id, "medium"),
                max_steps=self._max_steps,
                minister_names=self._negotiation.get_minister_names(),
            )

        self._state_obj = State(
            episode_id=episode_id or str(uuid4()),
            step_count=0,
        )

        obs_meta = self._build_observation_metadata(reward_breakdown=None)

        # v3: Generate initial negotiation context
        if self._negotiation_enabled:
            init_ctx = self._negotiation.phase_propose(
                self._world, [], 0, briefing="Welcome. The council awaits your leadership."
            )
            obs_meta["negotiation"] = init_ctx.to_dict()
            obs_meta["negotiation_narrative"] = init_ctx.to_narrative()
            self._last_negotiation_context = init_ctx

        self._trajectory.append(copy.deepcopy(obs_meta))

        return Observation(done=False, reward=0.0, metadata=obs_meta)

    # =================================================================
    # step()
    # =================================================================

    def step(
        self,
        action: Any,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> Observation:
        if self._done:
            obs_meta = self._build_observation_metadata(reward_breakdown=None)
            return Observation(done=True, reward=0.0, metadata=obs_meta)

        # --- Parse action (supports structured v3 input) ---
        action_str, agent_response = self._parse_action_v3(action)

        # --- Snapshot previous state ---
        self._prev_world = copy.deepcopy(self._world)

        # --- v3: Negotiation resolution ---
        negotiation_outcome = None
        tom_reward = 0.0
        if self._negotiation_enabled and agent_response:
            negotiation_outcome = self._negotiation.phase_resolve(
                self._world, agent_response
            )
            tom_reward = negotiation_outcome.tom_reward
            self._last_negotiation_outcome = negotiation_outcome
            # If vetoed, use the fallback action from negotiation
            if negotiation_outcome.vetoed:
                action_str = negotiation_outcome.final_action

        # --- Council step (determines or confirms action) ---
        num_ministers = self._task_cfg.get("num_ministers", 1)
        if num_ministers <= 1:
            forced = action_str if action_str in VALID_ACTIONS else None
        else:
            forced = None
        council_result = self._council.step(
            state=self._world,
            forced_action=forced,
            utility_volatility=self._events.utility_volatility,
        )
        if num_ministers > 1 and not self._negotiation_enabled:
            action_str = council_result["action"]
        elif action_str == "council":
            action_str = council_result["action"]

        # --- Drift step ---
        drift_vars = self._drift.step(chaos_this_step=len(self._events.active_event_names))

        # --- Apply transition engine (Layers 1-4) ---
        policy_fatigue = drift_vars.get("policy_fatigue", 0.1)
        resolved_delayed = self._transition.apply(
            self._world, action_str, self._state_obj.step_count,
            drift_vars=drift_vars,
            policy_fatigue=policy_fatigue,
        )
        self._total_resolved_delayed += resolved_delayed
        self._reward_eng.record_resolved_effects(resolved_delayed)

        # --- Satisfaction floor damping (calibrated regime) ---
        floor_damp = self._task_cfg.get("satisfaction_floor_damping", 0)
        if floor_damp > 0:
            sat_post_trans = self._world.get("public_satisfaction", 50)
            sat_prev = self._prev_world.get("public_satisfaction", 50)
            if sat_post_trans < 35 and sat_post_trans < sat_prev:
                loss = sat_prev - sat_post_trans
                gradient = min(1.0, floor_damp + (35 - sat_post_trans) / 35 * 0.15)
                self._world["public_satisfaction"] = sat_prev - loss * (1.0 - gradient)

        # --- Apply event engine ---
        sat_pre_events = self._world.get("public_satisfaction", 50)
        active_events = self._events.step(self._world)

        # Track black-swan events
        for evt in active_events:
            if evt in ("pandemic", "natural_disaster", "economic_recession", "climate_crisis"):
                if evt not in self._black_swan_events:
                    self._black_swan_events.append(evt)

        if floor_damp > 0:
            sat_post_events = self._world.get("public_satisfaction", 50)
            if sat_post_events < 35 and sat_post_events < sat_pre_events:
                loss = sat_pre_events - sat_post_events
                gradient = min(1.0, floor_damp + (35 - sat_post_events) / 35 * 0.15)
                self._world["public_satisfaction"] = sat_pre_events - loss * (1.0 - gradient)

        # --- Emergency crisis response bonus ---
        crisis_bonus = self._task_cfg.get("crisis_welfare_bonus", 0)
        social_actions = {"increase_welfare", "invest_in_healthcare", "invest_in_education"}
        if crisis_bonus > 0 and action_str in social_actions:
            sat_cur = self._world.get("public_satisfaction", 50)
            if sat_cur < 40:
                bonus = crisis_bonus if action_str == "increase_welfare" else crisis_bonus * 0.5
                self._world["public_satisfaction"] += bonus

        # --- Handle meta-action effects on institutional trust ---
        if action_str == "reset_institutional_trust":
            # Trust resets towards 0.5 from current value (costly but can help)
            current_trust = drift_vars.get("institutional_trust", 0.6)
            # Reset: jumps to 0.4 (lower at first, then rebuilds naturally)
            self._council.update_institutional_trust(0.4 - current_trust)
        elif action_str == "propose_global_policy_package":
            # Package proposal slightly boosts institutional trust
            self._council.update_institutional_trust(0.03)
        elif council_result.get("coalition_formed"):
            # Successful coalition also builds trust slightly
            self._council.update_institutional_trust(0.01)

        # Vetoes erode institutional trust (creates long-horizon memory)
        vetoes_this_step = council_result.get("vetoes", [])
        if vetoes_this_step:
            self._council.update_institutional_trust(-0.005 * len(vetoes_this_step))

        # --- Re-clamp after events ---
        TransitionEngine._clamp(self._world)

        # --- Track action history ---
        self._last_actions.append(action_str)
        if len(self._last_actions) > 5:
            self._last_actions.pop(0)

        # --- Update council with coalition survival ---
        self._reward_eng.update_coalition_survival(
            self._council.get_coalition_survival_ratio()
        )

        # --- v3: Briefing engine step ---
        briefing_text = ""
        briefing_reward = 0.0
        if self._briefing_enabled:
            briefing_text, briefing_reward = self._briefing.step(
                self._state_obj.step_count, self._world
            )

        # --- Compute reward ---
        is_terminal = (self._state_obj.step_count + 1 >= self._max_steps)
        alignment_score = council_result.get("alignment_score", 50.0)
        reward_info = self._reward_eng.compute(
            self._world, self._prev_world, action_str,
            is_terminal=is_terminal,
            alignment_score=alignment_score,
        )
        reward = reward_info["total_reward"]

        # v3: Add theory-of-mind and briefing rewards
        reward += tom_reward
        reward += briefing_reward
        if tom_reward != 0.0:
            reward_info["tom_reward"] = round(tom_reward, 4)
        if briefing_reward != 0.0:
            reward_info["briefing_reward"] = round(briefing_reward, 4)

        # --- Generate causal explanation ---
        explanation = self._explainer.explain(
            action=action_str,
            prev_state=self._prev_world,
            curr_state=self._world,
            active_events=active_events,
            step=self._state_obj.step_count,
            council_result=council_result,
            drift_vars=drift_vars,
            resolved_delayed=resolved_delayed,
        )

        # --- Advance step counter ---
        self._state_obj.step_count += 1

        # --- Check termination ---
        collapsed = self._check_collapse()
        reached_limit = self._state_obj.step_count >= self._max_steps
        self._done = collapsed or reached_limit

        # --- Build observation ---
        obs_meta = self._build_observation_metadata(
            reward_breakdown=reward_info,
            active_events=active_events,
            explanation=explanation,
            council_result=council_result,
            drift_vars=drift_vars,
        )

        # v3: Add negotiation context for NEXT step
        if self._negotiation_enabled and not self._done:
            next_ctx = self._negotiation.phase_propose(
                self._world, active_events, self._state_obj.step_count,
                briefing=briefing_text,
            )
            obs_meta["negotiation"] = next_ctx.to_dict()
            obs_meta["negotiation_narrative"] = next_ctx.to_narrative()
            self._last_negotiation_context = next_ctx

        # v3: Add active briefings
        if self._briefing_enabled:
            obs_meta["active_briefings"] = self._briefing.get_active_briefings(
                self._state_obj.step_count
            )
            if briefing_text:
                obs_meta["new_briefing"] = briefing_text

        # v3: Add negotiation outcome from this step
        if negotiation_outcome:
            obs_meta["negotiation_outcome"] = negotiation_outcome.to_dict()

        self._trajectory.append(copy.deepcopy(obs_meta))

        # --- If done, include final grader score ---
        if self._done:
            final_score = grade_trajectory(self._task_id, self._trajectory)
            obs_meta["final_score"] = final_score
            obs_meta["collapsed"] = collapsed
            obs_meta["total_steps"] = self._state_obj.step_count
            obs_meta["council_summary"] = self._council.get_episode_summary()
            obs_meta["black_swan_events"] = list(self._black_swan_events)
            obs_meta["total_resolved_delayed"] = self._total_resolved_delayed
            if self._briefing_enabled:
                obs_meta["briefing_stats"] = self._briefing.get_stats()

        return Observation(done=self._done, reward=reward, metadata=obs_meta)

    # =================================================================
    # state property
    # =================================================================

    @property
    def state(self) -> State:
        return self._state_obj

    def close(self) -> None:
        pass

    # =================================================================
    # Extra public methods
    # =================================================================

    def get_trajectory(self) -> List[Dict]:
        return list(self._trajectory)

    def get_valid_actions(self) -> List[str]:
        return list(VALID_ACTIONS)

    def get_action_descriptions(self) -> Dict[str, str]:
        return dict(ACTION_DESCRIPTIONS)

    def get_augmented_observation_vector(self) -> List[float]:
        """
        Return the full 55-dimensional observation vector for RL agents.
        [21 core | 5 influence | 15 risk | 8 history | 1 trust | 5 coalition]
        """
        from .config import STATE_BOUNDS, OBS_CORE_DIM
        # Core 21 dims (normalised)
        STATE_KEYS = [
            "pollution_index", "carbon_emission_rate", "renewable_energy_ratio",
            "ecological_stability", "gdp_index", "industrial_output",
            "unemployment_rate", "inflation_rate", "trade_balance",
            "foreign_investment", "public_satisfaction", "healthcare_index",
            "education_index", "inequality_index", "energy_efficiency",
            "transport_efficiency", "tax_rate", "regulation_strength",
            "welfare_spending", "green_subsidies", "interest_rate",
        ]
        core_vec = []
        for key in STATE_KEYS:
            val = self._world.get(key, 0.0)
            lo, hi = STATE_BOUNDS.get(key, (0, 100))
            norm = 2.0 * (val - lo) / (hi - lo) - 1.0 if hi > lo else 0.0
            core_vec.append(max(-3.0, min(3.0, norm)))

        # Influence vector (5 dims)
        influence_vec = self._council.get_influence_vector()

        # Risk heatmap (15 dims)
        risk_vec = self._drift.compute_risk_heatmap(self._world)

        # Action history encoded (8 dims)
        history_vec = self._council.get_action_history_encoded()

        # Institutional trust (1 dim)
        trust = self._drift.get("institutional_trust", 0.6)
        trust_vec = [trust]

        # Coalition status (5 dims)
        council_meta = self._council.step.__doc__  # just to reference
        # Get from last council result (stored in obs_meta or recompute)
        coalition_vec = self._council._get_coalition_status()

        full_vec = core_vec + influence_vec + risk_vec + history_vec + trust_vec + coalition_vec
        assert len(full_vec) == OBS_TOTAL_DIM, f"Obs dim mismatch: {len(full_vec)} != {OBS_TOTAL_DIM}"
        return full_vec

    # =================================================================
    # Internal helpers
    # =================================================================

    def _parse_action(self, action: Any) -> str:
        """Legacy action parser — returns action string only."""
        action_str = "no_action"

        if isinstance(action, str):
            action_str = action
        elif hasattr(action, "action"):
            action_str = getattr(action, "action", "no_action")
        elif isinstance(action, dict):
            action_str = action.get("action", "no_action")
        elif hasattr(action, "metadata"):
            meta = getattr(action, "metadata", {})
            if isinstance(meta, dict):
                action_str = meta.get("action", "no_action")

        if action_str not in VALID_ACTIONS and action_str != "council":
            action_str = "no_action"

        return action_str

    def _parse_action_v3(self, action: Any) -> tuple:
        """
        v3 action parser — returns (action_str, AgentResponse or None).
        Supports both legacy string actions and structured v3 actions.
        """
        if isinstance(action, dict) and "reasoning" in action:
            # Structured v3 action
            action_str = action.get("action", "no_action")
            if action_str not in VALID_ACTIONS:
                action_str = "no_action"
            agent_resp = AgentResponse(
                action=action_str,
                reasoning=action.get("reasoning", ""),
                coalition_target=action.get("coalition_target", []),
                negotiation_argument=action.get("negotiation_argument", ""),
                veto_prediction=action.get("veto_prediction", []),
                stance=action.get("stance", "cooperative"),
            )
            return action_str, agent_resp
        else:
            # Legacy action — wrap in AgentResponse for negotiation
            action_str = self._parse_action(action)
            if self._negotiation_enabled:
                agent_resp = AgentResponse(action=action_str)
                return action_str, agent_resp
            return action_str, None

    def _check_collapse(self) -> bool:
        for cond_name, (metric, threshold) in COLLAPSE_CONDITIONS.items():
            val = self._world.get(metric, 50.0)
            if cond_name == "gdp_collapse" and val < threshold:
                return True
            if cond_name == "eco_collapse" and val > threshold:
                return True
            if cond_name == "social_collapse" and val < threshold:
                return True
        return False

    def _build_observation_metadata(
        self,
        reward_breakdown: Optional[Dict] = None,
        active_events: Optional[List[str]] = None,
        explanation: Optional[Dict] = None,
        council_result: Optional[Dict] = None,
        drift_vars: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        cfg = TASK_CONFIGS[self._task_id]
        meta: Dict[str, Any] = {}

        # All world-state metrics
        for key, val in self._world.items():
            meta[key] = round(val, 4)

        # Temporal context
        meta["step_number"] = self._state_obj.step_count
        meta["max_steps"] = self._max_steps
        meta["last_actions"] = list(self._last_actions)
        meta["active_events"] = active_events or []

        # Task info
        meta["task_id"] = self._task_id
        meta["task_description"] = cfg["description"]

        # Reward breakdown
        if reward_breakdown:
            meta["reward_breakdown"] = reward_breakdown

        # Explainability -- causal reasoning chain
        if explanation:
            meta["explanation"] = explanation

        # Council / multi-agent metadata
        if council_result:
            meta["council"] = {
                "action": council_result.get("action"),
                "coalition_formed": council_result.get("coalition_formed", False),
                "coalition_strength": council_result.get("coalition_strength", 0.0),
                "vetoes": council_result.get("vetoes", []),
                "influence_vector": council_result.get("influence_vector", []),
                "alignment_score": council_result.get("alignment_score", 50.0),
                "credit_deltas": council_result.get("credit_deltas", {}),
                "coalition_status": council_result.get("coalition_status", []),
            }

        # Drift variables
        if drift_vars:
            meta["drift_vars"] = {k: round(v, 4) for k, v in drift_vars.items()}

        return meta
