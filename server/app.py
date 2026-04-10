"""
AI Policy Engine -- FastAPI Application

Exposes the PolicyEnvironment over HTTP and WebSocket endpoints
compatible with the OpenEnv framework and HuggingFace Spaces.

Dual-mode: uses openenv-core create_app when available,
falls back to a standalone FastAPI server when it is not.
"""

from __future__ import annotations

import os
import sys

# Ensure the project root is on sys.path so `server.*` and `models` resolve
# regardless of how uvicorn is invoked (module vs. directory).
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


# ------------------------------------------------------------------
# Try openenv-core first; fall back to standalone FastAPI
# ------------------------------------------------------------------

try:
    from openenv.core.env_server.http_server import create_app
    from openenv.core.env_server.types import Observation

    from server.policy_environment import PolicyEnvironment
    from models import PolicyAction

    app = create_app(
        PolicyEnvironment,
        PolicyAction,
        Observation,
        env_name="ai_policy_engine",
    )

except ImportError:
    import copy
    from typing import Any, Dict, List, Optional

    from fastapi import FastAPI
    from pydantic import BaseModel, Field

    from server.policy_environment import PolicyEnvironment

    app = FastAPI(
        title="AI Policy Engine",
        description=(
            "A multi-objective, event-driven governance simulation "
            "environment for reinforcement learning agents. "
            "This environment enables benchmarking of LLM-based policy "
            "agents under multi-objective, temporally-dependent decision "
            "constraints."
        ),
        version="1.0.0",
    )

    # ── Request / Response models ──

    class ResetRequest(BaseModel):
        seed: Optional[int] = Field(default=None, description="Random seed for reproducibility")
        episode_id: Optional[str] = Field(default=None, description="Episode ID")
        task_id: Optional[str] = Field(
            default="environmental_recovery",
            description="Task: environmental_recovery | balanced_economy | sustainable_governance",
        )

    class StepRequest(BaseModel):
        action: Dict[str, Any] = Field(
            ..., description='Action dict, e.g. {"action": "subsidize_renewables"}'
        )

    class ObservationResponse(BaseModel):
        observation: Dict[str, Any]
        reward: Optional[float] = None
        done: bool = False

    class StateResponse(BaseModel):
        episode_id: Optional[str] = None
        step_count: int = 0

    class HealthResponse(BaseModel):
        status: str = "healthy"

    # ── Session storage ──

    _envs: Dict[str, PolicyEnvironment] = {}

    # ── Endpoints ──

    @app.get("/health")
    async def health():
        return HealthResponse(status="healthy")

    @app.post("/reset", response_model=ObservationResponse)
    async def reset(request: ResetRequest = None):
        if request is None:
            request = ResetRequest()
        env = PolicyEnvironment()
        obs = env.reset(
            seed=request.seed,
            episode_id=request.episode_id,
            task_id=request.task_id,
        )
        eid = env.state.episode_id
        _envs[eid] = env
        return ObservationResponse(
            observation=obs.metadata,
            reward=obs.reward,
            done=obs.done,
        )

    @app.post("/step", response_model=ObservationResponse)
    async def step(request: StepRequest):
        if _envs:
            env = list(_envs.values())[-1]
        else:
            env = PolicyEnvironment()
            env.reset()
            _envs[env.state.episode_id] = env

        obs = env.step(request.action)
        return ObservationResponse(
            observation=obs.metadata,
            reward=obs.reward,
            done=obs.done,
        )

    @app.get("/state", response_model=StateResponse)
    async def get_state():
        if _envs:
            env = list(_envs.values())[-1]
            s = env.state
            return StateResponse(episode_id=s.episode_id, step_count=s.step_count)
        return StateResponse()

    @app.get("/schema")
    async def schema():
        from models import PolicyAction, PolicyObservationSchema
        return {
            "action": PolicyAction.model_json_schema(),
            "observation": PolicyObservationSchema.model_json_schema(),
        }

    @app.get("/tasks")
    async def list_tasks():
        from server.config import TASK_CONFIGS
        return {
            tid: {"description": cfg["description"], "max_steps": cfg["max_steps"]}
            for tid, cfg in TASK_CONFIGS.items()
        }


def main():
    """Entry point for direct execution."""
    import uvicorn
    port = int(os.environ.get("PORT", "7860"))
    uvicorn.run(app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()
