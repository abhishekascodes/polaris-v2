"""
AI Policy Engine — EnvClient

Provides a thin client class for connecting to a remote
PolicyEnvironment server over WebSocket (OpenEnv protocol).
"""

try:
    from openenv.core.env_client import EnvClient
    from openenv.core.env_server.types import Observation

    class PolicyEngineClient(EnvClient):
        """Client for connecting to a remote AI Policy Engine instance."""

        async def _deserialize_observation(self, data: dict) -> Observation:
            return Observation(**data)

except ImportError:
    # Standalone fallback — simple HTTP client
    import requests
    from typing import Optional

    class PolicyEngineClient:  # type: ignore[no-redef]
        """Simple HTTP client for the AI Policy Engine."""

        def __init__(self, base_url: str = "http://localhost:7860"):
            self.base_url = base_url.rstrip("/")

        def reset(self, seed: int = 42, task_id: str = "environmental_recovery") -> dict:
            resp = requests.post(f"{self.base_url}/reset", json={
                "seed": seed,
                "task_id": task_id,
            })
            resp.raise_for_status()
            return resp.json()

        def step(self, action: str) -> dict:
            resp = requests.post(f"{self.base_url}/step", json={
                "action": {"action": action},
            })
            resp.raise_for_status()
            return resp.json()

        def state(self) -> dict:
            resp = requests.get(f"{self.base_url}/state")
            resp.raise_for_status()
            return resp.json()
