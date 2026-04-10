"""
AI Policy Engine — Pydantic Typed Models

Defines the typed Action model for the policy simulation.
The environment uses the base OpenEnv Observation & State types,
enriching them via the metadata dict for maximum compatibility.
"""

from typing import List, Optional

from pydantic import BaseModel, Field

# ─────────────────────────────────────────────────────────────
# Action Model
# ─────────────────────────────────────────────────────────────

try:
    from openenv.core.env_server.types import Action as BaseAction
except ImportError:
    # Fallback for standalone usage / inference without openenv-core
    class BaseAction(BaseModel):  # type: ignore[no-redef]
        """Minimal Action base when openenv-core is not installed."""
        metadata: dict = Field(default_factory=dict)


class PolicyAction(BaseAction):
    """
    A single policy lever the agent can pull each turn.

    The ``action`` field must be one of the 16 valid action strings
    defined in ``server.config.VALID_ACTIONS``.
    """

    action: str = Field(
        ...,
        description=(
            "The policy action to execute. Must be one of: "
            "no_action, increase_tax, decrease_tax, stimulate_economy, "
            "reduce_interest_rates, expand_industry, restrict_polluting_industries, "
            "incentivize_clean_tech, enforce_emission_limits, subsidize_renewables, "
            "implement_carbon_tax, increase_welfare, invest_in_healthcare, "
            "invest_in_education, upgrade_energy_grid, invest_in_transport"
        ),
    )


# ─────────────────────────────────────────────────────────────
# Reward Breakdown (informational — returned in observation metadata)
# ─────────────────────────────────────────────────────────────

class RewardBreakdown(BaseModel):
    """Breakdown of the multi-objective reward signal."""

    economic_score: float = Field(description="Economic performance component (0–1)")
    environmental_score: float = Field(description="Environmental health component (0–1)")
    social_score: float = Field(description="Social welfare component (0–1)")
    stability_score: float = Field(description="Volatility penalty component (0–1)")
    penalties: float = Field(default=0.0, description="Sum of all penalties applied")
    total_reward: float = Field(description="Final weighted reward value")


# ─────────────────────────────────────────────────────────────
# Observation Schema (documentation — actual data in Observation.metadata)
# ─────────────────────────────────────────────────────────────

class PolicyObservationSchema(BaseModel):
    """
    Schema documenting the structure of Observation.metadata
    returned by the AI Policy Engine environment.

    This model is NOT used at runtime for serialisation — it exists
    purely for documentation, schema generation, and the README.
    """

    # Environmental
    pollution_index: float = Field(description="Pollution level (0–300). Above 200 causes exponential health damage.")
    carbon_emission_rate: float = Field(description="CO₂ emission rate (0–100).")
    renewable_energy_ratio: float = Field(description="Fraction of energy from renewables (0.0–1.0).")
    ecological_stability: float = Field(description="Ecosystem health score (0–100).")

    # Economic
    gdp_index: float = Field(description="GDP index (0–200, baseline=100).")
    industrial_output: float = Field(description="Industrial production level (0–100).")
    unemployment_rate: float = Field(description="Unemployment percentage (0–50).")
    inflation_rate: float = Field(description="Inflation rate (−10 to 30).")
    trade_balance: float = Field(description="Trade surplus/deficit (−100 to 100).")
    foreign_investment: float = Field(description="Foreign direct investment index (0–100).")

    # Social
    public_satisfaction: float = Field(description="Public approval rating (0–100).")
    healthcare_index: float = Field(description="Healthcare system quality (0–100).")
    education_index: float = Field(description="Education system quality (0–100).")
    inequality_index: float = Field(description="Income inequality (0–100, lower = more equal).")

    # Infrastructure
    energy_efficiency: float = Field(description="Energy grid efficiency (0–100).")
    transport_efficiency: float = Field(description="Transport network efficiency (0–100).")

    # Policy knobs
    tax_rate: float = Field(description="Current tax rate (0–50%).")
    regulation_strength: float = Field(description="Regulatory strictness (0–100).")
    welfare_spending: float = Field(description="Welfare budget level (0–100).")
    green_subsidies: float = Field(description="Green subsidy budget level (0–100).")
    interest_rate: float = Field(description="Central bank interest rate (0–20%).")

    # Temporal context
    step_number: int = Field(description="Current step in the episode.")
    max_steps: int = Field(description="Maximum steps for this task.")
    last_actions: List[str] = Field(description="Last 5 actions taken by the agent.")
    active_events: List[str] = Field(description="Currently active random events.")

    # Task
    task_id: str = Field(description="Identifier of the current task.")
    task_description: str = Field(description="Human-readable task objective.")

    # Reward
    reward_breakdown: Optional[dict] = Field(default=None, description="Detailed reward component breakdown.")
