"""
AI Policy Engine — Configuration & Constants (v2 — Nuclear Upgrade)

Defines the action space (16 core + 3 meta), state bounds, initial conditions
for each task, and all tunable parameters for the governance simulation.
"""

from typing import Dict, List, Tuple

# ─────────────────────────────────────────────────────────────
# Action Space  (16 core + 3 meta = 19 total)
# ─────────────────────────────────────────────────────────────

VALID_ACTIONS: List[str] = [
    # ── Core policy actions (original 16) ──
    "no_action",
    "increase_tax",
    "decrease_tax",
    "stimulate_economy",
    "reduce_interest_rates",
    "expand_industry",
    "restrict_polluting_industries",
    "incentivize_clean_tech",
    "enforce_emission_limits",
    "subsidize_renewables",
    "implement_carbon_tax",
    "increase_welfare",
    "invest_in_healthcare",
    "invest_in_education",
    "upgrade_energy_grid",
    "invest_in_transport",
    # ── Meta-actions (council-level) ──
    "propose_global_policy_package",   # bundles 2-3 actions voted as one unit
    "force_emergency_coalition_vote",  # triggers immediate coalition vote
    "reset_institutional_trust",       # high cost, high reward if trust recovers
]

# Core actions (original 16) — used by baselines & validation
CORE_ACTIONS: List[str] = VALID_ACTIONS[:16]

# Meta-actions subset
META_ACTIONS: List[str] = VALID_ACTIONS[16:]

ACTION_DESCRIPTIONS: Dict[str, str] = {
    "no_action": "Take no policy action this turn.",
    "increase_tax": "Raise tax rates — boosts revenue but discourages investment.",
    "decrease_tax": "Lower tax rates — stimulates growth but reduces public services.",
    "stimulate_economy": "Inject stimulus — lowers unemployment but raises inflation and pollution.",
    "reduce_interest_rates": "Cut interest rates — cheaper borrowing, risk of inflation.",
    "expand_industry": "Expand industrial capacity — GDP and jobs up, pollution up.",
    "restrict_polluting_industries": "Restrict dirty industry — pollution down, jobs lost.",
    "incentivize_clean_tech": "Fund clean-tech R&D — delayed green benefits, upfront cost.",
    "enforce_emission_limits": "Impose strict emissions caps — fast pollution drop, industry hit.",
    "subsidize_renewables": "Subsidize renewable energy — large delayed green gains.",
    "implement_carbon_tax": "Tax carbon emissions — strong pollution reducer, hurts industry.",
    "increase_welfare": "Boost welfare spending — satisfaction up, fiscal cost.",
    "invest_in_healthcare": "Invest in healthcare — delayed health gains.",
    "invest_in_education": "Invest in education — long-term GDP and equality gains.",
    "upgrade_energy_grid": "Modernize energy infrastructure — delayed efficiency gains.",
    "invest_in_transport": "Improve transport networks — delayed efficiency and GDP.",
    # Meta-actions
    "propose_global_policy_package": "Propose a bundled package of 2-3 actions for coalition vote.",
    "force_emergency_coalition_vote": "Override normal debate; force immediate coalition decision.",
    "reset_institutional_trust": "Costly reset of institutional trust — high risk, high reward.",
}

# ─────────────────────────────────────────────────────────────
# State Metric Bounds  (metric_name -> (min, max))
# ─────────────────────────────────────────────────────────────

STATE_BOUNDS: Dict[str, Tuple[float, float]] = {
    # Environmental
    "pollution_index": (0.0, 300.0),
    "carbon_emission_rate": (0.0, 100.0),
    "renewable_energy_ratio": (0.0, 1.0),
    "ecological_stability": (0.0, 100.0),
    # Economic
    "gdp_index": (0.0, 200.0),
    "industrial_output": (0.0, 100.0),
    "unemployment_rate": (0.0, 50.0),
    "inflation_rate": (-10.0, 30.0),
    "trade_balance": (-100.0, 100.0),
    "foreign_investment": (0.0, 100.0),
    # Social
    "public_satisfaction": (0.0, 100.0),
    "healthcare_index": (0.0, 100.0),
    "education_index": (0.0, 100.0),
    "inequality_index": (0.0, 100.0),
    # Infrastructure
    "energy_efficiency": (0.0, 100.0),
    "transport_efficiency": (0.0, 100.0),
    # Policy knobs
    "tax_rate": (0.0, 50.0),
    "regulation_strength": (0.0, 100.0),
    "welfare_spending": (0.0, 100.0),
    "green_subsidies": (0.0, 100.0),
    "interest_rate": (0.0, 20.0),
}

# ─────────────────────────────────────────────────────────────
# Non-stationary drift variable bounds
# ─────────────────────────────────────────────────────────────

DRIFT_BOUNDS: Dict[str, Tuple[float, float]] = {
    "climate_sensitivity":     (0.5, 2.0),   # multiplier on climate events
    "inequality_tolerance":    (0.2, 1.0),   # public tolerance of inequality
    "public_trust_decay":      (0.0, 0.5),   # per-step satisfaction drain
    "supply_chain_resilience": (0.3, 1.0),   # dampens trade/industry shocks
    "institutional_trust":     (0.0, 1.0),   # slow global trust state
    "policy_fatigue":          (0.0, 1.0),   # reduces effect of repeated actions
}

DRIFT_INITIAL: Dict[str, float] = {
    "climate_sensitivity":     1.0,
    "inequality_tolerance":    0.6,
    "public_trust_decay":      0.05,
    "supply_chain_resilience": 0.7,
    "institutional_trust":     0.6,
    "policy_fatigue":          0.1,
}

# ─────────────────────────────────────────────────────────────
# Observation augmentation dimensions
# ─────────────────────────────────────────────────────────────

OBS_CORE_DIM       = 21   # original state keys
OBS_INFLUENCE_DIM  = 5    # per-minister influence scores
OBS_RISK_DIM       = 15   # collapse prob at 5/10/20 steps x 5 metrics
OBS_HISTORY_DIM    = 8    # rolling last-8 joint actions (encoded)
OBS_TRUST_DIM      = 1    # institutional_trust scalar
OBS_COALITION_DIM  = 5    # coalition strength per minister
OBS_TOTAL_DIM      = (
    OBS_CORE_DIM + OBS_INFLUENCE_DIM + OBS_RISK_DIM
    + OBS_HISTORY_DIM + OBS_TRUST_DIM + OBS_COALITION_DIM
)   # = 55

# ─────────────────────────────────────────────────────────────
# Default (baseline) initial state
# ─────────────────────────────────────────────────────────────

DEFAULT_STATE: Dict[str, float] = {
    "pollution_index": 100.0,
    "carbon_emission_rate": 50.0,
    "renewable_energy_ratio": 0.20,
    "ecological_stability": 70.0,
    "gdp_index": 100.0,
    "industrial_output": 60.0,
    "unemployment_rate": 8.0,
    "inflation_rate": 3.0,
    "trade_balance": 5.0,
    "foreign_investment": 50.0,
    "public_satisfaction": 60.0,
    "healthcare_index": 55.0,
    "education_index": 50.0,
    "inequality_index": 40.0,
    "energy_efficiency": 50.0,
    "transport_efficiency": 50.0,
    "tax_rate": 25.0,
    "regulation_strength": 40.0,
    "welfare_spending": 30.0,
    "green_subsidies": 10.0,
    "interest_rate": 5.0,
}

# ─────────────────────────────────────────────────────────────
# Collapse thresholds — episode ends with failure
# ─────────────────────────────────────────────────────────────

COLLAPSE_CONDITIONS: Dict[str, Tuple[str, float]] = {
    "gdp_collapse": ("gdp_index", 15.0),            # GDP drops below 15
    "eco_collapse": ("pollution_index", 290.0),      # Pollution exceeds 290
    "social_collapse": ("public_satisfaction", 5.0), # Satisfaction below 5
}

# ─────────────────────────────────────────────────────────────
# Reward weights
# ─────────────────────────────────────────────────────────────

REWARD_WEIGHTS = {
    "economic": 0.30,
    "environmental": 0.30,
    "social": 0.25,
    "stability": 0.15,
}

# Pareto shaping parameters
PARETO_BONUS_MAX = 0.15       # maximum bonus from Pareto proximity
PARETO_UPDATE_FREQ = 10       # steps between Pareto front updates

# Long-horizon credit parameters
LONG_HORIZON_WINDOW = 80      # last N steps for terminal credit
LONG_HORIZON_BONUS  = 0.12   # bonus per resolved delayed effect

# Cooperation multiplier cap
COOPERATION_MULTIPLIER_MAX = 1.30

# Key metrics tracked for stability / volatility scoring
KEY_METRICS: List[str] = [
    "pollution_index", "gdp_index", "public_satisfaction",
    "healthcare_index", "unemployment_rate", "renewable_energy_ratio",
]

# ─────────────────────────────────────────────────────────────
# Task-specific initial states & parameters
# ─────────────────────────────────────────────────────────────

TASK_CONFIGS: Dict[str, dict] = {
    "environmental_recovery": {
        "description": (
            "Environmental Recovery (Easy): Reduce dangerously high pollution to "
            "safe levels (below 80) while keeping GDP above 60. "
            "Events are disabled — focus purely on green policy."
        ),
        "max_steps": 50,
        "events_enabled": False,
        "event_frequency_multiplier": 0.0,
        "chaos_level": 0.0,
        "drift_enabled": False,
        "num_ministers": 1,
        "initial_state_overrides": {
            "pollution_index": 170.0,
            "carbon_emission_rate": 65.0,
            "renewable_energy_ratio": 0.15,
            "ecological_stability": 55.0,
            "gdp_index": 105.0,
            "industrial_output": 65.0,
            "unemployment_rate": 5.0,
            "public_satisfaction": 70.0,
            "healthcare_index": 55.0,
            "inequality_index": 30.0,
        },
    },
    "balanced_economy": {
        "description": (
            "Balanced Economy (Medium): Simultaneously maintain GDP > 80, "
            "pollution < 100, and public satisfaction > 60 over 100 steps. "
            "Random events occur at reduced frequency."
        ),
        "max_steps": 100,
        "events_enabled": True,
        "event_frequency_multiplier": 0.5,
        "chaos_level": 0.3,
        "drift_enabled": False,
        "num_ministers": 1,
        "initial_state_overrides": {
            "pollution_index": 140.0,
            "carbon_emission_rate": 60.0,
            "renewable_energy_ratio": 0.15,
            "ecological_stability": 55.0,
            "gdp_index": 85.0,
            "industrial_output": 55.0,
            "unemployment_rate": 12.0,
            "inflation_rate": 5.0,
            "public_satisfaction": 48.0,
            "healthcare_index": 48.0,
            "education_index": 45.0,
            "inequality_index": 48.0,
        },
    },
    "sustainable_governance": {
        "description": (
            "Sustainable Governance (Hard): Maintain stability across all dimensions "
            "for 200 steps under stochastic events with calibrated intensity. "
            "Satisfaction shocks are dampened, making survival possible but non-trivial. "
            "Agents must actively manage social stability to avoid collapse."
        ),
        "max_steps": 200,
        "events_enabled": True,
        "event_frequency_multiplier": 1.0,
        "satisfaction_event_scale": 0.4,    # calibrated: 40% event satisfaction impact
        "satisfaction_floor_damping": 0.8,  # when sat < 35, absorb 80% of losses
        "crisis_welfare_bonus": 8.0,        # extra +8.0 sat when social actions used during crisis
        "chaos_level": 0.6,
        "drift_enabled": True,
        "num_ministers": 3,
        "initial_state_overrides": {
            "pollution_index": 130.0,
            "carbon_emission_rate": 55.0,
            "renewable_energy_ratio": 0.18,
            "ecological_stability": 60.0,
            "gdp_index": 90.0,
            "industrial_output": 58.0,
            "unemployment_rate": 7.0,
            "inflation_rate": 4.0,
            "trade_balance": 0.0,
            "foreign_investment": 45.0,
            "public_satisfaction": 65.0,
            "healthcare_index": 50.0,
            "education_index": 48.0,
            "inequality_index": 38.0,
            "energy_efficiency": 45.0,
            "transport_efficiency": 45.0,
            "tax_rate": 28.0,
            "regulation_strength": 35.0,
            "welfare_spending": 25.0,
            "green_subsidies": 8.0,
            "interest_rate": 6.0,
        },
    },
    "sustainable_governance_extreme": {
        "description": (
            "Sustainable Governance — EXTREME: Unstable regime with irreversible "
            "cascade dynamics under full event pressure. All tested strategies "
            "(Random, Heuristic, RL) collapse 100% of the time. Satisfaction shocks "
            "dominate, revealing a structural instability that no reactive policy "
            "can overcome. Serves as a failure-mode analysis benchmark."
        ),
        "max_steps": 200,
        "events_enabled": True,
        "event_frequency_multiplier": 1.0,
        "satisfaction_event_scale": 1.0,    # full intensity: structural collapse
        "chaos_level": 1.0,
        "drift_enabled": True,
        "num_ministers": 5,
        "initial_state_overrides": {
            "pollution_index": 130.0,
            "carbon_emission_rate": 55.0,
            "renewable_energy_ratio": 0.18,
            "ecological_stability": 60.0,
            "gdp_index": 90.0,
            "industrial_output": 58.0,
            "unemployment_rate": 10.0,
            "inflation_rate": 4.0,
            "trade_balance": 0.0,
            "foreign_investment": 45.0,
            "public_satisfaction": 52.0,
            "healthcare_index": 50.0,
            "education_index": 48.0,
            "inequality_index": 45.0,
            "energy_efficiency": 45.0,
            "transport_efficiency": 45.0,
            "tax_rate": 28.0,
            "regulation_strength": 35.0,
            "welfare_spending": 25.0,
            "green_subsidies": 8.0,
            "interest_rate": 6.0,
        },
    },
    "multi_agent_council": {
        "description": (
            "Multi-Agent Council (Extreme+): 5 ministers with emergent negotiation, "
            "coalition formation, vetoes, and dynamic utility vectors. Full chaos "
            "and non-stationary drift. Designed for multi-agent RL research."
        ),
        "max_steps": 300,
        "events_enabled": True,
        "event_frequency_multiplier": 1.2,
        "satisfaction_event_scale": 0.7,
        "satisfaction_floor_damping": 0.5,
        "crisis_welfare_bonus": 6.0,
        "chaos_level": 0.8,
        "drift_enabled": True,
        "num_ministers": 5,
        "initial_state_overrides": {
            "pollution_index": 120.0,
            "carbon_emission_rate": 52.0,
            "renewable_energy_ratio": 0.20,
            "ecological_stability": 62.0,
            "gdp_index": 92.0,
            "industrial_output": 60.0,
            "unemployment_rate": 8.0,
            "inflation_rate": 3.5,
            "trade_balance": 2.0,
            "foreign_investment": 50.0,
            "public_satisfaction": 58.0,
            "healthcare_index": 52.0,
            "education_index": 50.0,
            "inequality_index": 40.0,
            "energy_efficiency": 48.0,
            "transport_efficiency": 48.0,
            "tax_rate": 26.0,
            "regulation_strength": 38.0,
            "welfare_spending": 28.0,
            "green_subsidies": 10.0,
            "interest_rate": 5.5,
        },
    },
}
