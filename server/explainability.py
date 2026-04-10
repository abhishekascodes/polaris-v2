"""
AI Policy Engine — Research-Grade Explainability Layer (v2 — Nuclear Upgrade)

Upgrades:
  - Structured causal chain with cross-layer interactions from new engines.
  - Counterfactual analysis for last 3 major decisions (veto, coalition,
    high-impact action) with quantitative deltas.
  - Per-agent credit attribution per step.
  - Alignment Score (0–100): balance between short-term individual utilities
    and long-term global objectives.
  - Natural language summary generator for latest causal chain + counterfactuals.
"""

from __future__ import annotations

import math
from collections import deque
from typing import Any, Deque, Dict, List, Optional, Tuple

from .config import COLLAPSE_CONDITIONS, STATE_BOUNDS


# ------------------------------------------------------------------
# Causal chain types
# ------------------------------------------------------------------

class CausalLink:
    """One link in the causal reasoning chain."""

    __slots__ = ("layer", "trigger", "effect", "severity")

    def __init__(self, layer: str, trigger: str, effect: str, severity: str = "info"):
        self.layer = layer
        self.trigger = trigger
        self.effect = effect
        self.severity = severity

    def to_dict(self) -> dict:
        return {
            "layer": self.layer,
            "trigger": self.trigger,
            "effect": self.effect,
            "severity": self.severity,
        }

    def __repr__(self) -> str:
        tag = {" info": " ", "warning": "!", "critical": "X"}.get(self.severity, "?")
        return f"[{tag}] [{self.layer}] {self.trigger} -> {self.effect}"


# ------------------------------------------------------------------
# Decision record (for counterfactual tracking)
# ------------------------------------------------------------------

class MajorDecision:
    """A major decision event recorded for counterfactual analysis."""
    __slots__ = ("step", "decision_type", "action", "state_snapshot", "outcome_delta")

    def __init__(
        self, step: int, decision_type: str, action: str,
        state_snapshot: Dict[str, float], outcome_delta: Optional[Dict[str, float]] = None
    ):
        self.step = step
        self.decision_type = decision_type  # "veto" | "coalition" | "high_impact"
        self.action = action
        self.state_snapshot = state_snapshot
        self.outcome_delta = outcome_delta or {}


# ------------------------------------------------------------------
# Explainability engine
# ------------------------------------------------------------------

class ExplainabilityEngine:
    """
    Analyses pre/post state diffs and action context to produce
    structured causal explanations, counterfactual analysis,
    credit attribution, and alignment scores.
    """

    def __init__(self) -> None:
        self._major_decisions: Deque[MajorDecision] = deque(maxlen=3)
        self._step_credits: List[Dict[str, float]] = []

    def reset(self) -> None:
        self._major_decisions.clear()
        self._step_credits.clear()

    def explain(
        self,
        action: str,
        prev_state: Optional[Dict[str, float]],
        curr_state: Dict[str, float],
        active_events: List[str],
        step: int,
        council_result: Optional[Dict] = None,
        drift_vars: Optional[Dict[str, float]] = None,
        resolved_delayed: int = 0,
    ) -> Dict:
        """
        Generate a full explanation for one step.

        Returns a dict with:
          - causal_chain: list of CausalLink dicts
          - summary: human-readable 2-sentence summary
          - risk_alerts: approaching-threshold warnings
          - delta_report: top metric changes
          - counterfactuals: analysis of last 3 major decisions
          - credit_attribution: per-agent credits
          - alignment_score: 0–100
          - nl_narrative: natural language narrative
        """
        if prev_state is None:
            return {
                "causal_chain": [],
                "summary": "Episode initialised.",
                "risk_alerts": [],
                "delta_report": {},
                "counterfactuals": [],
                "credit_attribution": {},
                "alignment_score": 50.0,
                "nl_narrative": "The episode has just started. No actions taken yet.",
            }

        chain: List[CausalLink] = []
        deltas = self._compute_deltas(prev_state, curr_state)

        # -- Layer 1: Deterministic action effects --
        chain.extend(self._explain_action(action, deltas))

        # -- Layer 2: Non-linear threshold effects --
        chain.extend(self._explain_nonlinear(prev_state, curr_state, drift_vars or {}))

        # -- Layer 4: Feedback loops --
        chain.extend(self._explain_feedback(curr_state, drift_vars or {}))

        # -- Events --
        chain.extend(self._explain_events(active_events))

        # -- Cross-layer: drift interactions --
        if drift_vars:
            chain.extend(self._explain_drift(drift_vars, curr_state))

        # -- Delayed effects materialising --
        if resolved_delayed > 0:
            chain.append(CausalLink(
                "delayed",
                f"{resolved_delayed} queued effect(s) fired",
                "State updated from previously-invested policies",
                "info",
            ))

        # -- Council / negotiation --
        if council_result:
            chain.extend(self._explain_council(council_result, action))

        # -- Risk alerts --
        risk_alerts = self._check_risk_proximity(curr_state)

        # -- Record major decisions --
        self._record_major_decision(step, action, deltas, curr_state, council_result)

        # -- Counterfactual analysis --
        counterfactuals = self._compute_counterfactuals()

        # -- Credit attribution --
        credit_attribution = {}
        if council_result:
            credit_attribution = council_result.get("credit_deltas", {})
            self._step_credits.append(credit_attribution)

        # -- Alignment score (from council or default) --
        alignment_score = 50.0
        if council_result:
            alignment_score = council_result.get("alignment_score", 50.0)

        # -- Build summary --
        summary = self._build_summary(action, deltas, chain, risk_alerts)

        # -- Top deltas --
        sorted_deltas = sorted(deltas.items(), key=lambda x: abs(x[1]), reverse=True)
        delta_report = {k: round(v, 2) for k, v in sorted_deltas[:8] if abs(v) > 0.5}

        # -- Natural language narrative --
        nl_narrative = self._generate_nl_narrative(
            action, deltas, chain, risk_alerts, counterfactuals, alignment_score
        )

        return {
            "causal_chain": [c.to_dict() for c in chain],
            "summary": summary,
            "risk_alerts": risk_alerts,
            "delta_report": delta_report,
            "counterfactuals": counterfactuals,
            "credit_attribution": credit_attribution,
            "alignment_score": round(alignment_score, 2),
            "nl_narrative": nl_narrative,
        }

    # ------------------------------------------------------------------
    # Internals — causal chain builders
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_deltas(prev: Dict[str, float], curr: Dict[str, float]) -> Dict[str, float]:
        deltas = {}
        for k in curr:
            if k in prev and isinstance(curr[k], (int, float)):
                d = curr[k] - prev[k]
                if abs(d) > 0.01:
                    deltas[k] = d
        return deltas

    @staticmethod
    def _explain_action(action: str, deltas: Dict[str, float]) -> List[CausalLink]:
        chain = []
        if action == "no_action":
            return chain

        green_actions = {
            "restrict_polluting_industries", "enforce_emission_limits",
            "subsidize_renewables", "implement_carbon_tax", "incentivize_clean_tech",
        }
        econ_actions = {
            "increase_tax", "decrease_tax", "stimulate_economy",
            "reduce_interest_rates", "expand_industry",
        }
        social_actions = {"increase_welfare", "invest_in_healthcare", "invest_in_education"}
        infra_actions = {"upgrade_energy_grid", "invest_in_transport"}
        meta_actions = {"propose_global_policy_package", "force_emergency_coalition_vote",
                        "reset_institutional_trust"}

        if action in green_actions:
            if "pollution_index" in deltas and deltas["pollution_index"] < 0:
                chain.append(CausalLink(
                    "deterministic", f"Action '{action}' executed",
                    f"Pollution reduced by {abs(deltas['pollution_index']):.1f} points",
                ))
            if "gdp_index" in deltas and deltas["gdp_index"] < 0:
                chain.append(CausalLink(
                    "deterministic", "Green regulation cost",
                    f"GDP decreased by {abs(deltas['gdp_index']):.1f} (trade-off)",
                    "warning",
                ))
            if "ecological_stability" in deltas and deltas["ecological_stability"] > 0:
                chain.append(CausalLink(
                    "deterministic", "Ecological benefit",
                    f"Ecological stability improved by {deltas['ecological_stability']:.1f}",
                ))
        elif action in econ_actions:
            if "gdp_index" in deltas and deltas["gdp_index"] > 0:
                chain.append(CausalLink(
                    "deterministic", f"Action '{action}' executed",
                    f"GDP increased by {deltas['gdp_index']:.1f} points",
                ))
            if "pollution_index" in deltas and deltas["pollution_index"] > 0:
                chain.append(CausalLink(
                    "deterministic", "Economic expansion side-effect",
                    f"Pollution increased by {deltas['pollution_index']:.1f} (externality)",
                    "warning",
                ))
        elif action in social_actions:
            chain.append(CausalLink(
                "deterministic", f"Action '{action}' executed",
                "Social investment initiated (effects may be delayed)",
            ))
        elif action in infra_actions:
            chain.append(CausalLink(
                "deterministic", f"Action '{action}' executed",
                "Infrastructure investment initiated (delayed returns expected)",
            ))
        elif action in meta_actions:
            chain.append(CausalLink(
                "deterministic", f"Meta-action '{action}' executed",
                "Council-level coordination triggered — institutional effects apply",
                "info",
            ))

        return chain

    @staticmethod
    def _explain_nonlinear(
        prev: Dict[str, float], curr: Dict[str, float], drift_vars: Dict[str, float]
    ) -> List[CausalLink]:
        chain = []
        climate_sens = drift_vars.get("climate_sensitivity", 1.0)

        if prev.get("pollution_index", 0) <= 200 < curr.get("pollution_index", 0):
            chain.append(CausalLink(
                "nonlinear",
                "Pollution exceeded 200 (safe threshold)",
                "Healthcare and ecological stability now degrading exponentially",
                "critical",
            ))
        elif curr.get("pollution_index", 0) > 200:
            chain.append(CausalLink(
                "nonlinear",
                f"Pollution at {curr['pollution_index']:.0f} (above 200, sens={climate_sens:.1f}x)",
                "Ongoing exponential health damage",
                "warning",
            ))

        if curr.get("tax_rate", 0) > 40:
            chain.append(CausalLink(
                "nonlinear",
                f"Tax rate at {curr['tax_rate']:.0f}% (above 40% threshold)",
                "Quadratic GDP suppression active",
                "warning",
            ))

        if curr.get("gdp_index", 100) < 40:
            chain.append(CausalLink(
                "nonlinear",
                f"GDP at {curr['gdp_index']:.0f} (below 40 depression threshold)",
                "Unemployment rising, satisfaction dropping from economic depression",
                "critical",
            ))

        if curr.get("unemployment_rate", 0) > 25:
            chain.append(CausalLink(
                "nonlinear",
                f"Unemployment at {curr['unemployment_rate']:.0f}% (crisis level)",
                "Severe satisfaction drain and rising inequality",
                "critical",
            ))

        # NEW: pollution->education ROI cross-layer
        if curr.get("pollution_index", 0) > 180:
            chain.append(CausalLink(
                "nonlinear",
                f"Pollution at {curr['pollution_index']:.0f} > 180 (education ROI threshold)",
                "Education returns reduced 30% — learning environments degraded",
                "warning",
            ))

        return chain

    @staticmethod
    def _explain_feedback(
        curr: Dict[str, float], drift_vars: Dict[str, float]
    ) -> List[CausalLink]:
        chain = []
        inst_trust = drift_vars.get("institutional_trust", 0.6)
        trust_decay = drift_vars.get("public_trust_decay", 0.05)

        if curr.get("healthcare_index", 50) < 30:
            chain.append(CausalLink(
                "feedback",
                f"Healthcare index at {curr['healthcare_index']:.0f} (critically low)",
                "Health-Productivity loop: industrial output reduced, unemployment rising",
                "warning",
            ))

        if curr.get("education_index", 50) > 70:
            chain.append(CausalLink(
                "feedback",
                f"Education index at {curr['education_index']:.0f} (high)",
                "Education-Innovation loop: GDP receiving innovation bonus each step",
            ))

        if curr.get("inequality_index", 40) > 60:
            chain.append(CausalLink(
                "feedback",
                f"Inequality at {curr['inequality_index']:.0f} (above 60 threshold)",
                "Inequality-Satisfaction loop: public satisfaction being eroded",
                "warning",
            ))

        rer = curr.get("renewable_energy_ratio", 0)
        if rer > 0.3:
            chain.append(CausalLink(
                "feedback",
                f"Renewable ratio at {rer:.0%} (above 30% threshold)",
                "Renewable dividend: automatic pollution and emission reduction",
            ))

        if trust_decay > 0.1:
            chain.append(CausalLink(
                "feedback",
                f"Public trust decay at {trust_decay:.3f} (elevated)",
                "Baseline satisfaction erosion active each step",
                "warning",
            ))

        if inst_trust < 0.3:
            chain.append(CausalLink(
                "feedback",
                f"Institutional trust at {inst_trust:.2f} (critically low)",
                "Foreign investment fleeing, satisfaction under persistent pressure",
                "critical",
            ))

        return chain

    @staticmethod
    def _explain_drift(drift_vars: Dict[str, float], curr: Dict[str, float]) -> List[CausalLink]:
        chain = []
        cs = drift_vars.get("climate_sensitivity", 1.0)
        pf = drift_vars.get("policy_fatigue", 0.1)
        sr = drift_vars.get("supply_chain_resilience", 0.7)

        if cs > 1.4:
            chain.append(CausalLink(
                "drift",
                f"Climate sensitivity drifted to {cs:.2f} (elevated)",
                "Climate and pollution events now amplified beyond baseline",
                "warning",
            ))
        if pf > 0.5:
            chain.append(CausalLink(
                "drift",
                f"Policy fatigue at {pf:.2f} (high)",
                "Repeated policy actions losing up to 30% effectiveness",
                "warning",
            ))
        if sr < 0.4:
            chain.append(CausalLink(
                "drift",
                f"Supply chain resilience at {sr:.2f} (fragile)",
                "Trade shocks and industrial disruptions amplified",
                "warning",
            ))

        return chain

    @staticmethod
    def _explain_events(active_events: List[str]) -> List[CausalLink]:
        chain = []
        event_impacts = {
            "pandemic": "GDP falling, unemployment rising, healthcare strained",
            "industrial_boom": "GDP and industry surging but pollution increasing",
            "climate_crisis": "Pollution spiking, ecology destabilised",
            "public_protest": "Satisfaction dropping, foreign investment fleeing",
            "tech_breakthrough": "Renewables and efficiency accelerating",
            "trade_war": "Trade balance collapsing, investment declining",
            "natural_disaster": "Infrastructure damaged, GDP hit, public distress",
            "economic_recession": "Broad economic contraction, unemployment rising",
        }
        for event in active_events:
            impact = event_impacts.get(event, "Unknown effects")
            chain.append(CausalLink(
                "event",
                f"Event active: {event.replace('_', ' ').upper()}",
                impact,
                "critical" if event in ("pandemic", "natural_disaster", "economic_recession")
                else "warning",
            ))
        return chain

    @staticmethod
    def _explain_council(council_result: Dict, action: str) -> List[CausalLink]:
        chain = []
        vetoes = council_result.get("vetoes", [])
        coalition = council_result.get("coalition_formed", False)
        alignment = council_result.get("alignment_score", 50.0)

        if vetoes:
            chain.append(CausalLink(
                "council",
                f"Ministers {vetoes} vetoed action '{action}'",
                "Action proceeded despite opposition — potential future instability",
                "warning",
            ))
        if coalition:
            strength = council_result.get("coalition_strength", 0.0)
            chain.append(CausalLink(
                "council",
                f"Coalition formed (strength={strength:.2f})",
                f"Consensus reached on '{action}' — cooperation bonus applied",
                "info",
            ))
        if alignment < 40:
            chain.append(CausalLink(
                "council",
                f"Alignment score low ({alignment:.0f}/100)",
                "Individual ministerial utilities diverging from global objectives",
                "warning",
            ))

        return chain

    # ------------------------------------------------------------------
    # Counterfactual Analysis
    # ------------------------------------------------------------------

    def _record_major_decision(
        self,
        step: int,
        action: str,
        deltas: Dict[str, float],
        curr_state: Dict[str, float],
        council_result: Optional[Dict],
    ) -> None:
        """Record a decision as 'major' if it's a veto, coalition, or high-impact."""
        decision_type = None

        if council_result:
            if council_result.get("vetoes"):
                decision_type = "veto"
            elif council_result.get("coalition_formed"):
                decision_type = "coalition"

        # High-impact: top delta > 10
        if decision_type is None:
            max_delta = max((abs(v) for v in deltas.values()), default=0)
            if max_delta > 10:
                decision_type = "high_impact"

        if decision_type:
            self._major_decisions.append(MajorDecision(
                step=step,
                decision_type=decision_type,
                action=action,
                state_snapshot=dict(curr_state),
                outcome_delta=dict(deltas),
            ))

    def _compute_counterfactuals(self) -> List[Dict]:
        """
        Compute counterfactual analysis for last 3 major decisions.
        Uses analytical approximations: alternative action would have had
        opposite sign on the largest delta metrics.
        """
        results = []
        for decision in self._major_decisions:
            # Estimate alternative outcome: inverse of top 3 deltas
            alt_deltas = {}
            sorted_deltas = sorted(
                decision.outcome_delta.items(), key=lambda x: abs(x[1]), reverse=True
            )[:3]
            for k, v in sorted_deltas:
                # Counterfactual: opposite direction with 60% magnitude
                alt_deltas[k] = -v * 0.6

            # Estimate delta-reward from alternative
            actual_sat_delta = decision.outcome_delta.get("public_satisfaction", 0.0)
            actual_gdp_delta = decision.outcome_delta.get("gdp_index", 0.0)
            actual_poll_delta = decision.outcome_delta.get("pollution_index", 0.0)

            cf_sat_delta = alt_deltas.get("public_satisfaction", 0.0)
            cf_gdp_delta = alt_deltas.get("gdp_index", 0.0)
            cf_poll_delta = alt_deltas.get("pollution_index", 0.0)

            delta_reward = (
                0.3 * (cf_gdp_delta - actual_gdp_delta) / 100.0
                - 0.3 * (cf_poll_delta - actual_poll_delta) / 200.0
                + 0.25 * (cf_sat_delta - actual_sat_delta) / 50.0
            )

            results.append({
                "step": decision.step,
                "decision_type": decision.decision_type,
                "actual_action": decision.action,
                "actual_top_deltas": {k: round(v, 2) for k, v in sorted_deltas},
                "counterfactual_deltas": {k: round(v, 2) for k, v in alt_deltas.items()},
                "estimated_delta_reward": round(delta_reward, 4),
                "interpretation": (
                    f"Alternative approach would have yielded ~{delta_reward:+.3f} "
                    f"reward delta vs actual action '{decision.action}'."
                ),
            })

        return results

    # ------------------------------------------------------------------
    # Risk alerts
    # ------------------------------------------------------------------

    @staticmethod
    def _check_risk_proximity(curr: Dict[str, float]) -> List[str]:
        alerts = []

        gdp = curr.get("gdp_index", 100)
        if gdp < 35:
            alerts.append(f"CRITICAL: GDP at {gdp:.0f}, collapse threshold is 15")
        elif gdp < 50:
            alerts.append(f"WARNING: GDP at {gdp:.0f}, approaching danger zone")

        poll = curr.get("pollution_index", 100)
        if poll > 260:
            alerts.append(f"CRITICAL: Pollution at {poll:.0f}, collapse threshold is 290")
        elif poll > 220:
            alerts.append(f"WARNING: Pollution at {poll:.0f}, approaching danger zone")

        sat = curr.get("public_satisfaction", 50)
        if sat < 15:
            alerts.append(f"CRITICAL: Satisfaction at {sat:.0f}, collapse threshold is 5")
        elif sat < 25:
            alerts.append(f"WARNING: Satisfaction at {sat:.0f}, approaching danger zone")

        return alerts

    # ------------------------------------------------------------------
    # Summary & NL Narrative
    # ------------------------------------------------------------------

    @staticmethod
    def _build_summary(
        action: str,
        deltas: Dict[str, float],
        chain: List[CausalLink],
        risk_alerts: List[str],
    ) -> str:
        parts = []
        crits = sum(1 for c in chain if c.severity == "critical")
        warns = sum(1 for c in chain if c.severity == "warning")

        if crits > 0:
            parts.append(f"{crits} critical condition(s) active")
        if warns > 0:
            parts.append(f"{warns} warning(s)")

        if deltas:
            top_k = sorted(deltas.keys(), key=lambda k: abs(deltas[k]), reverse=True)[0]
            direction = "rose" if deltas[top_k] > 0 else "fell"
            parts.append(f"{top_k} {direction} by {abs(deltas[top_k]):.1f}")

        if risk_alerts:
            parts.append(f"{len(risk_alerts)} risk alert(s)")

        if not parts:
            return f"Action '{action}' applied with minimal state change."

        return f"After '{action}': " + "; ".join(parts) + "."

    @staticmethod
    def _generate_nl_narrative(
        action: str,
        deltas: Dict[str, float],
        chain: List[CausalLink],
        risk_alerts: List[str],
        counterfactuals: List[Dict],
        alignment_score: float,
    ) -> str:
        """Generate a 2-sentence natural language narrative."""
        # Sentence 1: What happened
        crits = [c for c in chain if c.severity == "critical"]
        warns = [c for c in chain if c.severity == "warning"]

        if crits:
            s1 = f"Policy action '{action.replace('_', ' ')}' triggered {len(crits)} critical cascade(s): {crits[0].effect.lower()}"
        elif warns:
            top_warn = warns[0]
            s1 = f"'{action.replace('_', ' ')}' was applied, causing {top_warn.trigger.lower()} with effect: {top_warn.effect.lower()}"
        elif deltas:
            top_k = sorted(deltas.keys(), key=lambda k: abs(deltas[k]), reverse=True)[0]
            direction = "rose" if deltas[top_k] > 0 else "fell"
            s1 = f"Policy '{action.replace('_', ' ')}' caused {top_k.replace('_', ' ')} to {direction} by {abs(deltas[top_k]):.1f} points"
        else:
            s1 = f"Policy '{action.replace('_', ' ')}' was applied with limited immediate effects"

        # Sentence 2: Counterfactual or alignment insight
        if counterfactuals:
            cf = counterfactuals[-1]
            dr = cf["estimated_delta_reward"]
            s2 = (f"Counterfactual analysis suggests an alternative to '{cf['actual_action'].replace('_', ' ')}' "
                  f"would have changed reward by {dr:+.3f}; council alignment is {alignment_score:.0f}/100.")
        elif risk_alerts:
            s2 = f"Risk monitoring shows {len(risk_alerts)} active alert(s) — {risk_alerts[0].lower()}"
        else:
            s2 = f"Council alignment score is {alignment_score:.0f}/100, indicating {'strong' if alignment_score > 70 else 'moderate' if alignment_score > 40 else 'weak'} consensus."

        return s1 + ". " + s2 + "."
