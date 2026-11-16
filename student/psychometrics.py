"""
Psychometric State Vector & Behavioral Adaptation Engine for AgentSchool.
Theoretical Grounding:
1. Study Demands-Resources (SD-R) Framework (Lesener et al., 2020; Schaufeli et al., 2002)
2. Social Cognitive Theory / Self-Efficacy (Bandura, 1997)
3. Transactional Model of Stress and Coping (Lazarus & Folkman, 1984)
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Literal
import math
import random

CopingStyle = Literal["problem_focused", "support_seeking", "avoidant"]


def _clamp(value: float, low: float = 0.0, high: float = 100.0) -> float:
    return max(low, min(high, value))


@dataclass
class PsychometricState:
    """
    Maintains baseline psychometric priors (calibrated from Phase 1 N=120 empirical data)
    and dynamic weekly states (normalized 0-100 scale) for a simulated student agent.
    """
    student_id: str
    archetype: str  # e.g., "high_resilience", "moderate_vulnerability", "high_burnout_risk"

    # Static Calibration Priors (from BFI-44, Brief COPE, and baseline diagnostics)
    stress_sensitivity: float = 1.0        # Multiplier (0.6 = resilient, 1.4 = high reactivity)
    coping_style: CopingStyle = "problem_focused"
    recovery_rate: float = 6.0             # Baseline coping replenishment during low-demand intervals

    # Dynamic Psychological State Variables (0.0 to 100.0)
    emotional_exhaustion: float = 20.0     # MBI-SS Exhaustion dimension
    cynicism: float = 15.0                 # MBI-SS Cynicism / detachment dimension
    perceived_stress: float = 25.0         # PSS-10 normalized load
    academic_self_efficacy: float = 70.0   # Bandura domain efficacy
    coping_reserve: float = 80.0           # Available self-regulatory energy

    # Longitudinal History Log
    weekly_snapshots: List[Dict] = field(default_factory=list)

    def update_on_academic_event(
        self,
        demand_load: float,
        mastery_score: float,
        scaffolding_support: float = 0.5,
        feedback_delay_days: float = 2.0,
    ) -> Dict[str, float]:
        """
        Updates psychological states after a lesson, reading load, or assessment event.

        Args:
            demand_load: Curricular workload intensity (0.0 to 1.0).
            mastery_score: Performance/comprehension on task (0.0 to 1.0).
            scaffolding_support: Instructor ZPD support & clarity (0.0 to 1.0).
            feedback_delay_days: Days elapsed before formative feedback (0 to 14).
        """
        # 1. Transactional Appraisal (Lazarus & Folkman, 1984)
        # Demand threat increases if self-efficacy is low or feedback is delayed
        ambiguity_penalty = min(0.35, feedback_delay_days * 0.03)
        effective_demand = (demand_load + ambiguity_penalty) * self.stress_sensitivity
        resource_buffer = (scaffolding_support * 0.6) + (self.coping_reserve / 100.0 * 0.4)

        net_strain = effective_demand - resource_buffer

        # 2. Perceived Stress & Coping Reserve Depletion
        if net_strain > 0:
            stress_delta = net_strain * 18.0
            reserve_drain = net_strain * 14.0
            self.perceived_stress = _clamp(self.perceived_stress + stress_delta)
            self.coping_reserve = _clamp(self.coping_reserve - reserve_drain)
        else:
            # Adaptive recovery when resources exceed demands
            recovery = abs(net_strain) * self.recovery_rate
            self.perceived_stress = _clamp(self.perceived_stress - recovery * 1.2)
            self.coping_reserve = _clamp(self.coping_reserve + recovery)

        # 3. Academic Self-Efficacy Update (Bandura, 1997)
        # Mastery experiences strengthen efficacy; un-scaffolded failure erodes it
        mastery_gap = mastery_score - 0.60
        if mastery_gap >= 0:
            efficacy_delta = mastery_gap * (8.0 + 4.0 * scaffolding_support)
        else:
            # Failure hurts more when scaffolding is absent and stress is high
            vulnerability = 1.0 + (self.perceived_stress / 100.0) * 0.5
            efficacy_delta = mastery_gap * 14.0 * vulnerability

        self.academic_self_efficacy = _clamp(self.academic_self_efficacy + efficacy_delta)

        # 4. Academic Burnout Accumulation (SD-R Model / MBI-SS)
        # Chronic high stress + depleted coping drives emotional exhaustion
        if self.perceived_stress > 55.0 or self.coping_reserve < 35.0:
            exhaustion_gain = ((self.perceived_stress - 45.0) * 0.14) + ((40.0 - self.coping_reserve) * 0.08)
            self.emotional_exhaustion = _clamp(self.emotional_exhaustion + max(0.0, exhaustion_gain))
        else:
            self.emotional_exhaustion = _clamp(self.emotional_exhaustion - (self.recovery_rate * 0.35))

        # Cynicism rises when exhaustion is high and academic self-efficacy drops
        if self.emotional_exhaustion > 60.0 and self.academic_self_efficacy < 45.0:
            cynicism_gain = (self.emotional_exhaustion - 55.0) * 0.12
            self.cynicism = _clamp(self.cynicism + cynicism_gain)
        elif self.academic_self_efficacy > 65.0:
            self.cynicism = _clamp(self.cynicism - 1.5)

        return self.get_current_metrics()

    def get_behavioral_probabilities(self) -> Dict[str, float]:
        """
        Translates internal psychological state into observable behavioral adaptation
        probabilities (used to gate agent actions and compare against real LMS traces).
        """
        exhaustion_norm = self.emotional_exhaustion / 100.0
        efficacy_norm = self.academic_self_efficacy / 100.0
        cynicism_norm = self.cynicism / 100.0

        # Procrastination rises with exhaustion and low self-efficacy
        procrastination = _clamp((0.55 * exhaustion_norm + 0.45 * (1.0 - efficacy_norm)) * 100.0)
        if self.coping_style == "avoidant":
            procrastination = _clamp(procrastination * 1.25)

        # Help-seeking depends on coping style and drops when cynicism is high
        base_help = 0.65 if self.coping_style == "support_seeking" else 0.35
        help_seeking = _clamp((base_help * (1.0 - 0.6 * cynicism_norm) * (0.5 + 0.5 * exhaustion_norm)) * 100.0)

        # Disengagement / task avoidance probability
        disengagement = _clamp((0.5 * cynicism_norm + 0.5 * exhaustion_norm) * 90.0)

        return {
            "procrastination_prob": round(procrastination / 100.0, 3),
            "help_seeking_prob": round(help_seeking / 100.0, 3),
            "disengagement_prob": round(disengagement / 100.0, 3),
        }

    def sample_behavioral_action(self) -> str:
        """Samples a concrete behavioral adaptation state for the current simulation step."""
        probs = self.get_behavioral_probabilities()
        roll = random.random()
        if roll < probs["disengagement_prob"] * 0.5:
            return "task_avoidance_or_late_submission"
        if roll < probs["procrastination_prob"]:
            return "delayed_engagement_procrastination"
        if roll < probs["procrastination_prob"] + probs["help_seeking_prob"] * 0.5:
            return "active_help_seeking"
        return "timely_task_engagement"

    def to_system_prompt_modifier(self) -> str:
        """
        Generates a psychological state block injected into the AgentSchool student agent's
        `reflect` and `plan` prompts to prevent idealized LLM over-compliance.
        """
        action_tendency = self.sample_behavioral_action()
        return (
            f"[PSYCHOLOGICAL STATE PROFILE - {self.archetype.upper()}]\n"
            f"- Emotional Exhaustion: {self.emotional_exhaustion:.1f}/100 | "
            f"Cynicism: {self.cynicism:.1f}/100\n"
            f"- Perceived Stress: {self.perceived_stress:.1f}/100 | "
            f"Academic Self-Efficacy: {self.academic_self_efficacy:.1f}/100\n"
            f"- Coping Style: {self.coping_style} (Reserve: {self.coping_reserve:.1f}/100)\n"
            f"- Current Behavioral Tendency: {action_tendency}\n"
            f"INSTRUCTION: Reflect and act authentically according to these scores. "
            f"If exhaustion is above 65 or self-efficacy is below 40, express realistic cognitive fatigue, "
            f"hesitation, self-doubt, or avoidance rather than effortless mastery."
        )

    def record_week_snapshot(self, week: int) -> Dict:
        snapshot = {
            "student_id": self.student_id,
            "archetype": self.archetype,
            "week": week,
            **self.get_current_metrics(),
            **self.get_behavioral_probabilities(),
        }
        self.weekly_snapshots.append(snapshot)
        return snapshot

    def get_current_metrics(self) -> Dict[str, float]:
        return {
            "emotional_exhaustion": round(self.emotional_exhaustion, 2),
            "cynicism": round(self.cynicism, 2),
            "perceived_stress": round(self.perceived_stress, 2),
            "academic_self_efficacy": round(self.academic_self_efficacy, 2),
            "coping_reserve": round(self.coping_reserve, 2),
        }

    def to_dict(self) -> Dict:
        return asdict(self)