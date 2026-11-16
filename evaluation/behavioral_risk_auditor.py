"""
Behavioral Risk Auditor & Longitudinal SEM Exporter for AgentSchool.
Flags structural curriculum burnout bottlenecks and exports wide/long CSVs
for Latent Growth Curve Modeling (LGCM) and Structural Equation Modeling (SEM).
"""

import csv
import json
from pathlib import Path
from typing import Dict, List
from statistics import mean, stdev


class BehavioralRiskAuditor:
    def __init__(
        self,
        exhaustion_alert_threshold: float = 65.0,
        efficacy_drop_threshold: float = 45.0,
    ):
        self.exhaustion_alert_threshold = exhaustion_alert_threshold
        self.efficacy_drop_threshold = efficacy_drop_threshold

    def audit_curriculum_run(
        self,
        student_snapshots: List[Dict],
        curriculum_condition: str = "standard_syllabus",
    ) -> Dict:
        """
        Aggregates longitudinal trajectories by week and identifies structural
        bottleneck weeks where burnout spikes or self-efficacy collapses.
        """
        by_week: Dict[int, List[Dict]] = {}
        for row in student_snapshots:
            by_week.setdefault(row["week"], []).append(row)

        weekly_summary = []
        bottleneck_weeks = []

        for week in sorted(by_week.keys()):
            rows = by_week[week]
            mean_ex = mean(r["emotional_exhaustion"] for r in rows)
            mean_cy = mean(r["cynicism"] for r in rows)
            mean_st = mean(r["perceived_stress"] for r in rows)
            mean_ef = mean(r["academic_self_efficacy"] for r in rows)
            mean_proc = mean(r["procrastination_prob"] for r in rows)
            mean_diseng = mean(r["disengagement_prob"] for r in rows)

            high_risk_count = sum(
                1 for r in rows
                if r["emotional_exhaustion"] >= self.exhaustion_alert_threshold
                or r["academic_self_efficacy"] <= self.efficacy_drop_threshold
            )
            risk_prevalence = round((high_risk_count / len(rows)) * 100.0, 1)

            is_bottleneck = (
                mean_ex >= self.exhaustion_alert_threshold
                or mean_ef <= self.efficacy_drop_threshold
                or risk_prevalence >= 35.0
            )
            if is_bottleneck:
                bottleneck_weeks.append(week)

            weekly_summary.append({
                "condition": curriculum_condition,
                "week": week,
                "n_agents": len(rows),
                "mean_exhaustion": round(mean_ex, 2),
                "sd_exhaustion": round(stdev(r["emotional_exhaustion"] for r in rows), 2) if len(rows) > 1 else 0.0,
                "mean_cynicism": round(mean_cy, 2),
                "mean_stress": round(mean_st, 2),
                "mean_self_efficacy": round(mean_ef, 2),
                "mean_procrastination_prob": round(mean_proc, 3),
                "mean_disengagement_prob": round(mean_diseng, 3),
                "high_risk_prevalence_pct": risk_prevalence,
                "structural_bottleneck_flag": is_bottleneck,
            })

        return {
            "curriculum_condition": curriculum_condition,
            "detected_bottleneck_weeks": bottleneck_weeks,
            "weekly_trajectories": weekly_summary,
        }

    def export_long_csv(self, student_snapshots: List[Dict], output_path: str) -> None:
        """Exports person-period (long) CSV for multilevel modeling / LGCM."""
        if not student_snapshots:
            return
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = list(student_snapshots[0].keys())
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(student_snapshots)

    def export_wide_sem_csv(
        self,
        student_snapshots: List[Dict],
        target_weeks: List[int],
        output_path: str,
    ) -> None:
        """
        Exports person-level (wide) CSV across target measurement waves (e.g., Weeks 1, 6, 12, 16)
        ready for lavaan / Mplus / AMOS Structural Equation Modeling.
        """
        by_student: Dict[str, Dict] = {}
        metrics = [
            "emotional_exhaustion",
            "cynicism",
            "perceived_stress",
            "academic_self_efficacy",
            "procrastination_prob",
            "help_seeking_prob",
            "disengagement_prob",
        ]

        for row in student_snapshots:
            sid = row["student_id"]
            if sid not in by_student:
                by_student[sid] = {
                    "student_id": sid,
                    "archetype": row["archetype"],
                }
            wk = row["week"]
            if wk in target_weeks:
                for m in metrics:
                    by_student[sid][f"{m}_w{wk}"] = row[m]

        rows = list(by_student.values())
        if not rows:
            return

        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    # Quick smoke test demonstrating pre-deployment stress-test comparison
    from student.psychometrics import PsychometricState

    cohort = [
        PsychometricState("S01", "high_resilience", stress_sensitivity=0.7, coping_style="problem_focused", academic_self_efficacy=82.0),
        PsychometricState("S02", "moderate_vulnerability", stress_sensitivity=1.0, coping_style="support_seeking", academic_self_efficacy=65.0),
        PsychometricState("S03", "high_burnout_risk", stress_sensitivity=1.35, coping_style="avoidant", academic_self_efficacy=48.0),
    ]

    # Simulate a 16-week syllabus with clustered midterm/project spikes at Weeks 6 and 12
    weekly_syllabus = [
        (wk, 0.92 if wk in (6, 12, 15) else 0.45, 0.35 if wk in (6, 12) else 0.72, 7.0 if wk in (6, 12) else 1.5)
        for wk in range(1, 17)
    ]

    all_snapshots = []
    for week, demand, scaffolding, fb_delay in weekly_syllabus:
        for agent in cohort:
            mastery = max(0.25, min(0.95, (agent.academic_self_efficacy / 100.0) - (demand * 0.25)))
            agent.update_on_academic_event(
                demand_load=demand,
                mastery_score=mastery,
                scaffolding_support=scaffolding,
                feedback_delay_days=fb_delay,
            )
            all_snapshots.append(agent.record_week_snapshot(week))

    auditor = BehavioralRiskAuditor()
    report = auditor.audit_curriculum_run(all_snapshots, curriculum_condition="unoptimized_baseline")
    print(json.dumps(report, indent=2))
    auditor.export_wide_sem_csv(all_snapshots, [1, 6, 12, 16], "logs/sem_wide_baseline.csv")