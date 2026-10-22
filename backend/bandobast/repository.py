"""Audit trail and scenario replay.

Persists scenario snapshots and the resulting plans as JSON so any historical
recommendation can be replayed and defended after the event (proposal
Section 4, "Reproducible, Auditable Decisions"). The proposal's production
target is PostGIS; this file-backed store keeps the prototype fully offline on
a single machine behind the same interface.
"""

from __future__ import annotations

from pathlib import Path

from .domain import Plan, Scenario


class Repository:
    def __init__(self, root: str | Path | None = None):
        self.root = Path(root) if root else None
        self._scenarios: dict[str, Scenario] = {}
        self._plans: dict[str, Plan] = {}
        if self.root:
            (self.root / "scenarios").mkdir(parents=True, exist_ok=True)
            (self.root / "plans").mkdir(parents=True, exist_ok=True)
            self._load()

    def _load(self) -> None:
        for p in (self.root / "scenarios").glob("*.json"):
            s = Scenario.model_validate_json(p.read_text())
            self._scenarios[s.id] = s
        for p in (self.root / "plans").glob("*.json"):
            pl = Plan.model_validate_json(p.read_text())
            self._plans[pl.plan_id] = pl

    def save_scenario(self, scenario: Scenario) -> None:
        self._scenarios[scenario.id] = scenario
        if self.root:
            (self.root / "scenarios" / f"{scenario.id}.json").write_text(
                scenario.model_dump_json(indent=2))

    def save_plan(self, plan: Plan) -> None:
        self._plans[plan.plan_id] = plan
        if self.root:
            (self.root / "plans" / f"{plan.plan_id}.json").write_text(
                plan.model_dump_json(indent=2))

    def get_scenario(self, scenario_id: str) -> Scenario | None:
        return self._scenarios.get(scenario_id)

    def get_plan(self, plan_id: str) -> Plan | None:
        return self._plans.get(plan_id)

    def list_scenarios(self) -> list[str]:
        return list(self._scenarios)

    def list_plans(self) -> list[str]:
        return list(self._plans)
