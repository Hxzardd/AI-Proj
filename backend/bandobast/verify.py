"""Independent feasibility checker.

Re-derives the hard feasibility constraints from the plan outputs, without
trusting the solver. Used by the property tests to assert that no returned plan
violates a declared hard constraint, and callable at runtime as a feasibility
certificate.

Coverage requirements (critical / medical / equity) are lexicographic soft
targets, not hard constraints, so they are reported via AllocationResult.
coverage rather than treated as violations here.
"""

from __future__ import annotations

from collections import defaultdict

from .domain import AllocationResult, Override, RouteResult, Scenario, Tier


def constraint_violations(
    scenario: Scenario,
    routes: dict[tuple[str, str], RouteResult],
    tiers: dict[str, Tier],
    allocation: AllocationResult,
    overrides: list[Override] | None = None,
) -> list[str]:
    """Return a list of hard-constraint violations; empty == feasible plan."""
    overrides = overrides or scenario.overrides
    units = {u.id: u for u in scenario.units}
    zones = {z.id: z for z in scenario.zones}
    problems: list[str] = []

    per_unit: dict[str, int] = defaultdict(int)
    cap_per_zone: dict[str, int] = defaultdict(int)
    assigned_pairs: set[tuple[str, str]] = set()
    for a in allocation.assignments:
        per_unit[a.unit_id] += 1
        cap_per_zone[a.zone_id] += units[a.unit_id].capacity
        assigned_pairs.add((a.unit_id, a.zone_id))
        r = routes.get((a.unit_id, a.zone_id))
        if not (r and r.reachable):
            problems.append(f"{a.unit_id}->{a.zone_id} assigned but unreachable")

    # 1. each unit at most once
    for u_id, n in per_unit.items():
        if n > 1:
            problems.append(f"unit {u_id} assigned {n} times")

    # 2. capacity <= requirement
    for z_id, cap in cap_per_zone.items():
        if cap > zones[z_id].requirement:
            problems.append(f"zone {z_id} capacity {cap} > requirement {zones[z_id].requirement}")

    # 3. overrides honoured
    for ov in overrides:
        key = (ov.unit_id, ov.zone_id)
        if key not in routes or not routes[key].reachable:
            continue
        if ov.action == "pin" and key not in assigned_pairs:
            problems.append(f"pinned {ov.unit_id}->{ov.zone_id} not in plan")
        if ov.action == "veto" and key in assigned_pairs:
            problems.append(f"vetoed {ov.unit_id}->{ov.zone_id} present in plan")

    return problems
