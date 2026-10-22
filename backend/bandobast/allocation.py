"""Constrained allocation (OR-Tools CP-SAT).

The dispatch decision is a CP/ILP model with binary variables x[u][z] = 1 when
unit u is sent to zone z. It returns the assignment together with the solver's
bound, so the system can state how far a plan is from optimal -- a claim no
heuristic-only approach can make (proposal Section 7.5).

Hard feasibility constraints (never violated):
  * each unit is assigned at most once;
  * a zone's assigned capacity does not exceed its requirement;
  * a unit can only go to a zone it can reach;
  * operator pins/vetoes are absorbed as equality constraints.

Coverage targets -- critical-tier coverage, a timely ambulance for medical
emergencies, and the equity bound on starved zones -- are enforced
lexicographically: a large bonus makes the solver meet every coverable target
first, then optimise the real objective. Under scarcity (more mandatory zones
than units) this degrades gracefully, meeting as many targets as possible and
reporting the rest, instead of returning a bare INFEASIBLE and deploying
nothing.
"""

from __future__ import annotations

from ortools.sat.python import cp_model

from .config import SolveParams
from .domain import (
    AllocationResult,
    Assignment,
    CoverageReport,
    Override,
    RouteResult,
    Scenario,
    Tier,
    Unit,
    UnitKind,
    Zone,
)

# Units that count as "qualifying" for rescuing a critical zone.
_QUALIFYING = {UnitKind.TEAM, UnitKind.BOAT, UnitKind.AMBULANCE}

# Lexicographic bonus (in real-objective units) for meeting one coverage
# target. Chosen to dominate any achievable real-objective difference so
# coverage is optimised strictly before reach/travel.
COVERAGE_BONUS = 1_000_000.0


def _contribution(
    unit: Unit, zone: Zone, eta_min: float, delta: float, params: SolveParams
) -> float:
    """Real objective value of sending `unit` to `zone`."""
    reach = min(unit.capacity, zone.requirement)
    weight = 1.0 + params.escalation_weight * max(delta, 0.0) + zone.severity
    return weight * reach - params.travel_penalty_per_min * eta_min


def allocate(
    scenario: Scenario,
    routes: dict[tuple[str, str], RouteResult],
    forecast_delta: dict[str, float],
    tiers: dict[str, Tier],
    overrides: list[Override] | None = None,
) -> AllocationResult:
    params = scenario.params
    overrides = overrides or scenario.overrides
    model = cp_model.CpModel()

    zones = {z.id: z for z in scenario.zones}
    units = {u.id: u for u in scenario.units}

    def reachable(u_id: str, z_id: str) -> bool:
        r = routes.get((u_id, z_id))
        return bool(r and r.reachable)

    def eta(u_id: str, z_id: str) -> float:
        return routes[(u_id, z_id)].eta_min

    # Variables only for reachable pairs; unreachable pairs are simply absent.
    x: dict[tuple[str, str], cp_model.IntVar] = {}
    for u_id in units:
        for z_id in zones:
            if reachable(u_id, z_id):
                x[(u_id, z_id)] = model.NewBoolVar(f"x_{u_id}_{z_id}")

    constraints_listed: list[str] = []

    # --- hard feasibility constraints ------------------------------------
    for u_id in units:  # 1. each unit assigned at most once
        vars_u = [x[(u_id, z_id)] for z_id in zones if (u_id, z_id) in x]
        if vars_u:
            model.Add(sum(vars_u) <= 1)
    constraints_listed.append("each unit assigned at most once")

    for z_id, z in zones.items():  # 2. capacity <= requirement
        terms = [units[u_id].capacity * x[(u_id, z_id)]
                 for u_id in units if (u_id, z_id) in x]
        if terms:
            model.Add(sum(terms) <= z.requirement)
    constraints_listed.append("assigned capacity <= zone requirement")

    for ov in overrides:  # operator overrides
        key = (ov.unit_id, ov.zone_id)
        if key in x:
            model.Add(x[key] == (1 if ov.action == "pin" else 0))
    if overrides:
        constraints_listed.append(f"{len(overrides)} operator override(s) injected")

    # --- coverage targets (lexicographic soft) ---------------------------
    cover_bonus_terms = []
    cov = CoverageReport()
    unreached_critical: list[str] = []

    # Critical-tier coverage by a qualifying unit.
    for z_id in zones:
        if tiers.get(z_id) != Tier.CRITICAL:
            continue
        qual = [x[(u_id, z_id)] for u_id in units
                if (u_id, z_id) in x and units[u_id].kind in _QUALIFYING]
        if not qual:
            unreached_critical.append(z_id)  # structurally impossible to cover
            continue
        cov.critical_total += 1
        covered = model.NewBoolVar(f"cov_crit_{z_id}")
        model.Add(sum(qual) >= 1).OnlyEnforceIf(covered)
        model.Add(sum(qual) == 0).OnlyEnforceIf(covered.Not())
        cover_bonus_terms.append((covered, z_id, "critical"))
    constraints_listed.append("critical zones covered by a qualifying unit (lexicographic)")

    # Medical emergencies get a timely ambulance.
    for z_id, z in zones.items():
        if z.medical_cases <= 0:
            continue
        ambs = [x[(u_id, z_id)] for u_id in units
                if (u_id, z_id) in x and units[u_id].kind == UnitKind.AMBULANCE
                and eta(u_id, z_id) <= params.medical_eta_bound_min]
        if not ambs:
            continue
        cov.medical_total += 1
        covered = model.NewBoolVar(f"cov_med_{z_id}")
        model.Add(sum(ambs) >= 1).OnlyEnforceIf(covered)
        model.Add(sum(ambs) == 0).OnlyEnforceIf(covered.Not())
        cover_bonus_terms.append((covered, z_id, "medical"))
    constraints_listed.append(
        f"medical zones get an ambulance within {params.medical_eta_bound_min:g} min (lexicographic)")

    # Equity: zones past the unserved bound must be served.
    for z_id, z in zones.items():
        if z.unserved_min <= params.max_unserved_min:
            continue
        served = [x[(u_id, z_id)] for u_id in units if (u_id, z_id) in x]
        if not served:
            continue
        cov.equity_total += 1
        covered = model.NewBoolVar(f"cov_eq_{z_id}")
        model.Add(sum(served) >= 1).OnlyEnforceIf(covered)
        model.Add(sum(served) == 0).OnlyEnforceIf(covered.Not())
        cover_bonus_terms.append((covered, z_id, "equity"))
    constraints_listed.append(
        f"zones unserved > {params.max_unserved_min:g} min served (lexicographic)")

    # --- objective -------------------------------------------------------
    scale = params.objective_scale
    contrib_float: dict[tuple[str, str], float] = {}
    obj_terms = []
    for (u_id, z_id), var in x.items():
        c = _contribution(units[u_id], zones[z_id], eta(u_id, z_id),
                          forecast_delta.get(z_id, 0.0), params)
        contrib_float[(u_id, z_id)] = c
        obj_terms.append(round(c * scale) * var)
    for covered, _z, _kind in cover_bonus_terms:
        obj_terms.append(round(COVERAGE_BONUS * scale) * covered)
    model.Maximize(sum(obj_terms))

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = params.time_limit_s
    solver.parameters.random_seed = params.seed
    status = solver.Solve(model)
    status_name = solver.StatusName(status)

    assignments: list[Assignment] = []
    real_objective = 0.0
    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        pinned = {(o.unit_id, o.zone_id) for o in overrides if o.action == "pin"}
        for (u_id, z_id), var in x.items():
            if solver.Value(var) == 1:
                z = zones[z_id]
                real_objective += contrib_float[(u_id, z_id)]
                assignments.append(Assignment(
                    unit_id=u_id, zone_id=z_id, eta_min=eta(u_id, z_id),
                    persons_reached=float(min(units[u_id].capacity, z.requirement)),
                    objective_contribution=contrib_float[(u_id, z_id)],
                    pinned=(u_id, z_id) in pinned,
                ))
        for covered, z_id, kind in cover_bonus_terms:
            if solver.Value(covered) == 1:
                setattr(cov, f"{kind}_met", getattr(cov, f"{kind}_met") + 1)
            else:
                cov.unmet.append(f"{kind}:{z_id}")

    solver_obj = solver.ObjectiveValue() / scale if assignments or status in (
        cp_model.OPTIMAL, cp_model.FEASIBLE) else 0.0
    bound = solver.BestObjectiveBound() / scale if status in (
        cp_model.OPTIMAL, cp_model.FEASIBLE) else 0.0
    gap = 0.0 if abs(bound) < 1e-9 else abs(bound - solver_obj) / abs(bound)

    return AllocationResult(
        status=status_name, assignments=assignments,
        objective_value=real_objective, solver_objective=solver_obj,
        best_bound=bound, optimality_gap=gap, wall_time_s=solver.WallTime(),
        coverage=cov, unreached_critical=unreached_critical,
        constraints_listed=constraints_listed,
    )
