"""Genetic Algorithm comparative baseline (DEAP).

Runs selection/crossover/mutation against the *same* objective as the CP-SAT
model, and plots fitness convergence against the exact solver's bound. It is
retained purely so the report can show what a metaheuristic costs relative to
the exact solution -- not because one is needed (proposal Section 7.5).
"""

from __future__ import annotations

import random

import numpy as np
from deap import base, creator, tools

from .allocation import _QUALIFYING, COVERAGE_BONUS, _contribution
from .config import SolveParams
from .domain import GAResult, Override, RouteResult, Scenario, Tier, UnitKind

# Penalty per hard-constraint violation; must dominate every reward term so
# the GA is pushed into the same feasible region CP-SAT solves over.
_VIOLATION_PENALTY = 1e9
_NR = RouteResult(source=-1, target=-1, reachable=False)

# DEAP registers classes on a module-global; guard against re-registration.
if not hasattr(creator, "FitnessMax"):
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
if not hasattr(creator, "Individual"):
    creator.create("Individual", list, fitness=creator.FitnessMax)


def _evaluate(ind, scenario, routes, forecast_delta, tiers, overrides, params):
    """Same objective as CP-SAT (real contributions + lexicographic coverage
    bonus) minus a dominating penalty for each hard-constraint violation. An
    individual maps unit index -> zone index, where index == n_zones means
    'unassigned'. Unit-at-most-once is guaranteed by the encoding."""
    zones = scenario.zones
    units = scenario.units
    n_zones = len(zones)
    score = 0.0
    violations = 0

    assigned_cap: dict[str, int] = {z.id: 0 for z in zones}
    assigned_units: dict[str, list] = {z.id: [] for z in zones}

    for u_idx, z_idx in enumerate(ind):
        if z_idx >= n_zones:
            continue  # unassigned
        unit, zone = units[u_idx], zones[z_idx]
        r = routes.get((unit.id, zone.id))
        if not (r and r.reachable):
            violations += 1  # cannot send a unit where it can't reach
            continue
        score += _contribution(unit, zone, r.eta_min,
                               forecast_delta.get(zone.id, 0.0), params)
        assigned_cap[zone.id] += unit.capacity
        assigned_units[zone.id].append(unit)

    for z in zones:
        if assigned_cap[z.id] > z.requirement:  # hard capacity ceiling
            violations += 1
        # Coverage bonuses (soft, mirror the CP-SAT objective).
        if tiers.get(z.id) == Tier.CRITICAL:
            coverable = any(u.kind in _QUALIFYING for u in units
                            if (routes.get((u.id, z.id)) or _NR).reachable)
            if coverable and any(u.kind in _QUALIFYING for u in assigned_units[z.id]):
                score += COVERAGE_BONUS
        if z.medical_cases > 0:
            coverable = any(u.kind == UnitKind.AMBULANCE
                            and (routes.get((u.id, z.id)) or _NR).reachable
                            and routes[(u.id, z.id)].eta_min <= params.medical_eta_bound_min
                            for u in units)
            if coverable and any(
                    u.kind == UnitKind.AMBULANCE
                    and routes[(u.id, z.id)].eta_min <= params.medical_eta_bound_min
                    for u in assigned_units[z.id]):
                score += COVERAGE_BONUS
        if z.unserved_min > params.max_unserved_min:
            coverable = any((routes.get((u.id, z.id)) or _NR).reachable for u in units)
            if coverable and assigned_units[z.id]:
                score += COVERAGE_BONUS

    # overrides (hard)
    zone_index = {z.id: i for i, z in enumerate(zones)}
    unit_index = {u.id: i for i, u in enumerate(units)}
    for ov in overrides:
        if ov.unit_id not in unit_index or ov.zone_id not in zone_index:
            continue
        is_set = ind[unit_index[ov.unit_id]] == zone_index[ov.zone_id]
        if ov.action == "pin" and not is_set:
            violations += 1
        if ov.action == "veto" and is_set:
            violations += 1

    return (score - _VIOLATION_PENALTY * violations,)


def _real_and_cover(ind, scenario, routes, forecast_delta, tiers, params):
    """Real objective (reach - travel, no bonus/penalty) and coverage count for
    a given individual -- used to compare the GA's best solution to CP-SAT on
    the meaningful objective rather than the coverage-dominated internal one."""
    zones, units, n_zones = scenario.zones, scenario.units, len(scenario.zones)
    real = 0.0
    covered = 0
    assigned = {z.id: [] for z in zones}
    for u_idx, z_idx in enumerate(ind):
        if z_idx >= n_zones:
            continue
        unit, zone = units[u_idx], zones[z_idx]
        r = routes.get((unit.id, zone.id))
        if not (r and r.reachable):
            continue
        real += _contribution(unit, zone, r.eta_min, forecast_delta.get(zone.id, 0.0), params)
        assigned[zone.id].append(unit)
    for z in zones:
        if tiers.get(z.id) == Tier.CRITICAL:
            coverable = any(u.kind in _QUALIFYING for u in units
                            if (routes.get((u.id, z.id)) or _NR).reachable)
            if coverable and any(u.kind in _QUALIFYING for u in assigned[z.id]):
                covered += 1
        if z.medical_cases > 0:
            coverable = any(u.kind == UnitKind.AMBULANCE
                            and (routes.get((u.id, z.id)) or _NR).reachable
                            and routes[(u.id, z.id)].eta_min <= params.medical_eta_bound_min
                            for u in units)
            if coverable and any(u.kind == UnitKind.AMBULANCE
                                 and routes[(u.id, z.id)].eta_min <= params.medical_eta_bound_min
                                 for u in assigned[z.id]):
                covered += 1
        if z.unserved_min > params.max_unserved_min and (
            any((routes.get((u.id, z.id)) or _NR).reachable for u in units)
            and assigned[z.id]
        ):
            covered += 1
    return real, covered


def run_ga(
    scenario: Scenario,
    routes: dict[tuple[str, str], RouteResult],
    forecast_delta: dict[str, float],
    tiers: dict[str, Tier],
    cpsat_objective: float,
    cpsat_real: float = 0.0,
    cpsat_covered: int = 0,
    overrides: list[Override] | None = None,
    generations: int = 40,
    pop_size: int = 60,
) -> GAResult:
    params: SolveParams = scenario.params
    overrides = overrides or scenario.overrides
    random.seed(params.seed)
    np.random.seed(params.seed)

    n_units = len(scenario.units)
    n_zones = len(scenario.zones)

    toolbox = base.Toolbox()
    toolbox.register("gene", random.randint, 0, n_zones)  # n_zones == unassigned
    toolbox.register("individual", tools.initRepeat, creator.Individual,
                     toolbox.gene, n=n_units)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", _evaluate, scenario=scenario, routes=routes,
                     forecast_delta=forecast_delta, tiers=tiers,
                     overrides=overrides, params=params)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutUniformInt, low=0, up=n_zones, indpb=0.2)
    toolbox.register("select", tools.selTournament, tournsize=3)

    pop = toolbox.population(n=pop_size)
    for ind in pop:
        ind.fitness.values = toolbox.evaluate(ind)
    best_ind = toolbox.clone(tools.selBest(pop, 1)[0])
    best_fit = best_ind.fitness.values[0]

    convergence: list[float] = []
    cxpb, mutpb = 0.6, 0.3
    for _ in range(generations):
        offspring = list(map(toolbox.clone, toolbox.select(pop, len(pop))))
        for c1, c2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < cxpb:
                toolbox.mate(c1, c2)
                del c1.fitness.values, c2.fitness.values
        for mut in offspring:
            if random.random() < mutpb:
                toolbox.mutate(mut)
                del mut.fitness.values
        for ind in offspring:
            if not ind.fitness.valid:
                ind.fitness.values = toolbox.evaluate(ind)
        pop[:] = offspring
        gen_best = tools.selBest(pop, 1)[0]
        if gen_best.fitness.values[0] > best_fit:
            best_fit = gen_best.fitness.values[0]
            best_ind = toolbox.clone(gen_best)
        convergence.append(best_fit)  # best-so-far (monotonic)

    ga_real, ga_cover = _real_and_cover(best_ind, scenario, routes, forecast_delta, tiers, params)
    gap_full = 0.0 if abs(cpsat_objective) < 1e-9 else (cpsat_objective - best_fit) / abs(cpsat_objective)
    real_gap = 0.0 if abs(cpsat_real) < 1e-9 else (cpsat_real - ga_real) / abs(cpsat_real)
    return GAResult(
        best_fitness=best_fit, convergence=convergence,
        cpsat_objective=cpsat_objective, gap_to_cpsat=gap_full,
        ga_objective=ga_real, cpsat_real=cpsat_real, real_gap=real_gap,
        coverage_match=ga_cover >= cpsat_covered,
    )
