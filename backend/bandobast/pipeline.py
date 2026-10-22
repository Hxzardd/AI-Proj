"""Decision orchestrator: Ingest -> Forecast -> Rank -> Route -> Allocate -> Explain.

Each stage hands the next a typed, inspectable object and appends a StageTrace,
so the final plan carries the full Ingest->...->Explain trace the operator
workspace renders. No free-form generation anywhere in the path.
"""

from __future__ import annotations

import uuid

from .allocation import allocate
from .clustering import cluster_zones
from .config import MODEL_VERSION
from .domain import (
    ExecutionTrace,
    Plan,
    RouteResult,
    Scenario,
    StageTrace,
    Tier,
)
from .forecast import EscalationForecaster
from .ga_baseline import run_ga
from .priority import PriorityClassifier
from .routing import Network, route_with_groundtruth


def _route_all(scenario: Scenario) -> dict[tuple[str, str], RouteResult]:
    net = Network.from_graph(scenario.graph)
    speed = scenario.params.max_speed_kmph
    routes: dict[tuple[str, str], RouteResult] = {}
    for u in scenario.units:
        for z in scenario.zones:
            routes[(u.id, z.id)] = route_with_groundtruth(net, u.depot_node, z.node, speed)
    return routes


def solve_scenario(
    scenario: Scenario,
    *,
    forecaster: EscalationForecaster | None = None,
    classifier: PriorityClassifier | None = None,
    run_ga_baseline: bool = True,
    run_clustering: bool = True,
) -> Plan:
    """Run the full pipeline and return a replayable Plan.

    Trained models may be injected (so the API can fit once and reuse); if not
    given, they are fitted on the scenario seed, keeping the run deterministic.
    """
    trace = ExecutionTrace()
    seed = scenario.seed

    # 1. Ingest.
    trace.stages.append(StageTrace(
        stage="ingest",
        summary=f"{len(scenario.zones)} zones, {len(scenario.units)} units, "
                f"{len(scenario.graph.edges)} road edges",
        artefact={"zone_ids": [z.id for z in scenario.zones],
                  "unit_ids": [u.id for u in scenario.units]},
    ))

    # 2. Forecast.
    forecaster = forecaster or EscalationForecaster(seed=seed).fit()
    forecast = forecaster.report(scenario)
    forecast_delta = {f.zone_id: f.predicted_delta for f in forecast.forecasts}
    trace.stages.append(StageTrace(
        stage="forecast",
        summary=f"model={forecast.model} rmse={forecast.rmse:.4f} "
                f"vs persistence {forecast.persistence_rmse:.4f} "
                f"(beats={forecast.beats_persistence})",
        artefact={"feature_importances": forecast.feature_importances},
    ))

    # 3. Rank.
    classifier = classifier or PriorityClassifier(seed=seed).fit()
    priority = classifier.report(scenario, forecast_delta)
    tiers: dict[str, Tier] = {p.zone_id: p.tier for p in priority.results}
    trace.stages.append(StageTrace(
        stage="rank",
        summary=f"tree depth={priority.max_depth} accuracy={priority.accuracy:.3f}; "
                f"queue={[p.zone_id for p in priority.results]}",
        artefact={"tiers": {k: v.value for k, v in tiers.items()}},
    ))

    # 4. Route.
    routes = _route_all(scenario)
    reachable = sum(1 for r in routes.values() if r.reachable)
    mismatches = [
        f"{k[0]}->{k[1]}" for k, r in routes.items()
        if r.reachable and r.dijkstra_cost_min is not None
        and abs(r.cost_min - r.dijkstra_cost_min) > 1e-6
    ]
    trace.stages.append(StageTrace(
        stage="route",
        summary=f"{reachable}/{len(routes)} unit-zone pairs reachable; "
                f"A* matches Dijkstra on all reachable pairs "
                f"({len(mismatches)} mismatches)",
        artefact={"astar_dijkstra_mismatches": mismatches},
    ))

    # 5. Allocate (exact).
    allocation = allocate(scenario, routes, forecast_delta, tiers, scenario.overrides)
    trace.stages.append(StageTrace(
        stage="allocate",
        summary=f"status={allocation.status} objective={allocation.objective_value:.2f} "
                f"bound={allocation.best_bound:.2f} gap={allocation.optimality_gap:.4f} "
                f"({len(allocation.assignments)} assignments)",
        artefact={"constraints": allocation.constraints_listed,
                  "unreached_critical": allocation.unreached_critical},
    ))

    # 5b. GA comparative baseline.
    ga = None
    if run_ga_baseline:
        cov = allocation.coverage
        ga = run_ga(scenario, routes, forecast_delta, tiers,
                    allocation.solver_objective,
                    cpsat_real=allocation.objective_value,
                    cpsat_covered=cov.critical_met + cov.medical_met + cov.equity_met,
                    overrides=scenario.overrides)
        trace.stages.append(StageTrace(
            stage="ga_baseline",
            summary=f"GA real objective {ga.ga_objective:.1f} vs CP-SAT {ga.cpsat_real:.1f} "
                    f"(real gap {ga.real_gap*100:.1f}%, coverage_match={ga.coverage_match})",
            artefact={"generations": len(ga.convergence)},
        ))

    # Secondary: clustering.
    clusters = None
    if run_clustering and len(scenario.zones) >= 3:
        clusters = cluster_zones(scenario, forecast_delta, seed=seed)
        trace.stages.append(StageTrace(
            stage="cluster",
            summary=f"k={clusters.k} silhouette={clusters.silhouette:.3f}",
            artefact={"assignments": clusters.assignments},
        ))

    # 6. Explain is the trace itself; the plan bundles every artefact.
    return Plan(
        plan_id=str(uuid.uuid4()),
        scenario_id=scenario.id,
        seed=seed,
        model_version=MODEL_VERSION,
        params=scenario.params,
        forecast=forecast,
        priority=priority,
        clusters=clusters,
        routes=list(routes.values()),
        allocation=allocation,
        ga_baseline=ga,
        trace=trace,
    )
