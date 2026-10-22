from bandobast.pipeline import _route_all, solve_scenario
from bandobast.synth import generate_scenario
from bandobast.verify import constraint_violations


def test_pipeline_produces_full_trace():
    plan = solve_scenario(generate_scenario(seed=42))
    stages = [s.stage for s in plan.trace.stages]
    assert stages == ["ingest", "forecast", "rank", "route", "allocate", "ga_baseline", "cluster"]
    assert plan.model_version
    assert plan.allocation.status in ("OPTIMAL", "FEASIBLE")


def test_pipeline_plan_is_feasible():
    scenario = generate_scenario(seed=21)
    plan = solve_scenario(scenario)
    routes = _route_all(scenario)
    tiers = {p.zone_id: p.tier for p in plan.priority.results}
    assert constraint_violations(scenario, routes, tiers, plan.allocation) == []


def test_pipeline_deterministic():
    s1 = generate_scenario(seed=7)
    s2 = generate_scenario(seed=7)
    p1 = solve_scenario(s1, run_ga_baseline=False)
    p2 = solve_scenario(s2, run_ga_baseline=False)
    assert p1.allocation.objective_value == p2.allocation.objective_value
    assert ({(a.unit_id, a.zone_id) for a in p1.allocation.assignments}
            == {(a.unit_id, a.zone_id) for a in p2.allocation.assignments})
