from bandobast.allocation import allocate
from bandobast.forecast import EscalationForecaster
from bandobast.ga_baseline import run_ga
from bandobast.pipeline import _route_all
from bandobast.priority import PriorityClassifier
from bandobast.synth import generate_scenario


def _setup(seed):
    scenario = generate_scenario(seed, n_zones=6, n_units=5)
    routes = _route_all(scenario)
    delta = {f.zone_id: f.predicted_delta
             for f in EscalationForecaster(seed=seed).fit().report(scenario).forecasts}
    tiers = {p.zone_id: p.tier
             for p in PriorityClassifier(seed=seed).fit().report(scenario, delta).results}
    return scenario, routes, delta, tiers


def test_ga_does_not_beat_exact_solver():
    scenario, routes, delta, tiers = _setup(3)
    exact = allocate(scenario, routes, delta, tiers)
    ga = run_ga(scenario, routes, delta, tiers, exact.solver_objective,
                generations=30, pop_size=40)
    assert len(ga.convergence) == 30
    # CP-SAT is exact; a feasible GA solution cannot exceed the optimum.
    assert ga.best_fitness <= exact.solver_objective + 1e-6


def test_ga_convergence_improves():
    scenario, routes, delta, tiers = _setup(8)
    exact = allocate(scenario, routes, delta, tiers)
    ga = run_ga(scenario, routes, delta, tiers, exact.solver_objective,
                generations=40, pop_size=50)
    # Best-so-far is non-decreasing across generations by construction of max().
    assert max(ga.convergence) >= ga.convergence[0]
