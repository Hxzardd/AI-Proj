import pytest

from bandobast.allocation import allocate
from bandobast.domain import Override
from bandobast.forecast import EscalationForecaster
from bandobast.pipeline import _route_all
from bandobast.priority import PriorityClassifier
from bandobast.synth import generate_scenario
from bandobast.verify import constraint_violations


def _setup(seed, n_zones=8, n_units=6):
    scenario = generate_scenario(seed, n_zones, n_units)
    routes = _route_all(scenario)
    forecast = EscalationForecaster(seed=seed).fit().report(scenario)
    delta = {f.zone_id: f.predicted_delta for f in forecast.forecasts}
    tiers = {p.zone_id: p.tier
             for p in PriorityClassifier(seed=seed).fit().report(scenario, delta).results}
    return scenario, routes, delta, tiers


@pytest.mark.parametrize("seed", [1, 2, 3, 7, 13, 21, 42, 99])
def test_no_constraint_violations(seed):
    """Property test: a returned plan violates no declared constraint."""
    scenario, routes, delta, tiers = _setup(seed)
    result = allocate(scenario, routes, delta, tiers)
    assert result.status in ("OPTIMAL", "FEASIBLE")
    assert constraint_violations(scenario, routes, tiers, result) == []


def test_small_instance_proven_optimal():
    scenario, routes, delta, tiers = _setup(42, n_zones=4, n_units=3)
    result = allocate(scenario, routes, delta, tiers)
    assert result.status == "OPTIMAL"
    assert result.optimality_gap < 1e-6


def test_override_pin_and_veto_honoured():
    scenario, routes, delta, tiers = _setup(5)
    # find a reachable pair to pin and another to veto
    reachable = [(u, z) for (u, z), r in routes.items() if r.reachable]
    pin_u, pin_z = reachable[0]
    veto = next((p for p in reachable if p[0] != pin_u), reachable[1])
    overrides = [Override(unit_id=pin_u, zone_id=pin_z, action="pin"),
                 Override(unit_id=veto[0], zone_id=veto[1], action="veto")]
    result = allocate(scenario, routes, delta, tiers, overrides)
    assert constraint_violations(scenario, routes, tiers, result, overrides) == []
    pairs = {(a.unit_id, a.zone_id) for a in result.assignments}
    assert (pin_u, pin_z) in pairs
    assert veto not in pairs


def test_gap_zero_means_optimal():
    scenario, routes, delta, tiers = _setup(2)
    result = allocate(scenario, routes, delta, tiers)
    if result.status == "OPTIMAL":
        assert result.optimality_gap < 1e-6
