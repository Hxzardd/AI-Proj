from bandobast.clustering import cluster_zones
from bandobast.synth import generate_scenario


def test_cluster_report_well_formed():
    scenario = generate_scenario(seed=42, n_zones=8)
    delta = {z.id: 0.1 for z in scenario.zones}
    report = cluster_zones(scenario, delta, seed=42)
    assert 2 <= report.k <= 6
    assert -1.0 <= report.silhouette <= 1.0
    assert set(report.assignments) == {z.id for z in scenario.zones}
    assert len(report.inertia_by_k) >= 1
