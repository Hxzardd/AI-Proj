from bandobast.priority import PriorityClassifier
from bandobast.synth import generate_scenario


def test_priority_accuracy_and_paths():
    clf = PriorityClassifier(seed=42, max_depth=3).fit()
    scenario = generate_scenario(seed=42)
    delta = {z.id: 0.1 for z in scenario.zones}
    report = clf.report(scenario, delta)
    assert report.accuracy > 0.75
    assert report.max_depth == 3
    for r in report.results:
        assert r.decision_path  # every zone gets a readable rule chain
        assert r.tier.value in report.labels or r.tier.value in {"critical", "high", "moderate", "low"}


def test_priority_queue_is_tier_ordered():
    clf = PriorityClassifier(seed=9).fit()
    scenario = generate_scenario(seed=9)
    delta = {z.id: 0.0 for z in scenario.zones}
    order = [r.tier for r in clf.report(scenario, delta).results]
    rank = {"critical": 0, "high": 1, "moderate": 2, "low": 3}
    ranks = [rank[t.value] for t in order]
    assert ranks == sorted(ranks)
