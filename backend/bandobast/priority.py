"""Explainable priority ranking (depth-limited decision tree).

Converts current severity, forecast escalation, exposed population, medical
load, and accessibility into an urgency tier. Depth is constrained so the tree
stays readable, and the decision path for any zone is surfaced verbatim -- this
is how the queue answers "why is Zone C above Zone A" (proposal Section 7.3).
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text

from .domain import PriorityReport, PriorityResult, Scenario, Tier
from .synth import PRIORITY_FEATURES, priority_feature_vector, priority_training_data

# Lower index == higher urgency, used to rank the queue.
TIER_ORDER = {Tier.CRITICAL: 0, Tier.HIGH: 1, Tier.MODERATE: 2, Tier.LOW: 3}


class PriorityClassifier:
    def __init__(self, seed: int = 42, max_depth: int = 3):
        self.seed = seed
        self.max_depth = max_depth
        self.model: DecisionTreeClassifier | None = None
        self._stats: dict = {}

    def fit(self) -> PriorityClassifier:
        X, y = priority_training_data(self.seed)
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=self.seed)
        clf = DecisionTreeClassifier(max_depth=self.max_depth, random_state=self.seed)
        clf.fit(X_tr, y_tr)
        self.model = clf
        labels = sorted(np.unique(y).tolist())
        pred = clf.predict(X_te)
        self._stats = {
            "accuracy": float((pred == y_te).mean()),
            "labels": labels,
            "confusion": confusion_matrix(y_te, pred, labels=labels).tolist(),
            "tree_text": export_text(clf, feature_names=PRIORITY_FEATURES),
        }
        return self

    def _decision_path(self, features: list[float]) -> list[str]:
        """Readable rule at each split from root to the predicted leaf."""
        tree = self.model.tree_
        node_indicator = self.model.decision_path([features])
        nodes = node_indicator.indices[node_indicator.indptr[0]:node_indicator.indptr[1]]
        rules: list[str] = []
        for node in nodes:
            if tree.feature[node] < 0:  # leaf
                continue
            name = PRIORITY_FEATURES[tree.feature[node]]
            thr = tree.threshold[node]
            val = features[tree.feature[node]]
            op = "<=" if val <= thr else ">"
            rules.append(f"{name} {op} {thr:.3f} (value={val:.3f})")
        return rules

    def classify(self, features: list[float]) -> tuple[Tier, float]:
        proba = self.model.predict_proba([features])[0]
        classes = self.model.classes_
        idx = int(np.argmax(proba))
        return Tier(classes[idx]), float(proba[idx])

    def report(self, scenario: Scenario, forecast_delta: dict[str, float]) -> PriorityReport:
        if self.model is None:
            self.fit()
        results: list[PriorityResult] = []
        for z in scenario.zones:
            f = priority_feature_vector(
                z.severity, forecast_delta.get(z.id, 0.0),
                z.population, z.medical_cases, z.accessible,
            )
            tier, score = self.classify(f)
            results.append(PriorityResult(
                zone_id=z.id, tier=tier, score=score,
                decision_path=self._decision_path(f),
            ))
        # Rank: tier first, then classifier confidence within tier.
        results.sort(key=lambda r: (TIER_ORDER[r.tier], -r.score))
        s = self._stats
        return PriorityReport(
            max_depth=self.max_depth, accuracy=s["accuracy"],
            confusion_matrix=s["confusion"], labels=s["labels"],
            tree_text=s["tree_text"], results=results,
        )
