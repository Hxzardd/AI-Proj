"""Risk grouping (K-Means) -- secondary analysis layer.

Groups zones with similar risk profiles for grouped dispatch. Reports the
silhouette score and an elbow curve so the chosen k is defensible rather than
assumed (proposal Section 5, scope note).
"""

from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from .domain import ClusterReport, Scenario


def cluster_zones(
    scenario: Scenario, forecast_delta: dict[str, float], seed: int = 42, k: int | None = None
) -> ClusterReport:
    zones = scenario.zones
    X = np.array([
        [z.severity, forecast_delta.get(z.id, 0.0),
         z.population, float(z.medical_cases)]
        for z in zones
    ])
    Xs = StandardScaler().fit_transform(X)
    n = len(zones)

    # Elbow: inertia across feasible k (silhouette needs 2..n-1 clusters).
    k_max = min(6, n - 1)
    inertia_by_k: dict[int, float] = {}
    best_k, best_sil = 2, -1.0
    for kk in range(2, max(3, k_max + 1)):
        if kk >= n:
            break
        km = KMeans(n_clusters=kk, random_state=seed, n_init=10).fit(Xs)
        inertia_by_k[kk] = float(km.inertia_)
        sil = float(silhouette_score(Xs, km.labels_)) if kk < n else -1.0
        if sil > best_sil:
            best_sil, best_k = sil, kk

    chosen = k or best_k
    final = KMeans(n_clusters=chosen, random_state=seed, n_init=10).fit(Xs)
    sil = float(silhouette_score(Xs, final.labels_)) if 1 < chosen < n else 0.0
    return ClusterReport(
        k=chosen, silhouette=sil, inertia_by_k=inertia_by_k,
        assignments={z.id: int(c) for z, c in zip(zones, final.labels_)},
    )
