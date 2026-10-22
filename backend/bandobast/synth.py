"""Seeded, reproducible synthetic data.

Review 1 operates on seeded scenarios rather than live telemetry. These
generators use a *known* generative process so (a) the regression model can
genuinely recover signal and beat the persistence baseline, and (b) the
decision tree can recover a readable rule. Real EM-DAT/IMD/OSM extracts plug
in at the same feature boundary (see FORECAST_FEATURES / PRIORITY_FEATURES).

Everything is a pure function of the seed, which is what makes a plan
replayable.
"""

from __future__ import annotations

import numpy as np

from .domain import Edge, RoadGraph, Scenario, Unit, UnitKind, Zone
from .routing import haversine_km

FORECAST_FEATURES = [
    "severity", "rain_mean", "rain_trend", "pop_density_norm",
    "elevation_norm", "medical_load",
]
PRIORITY_FEATURES = [
    "severity", "forecast_delta", "pop_exposure_norm", "medical_load", "accessible",
]

_MAX_DENSITY = 20000.0   # persons / km^2 normaliser
_MAX_ELEV = 500.0        # metres normaliser
_MAX_POP = 50000.0       # persons normaliser


# --- feature construction (shared by training + inference) ---------------

def forecast_feature_vector(
    severity: float, hazard_series: list[float], pop_density: float,
    elevation_m: float, medical_cases: int, population: int,
) -> list[float]:
    series = np.asarray(hazard_series, dtype=float) if hazard_series else np.zeros(1)
    rain_mean = float(series.mean())
    rain_trend = float(series[-1] - series[0]) if len(series) > 1 else 0.0
    return [
        severity,
        rain_mean,
        rain_trend,
        min(pop_density / _MAX_DENSITY, 1.0),
        min(elevation_m / _MAX_ELEV, 1.0),
        min(medical_cases / max(population, 1) * 100.0, 1.0),
    ]


def true_escalation(features: list[float]) -> float:
    """Generative rule for next-window severity change. The regression model
    is trained to approximate this; persistence (delta=0) cannot."""
    severity, rain_mean, rain_trend, dens, elev, med = features
    delta = (
        0.35 * rain_trend
        + 0.20 * rain_mean
        + 0.10 * dens
        - 0.25 * elev            # higher ground escalates less
        - 0.20 * severity        # already-severe zones have less room to rise
        + 0.05 * med
    )
    return float(np.clip(delta, -0.5, 0.5))


def priority_feature_vector(
    severity: float, forecast_delta: float, population: int,
    medical_cases: int, accessible: bool,
) -> list[float]:
    return [
        severity,
        forecast_delta,
        min(population / _MAX_POP, 1.0),
        min(medical_cases / max(population, 1) * 100.0, 1.0),
        1.0 if accessible else 0.0,
    ]


def true_tier(features: list[float]) -> str:
    """Generative rule for the urgency tier (what the tree must recover)."""
    severity, delta, pop, med, _accessible = features
    projected = severity + delta
    if projected >= 0.7 or med >= 0.6:
        return "critical"
    if projected >= 0.5 or (pop >= 0.5 and projected >= 0.4):
        return "high"
    if projected >= 0.3:
        return "moderate"
    return "low"


# --- training sets -------------------------------------------------------

def forecast_training_data(seed: int, n: int = 2000) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X, y = [], []
    for _ in range(n):
        severity = float(rng.uniform(0, 1))
        series = list(rng.uniform(0, 1, size=6))
        pop_density = float(rng.uniform(100, _MAX_DENSITY))
        elevation = float(rng.uniform(0, _MAX_ELEV))
        population = int(rng.integers(500, int(_MAX_POP)))
        medical = int(rng.integers(0, max(1, population // 50)))
        f = forecast_feature_vector(severity, series, pop_density, elevation, medical, population)
        X.append(f)
        y.append(true_escalation(f) + float(rng.normal(0, 0.03)))  # observation noise
    return np.array(X), np.array(y)


def priority_training_data(seed: int, n: int = 2000) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed + 1)
    X, y = [], []
    for _ in range(n):
        severity = float(rng.uniform(0, 1))
        delta = float(rng.uniform(-0.3, 0.4))
        population = int(rng.integers(500, int(_MAX_POP)))
        medical = int(rng.integers(0, max(1, population // 40)))
        accessible = bool(rng.random() > 0.15)
        f = priority_feature_vector(severity, delta, population, medical, accessible)
        X.append(f)
        y.append(true_tier(f))
    return np.array(X), np.array(y)


# --- scenario generation -------------------------------------------------

def build_grid_graph(seed: int, rows: int = 6, cols: int = 6, spacing_km: float = 1.2) -> RoadGraph:
    """A lat/lon road grid with 4-neighbour streets; a few edges start blocked
    to exercise the A* blocked-edge handling."""
    rng = np.random.default_rng(seed)
    base_lat, base_lon = 19.07, 72.87  # Mumbai-ish anchor; arbitrary but real-world scale
    dlat = spacing_km / 111.0
    dlon = spacing_km / (111.0 * np.cos(np.radians(base_lat)))

    coords: dict[int, tuple[float, float]] = {}
    for r in range(rows):
        for c in range(cols):
            node = r * cols + c
            coords[node] = (base_lat + r * dlat, base_lon + c * dlon)

    edges: list[Edge] = []
    for r in range(rows):
        for c in range(cols):
            node = r * cols + c
            for dr, dc in ((0, 1), (1, 0)):  # east and south; undirected
                nr, nc = r + dr, c + dc
                if nr < rows and nc < cols:
                    nbr = nr * cols + nc
                    length_m = haversine_km(coords[node], coords[nbr]) * 1000.0
                    speed = float(rng.choice([30, 40, 50, 60]))
                    blocked = bool(rng.random() < 0.08)  # ~8% of streets submerged
                    edges.append(Edge(u=node, v=nbr, length_m=length_m,
                                      speed_kmph=speed, blocked=blocked))
    return RoadGraph(coords=coords, edges=edges)


def generate_scenario(
    seed: int = 42, n_zones: int = 8, n_units: int = 6,
    scenario_id: str | None = None,
) -> Scenario:
    rng = np.random.default_rng(seed)
    graph = build_grid_graph(seed)
    nodes = list(graph.coords.keys())

    zone_nodes = list(rng.choice(nodes, size=min(n_zones, len(nodes)), replace=False))
    zones: list[Zone] = []
    for i, node in enumerate(zone_nodes):
        lat, lon = graph.coords[int(node)]
        population = int(rng.integers(800, 40000))
        severity = float(rng.uniform(0.1, 0.9))
        medical = int(rng.integers(0, 40))
        zones.append(Zone(
            id=f"Z{i}", name=f"Zone-{chr(65 + i)}", node=int(node), lat=lat, lon=lon,
            population=population, area_km2=float(rng.uniform(0.5, 3.0)),
            severity=severity, casualties=int(rng.integers(0, 60)),
            medical_cases=medical, accessible=bool(rng.random() > 0.1),
            elevation_m=float(rng.uniform(0, _MAX_ELEV)),
            unserved_min=float(rng.choice([0, 0, 0, 45, 90, 150])),
            hazard_series=list(np.clip(rng.uniform(0.1, 0.9, size=6)
                                       + np.linspace(0, rng.uniform(-0.2, 0.3), 6), 0, 1)),
        ))

    depot_nodes = list(rng.choice(nodes, size=min(n_units, len(nodes)), replace=False))
    kinds = [UnitKind.AMBULANCE, UnitKind.TEAM, UnitKind.BOAT,
             UnitKind.AMBULANCE, UnitKind.TEAM, UnitKind.RELIEF]
    units: list[Unit] = []
    for i, node in enumerate(depot_nodes):
        lat, lon = graph.coords[int(node)]
        kind = kinds[i % len(kinds)]
        units.append(Unit(
            id=f"U{i}", kind=kind,
            capacity=int(rng.integers(20, 120)),
            depot_node=int(node), depot_lat=lat, depot_lon=lon,
        ))

    return Scenario(
        id=scenario_id or f"scn-{seed}",
        seed=seed, zones=zones, units=units, graph=graph,
    )
