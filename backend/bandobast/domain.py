"""Typed domain contracts.

Each pipeline stage hands the next a typed, inspectable object rather than
prose (proposal Section 7). These models are the "typed contracts" referenced
in the data-model row of the status table.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field

from .config import SolveParams


class UnitKind(str, Enum):
    TEAM = "team"            # rescue team
    AMBULANCE = "ambulance"  # qualifies for medical emergencies
    BOAT = "boat"
    RELIEF = "relief"        # relief-load carrier


class Tier(str, Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MODERATE = "moderate"
    LOW = "low"


class Zone(BaseModel):
    """A district sub-area with geometry, exposure, and hazard state."""

    id: str
    name: str
    node: int = Field(..., description="Road-graph node id this zone attaches to.")
    lat: float
    lon: float
    population: int
    area_km2: float = 1.0
    severity: float = Field(..., ge=0.0, le=1.0, description="Current severity 0..1.")
    casualties: int = 0
    medical_cases: int = 0
    accessible: bool = True
    elevation_m: float = 0.0
    unserved_min: float = Field(
        0.0, description="How long this zone has already gone unserved."
    )
    # Hazard telemetry: recent window of normalised rainfall/hazard readings.
    hazard_series: list[float] = Field(default_factory=list)

    @property
    def population_density(self) -> float:
        return self.population / max(self.area_km2, 1e-6)

    @property
    def requirement(self) -> int:
        """Upper bound on unit capacity the zone can usefully absorb this round.
        Scales with the exposed, severity-weighted population plus reported
        casualties and medical cases. Floored above the largest unit capacity
        so a single unit can always be assigned (the ceiling only prevents
        wasteful over-stacking of several units on one zone)."""
        exposed = int(self.population * self.severity / 50)
        return exposed + self.casualties + 2 * self.medical_cases + 150


class Unit(BaseModel):
    """A deployable resource stationed at a depot node."""

    id: str
    kind: UnitKind
    capacity: int = Field(..., gt=0, description="Persons this unit can serve.")
    depot_node: int
    depot_lat: float
    depot_lon: float

    @property
    def qualifies_medical(self) -> bool:
        return self.kind == UnitKind.AMBULANCE


class Edge(BaseModel):
    u: int
    v: int
    length_m: float
    speed_kmph: float = 40.0
    blocked: bool = False


class RoadGraph(BaseModel):
    """Serialisable road network. Coordinates are keyed by node id."""

    coords: dict[int, tuple[float, float]]  # node -> (lat, lon)
    edges: list[Edge]


class Override(BaseModel):
    """A commander decision injected as a constraint on re-solve."""

    unit_id: str
    zone_id: str
    action: str = Field(..., pattern="^(pin|veto)$")


class Scenario(BaseModel):
    """A versioned, immutable-once-solved snapshot of district state."""

    id: str
    seed: int
    zones: list[Zone]
    units: list[Unit]
    graph: RoadGraph
    params: SolveParams = Field(default_factory=SolveParams)
    overrides: list[Override] = Field(default_factory=list)


# --- Outputs -------------------------------------------------------------

class RouteResult(BaseModel):
    """A* result from a depot to a zone, with the inspectable search stats."""

    source: int
    target: int
    reachable: bool
    eta_min: float = 0.0
    path: list[int] = Field(default_factory=list)
    cost_min: float = 0.0
    expanded_nodes: int = 0
    max_frontier: int = 0
    dijkstra_cost_min: float | None = None   # ground-truth cross-check
    dijkstra_expanded: int | None = None     # nodes Dijkstra settled (efficiency comparison)


class ForecastResult(BaseModel):
    zone_id: str
    current_severity: float
    predicted_delta: float      # model's escalation forecast
    persistence_delta: float    # naive baseline (always 0)
    predicted_next: float


class ForecastReport(BaseModel):
    model: str
    rmse: float
    mae: float
    persistence_rmse: float
    persistence_mae: float
    beats_persistence: bool
    feature_importances: dict[str, float]
    # (predicted_bin_mean, actual_bin_mean) pairs for the calibration plot.
    calibration: list[tuple[float, float]] = Field(default_factory=list)
    forecasts: list[ForecastResult] = Field(default_factory=list)


class PriorityResult(BaseModel):
    zone_id: str
    tier: Tier
    score: float
    decision_path: list[str]  # readable rule at each split, root -> leaf


class PriorityReport(BaseModel):
    max_depth: int
    accuracy: float
    confusion_matrix: list[list[int]]
    labels: list[str]
    tree_text: str
    results: list[PriorityResult] = Field(default_factory=list)


class ClusterReport(BaseModel):
    k: int
    silhouette: float
    inertia_by_k: dict[int, float]  # elbow data
    assignments: dict[str, int]     # zone_id -> cluster


class Assignment(BaseModel):
    unit_id: str
    zone_id: str
    eta_min: float
    persons_reached: float
    objective_contribution: float
    pinned: bool = False


class CoverageReport(BaseModel):
    """How many lexicographic coverage targets were met. `total` counts only
    targets that are structurally coverable (a qualifying/ambulance/reachable
    unit exists); `unmet` lists coverable targets dropped due to scarcity."""

    critical_met: int = 0
    critical_total: int = 0
    medical_met: int = 0
    medical_total: int = 0
    equity_met: int = 0
    equity_total: int = 0
    unmet: list[str] = Field(default_factory=list)


class AllocationResult(BaseModel):
    status: str                 # OPTIMAL / FEASIBLE / INFEASIBLE
    assignments: list[Assignment]
    objective_value: float      # real objective: escalation-weighted reach - travel
    solver_objective: float     # full solver objective incl. coverage bonus
    best_bound: float           # bound on the full solver objective
    optimality_gap: float       # |bound - solver_objective|/|bound|, 0 == proven optimal
    wall_time_s: float
    coverage: CoverageReport = Field(default_factory=CoverageReport)
    unreached_critical: list[str] = Field(default_factory=list)
    constraints_listed: list[str] = Field(default_factory=list)


class GAResult(BaseModel):
    best_fitness: float          # full internal fitness (incl. coverage bonus)
    convergence: list[float]     # best full fitness per generation
    cpsat_objective: float       # full CP-SAT objective, for the convergence reference line
    gap_to_cpsat: float          # gap on the full objective (coverage-dominated)
    # Comparison on the *real* objective (escalation-weighted reach - travel),
    # which is the meaningful number: how much deployment value the heuristic
    # leaves on the table versus the exact solver.
    ga_objective: float = 0.0
    cpsat_real: float = 0.0
    real_gap: float = 0.0        # (cpsat_real - ga_real)/|cpsat_real|
    coverage_match: bool = True  # did the GA meet the same coverage targets?


class StageTrace(BaseModel):
    stage: str
    summary: str
    artefact: dict  # stage-specific inspectable artefact


class ExecutionTrace(BaseModel):
    stages: list[StageTrace] = Field(default_factory=list)


class Plan(BaseModel):
    """The full, replayable result of solving a scenario."""

    plan_id: str
    scenario_id: str
    seed: int
    model_version: str
    params: SolveParams
    forecast: ForecastReport
    priority: PriorityReport
    clusters: ClusterReport | None
    routes: list[RouteResult]
    allocation: AllocationResult
    ga_baseline: GAResult | None
    trace: ExecutionTrace
