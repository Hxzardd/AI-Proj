"""Tunable parameters for the decision pipeline.

Every value that influences a recommendation lives here so that a solved plan
can be reproduced from (scenario snapshot + seed + this parameter set + model
version). These are the knobs the proposal calls "a visible, adjustable
parameter instead of an emergent side effect".
"""

from __future__ import annotations

from pydantic import BaseModel, Field

# Bumped whenever the trained-model feature set or label rule changes, so a
# persisted plan records exactly which model produced it.
MODEL_VERSION = "forecast-lin+gbr-v1;priority-dtree-d3-v1"


class SolveParams(BaseModel):
    """Parameters handed to the allocation solver and routing layer."""

    # Routing.
    max_speed_kmph: float = Field(
        60.0, description="Upper bound on edge speed; drives the admissible A* heuristic."
    )

    # Objective weights.
    escalation_weight: float = Field(
        1.0, description="Multiplier on forecast escalation when valuing a zone."
    )
    travel_penalty_per_min: float = Field(
        2.0, description="Objective penalty per minute of ETA for an assignment."
    )

    # Hard constraints.
    medical_eta_bound_min: float = Field(
        30.0, description="Ambulances must reach medical-emergency zones within this ETA."
    )
    max_unserved_min: float = Field(
        120.0,
        description="Equity bound: a zone unserved longer than this must be served if reachable.",
    )

    # Solver.
    time_limit_s: float = Field(10.0, description="CP-SAT wall-clock budget.")
    seed: int = Field(42, description="Random seed for solver and any sampling.")

    # Objective scaling (CP-SAT is integer; we scale floats to integers).
    objective_scale: int = Field(
        1000, description="Float->int scale factor for the CP-SAT objective."
    )
