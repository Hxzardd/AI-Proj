# Bandobast — backend

Optimisation-first disaster response and resource allocation. No language model
participates in the decision path: every recommendation reduces to a risk
score, a path cost, a binding constraint, and an objective contribution.

This package implements the full decision pipeline from the Review-1 proposal:

```
Ingest → Forecast → Rank → Route → Allocate → Explain
```

Each stage hands the next a typed, inspectable object and appends a trace entry,
so the final plan carries the whole execution trace and is replayable.

## Pipeline ↔ proposal mapping

| Stage | Module | AI technique | Inspectable artefact |
|-------|--------|--------------|----------------------|
| Forecast | `forecast.py` | Supervised regression (linear vs gradient-boosted), persistence baseline | held-out RMSE/MAE, calibration, feature importances |
| Rank | `priority.py` | Depth-limited decision tree | rendered tree, per-zone decision path, confusion matrix |
| Route | `routing.py` | Custom A\* over a road graph, blocked edges, admissible haversine/max-speed heuristic | expanded-node count, frontier size, path cost, Dijkstra cross-check |
| Allocate | `allocation.py` | CP/ILP via OR-Tools CP-SAT | variable/constraint listing, feasibility certificate, objective value, optimality gap |
| Baseline | `ga_baseline.py` | Genetic Algorithm (DEAP) on the same objective | fitness-convergence curve vs the exact solver |
| Cluster | `clustering.py` | K-Means over normalised risk features (secondary) | silhouette score, elbow curve, cluster map |
| Verify | `verify.py` | independent re-derivation of hard constraints | list of violations (empty == certified feasible) |

`domain.py` holds the typed contracts; `pipeline.py` orchestrates; `repository.py`
is the audit/replay store; `api.py` is the FastAPI workspace with a served
operator dashboard (`static/index.html`).

## Allocation model

Hard feasibility constraints, independently checked by `verify.py` and never
violated by a returned plan:

* each unit is assigned at most once;
* a zone's assigned capacity does not exceed its requirement;
* a unit can only be sent to a zone it can reach;
* operator pins/vetoes are absorbed as equality constraints.

Coverage requirements — critical-tier coverage, a timely ambulance for medical
emergencies, and the equity bound on starved zones — are enforced
**lexicographically**: a dominating bonus makes CP-SAT meet every *coverable*
target before optimising the real objective (escalation-weighted persons
reached minus a travel-time penalty). This is deliberate: under scarcity (more
mandatory zones than units — the situation the system exists for) a bare
`INFEASIBLE` that deploys nothing is the wrong answer. The solver instead meets
as many targets as possible and reports the rest (`AllocationResult.coverage`).
In resource-sufficient instances this is identical to hard constraints.

The optimality gap is reported on the solver's full objective; `0.0` means the
plan is provably optimal.

## Running

```bash
# from this directory, with deps installed (see pyproject.toml)
uvicorn bandobast.api:app --reload
# open http://127.0.0.1:8000/  -> operator workspace
```

Key endpoints: `POST /scenarios/generate`, `POST /scenarios/{id}/solve`,
`POST /scenarios/{id}/override`, `POST /plans/{id}/replay`, `GET /plans/{id}`.

Set `BANDOBAST_DATA=/some/dir` to persist scenarios and plans to disk for the
audit trail; unset, the repository is in-memory.

## Review 1–3 feature coverage

Every feature committed across the three review checkpoints is implemented,
tested, and exercised by the end-to-end smoke test:

| Review | Committed feature | Module(s) | Verified by |
|--------|-------------------|-----------|-------------|
| **1** | Zone/unit/scenario typed data model | `domain.py`, `synth.py` | `test_pipeline`, smoke `[2]` |
| **1** | A\* routing over road graph, blocked edges, Dijkstra cross-check | `routing.py` | `test_routing`, smoke `[3]` |
| **1** | CP-SAT allocation with reported optimality gap | `allocation.py` | `test_allocation`, smoke `[3]` |
| **2** | Escalation forecast vs persistence baseline (RMSE/MAE, calibration) | `forecast.py` | `test_forecast`, smoke `[3]` |
| **2** | Depth-limited decision tree with per-zone decision path | `priority.py` | `test_priority`, smoke `[3]` |
| **2** | Genetic-algorithm comparative baseline | `ga_baseline.py` | `test_ga`, smoke `[3]` |
| **2** | Evaluation harness / execution trace + constraint-violation property test | `pipeline.py`, `verify.py` | `test_allocation`, smoke `[4]` |
| **2** | K-Means risk clustering (secondary) | `clustering.py` | `test_clustering`, smoke `[3]` |
| **3** | Operator override (pin/veto) + constrained re-solve | `api.py`, `allocation.py` | `test_allocation`, smoke `[5]` |
| **3** | Audit trail + deterministic scenario replay | `repository.py`, `api.py` | `test_pipeline`, `test_api`, smoke `[6]` |

## Tests

```bash
pytest -q
```

The suite includes the property test required by the proposal — for a range of
seeds, `verify.constraint_violations` confirms that **no returned plan violates
a declared hard constraint** — plus A\*-equals-Dijkstra correctness, heuristic
admissibility, blocked-edge handling, forecast-beats-persistence, decision-path
extraction, GA-cannot-beat-the-exact-solver, replay determinism, and the API
flow.

### End-to-end smoke test

```bash
python scripts/smoke_test.py
```

Drives the live FastAPI app through the whole decision flow — generate → solve →
verify → override/re-solve → replay — and asserts every review's committed
feature in one run. Exit code `0` means all checks passed.

## Scope decisions (stated, not incidental)

* **Data.** Review 1 operates on seeded scenarios, not live telemetry. `synth.py`
  generates data from a *known* generative process so the regression model can
  genuinely recover signal (and beat persistence) and the tree can recover a
  readable rule. Real EM-DAT / IMD / OSM / WorldPop / SRTM extracts plug in at
  the same feature boundary (`FORECAST_FEATURES`, `PRIORITY_FEATURES`) and the
  same `RoadGraph` contract. No accuracy claim is made against real events.
* **Store.** The proposal's production target is PostgreSQL + PostGIS. This
  prototype uses a file/in-memory repository behind the same interface so it
  runs fully offline on a single machine, as the proposal requires.
* **Frontend.** The proposal's production UI is Next.js 16 / React 19 / MapLibre
  / Recharts (a separate team member's responsibility). This repository ships a
  single self-contained operator dashboard served by FastAPI that exercises the
  entire pipeline — heatmap, priority queue with per-zone reasoning, deployment
  table with ETAs and override controls, inventory, forecast/GA charts, and the
  full execution trace — so the system is demonstrable and verifiable offline.
