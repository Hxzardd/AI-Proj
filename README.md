# Bandobast — Optimisation-First Disaster Response

Bandobast is a disaster-response and resource-allocation engine where **no
language model participates in the decision path**. Every recommendation reduces
to an inspectable, verifiable quantity: a risk score, a path cost, a binding
constraint, and an objective contribution.

The decision pipeline runs end to end:

```
Ingest → Forecast → Rank → Route → Allocate → Explain
```

Each stage hands the next a typed, inspectable object and appends a trace entry,
so the final plan carries its whole execution trace and is replayable.

## Repository layout

```
bandobast/
├── backend/            FastAPI service + full decision pipeline (Python)
│   ├── bandobast/      pipeline modules (forecast, priority, routing, ...)
│   │   └── static/     self-contained operator dashboard
│   └── tests/          property, correctness, and API tests
└── README.md
```

See [`backend/README.md`](backend/README.md) for the full pipeline ↔ proposal
mapping, the allocation model, and scope decisions.

## Technique map

| Stage | Module | AI technique |
|-------|--------|--------------|
| Forecast | `forecast.py` | Supervised regression vs persistence baseline |
| Rank | `priority.py` | Depth-limited decision tree |
| Route | `routing.py` | Custom A\* over a road graph with blocked edges |
| Allocate | `allocation.py` | CP/ILP via OR-Tools CP-SAT |
| Baseline | `ga_baseline.py` | Genetic Algorithm (DEAP) on the same objective |
| Cluster | `clustering.py` | K-Means over normalised risk features |
| Verify | `verify.py` | Independent re-derivation of hard constraints |

## Quick start

```bash
cd backend

# create a virtual environment and install (editable, with dev extras)
python -m venv .venv
.venv/Scripts/activate          # Windows
# source .venv/bin/activate     # macOS / Linux
pip install -e ".[dev]"

# run the service + operator dashboard
uvicorn bandobast.api:app --reload
# open http://127.0.0.1:8000/
```

Key endpoints: `POST /scenarios/generate`, `POST /scenarios/{id}/solve`,
`POST /scenarios/{id}/override`, `POST /plans/{id}/replay`, `GET /plans/{id}`.

Set `BANDOBAST_DATA=/some/dir` to persist scenarios and plans to disk for the
audit trail; unset, the repository is in-memory.

## Review coverage (Reviews 1–3)

Every feature committed across the three review checkpoints is implemented,
unit-tested, and exercised by a single end-to-end smoke test:

- **Review 1** — typed scenario/zone/unit data model, A\* routing over a road
  graph with blocked edges and a Dijkstra cross-check, and CP-SAT allocation
  with a reported optimality gap.
- **Review 2** — escalation forecasting vs a persistence baseline, a depth-limited
  decision-tree priority classifier with per-zone decision paths, a genetic-algorithm
  comparative baseline, K-Means risk clustering, and the evaluation harness
  (execution trace + constraint-violation property test).
- **Review 3** — operator override (pin/veto) with constrained re-solve, plus an
  audit trail and deterministic scenario replay.

See [`backend/README.md`](backend/README.md) for the per-feature mapping to
modules and tests.

## Tests

```bash
cd backend
pytest -q                     # 29 unit + property tests
python scripts/smoke_test.py  # end-to-end smoke test of the live API
```

The suite includes the property test that confirms **no returned plan violates a
declared hard constraint** across a range of seeds, plus A\*-equals-Dijkstra
correctness, heuristic admissibility, forecast-beats-persistence,
GA-cannot-beat-the-exact-solver, replay determinism, and the API flow. The smoke
test drives the whole decision flow (generate → solve → verify → override →
replay) and asserts every review's committed feature in one run.

## Requirements

Python ≥ 3.11. Dependencies are declared in
[`backend/pyproject.toml`](backend/pyproject.toml).
