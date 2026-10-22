"""FastAPI decision orchestrator.

Exposes the pipeline as a stateful workspace: generate or post a scenario,
solve it, inject operator overrides and re-solve, and replay any stored plan.
Trained models are fitted once per seed and reused across requests.
"""

from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .domain import Override, Plan, Scenario
from .forecast import EscalationForecaster
from .pipeline import solve_scenario
from .priority import PriorityClassifier
from .repository import Repository
from .synth import generate_scenario

app = FastAPI(title="Bandobast", version="0.1.0",
              description="Optimisation-first disaster response and resource allocation.")

repo = Repository(os.environ.get("BANDOBAST_DATA"))
_STATIC = Path(__file__).parent / "static"

# Fit-once model cache, keyed by seed.
_forecasters: dict[int, EscalationForecaster] = {}
_classifiers: dict[int, PriorityClassifier] = {}


def _models(seed: int):
    if seed not in _forecasters:
        _forecasters[seed] = EscalationForecaster(seed=seed).fit()
        _classifiers[seed] = PriorityClassifier(seed=seed).fit()
    return _forecasters[seed], _classifiers[seed]


def _solve(scenario: Scenario) -> Plan:
    forecaster, classifier = _models(scenario.seed)
    plan = solve_scenario(scenario, forecaster=forecaster, classifier=classifier)
    repo.save_plan(plan)
    return plan


class GenerateRequest(BaseModel):
    seed: int = 42
    n_zones: int = 8
    n_units: int = 6
    scenario_id: str | None = None


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "scenarios": len(repo.list_scenarios()), "plans": len(repo.list_plans())}


@app.post("/scenarios/generate", response_model=Scenario)
def generate(req: GenerateRequest) -> Scenario:
    scenario = generate_scenario(req.seed, req.n_zones, req.n_units, req.scenario_id)
    repo.save_scenario(scenario)
    return scenario


@app.post("/scenarios", response_model=Scenario)
def create_scenario(scenario: Scenario) -> Scenario:
    repo.save_scenario(scenario)
    return scenario


@app.get("/scenarios")
def list_scenarios() -> list[str]:
    return repo.list_scenarios()


@app.get("/scenarios/{scenario_id}", response_model=Scenario)
def get_scenario(scenario_id: str) -> Scenario:
    s = repo.get_scenario(scenario_id)
    if not s:
        raise HTTPException(404, "scenario not found")
    return s


@app.post("/scenarios/{scenario_id}/solve", response_model=Plan)
def solve(scenario_id: str) -> Plan:
    s = repo.get_scenario(scenario_id)
    if not s:
        raise HTTPException(404, "scenario not found")
    return _solve(s)


@app.post("/scenarios/{scenario_id}/override", response_model=Plan)
def override(scenario_id: str, overrides: list[Override]) -> Plan:
    s = repo.get_scenario(scenario_id)
    if not s:
        raise HTTPException(404, "scenario not found")
    s.overrides = overrides  # pins/vetoes become constraints on re-solve
    repo.save_scenario(s)
    return _solve(s)


@app.get("/plans/{plan_id}", response_model=Plan)
def get_plan(plan_id: str) -> Plan:
    p = repo.get_plan(plan_id)
    if not p:
        raise HTTPException(404, "plan not found")
    return p


@app.post("/plans/{plan_id}/replay", response_model=Plan)
def replay(plan_id: str) -> Plan:
    """Re-run the pipeline from the stored scenario + seed + params. The
    allocation is deterministic, so a replay reconstructs the same decision."""
    p = repo.get_plan(plan_id)
    if not p:
        raise HTTPException(404, "plan not found")
    s = repo.get_scenario(p.scenario_id)
    if not s:
        raise HTTPException(404, "scenario for plan not found")
    return _solve(s)


@app.get("/")
def index() -> FileResponse:
    # no-cache so a reload always serves the latest dashboard build
    return FileResponse(_STATIC / "index.html", headers={"Cache-Control": "no-cache"})


if _STATIC.exists():
    app.mount("/static", StaticFiles(directory=_STATIC), name="static")
