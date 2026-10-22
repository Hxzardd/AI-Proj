"""End-to-end smoke test of the Bandobast decision pipeline via the live API.

Drives the FastAPI app through a real ASGI TestClient (the same path uvicorn
serves) and asserts the invariants the proposal commits to, grouped by review:

  Review 1  scenario ingest + typed data model, A* routing (Dijkstra cross-check),
            CP-SAT allocation with a reported optimality gap
  Review 2  escalation forecast vs persistence baseline, decision-tree priority
            with per-zone decision paths, GA comparative baseline, the full
            Ingest->Explain execution trace (evaluation harness)
  Review 3  operator override (pin/veto) + constrained re-solve, audit trail,
            deterministic replay

Run from the backend directory:

    python scripts/smoke_test.py

Exit code 0 == every check passed.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Make the package importable whether run from backend/ or scripts/.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi.testclient import TestClient

from bandobast.api import app
from bandobast.domain import Override, Plan, Scenario
from bandobast.forecast import EscalationForecaster
from bandobast.pipeline import _route_all
from bandobast.priority import PriorityClassifier
from bandobast.verify import constraint_violations

client = TestClient(app)
_ok = True


def check(label: str, cond: bool) -> None:
    global _ok
    print(f"  [{'PASS' if cond else 'FAIL'}] {label}")
    _ok = _ok and bool(cond)


def rebuild_internals(scn: Scenario):
    """Reconstruct the (unit,zone)-keyed routes and priority tiers the verifier
    needs, independently of the API JSON, for the same seed (deterministic)."""
    seed = scn.seed
    routes = _route_all(scn)
    fc = EscalationForecaster(seed=seed).fit().report(scn)
    delta = {f.zone_id: f.predicted_delta for f in fc.forecasts}
    tiers = {p.zone_id: p.tier
             for p in PriorityClassifier(seed=seed).fit().report(scn, delta).results}
    return routes, tiers


def main() -> int:
    print("=" * 70)
    print("BANDOBAST END-TO-END SMOKE TEST")
    print("=" * 70)

    print("\n[0] Health")
    r = client.get("/health")
    check("GET /health -> 200", r.status_code == 200)
    check("status ok", r.json()["status"] == "ok")

    print("\n[1] Operator workspace (dashboard)")
    r = client.get("/")
    check("GET / -> 200", r.status_code == 200)
    check("serves HTML dashboard", "<html" in r.text.lower())

    print("\n[2] Review 1 - scenario ingest + typed data model")
    r = client.post("/scenarios/generate", json={"seed": 7, "n_zones": 8, "n_units": 6})
    check("POST /scenarios/generate -> 200", r.status_code == 200)
    scn = r.json()
    sid = scn["id"]
    check("8 zones generated", len(scn["zones"]) == 8)
    check("6 units generated", len(scn["units"]) == 6)
    check("road graph present", len(scn["graph"]["edges"]) > 0)

    print("\n[3] Reviews 1+2 - full pipeline solve")
    r = client.post(f"/scenarios/{sid}/solve")
    check("POST /scenarios/{id}/solve -> 200", r.status_code == 200)
    plan = r.json()
    pid = plan["plan_id"]
    alloc = plan["allocation"]

    route_stage = next(s for s in plan["trace"]["stages"] if s["stage"] == "route")
    check("A* routes computed", len(plan["routes"]) == 8 * 6)
    check("A* matches Dijkstra (0 mismatches)",
          route_stage["artefact"]["astar_dijkstra_mismatches"] == [])

    check("allocation status set", alloc["status"] in ("OPTIMAL", "FEASIBLE"))
    check("optimality gap reported", "optimality_gap" in alloc)
    check("constraints listed", len(alloc["constraints_listed"]) > 0)
    check("assignments produced", len(alloc["assignments"]) > 0)

    fc = plan["forecast"]
    check("forecast rmse reported", "rmse" in fc)
    check("persistence baseline reported", "persistence_rmse" in fc)
    check("forecast beats persistence", fc["beats_persistence"] is True)
    check("feature importances present", len(fc["feature_importances"]) > 0)
    check("calibration data present", len(fc["calibration"]) > 0)

    pr = plan["priority"]
    check("decision tree rendered", len(pr["tree_text"]) > 0)
    check("confusion matrix present", len(pr["confusion_matrix"]) > 0)
    check("per-zone decision paths",
          all(len(x["decision_path"]) > 0 for x in pr["results"]))

    ga = plan["ga_baseline"]
    check("GA baseline ran", ga is not None)
    check("GA convergence curve present", len(ga["convergence"]) > 0)
    check("GA cannot beat exact solver (real_gap >= 0)", ga["real_gap"] >= -1e-9)

    cl = plan["clusters"]
    check("K-Means clustering ran", cl is not None and "silhouette" in cl)

    stages = [s["stage"] for s in plan["trace"]["stages"]]
    check("full Ingest->Explain trace",
          stages[:5] == ["ingest", "forecast", "rank", "route", "allocate"])

    print("\n[4] Verification - no plan violates a hard constraint")
    scn_obj = Scenario.model_validate(scn)
    plan_obj = Plan.model_validate(plan)
    vroutes, vtiers = rebuild_internals(scn_obj)
    viols = constraint_violations(scn_obj, vroutes, vtiers, plan_obj.allocation)
    check("constraint_violations == [] (certified feasible)", viols == [])
    for v in viols:
        print("     violation:", v)

    print("\n[5] Review 3 - operator override + constrained re-solve")
    a0 = alloc["assignments"][0]
    pin = {"unit_id": a0["unit_id"], "zone_id": a0["zone_id"], "action": "pin"}
    r = client.post(f"/scenarios/{sid}/override", json=[pin])
    check("POST /scenarios/{id}/override -> 200", r.status_code == 200)
    plan2 = r.json()
    pinned = [a for a in plan2["allocation"]["assignments"]
              if a["unit_id"] == a0["unit_id"] and a["zone_id"] == a0["zone_id"]]
    check("pinned assignment honoured in re-solve", len(pinned) == 1)
    scn_pin = scn_obj.model_copy(update={"overrides": [Override(**pin)]})
    check("re-solve still feasible under pin",
          constraint_violations(scn_pin, vroutes, vtiers,
                                Plan.model_validate(plan2).allocation,
                                [Override(**pin)]) == [])

    v1 = alloc["assignments"][1]
    veto = {"unit_id": v1["unit_id"], "zone_id": v1["zone_id"], "action": "veto"}
    plan3 = client.post(f"/scenarios/{sid}/override", json=[veto]).json()
    vpresent = [a for a in plan3["allocation"]["assignments"]
                if a["unit_id"] == v1["unit_id"] and a["zone_id"] == v1["zone_id"]]
    check("vetoed assignment excluded in re-solve", len(vpresent) == 0)

    print("\n[6] Review 3 - audit trail + deterministic replay")
    check("GET /plans/{id} -> 200 (audit retrieval)",
          client.get(f"/plans/{pid}").status_code == 200)
    fresh = client.post("/scenarios/generate",
                        json={"seed": 7, "n_zones": 8, "n_units": 6}).json()
    p_a = client.post(f"/scenarios/{fresh['id']}/solve").json()
    rr = client.post(f"/plans/{p_a['plan_id']}/replay")
    check("POST /plans/{id}/replay -> 200", rr.status_code == 200)
    p_b = rr.json()
    check("replay reproduces objective value",
          abs(p_a["allocation"]["objective_value"]
              - p_b["allocation"]["objective_value"]) < 1e-9)
    check("replay reproduces identical assignment set",
          {(a["unit_id"], a["zone_id"]) for a in p_a["allocation"]["assignments"]}
          == {(a["unit_id"], a["zone_id"]) for a in p_b["allocation"]["assignments"]})
    check("audit store persisted plans", client.get("/health").json()["plans"] > 0)

    print("\n" + "=" * 70)
    print("RESULT:", "ALL SMOKE CHECKS PASSED" if _ok else "SOME CHECKS FAILED")
    print("=" * 70)
    return 0 if _ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
