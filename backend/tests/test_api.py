from fastapi.testclient import TestClient

from bandobast.api import app

client = TestClient(app)


def test_health():
    assert client.get("/health").json()["status"] == "ok"


def test_generate_solve_override_replay_flow():
    scn = client.post("/scenarios/generate",
                      json={"seed": 42, "n_zones": 6, "n_units": 5}).json()
    sid = scn["id"]

    plan = client.post(f"/scenarios/{sid}/solve").json()
    assert plan["allocation"]["status"] in ("OPTIMAL", "FEASIBLE")
    assert len(plan["trace"]["stages"]) == 7

    # Replay reconstructs the same decision.
    replay = client.post(f"/plans/{plan['plan_id']}/replay").json()
    assert replay["allocation"]["objective_value"] == plan["allocation"]["objective_value"]

    # Override: veto the first assignment and confirm it is honoured.
    first = plan["allocation"]["assignments"][0]
    ov = [{"unit_id": first["unit_id"], "zone_id": first["zone_id"], "action": "veto"}]
    re = client.post(f"/scenarios/{sid}/override", json=ov).json()
    pairs = {(a["unit_id"], a["zone_id"]) for a in re["allocation"]["assignments"]}
    assert (first["unit_id"], first["zone_id"]) not in pairs


def test_missing_scenario_404():
    assert client.post("/scenarios/nope/solve").status_code == 404
