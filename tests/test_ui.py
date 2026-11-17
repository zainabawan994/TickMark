import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from tickmark.ui.server import create_app

from .fakes import ScriptedModel, call, decide

SAMPLES = Path(__file__).parent.parent / "samples"


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # learned aliases are written under ./.tickmark
    for var in ("GOOGLE_API_KEY", "GEMINI_API_KEY", "ANTHROPIC_API_KEY", "TICKMARK_MODEL"):
        monkeypatch.delenv(var, raising=False)
    return TestClient(create_app(work_dir=tmp_path / "runs"))


def wait_for(client, run_id, statuses=("waiting", "done", "error"), timeout=20):
    end = time.time() + timeout
    while time.time() < end:
        snap = client.get(f"/api/runs/{run_id}").json()
        if snap["status"] in statuses:
            return snap
        time.sleep(0.05)
    raise AssertionError(f"run stuck in {snap['status']}")


def test_page_and_info(client):
    assert "Reconcile your bank statement" in client.get("/").text
    info = client.get("/api/info").json()
    assert info["ai_ready"] is False and "GOOGLE_API_KEY" in info["reason"] and info["samples"] is True


def test_full_flow_with_decisions(client):
    run_id = client.post("/api/runs/sample", data={"use_ai": "false", "auto": "false"}).json()["id"]
    snap = wait_for(client, run_id)
    assert snap["status"] == "waiting"
    assert [s["key"] for s in snap["steps"]] == ["read", "match", "ai", "journal", "decide", "report"]
    assert next(s for s in snap["steps"] if s["key"] == "ai")["skipped"] is True
    pending = snap["pending"]
    assert len(pending) == 3 and pending[0]["line"]["matched_to"].startswith("INV-2042")
    answers = {pending[0]["je_id"]: "approve", pending[1]["je_id"]: "reject", pending[2]["je_id"]: "skip"}
    assert client.post(f"/api/runs/{run_id}/decisions", json={"answers": answers}).json() == {"ok": True}
    done = wait_for(client, run_id, ("done", "error"))
    assert done["status"] == "done"
    m = done["summary"]
    assert (m["matched"], m["suspense"], m["waiting"], m["unresolved"]) == (12, 0, 1, 3)   # approved, skipped, rejected + unmatched
    assert m["matched"] + m["suspense"] + m["waiting"] + m["unresolved"] == m["lines"] == 16
    outcomes = {l["id"]: l["outcome"] for l in done["lines"]}
    assert (outcomes["B8"], outcomes["B10"], outcomes["B11"]) == ("matched", "unresolved", "waiting")
    keys = {d["key"] for d in done["downloads"]}
    assert {"approved_entries", "report", "review_queue", "summary"} <= keys
    csv_text = client.get(f"/api/runs/{run_id}/files/approved_entries").text
    assert csv_text.startswith("Entry,Date,Account") and "Cedar Point Dental" in csv_text
    assert client.post(f"/api/runs/{run_id}/decisions", json={"answers": {}}).status_code == 409


def test_upload_and_validation(client):
    with open(SAMPLES / "bank_statement.csv", "rb") as b, open(SAMPLES / "ledger.csv", "rb") as l:
        r = client.post("/api/runs", files={"bank": ("oct.csv", b), "ledger": ("ledger.csv", l)}, data={"use_ai": "false"})
    snap = wait_for(client, r.json()["id"])
    assert snap["files"] == {"bank": "oct.csv", "ledger": "ledger.csv"} and snap["status"] == "waiting"
    with open(SAMPLES / "bank_statement.csv", "rb") as b:
        bad = client.post("/api/runs", files={"bank": ("x.exe", b), "ledger": ("l.pdf", b)})
    assert bad.status_code == 400
    assert client.get("/api/runs/nope").status_code == 404


def test_ai_step_shows_progress(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    scripted = ScriptedModel(scripts={"B8": [call("search_open_items", amount=1275.0), decide("match", ["L9"], "high", "Only invoice for 1,275.00.")]})
    client = TestClient(create_app(model_factory=lambda s: scripted, work_dir=tmp_path / "runs"))
    assert client.get("/api/info").json()["ai_ready"] is True
    run_id = client.post("/api/runs/sample", data={"use_ai": "true", "auto": "true"}).json()["id"]
    snap = wait_for(client, run_id)
    ai = next(s for s in snap["steps"] if s["key"] == "ai")
    assert ai["done"] and ai["detail"] == "5 of 5 checked"
    pending = {p["bank_id"] for p in snap["pending"]}
    # With "approve confident AI matches", B8 no longer needs a person...
    assert "B8" not in pending
    # ...but an unsure AI doesn't override the engine's own proposals for B10 and B11: a person decides.
    assert pending == {"B10", "B11"}
