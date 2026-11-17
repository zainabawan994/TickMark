import csv
import uuid
from pathlib import Path

from langgraph.types import Command

from tickmark.config import RunContext, Settings
from tickmark.graph.build import build_graph, make_checkpointer

from .fakes import ScriptedModel, call, decide

SAMPLES = Path(__file__).parent.parent / "samples"
# Ledger ids follow row order in samples/ledger.csv:
L_INV2035, L_INV2040, L_INV2041, L_INV2042, L_INV2043, L_INV2045 = "L2", "L7", "L8", "L9", "L10", "L14"


def run(tmp_path, settings: Settings, model=None, bank="bank_statement.csv", ledger="ledger.csv"):
    settings.memory_path = str(tmp_path / "aliases.json")
    graph = build_graph(checkpointer=make_checkpointer())
    cfg = {"configurable": {"thread_id": uuid.uuid4().hex}, "max_concurrency": 4}
    ctx = RunContext(settings=settings, model=model)
    state = graph.invoke({"bank_path": str(SAMPLES / bank), "ledger_path": str(SAMPLES / ledger), "out_dir": str(tmp_path / "out")},
                         cfg, context=ctx)
    return graph, cfg, ctx, state


def resolver_model():
    return ScriptedModel(scripts={
        "B8": [call("search_open_items", amount=1275.0), decide("match", [L_INV2042], "high", "Only open invoice for 1,275.00.")],
        "B10": [call("party_open_items", party="Atlas Engineering"), decide("match", [L_INV2040], "medium", "Atlas overpaid INV-2040.")],
        "B11": [call("find_combinations", target_amount=6000.0, party="Quantum Retail"),
                decide("match", [L_INV2035, L_INV2043], "medium", "Oldest first; no exact combination.")],
        "B13": [call("search_open_items", amount=900.0), decide("suspense", [], "medium", "Two invoices for 900.00; cannot tell which.")],
        # A wrong suggestion: INV-2041 was already paid in full by B1. The guardrail must refuse it.
        "B14": [call("similar_names", name="J RIVERA"), decide("match", [L_INV2041], "high", "Hallucinated match.")],
    })


def test_deterministic_run_writes_all_outputs(tmp_path):
    _, _, _, state = run(tmp_path, Settings(use_llm=False))
    assert not state.get("errors")
    assert set(state["outputs"]) == {"approved_entries", "review_queue", "report", "outstanding", "summary", "run_json"}
    approved = [a for a in state["approvals"].values() if a["status"] == "approved"]
    assert len(approved) == 10 and len(state["approvals"]) == 13
    rows = list(csv.reader(open(state["outputs"]["approved_entries"])))
    debit = sum(float(r[5] or 0) for r in rows[1:])
    credit = sum(float(r[6] or 0) for r in rows[1:])
    assert round(debit, 2) == round(credit, 2) > 0


def test_ai_resolver_with_guardrails(tmp_path):
    model = resolver_model()
    _, _, _, state = run(tmp_path, Settings(), model=model)
    res = state["results"]
    assert len(state["resolutions"]) == 5                       # one parallel agent per exception
    assert res["B8"].source == "ai" and res["B8"].status == "review"   # AI matches wait for a person by default
    assert [a.ledger_id for a in res["B8"].allocations] == [L_INV2042]
    assert res["B13"].status == "suspense"
    assert res["B14"].status == "unmatched"                     # guardrail refused the hallucinated match
    assert "already fully paid" in state["resolution_notes"]["B14"]
    assert state["approvals"][next(je.id for je in state["journal"] if je.bank_id == "B8")]["status"] == "pending"
    assert state["narrative"] == model.narrative
    assert "Scripted narrative" in Path(state["outputs"]["summary"]).read_text()


def test_lights_out_policy(tmp_path):
    s = Settings(ai_auto_approve=True, auto_post_suspense=True)
    _, _, _, state = run(tmp_path, s, model=resolver_model())
    status = {je.bank_id: state["approvals"][je.id]["status"] for je in state["journal"]}
    assert status["B8"] == "approved"       # high-confidence AI match
    assert status["B13"] == "approved"      # suspense posted automatically
    assert status["B10"] == "pending"       # medium confidence still needs a person


def test_interactive_interrupt_resume_and_learning(tmp_path):
    graph, cfg, ctx, state = run(tmp_path, Settings(use_llm=False, interactive=True))
    request = state["__interrupt__"][0].value
    assert request["type"] == "approval_request" and len(request["items"]) == 3
    cedar = next(i for i in request["items"] if i["bank_id"] == "B8")
    answers = {i["je_id"]: "approve" for i in request["items"]}
    answers[next(i["je_id"] for i in request["items"] if i["bank_id"] == "B10")] = "reject"
    final = graph.invoke(Command(resume=answers), cfg, context=ctx)
    assert final["approvals"][cedar["je_id"]] == {"status": "approved", "by": "person", "reason": "approved by a person"}
    assert sum(a["status"] == "rejected" for a in final["approvals"].values()) == 1
    # "MOBILE DEP 4471" holds only bank channel words, so nothing is learned from it (it would
    # wrongly boost every future mobile deposit).
    assert "mobile" not in (tmp_path / "aliases.json").read_text()


def test_learns_name_variations_from_approvals(tmp_path):
    """A customer pays under the owner's personal name; once approved, next month it matches on its own."""
    bank = tmp_path / "oct.csv"
    bank.write_text("Date,Description,Amount\n05/10/2026,TRF FROM M KOWALSKI,750.00\n")
    ledger = tmp_path / "ledger.csv"
    ledger.write_text("Date,Type,Number,Name,Amount\n01/10/2026,Invoice,INV-9001,Northgate Plumbing Ltd,750.00\n"
                      "02/11/2026,Invoice,INV-9002,Northgate Plumbing Ltd,1200.00\n")
    graph, cfg, ctx, state = run(tmp_path, Settings(use_llm=False, interactive=True), bank=str(bank), ledger=str(ledger))
    item = state["__interrupt__"][0].value["items"][0]
    assert item["reason"].startswith("Amount only")
    graph.invoke(Command(resume={item["je_id"]: "approve"}), cfg, context=ctx)
    assert "kowalski" in (tmp_path / "aliases.json").read_text()
    nov = tmp_path / "nov.csv"
    nov.write_text("Date,Description,Amount\n20/11/2026,TRF FROM M KOWALSKI,600.00\n")
    nov_ledger = tmp_path / "nov_ledger.csv"  # INV-9001 was paid in October
    nov_ledger.write_text("Date,Type,Number,Name,Amount\n02/11/2026,Invoice,INV-9002,Northgate Plumbing Ltd,1200.00\n")
    _, _, _, again = run(tmp_path, Settings(use_llm=False), bank=str(nov), ledger=str(nov_ledger))
    r = again["results"]["B1"]
    assert r.status == "matched" and r.method == "Partial payment" and r.party == "Northgate Plumbing Ltd"


def test_llm_maps_unknown_columns(tmp_path):
    weird = tmp_path / "weird_bank.csv"
    weird.write_text("Col A,Col B,Col C,Col D\n" + "\n".join(
        ",".join(r[:4]) for r in list(csv.reader(open(SAMPLES / "bank_statement.csv")))[2:]))
    model = ScriptedModel(columns={"bank": {"date": 0, "description": 1, "debit": 2, "credit": 3}})
    _, _, _, state = run(tmp_path, Settings(), model=model, bank=str(weird))
    assert state["bank_info"]["method"] == "columns identified by LLM"
    assert len(state["bank"]) == 16


def test_missing_file_stops_cleanly(tmp_path):
    _, _, _, state = run(tmp_path, Settings(use_llm=False), bank="nope.csv")
    assert state["errors"] and "not found" in state["errors"][0]
    assert "The run stopped" in Path(state["outputs"]["summary"]).read_text()
