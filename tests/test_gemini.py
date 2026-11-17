"""Gemini integration, end to end, without network access.

The real ``ChatGoogleGenerativeAI`` class is used; only Google's HTTP endpoint is
simulated. This checks what LangChain actually sends to Gemini (tools, JSON schema,
temperature, token limits, thought signatures) and that the whole graph works on it.
"""
import json
import re
import uuid
from pathlib import Path

import httpx
import pytest

from tickmark.cli import main as cli_main
from tickmark.config import RunContext, Settings, default_model
from tickmark.graph.build import build_graph, make_checkpointer

SAMPLES = Path(__file__).parent.parent / "samples"


def _fc(name, args, signature=None):
    part = {"functionCall": {"name": name, "args": args}}
    if signature:
        part["thoughtSignature"] = signature
    return {"candidates": [{"content": {"role": "model", "parts": [part]}, "finishReason": "STOP"}],
            "usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 5, "totalTokenCount": 15}}


def _text(t):
    return {"candidates": [{"content": {"role": "model", "parts": [{"text": t}]}, "finishReason": "STOP"}],
            "usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 5, "totalTokenCount": 15}}


def _decide(action, ids=(), confidence="high", explanation="Scripted."):
    return _fc("ExceptionResolution", {"action": action, "ledger_ids": list(ids), "confidence": confidence, "explanation": explanation})


SCRIPTS = {
    "B8": [_fc("search_open_items", {"amount": 1275}, signature="c2lnLUI4"), _decide("match", ["L9"], "high", "Only invoice for 1,275.00.")],
    "B10": [_fc("party_open_items", {"party": "Atlas Engineering"}), _decide("match", ["L7"], "medium", "Atlas overpaid.")],
    "B11": [_fc("find_combinations", {"target_amount": 6000, "party": "Quantum Retail"}), _decide("match", ["L2", "L10"], "medium", "Oldest first.")],
    "B13": [_fc("search_open_items", {"amount": 900}), _decide("suspense", [], "medium", "Two invoices for 900.00.")],
    "B14": [_fc("similar_names", {"name": "J RIVERA"}), _decide("suspense", [], "low", "Unknown payer.")],
    # used by `tickmark check`
    "B1": [_fc("similar_names", {"name": "ACME TRADNG"}), _decide("match", ["L1"], "high", "Name and amount match.")],
}


class FakeGemini:
    """Answers generateContent requests like Google would, routed by what is being asked."""

    def __init__(self):
        self.requests: list[dict] = []

    def reply(self, body: dict) -> dict:
        self.requests.append(body)
        contents = body.get("contents", [])
        texts = " ".join(p.get("text", "") for c in contents for p in c.get("parts", []))
        if m := re.search(r"Bank line id: (B\d+)", texts):
            turns = sum(1 for c in contents if c.get("role") == "model")
            script = SCRIPTS[m.group(1)]
            return script[min(turns, len(script) - 1)]
        if "columns of a bank statement" in texts:
            return _text(json.dumps({"date": 0, "description": 1, "debit": 2, "credit": 3}))
        return _text("All routine items were settled; three payments need a person.")


@pytest.fixture
def gemini(monkeypatch):
    for var in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "TICKMARK_MODEL", "GEMINI_API_KEY", "TICKMARK_RPM"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
    fake = FakeGemini()

    def make_handler(lib):
        def handle(self, request):
            return lib.Response(200, json=fake.reply(json.loads(request.read() or b"{}")))
        return handle

    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", make_handler(httpx))
    try:
        import httpx2  # newer google-genai SDKs send through httpx2 when it's installed
        monkeypatch.setattr(httpx2.HTTPTransport, "handle_request", make_handler(httpx2))
    except ImportError:
        pass
    return fake


def test_default_model_follows_available_key(monkeypatch):
    for var in ("ANTHROPIC_API_KEY", "GOOGLE_API_KEY", "GEMINI_API_KEY", "OPENAI_API_KEY", "TICKMARK_MODEL"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("GEMINI_API_KEY", "x")
    assert default_model() == "google_genai:gemini-flash-latest"
    monkeypatch.delenv("GEMINI_API_KEY")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "x")
    assert default_model().startswith("anthropic:")
    monkeypatch.setenv("TICKMARK_MODEL", "google_genai:gemini-3.1-flash-lite")
    assert default_model() == "google_genai:gemini-3.1-flash-lite"


def test_missing_gemini_key_is_explained(monkeypatch):
    for var in ("GOOGLE_API_KEY", "GEMINI_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    ctx = RunContext(settings=Settings(model="google_genai:gemini-flash-latest"))
    assert ctx.get_model() is None
    assert "GOOGLE_API_KEY" in ctx.unavailable_reason and "GEMINI_API_KEY" in ctx.unavailable_reason


def test_full_graph_on_gemini(gemini, tmp_path):
    settings = Settings(model="google_genai:gemini-flash-latest", memory_path=str(tmp_path / "aliases.json"))
    ctx = RunContext(settings=settings)  # the real ChatGoogleGenerativeAI is built from settings
    graph = build_graph(checkpointer=make_checkpointer())
    state = graph.invoke({"bank_path": str(SAMPLES / "bank_statement.pdf"), "ledger_path": str(SAMPLES / "ledger.xlsx"),
                          "out_dir": str(tmp_path / "out")},
                         {"configurable": {"thread_id": uuid.uuid4().hex}, "max_concurrency": 4}, context=ctx)
    assert type(ctx.model).__name__ == "ChatGoogleGenerativeAI"
    assert not state.get("errors")
    assert len(state["resolutions"]) == 5 and all(r.decision for r in state["resolutions"])
    assert state["results"]["B8"].source == "ai" and [a.ledger_id for a in state["results"]["B8"].allocations] == ["L9"]
    assert state["results"]["B13"].status == "suspense"
    assert state["narrative"].startswith("All routine items")

    agent_calls = [b for b in gemini.requests if b.get("tools")]
    first = agent_calls[0]
    assert first["generationConfig"]["temperature"] == 1.0           # Google's recommendation for Gemini 3+
    assert first["generationConfig"]["maxOutputTokens"] == 32768     # room for thinking tokens
    assert first["toolConfig"]["functionCallingConfig"]["mode"] == "ANY"
    names = {d["name"] for t in first["tools"] for d in t["functionDeclarations"]}
    assert names == {"search_open_items", "similar_names", "party_open_items", "find_combinations", "ExceptionResolution"}
    # Gemini 3+ rejects multi-turn tool use unless thought signatures are sent back.
    b8_second_turn = next(b for b in agent_calls if "Bank line id: B8" in json.dumps(b) and any(c["role"] == "model" for c in b["contents"]))
    assert any(p.get("thoughtSignature") == "c2lnLUI4" for c in b8_second_turn["contents"] if c["role"] == "model" for p in c["parts"])


def test_check_command_on_gemini(gemini, capsys):
    assert cli_main(["check"]) == 0
    out = capsys.readouterr().out
    assert "google_genai:gemini-flash-latest" in out and "GOOGLE_API_KEY" in out
    assert "✓ Structured output" in out and "✓ Agent with tools: match ['L1']" in out
