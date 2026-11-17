"""Wire the reconciliation graph.

    START ─┬─ ingest_bank ───┐
           └─ ingest_ledger ─┴─> validate ─┬─> match ─┬─> resolve_exception (×N, parallel via Send)
                                           │          │          │
                                           │          │          v
                                           │          │   apply_resolutions (guardrails)
                                           │          │          │
                                           │          └─────> build_journal <┘
                                           │                     │
                                           │               approval_gate  (policy, or interrupt() for a person)
                                           │                     │
                                           └──────────────> report ─> END
"""
from __future__ import annotations

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.graph import END, START, StateGraph

from ..config import RunContext
from ..models import CHECKPOINT_TYPES
from . import nodes
from .state import RecInput, RecState


def build_graph(checkpointer=None):
    g = StateGraph(RecState, context_schema=RunContext, input_schema=RecInput)
    g.add_node("ingest_bank", nodes.ingest_bank)
    g.add_node("ingest_ledger", nodes.ingest_ledger)
    g.add_node("validate", nodes.validate)
    g.add_node("match", nodes.match)
    g.add_node("resolve_exception", nodes.resolve_exception)
    g.add_node("apply_resolutions", nodes.apply_resolutions)
    g.add_node("build_journal", nodes.build_journal)
    g.add_node("approval_gate", nodes.approval_gate)
    g.add_node("report", nodes.report)

    g.add_edge(START, "ingest_bank")              # the two files are read in parallel
    g.add_edge(START, "ingest_ledger")
    g.add_edge(["ingest_bank", "ingest_ledger"], "validate")  # join: waits for both
    g.add_conditional_edges("validate", nodes.route_after_validate, ["match", "report"])
    g.add_conditional_edges("match", nodes.route_after_match, ["resolve_exception", "build_journal"])
    g.add_edge("resolve_exception", "apply_resolutions")
    g.add_edge("apply_resolutions", "build_journal")
    g.add_edge("build_journal", "approval_gate")
    g.add_edge("approval_gate", "report")
    g.add_edge("report", END)
    return g.compile(checkpointer=checkpointer, name="tickmark")


def make_checkpointer() -> InMemorySaver:
    """In-memory checkpoints, needed to pause for a person and resume.

    Only Tickmark's own models may be deserialised (explicit allowlist). For runs that
    must survive a restart, swap in langgraph-checkpoint-sqlite or -postgres with the same serde.
    """
    return InMemorySaver(serde=JsonPlusSerializer(allowed_msgpack_modules=CHECKPOINT_TYPES))


# Exposed for `langgraph dev` / LangGraph Studio (see langgraph.json); the platform adds its own checkpointer.
graph = build_graph()
