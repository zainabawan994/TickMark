"""The exception-resolver agent: one investigation per bank line the matcher couldn't settle.

Built with LangChain's ``create_agent`` (which runs on LangGraph). The agent loops
over the read-only tools, then must answer with an ``ExceptionResolution``.
"""
from __future__ import annotations

from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.errors import GraphRecursionError

from ..models import ExceptionResolution, Resolution, ResolverTask
from .tools import make_resolver_tools

SYSTEM_PROMPT = """You are a meticulous accounts assistant reconciling a company's bank statement against its ledger.
An automatic matcher has already settled the easy lines. You are given ONE bank line it could not settle
confidently. Investigate it with the tools, then submit exactly one decision.

Rules
- Money in (positive amount) can only pay customer invoices (type AR). Money out (negative) can only pay
  supplier bills (type AP). The tools only show items in the right direction.
- Partial payments, one payment covering several invoices, bank fees deducted in transit, abbreviations,
  misspellings, trading names and owners' personal names are all normal. Check them with the tools.
- Choose "match" only when references, names, amounts and timing point to specific items. Use only ids
  returned by the tools, never invent one.
- Choose "bank_charge" or "interest" only when the wording shows the bank originated the line.
- Choose "suspense" when the data cannot identify the line, or when two or more items fit equally well.
  A wrong match is worse than suspense: someone will have to find and reverse it.
- If the matcher already proposed an allocation, confirm it only if the evidence supports it.
Be efficient: usually two to four tool calls are enough."""


def describe_task(task: ResolverTask) -> str:
    b = task.bank
    direction = "money IN (pays customer invoices)" if b.amount > 0 else "money OUT (pays supplier bills)"
    cands = ", ".join(task.candidate_ids) or "none"
    return (f"Bank line id: {b.id}\nDate: {b.date}\nDescription: {b.description}\n"
            f"Reference: {b.reference or '-'}\nAmount: {b.amount:.2f} ({direction})\n\n"
            f"What the automatic matcher concluded ({task.matcher_status}): {task.matcher_summary}\n"
            f"Items the matcher considered closest: {cands}\n"
            f"Bank-fee tolerance used by this company: {task.fee_tolerance:.2f}\n"
            f"There are {len(task.items)} open items in the right direction. Investigate, then submit your decision.")


def resolve_exception(model, task: ResolverTask, recursion_limit: int = 16) -> Resolution:
    agent = create_agent(model, tools=make_resolver_tools(task), system_prompt=SYSTEM_PROMPT,
                         response_format=ToolStrategy(ExceptionResolution), name="exception_resolver")
    try:
        out = agent.invoke({"messages": [HumanMessage(describe_task(task))]}, {"recursion_limit": recursion_limit})
    except GraphRecursionError:
        return Resolution(bank_id=task.bank.id, error="No decision within the step limit.")
    except Exception as e:  # network, rate limits, provider errors: keep the line for a person
        return Resolution(bank_id=task.bank.id, error=f"{type(e).__name__}: {e}"[:300])
    calls = sum(1 for m in out["messages"] if isinstance(m, AIMessage)
                for c in (m.tool_calls or []) if c["name"] != ExceptionResolution.__name__)
    decision = out.get("structured_response")
    if not isinstance(decision, ExceptionResolution):
        return Resolution(bank_id=task.bank.id, error="The agent finished without a structured decision.", tool_calls=calls)
    return Resolution(bank_id=task.bank.id, decision=decision, tool_calls=calls)
