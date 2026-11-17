"""A scripted chat model so the LLM paths can be tested without an API key.

Resolver agents run in parallel, so the script is chosen per bank line (read from
the agent's first message) and per step (number of AI turns so far).
"""
from __future__ import annotations

import re
import uuid
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult


def call(tool_name: str, /, **args: Any) -> AIMessage:
    return AIMessage(content="", tool_calls=[{"name": tool_name, "args": args, "id": f"call_{uuid.uuid4().hex[:8]}"}])


def decide(action: str, ledger_ids=(), confidence: str = "high", explanation: str = "Scripted decision.") -> AIMessage:
    return call("ExceptionResolution", action=action, ledger_ids=list(ledger_ids), confidence=confidence, explanation=explanation)


class ScriptedModel(BaseChatModel):
    scripts: dict[str, list[AIMessage]] = {}
    columns: dict[str, dict] = {}
    narrative: str = "Scripted narrative for the finance manager."
    seen_prompts: list[str] = []

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        first = next((m for m in messages if isinstance(m, HumanMessage)), messages[0])
        text = first.content if isinstance(first.content, str) else str(first.content)
        self.seen_prompts.append(text[:200])
        if m := re.search(r"Bank line id: (B\d+)", text):
            script = self.scripts.get(m.group(1), [decide("suspense", confidence="low", explanation="No script.")])
            step = sum(isinstance(x, AIMessage) for x in messages)
            msg = script[min(step, len(script) - 1)]
        elif "columns of a bank statement" in text and "bank" in self.columns:
            msg = call("BankColumns", **self.columns["bank"])
        elif "columns of an accounting ledger" in text and "ledger" in self.columns:
            msg = call("LedgerColumns", **self.columns["ledger"])
        else:
            msg = AIMessage(content=self.narrative)
        return ChatResult(generations=[ChatGeneration(message=msg.model_copy(deep=True))])
