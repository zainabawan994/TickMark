"""Tools for the exception-resolver agent.

They are deliberately read-only: the agent can search and do arithmetic over the
open ledger items it was given, but it cannot change anything. Its only output
is a structured decision, which the graph validates before using.
"""
from __future__ import annotations

import re

from langchain_core.tools import BaseTool, tool

from ..ingest.values import cents
from ..matching.engine import subset_sums
from ..matching.names import build_party_index, name_score, tokens_of
from ..models import LedgerItem, ResolverTask


def _view(l: LedgerItem) -> dict:
    return {"id": l.id, "number": l.ref, "name": l.party, "date": l.date, "open_amount": l.open, "type": l.type, "memo": l.description}


def make_resolver_tools(task: ResolverTask) -> list[BaseTool]:
    items = [l.model_copy() for l in task.items]
    parties = build_party_index(items)

    def party_similarity(text: str) -> list[tuple[str, float]]:
        qt = tokens_of(text, True) or tokens_of(text)
        ranked = [(k, name_score(qt, "".join(qt), p)) for k, p in parties.items()]
        return sorted(ranked, key=lambda x: -x[1])

    @tool
    def search_open_items(text: str = "", amount: float | None = None, tolerance: float = 0.01, limit: int = 12) -> list[dict]:
        """Search the open ledger items that this bank line could pay. Filter by `text` (an invoice/bill number,
        a customer or supplier name, or memo words) and/or by `amount` (matches open amounts within `tolerance`).
        Returns id, number, name, date, open_amount, type and memo for each hit."""
        q = re.sub(r"[^a-z0-9]", "", text.lower())
        qt = tokens_of(text)
        hits = []
        for l in items:
            if amount is not None and abs(l.open - abs(amount)) > max(tolerance, 0.004):
                continue
            if text.strip():
                hay = re.sub(r"[^a-z0-9]", "", f"{l.ref} {l.party} {l.description}".lower())
                s = 1.0 if q and q in hay else (name_score(qt, "".join(qt), parties[l.party_key]) if l.party_key in parties else 0.0)
                if s < 0.5:
                    continue
            hits.append(_view(l))
        return hits[: max(1, min(limit, 40))] or [{"result": "no open items match"}]

    @tool
    def similar_names(name: str, limit: int = 5) -> list[dict]:
        """Rank the customers/suppliers that have open items by name similarity (0-1) to `name`, which can be a raw
        bank description. Also returns how many open items each has and their total."""
        out = []
        for k, s in party_similarity(name)[: max(1, min(limit, 15))]:
            mine = [l for l in items if l.party_key == k]
            out.append({"name": parties[k].name, "similarity": round(s, 2), "open_items": len(mine), "open_total": round(sum(l.open for l in mine), 2)})
        return out

    @tool
    def party_open_items(party: str) -> dict:
        """List every open item, oldest first, for the customer/supplier whose name best matches `party`."""
        ranked = party_similarity(party)
        if not ranked or ranked[0][1] < 0.5:
            return {"result": f"no customer or supplier resembles '{party}'"}
        k, s = ranked[0]
        mine = sorted((l for l in items if l.party_key == k), key=lambda l: l.date)
        return {"party": parties[k].name, "similarity": round(s, 2), "total_open": round(sum(l.open for l in mine), 2),
                "items": [_view(l) for l in mine]}

    @tool
    def find_combinations(target_amount: float, party: str = "", max_items: int = 6, fee_shortfall_allowed: float = 0.0) -> list[dict]:
        """Find sets of 2+ open items whose open amounts add up to `target_amount` (a payment covering several
        invoices). Restrict to one customer/supplier with `party` (recommended). `fee_shortfall_allowed` lets the
        items total exceed the payment by up to that amount (a bank fee deducted in transit)."""
        pool = items
        if party.strip():
            ranked = party_similarity(party)
            if not ranked or ranked[0][1] < 0.5:
                return [{"result": f"no customer or supplier resembles '{party}'"}]
            pool = [l for l in items if l.party_key == ranked[0][0]]
        elif len(pool) > 30:
            return [{"result": "too many open items to search without a party; pass a party name"}]
        pool = sorted(pool, key=lambda l: l.date)[:24]
        target = cents(abs(target_amount))
        sols = subset_sums([cents(l.open) for l in pool], target, target + cents(max(0.0, fee_shortfall_allowed)),
                           max_size=max(2, min(max_items, 8)), limit=5)
        if not sols:
            return [{"result": "no combination adds up to that amount"}]
        return [{"ids": [pool[i].id for i in s], "numbers": [pool[i].ref for i in s],
                 "total": round(sum(pool[i].open for i in s), 2)} for s in sols]

    return [search_open_items, similar_names, party_open_items, find_combinations]
