"""LangGraph state for one reconciliation run."""
from __future__ import annotations

import operator
from typing import Annotated, TypedDict

from ..models import BankTxn, JournalEntry, LedgerItem, MatchResult, Override, Resolution, ResolverTask


class RecInput(TypedDict, total=False):
    bank_path: str
    ledger_path: str
    out_dir: str


class RecState(RecInput, total=False):
    # ingestion
    bank: list[BankTxn]
    ledger: list[LedgerItem]
    bank_info: dict
    ledger_info: dict
    # matching
    results: dict[str, MatchResult]
    outstanding: list[dict]
    open_after: dict[str, float]
    tasks: list[ResolverTask]
    # AI resolution (fan-out results are appended by each parallel agent)
    resolutions: Annotated[list[Resolution], operator.add]
    resolution_notes: dict[str, str]
    overrides: dict[str, Override]
    # journal + approval
    journal: list[JournalEntry]
    approvals: dict[str, dict]
    # output and audit trail
    outputs: dict[str, str]
    narrative: str
    errors: Annotated[list[str], operator.add]
    warnings: Annotated[list[str], operator.add]
    log: Annotated[list[str], operator.add]
