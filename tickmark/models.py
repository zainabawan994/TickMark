"""Data models shared by every part of the pipeline.

Amounts follow one convention everywhere:
* Bank lines are signed: positive = money into the account, negative = money out.
* Ledger items store absolute amounts; ``type`` says which way they move cash
  (AR = customer invoice, paid by money in; AP = supplier bill, paid by money out;
  GL = an entry already booked against the bank account, with ``signed`` amount).
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

LedgerType = Literal["AR", "AP", "GL"]
Status = Literal["matched", "recorded", "review", "unmatched", "bankitem", "suspense"]
Kind = Literal["match", "recorded", "fees", "interest", "suspense"]
Confidence = Literal["high", "medium", "low"]


class BankTxn(BaseModel):
    id: str
    date: str  # ISO yyyy-mm-dd
    description: str
    amount: float
    reference: str = ""


class LedgerItem(BaseModel):
    id: str
    date: str
    type: LedgerType
    type_label: str = ""
    ref: str = ""
    party: str = ""
    description: str = ""
    amount: float
    open: float
    signed: float | None = None
    party_key: str = ""


class Allocation(BaseModel):
    ledger_id: str
    amount: float
    before: float = 0.0
    after: float = 0.0


class MatchResult(BaseModel):
    bank_id: str
    status: Status = "unmatched"
    method: str = ""
    confidence: int = 0
    kind: Kind = "match"
    allocations: list[Allocation] = Field(default_factory=list)
    fee: float = 0.0
    excess: float = 0.0
    party: str | None = None
    gl_id: str | None = None
    reasons: list[str] = Field(default_factory=list)
    why: list[str] = Field(default_factory=list)
    suggestion: str = ""
    candidates: list[str] = Field(default_factory=list)
    source: Literal["engine", "ai", "user"] = "engine"


class Override(BaseModel):
    """A decision that takes priority over the matcher (from a person or from the AI)."""
    kind: Literal["match", "fees", "interest", "suspense"]
    ledger_ids: list[str] = Field(default_factory=list)
    source: Literal["ai", "user"] = "user"
    confidence: Confidence = "high"
    explanation: str = ""


class JournalLine(BaseModel):
    account: str
    debit: float = 0.0
    credit: float = 0.0
    party: str = ""
    ref: str = ""


class JournalEntry(BaseModel):
    id: str
    bank_id: str
    date: str
    memo: str
    lines: list[JournalLine]

    @property
    def total_debit(self) -> float:
        return round(sum(l.debit for l in self.lines), 2)

    @property
    def total_credit(self) -> float:
        return round(sum(l.credit for l in self.lines), 2)

    @property
    def balanced(self) -> bool:
        return abs(self.total_debit - self.total_credit) < 0.005


# ---------------------------------------------------------------------------
# Schemas the LLM fills in (structured output). Docstrings and field
# descriptions are sent to the model, so they are written as instructions.
# ---------------------------------------------------------------------------

class ExceptionResolution(BaseModel):
    """Your final decision for the bank line you investigated."""
    action: Literal["match", "bank_charge", "interest", "suspense"] = Field(
        description="match = pays specific open ledger items; bank_charge / interest = originated by the bank; "
                    "suspense = cannot be identified from the data, or several items fit equally well."
    )
    ledger_ids: list[str] = Field(
        default_factory=list,
        description="Ids of the open ledger items this line pays, exactly as returned by the tools. Only for action=match.",
    )
    confidence: Confidence = Field(
        description="high = a reference, or a unique name and amount; medium = strong name evidence but the allocation "
                    "is a judgement call; low = a guess."
    )
    explanation: str = Field(description="At most 45 words for the accountant: the evidence behind the decision.")


class Resolution(BaseModel):
    """What the graph stores for each exception the agent looked at."""
    bank_id: str
    decision: ExceptionResolution | None = None
    error: str | None = None
    tool_calls: int = 0
    applied: bool = False
    note: str = ""


class BankColumns(BaseModel):
    """Zero-based index of the column holding each field in a bank statement export. Use null when absent."""
    date: int | None = Field(default=None, description="Transaction or posting date")
    description: int | None = Field(default=None, description="Narrative, details, payee or memo")
    amount: int | None = Field(default=None, description="A single signed amount column, if the file has one")
    debit: int | None = Field(default=None, description="Money out / withdrawals column, if separate")
    credit: int | None = Field(default=None, description="Money in / deposits column, if separate")
    drcr: int | None = Field(default=None, description="A column holding only DR/CR or D/C markers")
    reference: int | None = Field(default=None, description="Cheque number or bank reference")


class LedgerColumns(BaseModel):
    """Zero-based index of the column holding each field in a ledger / open-items export. Use null when absent."""
    date: int | None = Field(default=None, description="Invoice, bill or entry date")
    type: int | None = Field(default=None, description="Document type, e.g. Invoice, Bill, Payment, Receipt")
    reference: int | None = Field(default=None, description="Invoice / bill / document number")
    party: int | None = Field(default=None, description="Customer or supplier name")
    description: int | None = Field(default=None, description="Memo or line description")
    amount: int | None = Field(default=None, description="Original document amount")
    open: int | None = Field(default=None, description="Open / outstanding / balance due amount")
    debit: int | None = Field(default=None, description="Debit column, if amounts are split")
    credit: int | None = Field(default=None, description="Credit column, if amounts are split")


class ExtractedTxn(BaseModel):
    date: str = Field(description="ISO date yyyy-mm-dd")
    description: str = Field(description="Full narrative, with reference numbers and payer/payee names")
    amount: float = Field(description="Positive for money into the account, negative for money out")


class ExtractedStatement(BaseModel):
    """Every transaction on the bank statement pages provided."""
    transactions: list[ExtractedTxn]
    opening_balance: float | None = Field(default=None, description="Opening balance, if printed")
    closing_balance: float | None = Field(default=None, description="Closing balance, if printed")


# Types that may be stored in LangGraph checkpoints (explicit allowlist).
CHECKPOINT_TYPES = [
    ("tickmark.models", n)
    for n in ("BankTxn", "LedgerItem", "Allocation", "MatchResult", "Override", "JournalLine", "JournalEntry",
              "ExceptionResolution", "Resolution", "ResolverTask")
]


class ResolverTask(BaseModel):
    """Payload sent (via LangGraph ``Send``) to one exception-resolver agent."""
    bank: BankTxn
    matcher_status: str
    matcher_summary: str
    candidate_ids: list[str] = Field(default_factory=list)
    items: list[LedgerItem] = Field(default_factory=list)  # open items in the right direction, open = available
    fee_tolerance: float = 50.0
