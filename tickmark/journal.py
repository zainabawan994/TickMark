"""Turn match results into journal entries. Deterministic: the LLM never writes entries."""
from __future__ import annotations

from .config import Accounts
from .ingest.values import r2
from .models import BankTxn, JournalEntry, JournalLine, LedgerItem, MatchResult


class UnbalancedEntryError(ValueError):
    pass


def journal_for(b: BankTxn, r: MatchResult, by_id: dict[str, LedgerItem], acc: Accounts, entry_id: str) -> JournalEntry | None:
    if r.status == "unmatched" or r.kind == "recorded":
        return None
    amt = abs(b.amount)
    lines: list[JournalLine] = []

    def add(account: str, debit: float = 0.0, credit: float = 0.0, party: str = "", ref: str = "") -> None:
        lines.append(JournalLine(account=account, debit=r2(debit), credit=r2(credit), party=party, ref=ref))

    money_in = b.amount > 0
    if r.kind == "fees":
        (add(acc.fees, debit=amt), add(acc.bank, credit=amt)) if not money_in else (add(acc.bank, debit=amt), add(acc.fees, credit=amt))
    elif r.kind == "interest":
        (add(acc.bank, debit=amt), add(acc.interest, credit=amt)) if money_in else (add(acc.interest_exp, debit=amt), add(acc.bank, credit=amt))
    elif r.kind == "suspense":
        (add(acc.bank, debit=amt), add(acc.suspense, credit=amt)) if money_in else (add(acc.suspense, debit=amt), add(acc.bank, credit=amt))
    elif money_in:
        add(acc.bank, debit=amt)
        if r.fee:
            add(acc.fees, debit=r.fee)
        for a in r.allocations:
            l = by_id[a.ledger_id]
            add(acc.ar, credit=a.amount, party=l.party, ref=l.ref)
        if r.excess:
            add(acc.credit, credit=r.excess, party=r.party or "")
    else:
        for a in r.allocations:
            l = by_id[a.ledger_id]
            add(acc.ap, debit=a.amount, party=l.party, ref=l.ref)
        if r.fee:
            add(acc.fees, debit=r.fee)
        if r.excess:
            add(acc.advances, debit=r.excess, party=r.party or "")
        add(acc.bank, credit=amt)

    refs = ", ".join(by_id[a.ledger_id].ref for a in r.allocations if by_id[a.ledger_id].ref)
    memo = f"{r.method}: {r.party or ''}{' ' + refs if refs else ''}".strip() if r.kind == "match" else f"{r.method}: {b.description}"
    je = JournalEntry(id=entry_id, bank_id=b.id, date=b.date, memo=memo, lines=lines)
    if not je.balanced:
        raise UnbalancedEntryError(f"{entry_id} for {b.id} does not balance: Dr {je.total_debit} / Cr {je.total_credit}")
    return je


def build_journal(bank: list[BankTxn], results: dict[str, MatchResult], ledger: list[LedgerItem], acc: Accounts) -> list[JournalEntry]:
    by_id = {l.id: l for l in ledger}
    out: list[JournalEntry] = []
    for b in sorted(bank, key=lambda x: x.date):
        je = journal_for(b, results[b.id], by_id, acc, f"JE-{len(out) + 1:03d}")
        if je:
            out.append(je)
    return out
