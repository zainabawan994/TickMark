"""Read CSV/Excel exports, find the header row and the columns, and normalise rows."""
from __future__ import annotations

import csv
import io
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..models import BankTxn, LedgerItem
from .values import detect_date_order, parse_amount, parse_date, r2

FIELD_PATTERNS: dict[str, dict[str, list[re.Pattern]]] = {
    "bank": {
        "date": [r"^(transaction |posting |post |value |trans\.? |txn |booking )?date$", r"date"],
        "description": [r"^(description|narrative|narration|details|transaction details|particulars|memo|payee|transaction description|transaction)$",
                        r"desc|narr|detail|partic|memo|payee"],
        "amount": [r"^(amount|value|transaction amount|amt|amount \(.*\))$", r"^amount"],
        "debit": [r"^(debit|debits|withdrawal|withdrawals|paid out|money out|dr|debit amount|withdrawal amt\.?)$", r"debit|withdraw|paid out|money out"],
        "credit": [r"^(credit|credits|deposit|deposits|paid in|money in|cr|credit amount|deposit amt\.?)$", r"credit|deposit|paid in|money in"],
        "drcr": [r"^(dr/cr|cr/dr|d/c|c/d|debit/credit|credit/debit|dr / cr|indicator|sign)$"],
        "reference": [r"^(reference|ref|cheque|check|cheque no\.?|check number|chq|ref no\.?)$", r"^ref|cheque|check no|chq"],
    },
    "ledger": {
        "date": [r"^(date|invoice date|bill date|doc date|transaction date|posting date|txn date)$", r"date"],
        "type": [r"^(type|doc type|transaction type|document type|kind|txn type)$", r"type"],
        "reference": [r"^(reference|ref|invoice|invoice no\.?|invoice number|inv no\.?|doc no\.?|document no\.?|number|bill no\.?|num|no\.?|#)$",
                      r"invoice|^ref|doc|number|^num"],
        "party": [r"^(customer|vendor|supplier|party|name|contact|payee|client|account name|customer name|vendor name|supplier name)$",
                  r"customer|vendor|supplier|party|name|contact|client|payee"],
        "description": [r"^(description|memo|details|narration|narrative|particulars)$", r"desc|memo|detail|narr"],
        "open": [r"^(open|open balance|open amount|amount due|outstanding|balance due|balance|due|remaining)$", r"open|outstanding|due|remaining"],
        "amount": [r"^(amount|total|invoice amount|value|gross|total amount)$", r"amount|total"],
        "debit": [r"^(debit|dr)$"],
        "credit": [r"^(credit|cr)$"],
    },
}
_COMPILED = {k: {f: [re.compile(p, re.I) for p in ps] for f, ps in v.items()} for k, v in FIELD_PATTERNS.items()}
REQUIRED = {"bank": ["date", "description", "amount"], "ledger": ["date", "amount", "reference"]}


@dataclass
class Table:
    headers: list[str]
    rows: list[list[Any]]


@dataclass
class Normalized:
    rows: list
    skipped: list[dict] = field(default_factory=list)
    order: str = "dmy"


def read_rows(path: str | Path) -> list[list[Any]]:
    path = Path(path)
    ext = path.suffix.lower()
    if ext in (".xlsx", ".xlsm"):
        from openpyxl import load_workbook
        wb = load_workbook(path, read_only=True, data_only=True)
        ws = wb.worksheets[0]
        return [list(r) for r in ws.iter_rows(values_only=True)]
    if ext == ".xls":
        raise ValueError("Old .xls files aren't supported. Save the file as .xlsx or .csv.")
    text = path.read_bytes().decode("utf-8-sig", errors="replace")
    head = "\n".join(text.splitlines()[:8])
    delim = max([",", ";", "\t", "|"], key=lambda d: head.count(d))
    return [row for row in csv.reader(io.StringIO(text), delimiter=delim)]


def _clean(c: Any) -> str:
    return "" if c is None else str(c).strip()


def _looks_like_data(row: list[Any]) -> bool:
    """A row holding a real date is a transaction, never a header."""
    return any(c is not None and not isinstance(c, str) and parse_date(c) or
               isinstance(c, str) and re.match(r"\s*(\d{1,4}[/.\-]\d{1,2}[/.\-]\d{1,4})", c) for c in row)


def find_header_row(rows: list[list[Any]], kind: str) -> int:
    pats = [p for ps in _COMPILED[kind].values() for p in ps]
    best, best_score = 0, 0
    for i, row in enumerate(rows[:20]):
        if _looks_like_data(row):
            continue
        score = sum(1 for c in row if isinstance(c, str) and c.strip() and any(p.search(c.strip()) for p in pats))
        if score > best_score:
            best, best_score = i, score
    return best


def to_table(rows: list[list[Any]], kind: str) -> Table:
    rows = [r for r in rows if r and any(_clean(c) for c in r)]
    if not rows:
        return Table([], [])
    h = find_header_row(rows, kind)
    headers = [_clean(c) or f"Column {i + 1}" for i, c in enumerate(rows[h])]
    return Table(headers, rows[h + 1:])


def detect_columns(headers: list[str], kind: str) -> dict[str, int]:
    mapping: dict[str, int] = {}
    used: set[int] = set()
    for pass_ in (0, 1):
        for fld, pats in _COMPILED[kind].items():
            if fld in mapping or len(pats) <= pass_:
                continue
            for i, h in enumerate(headers):
                if i not in used and pats[pass_].search(h.strip()):
                    mapping[fld] = i
                    used.add(i)
                    break
    if kind == "bank" and "amount" in mapping and "debit" in mapping and "credit" in mapping:
        del mapping["amount"]
    return mapping


def missing_fields(mapping: dict[str, int], kind: str) -> list[str]:
    miss = []
    if "date" not in mapping:
        miss.append("date")
    if kind == "bank":
        if "description" not in mapping:
            miss.append("description")
        if not {"amount", "debit", "credit"} & mapping.keys():
            miss.append("amount")
    else:
        if not {"amount", "open", "debit"} & mapping.keys():
            miss.append("amount")
        if not {"reference", "party"} & mapping.keys():
            miss.append("number or name")
    return miss


def _cell(row: list[Any], mapping: dict[str, int], fld: str) -> Any:
    i = mapping.get(fld)
    return row[i] if i is not None and i < len(row) else None


BALANCE_LINE = re.compile(r"^(opening|closing) balance|balance (brought|carried) forward|^total|brought forward|carried forward", re.I)


def normalize_bank(table: Table, mapping: dict[str, int], order_setting: str = "auto") -> Normalized:
    order = detect_date_order(_cell(r, mapping, "date") for r in table.rows) if order_setting == "auto" else order_setting
    out, skipped = [], []
    for n, row in enumerate(table.rows, 1):
        desc = _clean(_cell(row, mapping, "description"))
        ref = _clean(_cell(row, mapping, "reference"))
        if BALANCE_LINE.search(desc):
            continue
        d = parse_date(_cell(row, mapping, "date"), order)
        amount = parse_amount(_cell(row, mapping, "amount")) if "amount" in mapping else None
        if amount is not None and "drcr" in mapping:
            ind = _clean(_cell(row, mapping, "drcr")).upper()
            amount = -abs(amount) if ind.startswith("D") else abs(amount) if ind.startswith("C") else amount
        if amount is None and ({"debit", "credit"} & mapping.keys()):
            dr = parse_amount(_cell(row, mapping, "debit"))
            cr = parse_amount(_cell(row, mapping, "credit"))
            if dr is not None or cr is not None:
                amount = r2(abs(cr or 0) - abs(dr or 0))
        if not d or not amount:
            if desc or amount:
                skipped.append({"row": n, "reason": "date not recognised" if not d else "no amount"})
            continue
        out.append(BankTxn(id=f"B{len(out) + 1}", date=d, description=desc or ref or "(no description)", reference=ref, amount=r2(amount)))
    return Normalized(out, skipped, order)


def infer_ledger_type(type_val: Any, signed: float) -> str:
    t = _clean(type_val).lower()
    if re.search(r"bill payment|payment made|vendor payment|supplier payment|cheque|check|withdrawal|receipt|deposit|"
                 r"payment received|customer payment|journal|transfer|bank|cash|^payment$", t):
        return "GL"
    if re.search(r"bill|purchase|vendor|supplier|payable|\bap\b|expense", t):
        return "AP"
    if re.search(r"invoice|sale|customer|receivable|\bar\b", t):
        return "AR"
    return "AP" if signed < 0 else "AR"


def gl_sign(type_val: Any, signed: float) -> float:
    if signed < 0:
        return signed
    t = _clean(type_val).lower()
    if re.search(r"receipt|deposit|received|customer payment|money in", t):
        return abs(signed)
    if re.search(r"payment|cheque|check|withdrawal|paid|expense|money out", t):
        return -abs(signed)
    return signed


def normalize_ledger(table: Table, mapping: dict[str, int], order_setting: str = "auto") -> Normalized:
    order = detect_date_order(_cell(r, mapping, "date") for r in table.rows) if order_setting == "auto" else order_setting
    out, skipped = [], []
    for n, row in enumerate(table.rows, 1):
        d = parse_date(_cell(row, mapping, "date"), order)
        amt = parse_amount(_cell(row, mapping, "amount")) if "amount" in mapping else None
        if amt is None and ({"debit", "credit"} & mapping.keys()):
            dr, cr = parse_amount(_cell(row, mapping, "debit")), parse_amount(_cell(row, mapping, "credit"))
            if dr is not None or cr is not None:
                amt = r2(abs(dr or 0) - abs(cr or 0))
        open_v = parse_amount(_cell(row, mapping, "open")) if "open" in mapping else None
        if amt is None and open_v is not None:
            amt = open_v
        type_val = _cell(row, mapping, "type")
        ref, party, desc = (_clean(_cell(row, mapping, f)) for f in ("reference", "party", "description"))
        if not d or not amt:
            if ref or party or amt:
                skipped.append({"row": n, "reason": "date not recognised" if not d else "no amount"})
            continue
        typ = infer_ledger_type(type_val, amt)
        item = LedgerItem(id=f"L{len(out) + 1}", date=d, type=typ, type_label=_clean(type_val), ref=ref, party=party,
                          description=desc, amount=r2(abs(amt)), open=r2(abs(amt)))
        if typ == "GL":
            item.signed = r2(gl_sign(type_val, amt))
        else:
            item.open = r2(abs(open_v)) if open_v is not None else item.amount
            if item.open < 0.005:
                continue  # fully paid already
        out.append(item)
    return Normalized(out, skipped, order)
