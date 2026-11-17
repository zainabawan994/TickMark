"""Single-shot LLM calls with structured output (no tools): reading what rules can't."""
from __future__ import annotations

from typing import Any

from langchain_core.messages import HumanMessage

from ..ingest.values import parse_date, r2
from ..models import BankColumns, BankTxn, ExtractedStatement, LedgerColumns

STATEMENT_RULES = """Extract every transaction from this bank statement.
- amount is POSITIVE for money into the account (credits, deposits, receipts) and NEGATIVE for money out
  (debits, withdrawals, payments, fees). Use the running balance, if printed, to decide the sign.
- Join multi-line descriptions into one string and keep reference numbers, cheque numbers and payer/payee names.
- Skip opening and closing balances, totals, headers and page footers, but report the opening and closing
  balances in their own fields when printed.
- Dates as yyyy-mm-dd. If a line has no year, infer it from the statement period."""


def map_columns(model, kind: str, headers: list[str], sample_rows: list[list[Any]]) -> dict[str, int]:
    schema = BankColumns if kind == "bank" else LedgerColumns
    what = "a bank statement export" if kind == "bank" else "an accounting ledger / open invoices and bills export"
    table = "\n".join(f"{i}: {h!r}  e.g. {[str(r[i]) if i < len(r) else '' for r in sample_rows[:4]]}" for i, h in enumerate(headers))
    prompt = (f"These are the columns of {what}, with example values.\n{table}\n\n"
              "Map each field to the index of the column that holds it, or null if there is no such column.")
    result = model.with_structured_output(schema).invoke(prompt)
    return {k: v for k, v in result.model_dump().items() if isinstance(v, int) and 0 <= v < len(headers)}


def _to_txns(st: ExtractedStatement) -> list[BankTxn]:
    out = []
    for t in st.transactions:
        d = parse_date(t.date, "dmy")
        if d and t.amount:
            out.append(BankTxn(id=f"B{len(out) + 1}", date=d, description=t.description.strip() or "(no description)", amount=r2(t.amount)))
    return out


def _merge(parts: list[ExtractedStatement]) -> ExtractedStatement:
    return ExtractedStatement(transactions=[t for p in parts for t in p.transactions],
                              opening_balance=next((p.opening_balance for p in parts if p.opening_balance is not None), None),
                              closing_balance=next((p.closing_balance for p in reversed(parts) if p.closing_balance is not None), None))


def extract_statement_from_text(model, text: str, chunk_chars: int = 24000) -> tuple[list[BankTxn], ExtractedStatement]:
    """Chunked so long statements fit; each chunk keeps whole lines."""
    chunks, cur = [], ""
    for line in text.splitlines(keepends=True):
        if len(cur) + len(line) > chunk_chars and cur:
            chunks.append(cur)
            cur = ""
        cur += line
    if cur:
        chunks.append(cur)
    extractor = model.with_structured_output(ExtractedStatement)
    parts = [extractor.invoke(f"{STATEMENT_RULES}\n\nSTATEMENT TEXT (part {i + 1} of {len(chunks)}):\n{c}") for i, c in enumerate(chunks)]
    merged = _merge(parts)
    return _to_txns(merged), merged


def extract_statement_from_images(model, images_b64: list[str], per_call: int = 5) -> tuple[list[BankTxn], ExtractedStatement]:
    """For scanned statements: send page images to a vision-capable model."""
    extractor = model.with_structured_output(ExtractedStatement)
    parts = []
    for start in range(0, len(images_b64), per_call):
        batch = images_b64[start:start + per_call]
        content: list[dict] = [{"type": "text", "text": f"{STATEMENT_RULES}\n\nThese are statement pages {start + 1}-{start + len(batch)}."}]
        content += [{"type": "image", "base64": b, "mime_type": "image/png"} for b in batch]
        parts.append(extractor.invoke([HumanMessage(content=content)]))
    merged = _merge(parts)
    return _to_txns(merged), merged


def write_narrative(model, facts: str) -> str:
    prompt = ("Write a short reconciliation note (at most 150 words, plain prose, no headings) for the finance manager. "
              "Say what was settled automatically, what needs a person and why, and any risk worth flagging "
              "(duplicates, overpayments, suspense). Use only these facts:\n\n" + facts)
    msg = model.invoke(prompt)
    return msg.text if hasattr(msg, "text") and isinstance(msg.text, str) else str(msg.content)
