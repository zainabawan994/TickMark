"""Read bank statement PDFs.

Strategy: rebuild text lines from word positions, parse rows with a date and
amounts, and work out each amount's direction from the running balance (or
from which column it sits under). The result is *verified* when opening
balance + transactions = closing balance. Unverified or unreadable statements
are handed to the LLM by the ingest node.
"""
from __future__ import annotations

import base64
import io
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from ..models import BankTxn
from .values import detect_date_order, parse_amount, parse_date, r2

MONEY_RE = re.compile(r"^\(?-?[$€£₹¥]?\s?\d{1,3}(?:[,\s]\d{3})*\.\d{2}\)?(?:\s?(?:CR|DR|Cr|Dr|-))?$|^\(?-?[$€£₹¥]?\d+\.\d{2}\)?(?:\s?(?:CR|DR|Cr|Dr|-))?$")
MONEY_IN_TEXT = re.compile(r"(?<![\d.,])\(?-?[$€£₹¥]?\d{1,3}(?:,\d{3})*\.\d{2}\)?(?:\s?(?:CR|DR|Cr|Dr))?(?![\d.])")
DATE_START = re.compile(r"^(\d{4}[/.\-]\d{1,2}[/.\-]\d{1,2}|\d{1,2}[/.\-]\d{1,2}(?:[/.\-]\d{2,4})?|\d{1,2}[\s\-]?[A-Za-z]{3,9}\.?(?:[\s\-,]+\d{2,4})?|[A-Za-z]{3,9}\.?\s\d{1,2}(?:,?\s\d{4})?)\b")
OUT_WORDS = re.compile(r"withdraw|debit|pmt to|payment to|paid to|purchase|\bfee\b|charge|\bchq\b|cheque|\bpos\b|\batm\b|direct debit|standing order|\bdd\b", re.I)
OPENING = re.compile(r"opening balance|balance brought forward|brought forward|\bb/f\b|previous balance", re.I)
CLOSING = re.compile(r"closing balance|balance carried forward|carried forward|\bc/f\b", re.I)


@dataclass
class Cell:
    x: float
    w: float
    text: str


@dataclass
class Line:
    text: str
    cells: list[Cell]


@dataclass
class PdfParse:
    rows: list[BankTxn]
    unsure: int = 0
    opening: float | None = None
    closing: float | None = None
    warnings: list[str] = field(default_factory=list)

    @property
    def verified(self) -> bool | None:
        """True/False when both balances are printed, None when they aren't."""
        if self.opening is None or self.closing is None:
            return None
        return abs(r2(self.opening + sum(t.amount for t in self.rows)) - self.closing) < 0.01


def pdf_lines(path: str | Path) -> list[Line]:
    import pdfplumber
    lines: list[Line] = []
    with pdfplumber.open(str(path)) as pdf:
        for page in pdf.pages:
            words = page.extract_words(x_tolerance=1.5, y_tolerance=2, keep_blank_chars=False)
            rows: list[list[dict]] = []
            for w in sorted(words, key=lambda w: (round(w["top"]), w["x0"])):
                if rows and abs(rows[-1][0]["top"] - w["top"]) < 3:
                    rows[-1].append(w)
                else:
                    rows.append([w])
            for r in rows:
                r.sort(key=lambda w: w["x0"])
                cells: list[Cell] = []
                for w in r:  # merge words separated by ordinary spacing into one cell
                    if cells and w["x0"] - (cells[-1].x + cells[-1].w) < 5:
                        c = cells[-1]
                        c.text += " " + w["text"]
                        c.w = w["x1"] - c.x
                    else:
                        cells.append(Cell(w["x0"], w["x1"] - w["x0"], w["text"]))
                lines.append(Line(" ".join(c.text for c in cells), cells))
    return lines


def page_images(path: str | Path, max_pages: int = 10, resolution: int = 150) -> list[str]:
    """Render pages to base64 PNG (for scanned statements read by a vision model)."""
    import pdfplumber
    out = []
    with pdfplumber.open(str(path)) as pdf:
        for page in pdf.pages[:max_pages]:
            buf = io.BytesIO()
            page.to_image(resolution=resolution).original.save(buf, format="PNG")
            out.append(base64.b64encode(buf.getvalue()).decode())
    return out


def parse_pdf_lines(lines: list[Line]) -> PdfParse:
    col_debit = col_credit = col_bal = None
    for ln in lines:
        low = ln.text.lower()
        if re.search(r"debit|withdrawal|paid out|money out", low) and re.search(r"credit|deposit|paid in|money in", low):
            for c in ln.cells:
                s, mid = c.text.lower(), c.x + c.w / 2
                if re.search(r"debit|withdrawal|paid out|money out", s) and col_debit is None:
                    col_debit = mid
                elif re.search(r"credit|deposit|paid in|money in", s) and col_credit is None:
                    col_credit = mid
                elif "balance" in s:
                    col_bal = mid
            break
    all_text = "\n".join(l.text for l in lines)
    years = Counter(re.findall(r"\b20\d{2}\b", all_text))
    default_year = int(years.most_common(1)[0][0]) if years else None
    order = detect_date_order(l.text for l in lines)

    result = PdfParse(rows=[])
    raw: list[dict] = []
    prev_bal: list[float | None] = [None]
    last: dict | None = None

    def money_of(ln: Line, text: str) -> list[dict]:
        m = [{"x": c.x + c.w / 2, "s": c.text.strip(), "v": parse_amount(c.text)} for c in ln.cells if MONEY_RE.match(c.text.strip())]
        if not m:
            dm0 = DATE_START.match(text)
            rest = text[dm0.end():] if dm0 else text
            m = [{"x": None, "s": s.strip(), "v": parse_amount(s)} for s in MONEY_IN_TEXT.findall(rest)]
        return m

    def assign(row: dict, money: list[dict], text: str) -> None:
        bal, amounts = None, money
        if len(money) >= 2 or (col_bal is not None and len(money) == 1 and money[0]["x"] is not None and abs(money[0]["x"] - col_bal) < 40):
            bal, amounts = money[-1]["v"], money[:-1]
        amt = None
        if bal is not None and prev_bal[0] is not None:
            diff = r2(bal - prev_bal[0])
            if not amounts or any(abs(abs(a["v"]) - abs(diff)) < 0.006 for a in amounts):
                amt = diff
        if amt is None and amounts:
            a = amounts[0]
            if col_debit is not None and col_credit is not None and a["x"] is not None:
                amt = -abs(a["v"]) if abs(a["x"] - col_debit) < abs(a["x"] - col_credit) else abs(a["v"])
            elif a["v"] < 0 or a["s"].lower().endswith("dr"):
                amt = -abs(a["v"])
            elif a["s"].lower().endswith("cr"):
                amt = abs(a["v"])
            else:
                amt = -abs(a["v"]) if OUT_WORDS.search(text) else abs(a["v"])
                result.unsure += 1
        if bal is not None:
            prev_bal[0] = bal
        row["amount"] = None if amt is None else r2(amt)

    for ln in lines:
        text = ln.text.strip()
        if not text:
            continue
        money = money_of(ln, text)
        if OPENING.search(text):
            if money:
                prev_bal[0] = money[-1]["v"]
                result.opening = money[-1]["v"]
            continue
        if CLOSING.search(text):
            if money:
                result.closing = money[-1]["v"]
            continue
        if re.search(r"\btotal\b", text, re.I):
            continue
        dm = DATE_START.match(text)
        d = parse_date(dm.group(1), order, default_year) if dm else None
        if not d:
            if last is not None and last["amount"] is None and money:
                assign(last, money, text)
            elif last is not None and not money and len(text) < 100 and not re.search(r"page \d|statement|account (no|number)", text, re.I):
                last["description"] += " " + text
            continue
        desc = MONEY_IN_TEXT.sub("", text[dm.end():])
        desc = re.sub(r"\s{2,}", " ", desc).strip()
        last = {"date": d, "description": desc, "amount": None}
        raw.append(last)
        if money:
            assign(last, money, text)
    result.rows = [BankTxn(id=f"B{i + 1}", date=r["date"], description=r["description"] or "(no description)", amount=r["amount"])
                   for i, r in enumerate(x for x in raw if x["amount"])]
    return result
