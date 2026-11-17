"""Parsing of the messy dates and amounts found in bank and ledger exports."""
from __future__ import annotations

import re
from datetime import date, datetime, timedelta
from decimal import ROUND_HALF_UP, Decimal
from typing import Any, Iterable

MONTHS = {m: i for i, m in enumerate(["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"], 1)}


def r2(x: float) -> float:
    """Round money half-up to cents (avoids binary float surprises like 2.675 -> 2.67)."""
    return float(Decimal(str(x)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


def cents(x: float) -> int:
    return int(Decimal(str(x)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP) * 100)


def _mk(y: int, m: int, d: int) -> str | None:
    try:
        return date(y, m, d).isoformat() if 1900 < y < 2200 else None
    except ValueError:
        return None


def detect_date_order(values: Iterable[Any]) -> str:
    """Decide whether ambiguous dates like 03/04/2026 are day-first or month-first."""
    dmy = mdy = 0
    for v in values:
        m = re.match(r"\s*(\d{1,2})[/.\-](\d{1,2})[/.\-](\d{2,4})", str(v or ""))
        if m:
            dmy += int(m.group(1)) > 12
            mdy += int(m.group(2)) > 12
    return "mdy" if mdy > dmy else "dmy"


def parse_date(v: Any, order: str = "dmy", default_year: int | None = None) -> str | None:
    if v is None or v == "":
        return None
    if isinstance(v, datetime):
        return v.date().isoformat()
    if isinstance(v, date):
        return v.isoformat()
    if isinstance(v, (int, float)):
        if 20000 < v < 80000:  # Excel serial date
            return (date(1899, 12, 30) + timedelta(days=int(v))).isoformat()
        return None
    s = str(v).strip()
    year = default_year or date.today().year
    if m := re.match(r"(\d{4})[/.\-](\d{1,2})[/.\-](\d{1,2})", s):
        return _mk(int(m[1]), int(m[2]), int(m[3]))
    if m := re.match(r"(\d{1,2})[/.\-](\d{1,2})[/.\-](\d{2,4})\b", s):
        a, b, y = int(m[1]), int(m[2]), int(m[3])
        y = y + 2000 if y < 100 else y
        if a > 12:
            return _mk(y, b, a)
        if b > 12:
            return _mk(y, a, b)
        return _mk(y, a, b) if order == "mdy" else _mk(y, b, a)
    if m := re.match(r"(\d{1,2})(?:st|nd|rd|th)?[\s\-/.]*([A-Za-z]{3,9})\.?(?:[\s\-/.,]+(\d{2,4}))?\b", s):
        mo = MONTHS.get(m[2][:3].lower())
        if mo:
            y = int(m[3]) if m[3] else year
            return _mk(y + 2000 if y < 100 else y, mo, int(m[1]))
    if m := re.match(r"([A-Za-z]{3,9})\.?[\s\-.]*(\d{1,2})(?:st|nd|rd|th)?(?:,?[\s\-.]+(\d{2,4}))?\b", s):
        mo = MONTHS.get(m[1][:3].lower())
        if mo:
            y = int(m[3]) if m[3] else year
            return _mk(y + 2000 if y < 100 else y, mo, int(m[2]))
    return None


def parse_amount(v: Any) -> float | None:
    """Parse '1,234.56', '(842.17)', '25.00 DR', '1.234,56', '-1,200' and similar."""
    if v is None:
        return None
    if isinstance(v, (int, float)):
        return float(v)
    s = str(v).strip()
    if not s:
        return None
    neg = False
    if s.startswith("(") and s.endswith(")"):
        neg, s = True, s[1:-1]
    if re.search(r"\s*(dr|debit)\.?\s*$", s, re.I):
        neg = True
    s = re.sub(r"\s*(cr|dr|credit|debit)\.?\s*$", "", s, flags=re.I)
    if re.search(r"-\s*$", s):
        neg, s = True, re.sub(r"-\s*$", "", s)
    if re.match(r"^[^\d]*-", s):
        neg = True
    s = re.sub(r"[^\d.,]", "", s)
    if not re.search(r"\d", s):
        return None
    if re.search(r"\.\d{3},\d{1,2}$", s) or re.fullmatch(r"\d{1,3}(\.\d{3})+,\d{1,2}", s):
        s = s.replace(".", "").replace(",", ".")
    elif re.fullmatch(r"\d+,\d{1,2}", s):
        s = s.replace(",", ".")
    else:
        s = s.replace(",", "")
    try:
        n = float(s)
    except ValueError:
        return None
    return -n if neg else n


def days_between(a: str, b: str) -> int:
    return (date.fromisoformat(a) - date.fromisoformat(b)).days
