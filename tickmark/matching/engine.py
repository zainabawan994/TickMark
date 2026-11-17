"""Deterministic reconciliation engine.

Passes run from most to least certain, and each pass consumes the open balances
it allocates, so later passes can't double-count an invoice:

0. overrides (decisions by a person or the AI resolver)
1. references quoted on the bank line (invoice, bill or cheque numbers)
2. entries already booked in the ledger (cash book)
3. exact amount + recognisable name
4. combined payments (subset sum over one party's open items)
5. single invoice paid net of a bank fee
6. partial payments, oldest-first allocation and overpayments
7. unique exact amount with no name evidence (always sent for review)
8. bank-originated items (fees, interest)
Anything left gets a written explanation of why nothing fit.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from ..ingest.values import cents, days_between, r2
from ..models import Allocation, BankTxn, LedgerItem, MatchResult, Override
from .names import NOISE, build_party_index, name_score, tok_sim, tokens_of

FEE_RE = re.compile(r"\b(fee|fees|charge|charges|chg|chgs|commission|levy|excise|service charge|maintenance|overdraft|wht|"
                    r"withholding|stamp duty|sms alert|annual card|ledger fee)\b", re.I)
INT_RE = re.compile(r"\b(interest|int\.? ?(cr|credit|paid)|profit paid|profit credit|savings profit)\b", re.I)


@dataclass
class MatchConfig:
    name_threshold: float = 0.72
    fee_tolerance: float = 50.0
    date_window: int = 90
    ai_auto_approve: bool = False


@dataclass
class ReconcileOutput:
    results: dict[str, MatchResult]
    outstanding: list[dict]
    open_after: dict[str, float]


def fmt(n: float) -> str:
    return f"{n:,.2f}"


def pct(s: float) -> str:
    return f"{round(s * 100)}%"


def ref_info(ref: str) -> tuple[str, str]:
    n = re.sub(r"[^A-Z0-9]", "", str(ref or "").upper())
    digits = "".join(re.findall(r"\d+", str(ref or ""))).lstrip("0")
    return n, digits


def bank_ref_tokens(text: str) -> tuple[set[str], set[str]]:
    toks = [t for t in re.split(r"[^A-Z0-9]+", str(text or "").upper()) if t]
    tokens = set(toks) | {toks[i] + toks[i + 1] for i in range(len(toks) - 1)}
    digits = {m.group(1).lstrip("0") for t in toks if (m := re.fullmatch(r"[A-Z]{0,4}(\d{3,})", t))}
    return tokens, digits


def subset_sums(values: list[int], lo: int, hi: int, max_size: int = 8, limit: int = 4) -> list[list[int]]:
    """Index sets (size >= 2) whose sum lies in [lo, hi]. Values are cents, in date order."""
    out: list[list[int]] = []
    n = len(values)
    suffix = [0] * (n + 1)
    for i in range(n - 1, -1, -1):
        suffix[i] = suffix[i + 1] + values[i]
    pick: list[int] = []
    budget = [300_000]

    def dfs(i: int, total: int) -> None:
        budget[0] -= 1
        if len(out) >= limit or budget[0] <= 0:
            return
        if len(pick) >= 2 and lo <= total <= hi:
            out.append(pick.copy())
            return
        if len(pick) == max_size or i == n or total + suffix[i] < lo:
            return
        for j in range(i, n):
            if total + values[j] > hi:
                continue
            pick.append(j)
            dfs(j + 1, total + values[j])
            pick.pop()
            if len(out) >= limit:
                return

    dfs(0, 0)
    return out


@dataclass
class _Pre:
    dt: list[str]
    d_compact: str
    desc_key: str
    name_text: str
    ref_tokens: set[str]
    ref_digits: set[str]
    dir: str
    abs: float
    scores: dict[str, float] = field(default_factory=dict)


def description_key(description: str) -> str:
    return " ".join(tokens_of(description, True))


def reconcile(bank: list[BankTxn], ledger: list[LedgerItem], cfg: MatchConfig,
              overrides: dict[str, Override] | None = None, rejected: set[str] | None = None,
              aliases: dict[str, str] | None = None, no_bank_item: set[str] | None = None) -> ReconcileOutput:
    overrides, rejected = overrides or {}, rejected or set()
    aliases, no_bank_item = aliases or {}, no_bank_item or set()
    th, fee_tol, win = cfg.name_threshold, cfg.fee_tolerance, cfg.date_window
    parties = build_party_index(ledger)
    by_id = {l.id: l for l in ledger}
    open_ = {l.id: l.open for l in ledger}
    used_gl: set[str] = set()
    results: dict[str, MatchResult] = {}
    ordered = sorted(bank, key=lambda b: b.date)

    pre: dict[str, _Pre] = {}
    for b in bank:
        dt = tokens_of(b.description, True)
        tokens, digits = bank_ref_tokens(f"{b.description} {b.reference}")
        name_text = " ".join(w for w in re.sub(r"[^A-Za-z&' ]+", " ", b.description).split() if w.lower() not in NOISE)
        pre[b.id] = _Pre(dt, "".join(dt), " ".join(dt), name_text, tokens, digits, "AR" if b.amount > 0 else "AP", abs(b.amount))

    def score(b: BankTxn, key: str) -> float:
        p = pre[b.id]
        if key not in p.scores:
            party = parties.get(key)
            s = name_score(p.dt, p.d_compact, party) if party else 0.0
            if party and p.desc_key and aliases.get(p.desc_key, "").lower() == key:
                s = 1.0
            p.scores[key] = s
        return p.scores[key]

    def is_rej(b, l) -> bool:
        return f"{b.id}|{l.id}" in rejected

    def in_window(b, l) -> bool:
        return -7 <= days_between(b.date, l.date) <= win

    def open_items(b, flt=None) -> list[LedgerItem]:
        d = pre[b.id].dir
        return [l for l in ledger if l.type == d and open_[l.id] > 0.005 and not is_rej(b, l) and in_window(b, l) and (flt is None or flt(l))]

    def best_party(b):
        best = second = None
        for k in {l.party_key for l in open_items(b)}:
            s = score(b, k)
            if best is None or s > best[1]:
                best, second = (k, s), best
            elif second is None or s > second[1]:
                second = (k, s)
        return best, second

    def ambiguous(best, second) -> bool:
        return bool(second and second[1] >= th and best[1] - second[1] < 0.05)

    name_of = lambda k: parties[k].name if k in parties else k
    quote = lambda b: f"“{b.description}”"
    pending = lambda: [b for b in ordered if b.id not in results]

    def res(b: BankTxn, status: str, allocations: list[tuple[str, float]] | None = None, **kw) -> MatchResult:
        allocs = []
        for lid, amt in allocations or []:
            before = open_[lid]
            after = r2(before - amt)
            open_[lid] = after
            allocs.append(Allocation(ledger_id=lid, amount=r2(amt), before=before, after=after))
        if status == "auto":
            status = "matched" if kw.get("confidence", 0) >= 80 else "review"
        r = MatchResult(bank_id=b.id, status=status, allocations=allocs, **kw)
        results[b.id] = r
        return r

    def fifo(items: list[LedgerItem], amount: float) -> tuple[list[tuple[str, float]], float]:
        allocs, rem = [], cents(amount)
        for it in sorted(items, key=lambda l: l.date):
            if rem <= 0:
                break
            take = min(cents(open_[it.id]), rem)
            if take > 0:
                allocs.append((it.id, take / 100))
                rem -= take
        return allocs, rem / 100

    def remaining_notes(r: MatchResult) -> None:
        for a in r.allocations:
            if a.after > 0.005:
                r.reasons.append(f"{by_id[a.ledger_id].ref} stays open with {fmt(a.after)} remaining.")

    # ---- Pass 0: overrides -------------------------------------------------
    for b in ordered:
        ov = overrides.get(b.id)
        if not ov:
            continue
        is_ai = ov.source == "ai"
        if ov.kind == "match":
            chosen = [by_id[i] for i in ov.ledger_ids if i in by_id]
            gl = next((l for l in chosen if l.type == "GL" and l.id not in used_gl), None)
            if gl:
                used_gl.add(gl.id)
                res(b, "recorded", kind="recorded", gl_id=gl.id, party=gl.party, confidence=100 if not is_ai else 80,
                    method="Already in ledger", source=ov.source,
                    reasons=[f"Matched to {gl.ref or gl.description}, which is already booked."] + ([f"Claude: {ov.explanation}"] if is_ai else []))
                continue
            items = [l for l in chosen if l.type == pre[b.id].dir and open_[l.id] > 0.005]
            if not items:
                continue
            total = r2(sum(open_[l.id] for l in items))
            allocs, rem = fifo(items, pre[b.id].abs)
            refs = ", ".join(l.ref or l.id for l in items)
            reasons = [f"Claude: {ov.explanation}"] if is_ai else [f"Matched by you to {refs}."]
            if rem > 0.005:
                reasons.append(f"The extra {fmt(rem)} is held as {'an unapplied customer credit' if b.amount > 0 else 'a supplier advance'}.")
            if is_ai:
                conf = {"high": 85, "medium": 65, "low": 45}[ov.confidence]
                status = "matched" if (cfg.ai_auto_approve and ov.confidence == "high") else "review"
            else:
                conf, status = 100, "matched"
            r = res(b, status, allocs, method="AI match" if is_ai else "Matched by you", confidence=conf,
                    excess=r2(rem) if rem > 0.005 else 0.0, party=items[0].party, reasons=reasons, source=ov.source)
            if cents(total) > cents(pre[b.id].abs):
                remaining_notes(r)
        else:
            label = {"fees": "Bank charge", "interest": "Interest", "suspense": "Suspense"}[ov.kind]
            conf = {"high": 85, "medium": 65, "low": 45}[ov.confidence] if is_ai else 100
            reasons = [f"Claude: {ov.explanation}"] if is_ai else (
                ["Parked in suspense until it can be identified."] if ov.kind == "suspense" else [f"Booked by you as {label.lower()}."])
            res(b, "suspense" if ov.kind == "suspense" else "bankitem", kind=ov.kind, method=label, confidence=conf,
                reasons=reasons, source=ov.source)

    # ---- Pass 1: references ------------------------------------------------
    for b in pending():
        p = pre[b.id]
        hits = []
        for l in ledger:
            if not l.ref or is_rej(b, l):
                continue
            n, digits = ref_info(l.ref)
            strength = 0.0
            if len(n) >= 3 and n in p.ref_tokens:
                strength = 1.0
            elif len(digits) >= 4 and digits in p.ref_digits:
                strength = 0.9
            elif len(digits) == 3 and digits in p.ref_digits:
                strength = 0.6
            if strength:
                hits.append((l, strength))
        if not hits:
            continue
        gl = next((l for l, s in hits if l.type == "GL" and l.id not in used_gl and cents(l.signed or 0) == cents(b.amount)), None)
        if gl:
            used_gl.add(gl.id)
            res(b, "recorded", kind="recorded", gl_id=gl.id, party=gl.party, confidence=98, method="Already in ledger",
                reasons=[f"The bank line quotes reference {gl.ref}, which is already booked in the ledger "
                         f"({gl.description or gl.type_label}) for the same amount.", "No new journal entry is needed."])
            continue
        items = [l for l, s in hits if l.type == p.dir and open_[l.id] > 0.005 and (s >= 0.7 or cents(open_[l.id]) == cents(p.abs))]
        if not items:
            continue
        items.sort(key=lambda l: l.date)
        total = r2(sum(open_[l.id] for l in items))
        refs = ", ".join(l.ref for l in items)
        pty = items[0].party
        ns = score(b, items[0].party_key)
        doc = "invoice" if p.dir == "AR" else "bill"
        base = [f"The bank line quotes {refs}, {'open ' + doc + 's' if len(items) > 1 else 'an open ' + doc} for {pty}."]
        if ns >= th:
            base.append(f"The name on the bank line also matches {pty} ({pct(ns)} similar).")
        full = [(l.id, open_[l.id]) for l in items]
        diff = r2(total - p.abs)
        if abs(diff) < 0.005:
            res(b, "auto", full, method="Combined payment" if len(items) > 1 else "Reference match",
                confidence=97 if len(items) > 1 else 99, party=pty,
                reasons=base + [f"{fmt(p.abs)} equals the {'combined ' if len(items) > 1 else ''}open balance exactly."])
        elif diff > 0:
            if diff <= fee_tol and diff <= 0.05 * total and p.dir == "AR":
                res(b, "auto", full, method="Paid net of bank fee", confidence=93, party=pty, fee=diff,
                    reasons=base + [f"Received {fmt(p.abs)} against {fmt(total)} due. The {fmt(diff)} shortfall is within the "
                                    f"{fmt(fee_tol)} bank-fee tolerance, so it is booked to bank charges and the invoice is closed."])
            else:
                allocs, _ = fifo(items, p.abs)
                r = res(b, "auto", allocs, method="Partial payment", confidence=85 if len(items) > 1 else 90, party=pty,
                        reasons=base + [f"{fmt(p.abs)} is less than the {fmt(total)} due, so it is treated as a partial payment."])
                remaining_notes(r)
        else:
            excess = r2(-diff)
            res(b, "auto", full, method="Overpayment", confidence=75, party=pty, excess=excess,
                reasons=base + [f"{pty} paid {fmt(p.abs)} but only {fmt(total)} is due on {refs}. The extra {fmt(excess)} is held as "
                                f"{'an unapplied customer credit' if p.dir == 'AR' else 'a supplier advance'} until someone decides what it is for."])

    # ---- Pass 2: already booked in the ledger --------------------------------
    for b in pending():
        p = pre[b.id]
        cands = [l for l in ledger if l.type == "GL" and l.id not in used_gl and not is_rej(b, l)
                 and cents(l.signed or 0) == cents(b.amount) and abs(days_between(b.date, l.date)) <= 7]
        if not cands:
            continue

        def sim(l: LedgerItem) -> float:
            lt = tokens_of(f"{l.description} {l.party}")
            if not lt or not p.dt:
                return 0.0
            return sum(1 for t in lt if any(tok_sim(t, d) >= 0.88 for d in p.dt)) / len(lt)

        scored = sorted(((l, sim(l), abs(days_between(b.date, l.date))) for l in cands), key=lambda x: (-x[1], x[2]))
        top, s_top, d_top = scored[0]
        used_gl.add(top.id)
        if len(scored) > 1 and s_top - scored[1][1] < 0.2:
            res(b, "review", kind="recorded", gl_id=top.id, party=top.party, confidence=60, method="Already in ledger?",
                reasons=[f"{len(scored)} booked ledger entries have exactly this amount within a week. The closest is "
                         f"{top.ref or top.description} on {top.date}.", "Confirm it is the right one."])
        else:
            ok = s_top >= 0.3
            res(b, "recorded" if ok else "review", kind="recorded", gl_id=top.id, party=top.party, confidence=90 if ok else 72,
                method="Already in ledger",
                reasons=[f"Booked in the ledger as {top.ref or top.description} on {top.date} for the same amount"
                         f"{f' ({d_top} day(s) apart)' if d_top else ''}.",
                         "The descriptions agree, so no new entry is needed." if ok else "The descriptions share no words, so please confirm."])

    # ---- Pass 3: exact amount + name ------------------------------------------
    for b in pending():
        p = pre[b.id]
        cands = [(l, score(b, l.party_key)) for l in open_items(b, lambda l: cents(open_[l.id]) == cents(p.abs))]
        cands = sorted([c for c in cands if c[1] >= th], key=lambda c: (-c[1], c[0].date))
        if not cands:
            continue
        top, s = cands[0]
        if any(l.party_key != top.party_key and s - s2 < 0.05 for l, s2 in cands[1:]):
            continue  # two parties fit equally well: explained later
        same = [c for c in cands if c[0].party_key == top.party_key]
        reasons = [f"{quote(b)} matches {top.party} ({pct(s)} similar{', allowing for spelling and abbreviations' if s < 0.999 else ''}).",
                   f"{fmt(p.abs)} equals the open balance on {top.ref} exactly."]
        if len(same) > 1:
            reasons.append(f"{top.party} has {len(same)} open items for this amount; applied to the oldest ({top.ref}).")
        res(b, "auto", [(top.id, open_[top.id])], method="Amount and name",
            confidence=min(96, round(78 + 18 * s)) - (6 if len(same) > 1 else 0), party=top.party, reasons=reasons)

    # ---- Pass 4: combined payments ---------------------------------------------
    for b in pending():
        p = pre[b.id]
        best, second = best_party(b)
        if not best or best[1] < th or ambiguous(best, second):
            continue
        items = sorted(open_items(b, lambda l: l.party_key == best[0]), key=lambda l: l.date)[:24]
        if len(items) < 2:
            continue
        values = [cents(open_[l.id]) for l in items]
        target = cents(p.abs)
        sols, fee = subset_sums(values, target, target), 0.0
        if not sols and p.dir == "AR":
            sols = subset_sums(values, target + 1, target + cents(fee_tol))
            if sols:
                fee = r2(sum(values[i] for i in sols[0]) / 100 - p.abs)
        if not sols:
            continue
        pick = [items[i] for i in sols[0]]
        pty = name_of(best[0])
        reasons = [f"{quote(b)} matches {pty} ({pct(best[1])} similar).",
                   f"{pty}'s open {'invoices' if p.dir == 'AR' else 'bills'} {' + '.join(l.ref for l in pick)} add up to "
                   f"{fmt(p.abs + fee) if fee else 'exactly ' + fmt(p.abs)}."]
        if fee:
            reasons.append(f"The {fmt(fee)} difference is within the bank-fee tolerance and is booked to bank charges.")
        if len(sols) > 1:
            reasons.append(f"Another combination of {pty}'s items also adds up to this amount; check which ones were meant.")
        res(b, "auto", [(l.id, open_[l.id]) for l in pick], method="Combined payment",
            confidence=70 if len(sols) > 1 else (85 if fee else 92), party=pty, fee=fee, reasons=reasons)

    # ---- Pass 5: paid net of a bank fee ------------------------------------------
    for b in pending():
        p = pre[b.id]

        def fee_gap(l):
            return (open_[l.id] - p.abs) if p.dir == "AR" else (p.abs - open_[l.id])

        cands = [(l, score(b, l.party_key)) for l in open_items(b, lambda l: 0.004 < fee_gap(l) <= fee_tol and fee_gap(l) <= 0.05 * open_[l.id])]
        cands = sorted([c for c in cands if c[1] >= th], key=lambda c: (-c[1], abs(open_[c[0].id] - p.abs)))
        if not cands:
            continue
        top, s = cands[0]
        fee = r2(abs(open_[top.id] - p.abs))
        due = open_[top.id]
        res(b, "auto", [(top.id, due)], method="Paid net of bank fee", confidence=86, party=top.party, fee=fee,
            reasons=[f"{quote(b)} matches {top.party} ({pct(s)} similar).",
                     (f"Received {fmt(p.abs)} against {fmt(due)} due on {top.ref}. The {fmt(fee)} shortfall looks like a bank or "
                      f"transfer fee deducted in transit (within the {fmt(fee_tol)} tolerance).") if p.dir == "AR" else
                     (f"Paid {fmt(p.abs)} against {fmt(due)} due on {top.ref}. The extra {fmt(fee)} looks like a transfer fee."),
                     f"{top.ref} is closed in full and the {fmt(fee)} goes to bank charges."])

    # ---- Pass 6: partial / oldest-first / overpayment ------------------------------
    for b in pending():
        p = pre[b.id]
        best, second = best_party(b)
        if not best or best[1] < th or ambiguous(best, second):
            continue
        items = sorted(open_items(b, lambda l: l.party_key == best[0]), key=lambda l: l.date)
        total = r2(sum(open_[l.id] for l in items))
        pty = name_of(best[0])
        head = (f"{quote(b)} matches {pty} ({pct(best[1])} similar), but no reference is quoted and the amount doesn't "
                f"equal any single open item or combination.")
        if p.abs > total + 0.004:
            excess = r2(p.abs - total)
            res(b, "auto", [(l.id, open_[l.id]) for l in items], method="Overpayment", confidence=68, party=pty, excess=excess,
                reasons=[head, f"{pty} owes {fmt(total)} in total ({', '.join(l.ref for l in items)}) but paid {fmt(p.abs)}.",
                         f"All open items are cleared and the extra {fmt(excess)} is held as "
                         f"{'an unapplied customer credit' if p.dir == 'AR' else 'a supplier advance'}. It may be a duplicate "
                         f"payment or meant for an invoice not yet raised."])
        elif len(items) == 1:
            r = res(b, "auto", [(items[0].id, p.abs)], method="Partial payment", confidence=80, party=pty,
                    reasons=[head, f"{pty} has one open item, {items[0].ref} for {fmt(total)}, so this is a partial payment against it."])
            remaining_notes(r)
        else:
            allocs, _ = fifo(items, p.abs)
            r = res(b, "auto", allocs, method="Partial payment (oldest first)" if len(allocs) > 1 else "Partial payment",
                    confidence=64, party=pty,
                    reasons=[head, f"{pty} has {len(items)} open items totalling {fmt(total)}. The payment is applied oldest first."])
            for a in r.allocations:
                r.reasons.append(f"{by_id[a.ledger_id].ref}: " + (f"{fmt(a.amount)} applied, {fmt(a.after)} still open." if a.after > 0.005 else "paid in full."))
            r.reasons.append("Check the remittance advice to confirm which invoices were intended.")

    # ---- Pass 7: unique exact amount, no name ---------------------------------------
    for b in pending():
        p = pre[b.id]
        if FEE_RE.search(b.description) or INT_RE.search(b.description):
            continue
        cands = open_items(b, lambda l: cents(open_[l.id]) == cents(p.abs))
        if len(cands) != 1:
            continue
        l = cands[0]
        res(b, "review", [(l.id, open_[l.id])], method="Amount only", confidence=55, party=l.party,
            reasons=[f"{quote(b)} has no reference or recognisable name.",
                     f"{l.ref} ({l.party}, {l.date}) is the only open {'invoice' if p.dir == 'AR' else 'bill'} for exactly "
                     f"{fmt(p.abs)}, so it is the likely match.", "Confirm before approving: amount alone isn't proof."])

    # ---- Pass 8: bank-originated items --------------------------------------------
    for b in pending():
        if b.id in no_bank_item:
            continue
        if b.amount < 0 and FEE_RE.search(b.description):
            res(b, "bankitem", kind="fees", method="Bank charge", confidence=88,
                reasons=[f"{quote(b)} reads as a bank fee or charge.", "Bank charges usually reach the ledger only from the statement, so a new entry is proposed."])
        elif INT_RE.search(b.description):
            res(b, "bankitem", kind="interest", method="Interest received" if b.amount > 0 else "Interest charged", confidence=88,
                reasons=[f"{quote(b)} reads as interest {'paid' if b.amount > 0 else 'charged'} by the bank.", "A new entry is proposed."])

    # ---- Remaining: explain why nothing fits ----------------------------------------
    for b in pending():
        p = pre[b.id]
        why: list[str] = []
        doc = "customer invoice" if p.dir == "AR" else "supplier bill"
        who = "customer" if p.dir == "AR" else "supplier"
        if not p.dt and not p.ref_digits:
            why.append(f"{quote(b)} has no name or reference to go on.")
        best, second = best_party(b)
        if best and ambiguous(best, second) and best[1] >= th:
            why.append(f"The name fits both {name_of(best[0])} and {name_of(second[0])} equally well.")
        elif best and 0.45 <= best[1] < th:
            why.append(f"The closest name is {name_of(best[0])} at {pct(best[1])} similar, below the {pct(th)} needed to trust it.")
        elif p.dt and (not best or best[1] < 0.45):
            why.append(f"No {who} in the ledger resembles “{p.name_text}”.")
        for pt in parties.values():
            if score(b, pt.key) >= th and not open_items(b, lambda l, k=pt.key: l.party_key == k):
                why.append(f"The name matches {pt.name}, but they have no open {doc}s in the date window. It may be a duplicate "
                           f"payment or for an invoice not yet entered.")
                break
        same_amt = [l for l in ledger if l.type == p.dir and open_[l.id] > 0.005 and cents(open_[l.id]) == cents(p.abs) and not is_rej(b, l)]
        in_win = [l for l in same_amt if in_window(b, l)]
        if len(in_win) > 1:
            why.append(f"{len(in_win)} open items are for exactly {fmt(p.abs)}: {', '.join(f'{l.ref} ({l.party})' for l in in_win)}. "
                       f"Without a name or reference there's no way to tell which.")
        elif not in_win and same_amt:
            why.append(f"{same_amt[0].ref} ({same_amt[0].party}) is for the same amount but dated {same_amt[0].date}, outside the {win}-day window.")
        elif not same_amt:
            why.append(f"No open {doc} is for {fmt(p.abs)}, alone or combined with others from the same {who}.")
        dup = next((o for o in bank if o.id != b.id and cents(o.amount) == cents(b.amount)
                    and abs(days_between(o.date, b.date)) <= 3 and description_key(o.description) == p.desc_key), None)
        if dup:
            why.append(f"Possible duplicate: {dup.date} has another line for the same amount and description.")
        cand = sorted((l for l in ledger if l.type == p.dir and open_[l.id] > 0.005 and not is_rej(b, l)),
                      key=lambda l: -(0.55 * score(b, l.party_key) + 0.45 * (1 - min(1, abs(open_[l.id] - p.abs) / max(p.abs, 1)))))[:6]
        suggestion = (f"Ask the bank for the payer's details, or check which of {' / '.join(l.ref for l in in_win)} was paid."
                      if len(in_win) > 1 else
                      f"Ask the {'payer for a remittance advice' if p.dir == 'AR' else 'person who made this payment what it was for'}, "
                      f"or park it in suspense so the books still balance.")
        res(b, "unmatched", method="No match", why=why, candidates=[l.id for l in cand], suggestion=suggestion)

    outstanding = []
    for l in ledger:
        remaining = l.amount if l.type == "GL" else open_[l.id]
        if (l.type == "GL" and l.id not in used_gl) or (l.type != "GL" and remaining > 0.005):
            outstanding.append({**l.model_dump(), "remaining": remaining, "touched": l.type != "GL" and remaining < l.open - 0.004})
    return ReconcileOutput(results, outstanding, dict(open_))
