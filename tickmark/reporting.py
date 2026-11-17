"""Write the run's outputs: files an accountant can import, review and file."""
from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

from .config import Settings

STATUS_LABEL = {"matched": "Matched", "recorded": "Already booked", "review": "Needs review", "unmatched": "Unmatched",
                "bankitem": "Bank item", "suspense": "Suspense"}


def _write_csv(path: Path, rows: list[list]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(rows)


def _matched_to(r, ledger_by_id) -> str:
    if r.kind == "recorded" and r.gl_id:
        g = ledger_by_id[r.gl_id]
        return f"Booked as {g.ref or g.description}"
    refs = ", ".join(ledger_by_id[a.ledger_id].ref for a in r.allocations)
    return f"{refs}{' for ' + r.party if r.party and refs else ''}"


def facts_for_narrative(state: dict) -> str:
    results = state.get("results") or {}
    bank = {b.id: b for b in state.get("bank") or []}
    approvals = state.get("approvals") or {}
    c = Counter(r.status for r in results.values())
    lines = [f"Bank lines: {len(bank)}. " + ", ".join(f"{STATUS_LABEL[k]}: {n}" for k, n in c.items()) + ".",
             f"Journal entries: {len(state.get('journal') or [])}; approved {sum(1 for a in approvals.values() if a['status'] == 'approved')}, "
             f"pending {sum(1 for a in approvals.values() if a['status'] == 'pending')}."]
    for bid, r in results.items():
        if r.status in ("review", "unmatched", "suspense") or r.excess or r.source == "ai":
            b = bank[bid]
            detail = " ".join(r.why or r.reasons)[:300]
            lines.append(f"- {b.date} {b.description} {b.amount:.2f}: {STATUS_LABEL[r.status]}, {r.method}. {detail}")
    lines += [f"Warning: {w}" for w in state.get("warnings") or []]
    return "\n".join(lines)


def write_reports(state: dict, settings: Settings, out_dir: Path, narrative: str = "") -> dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, str] = {}
    bank = state.get("bank") or []
    ledger_by_id = {l.id: l for l in state.get("ledger") or []}
    results = state.get("results") or {}
    journal = state.get("journal") or []
    approvals = state.get("approvals") or {}
    notes = state.get("resolution_notes") or {}
    je_by_bank = {je.bank_id: je for je in journal}

    if results:
        # 1. Approved journal entries, ready to import
        rows = [["Entry", "Date", "Account", "Party", "Reference", "Debit", "Credit", "Memo", "Approved by"]]
        for je in journal:
            a = approvals.get(je.id, {})
            if a.get("status") == "approved":
                for l in je.lines:
                    rows.append([je.id, je.date, l.account, l.party, l.ref, f"{l.debit:.2f}" if l.debit else "",
                                 f"{l.credit:.2f}" if l.credit else "", je.memo, a.get("by", "")])
        _write_csv(out_dir / "journal_entries_approved.csv", rows)
        outputs["approved_entries"] = str(out_dir / "journal_entries_approved.csv")

        # 2. Review queue: everything a person still has to decide
        rows = [["Item", "Bank date", "Bank description", "Amount", "Status", "Method", "Confidence", "Why", "AI note",
                 "Proposed entry", "Proposed lines"]]
        for b in sorted(bank, key=lambda x: x.date):
            r = results[b.id]
            je = je_by_bank.get(b.id)
            pending = je is not None and approvals.get(je.id, {}).get("status") == "pending"
            if not pending and r.status != "unmatched":
                continue
            lines = "; ".join(f"{'Dr' if l.debit else 'Cr'} {l.account} {l.debit or l.credit:.2f}" for l in je.lines) if je else ""
            rows.append([je.id if je else f"Unmatched {b.id}", b.date, b.description, f"{b.amount:.2f}", STATUS_LABEL[r.status],
                         r.method, f"{r.confidence}%" if r.confidence else "", " ".join(r.why or r.reasons) + (f" Next step: {r.suggestion}" if r.suggestion else ""),
                         notes.get(b.id, ""), je.memo if je else "", lines])
        _write_csv(out_dir / "review_queue.csv", rows)
        outputs["review_queue"] = str(out_dir / "review_queue.csv")

        # 3. Full line-by-line reconciliation report
        rows = [["Bank id", "Date", "Description", "Money in", "Money out", "Status", "Method", "Confidence", "Source",
                 "Matched to", "Fee", "Overpayment", "Explanation", "AI note", "Journal entry", "Entry status"]]
        for b in sorted(bank, key=lambda x: x.date):
            r = results[b.id]
            je = je_by_bank.get(b.id)
            rows.append([b.id, b.date, b.description, f"{b.amount:.2f}" if b.amount > 0 else "", f"{-b.amount:.2f}" if b.amount < 0 else "",
                         STATUS_LABEL[r.status], r.method, f"{r.confidence}%" if r.confidence else "", r.source, _matched_to(r, ledger_by_id),
                         f"{r.fee:.2f}" if r.fee else "", f"{r.excess:.2f}" if r.excess else "", " ".join(r.why or r.reasons),
                         notes.get(b.id, ""), je.id if je else "", approvals.get(je.id, {}).get("status", "") if je else ""])
        _write_csv(out_dir / "reconciliation_report.csv", rows)
        outputs["report"] = str(out_dir / "reconciliation_report.csv")

        # 4. Ledger items not on this statement
        rows = [["Date", "Number", "Name", "Kind", "Still open"]]
        for o in state.get("outstanding") or []:
            kind = {"AR": "Invoice", "AP": "Bill"}.get(o["type"]) or ("Deposit in transit" if (o.get("signed") or 0) > 0 else "Uncleared payment")
            rows.append([o["date"], o["ref"], o["party"] or o["description"], kind + (", part paid" if o.get("touched") else ""), f"{o['remaining']:.2f}"])
        _write_csv(out_dir / "outstanding_items.csv", rows)
        outputs["outstanding"] = str(out_dir / "outstanding_items.csv")

    # 5. Human-readable summary
    outputs["summary"] = str(out_dir / "summary.md")
    (out_dir / "summary.md").write_text(_summary_md(state, narrative), encoding="utf-8")

    # 6. Machine-readable dump of the whole run
    dump = {
        "bank_info": state.get("bank_info"), "ledger_info": state.get("ledger_info"),
        "errors": state.get("errors") or [], "warnings": state.get("warnings") or [], "log": state.get("log") or [],
        "results": {k: v.model_dump() for k, v in results.items()},
        "journal": [je.model_dump() for je in journal], "approvals": approvals,
        "resolutions": [r.model_dump() for r in state.get("resolutions") or []], "resolution_notes": notes,
        "settings": settings.model_dump(),
    }
    (out_dir / "run.json").write_text(json.dumps(dump, indent=2, default=str), encoding="utf-8")
    outputs["run_json"] = str(out_dir / "run.json")
    return outputs


def _summary_md(state: dict, narrative: str) -> str:
    bank = state.get("bank") or []
    results = state.get("results") or {}
    approvals = state.get("approvals") or {}
    md = ["# Bank reconciliation summary", ""]
    if state.get("errors"):
        md += ["**The run stopped:**", ""] + [f"- {e}" for e in state["errors"]] + [""]
        return "\n".join(md + ["## Audit trail", ""] + [f"- {l}" for l in state.get("log") or []]) + "\n"
    dates = sorted(b.date for b in bank)
    bi, li = state.get("bank_info") or {}, state.get("ledger_info") or {}
    md += [f"Statement **{bi.get('file', '')}** ({dates[0]} to {dates[-1]}, {len(bank)} lines, read by {bi.get('method', '')}) "
           f"against ledger **{li.get('file', '')}** ({li.get('count', 0)} items).", ""]
    if bi.get("balance_check") is not None:
        md += [f"Statement balance check: {'passed' if bi['balance_check'] else '**failed**'} "
               f"(opening {bi.get('opening'):,.2f}, closing {bi.get('closing'):,.2f}).", ""]
    c = Counter(r.status for r in results.values())
    unexplained = sum(abs(b.amount) for b in bank if results[b.id].status in ("unmatched", "suspense"))
    md += ["| Result | Lines |", "|---|---:|"] + [f"| {STATUS_LABEL[k]} | {c.get(k, 0)} |" for k in STATUS_LABEL] + [""]
    a = Counter(v["status"] for v in approvals.values())
    md += [f"Journal entries: **{len(approvals)}** drafted, **{a.get('approved', 0)}** approved, "
           f"**{a.get('pending', 0)}** waiting for a person, **{a.get('rejected', 0)}** rejected. "
           f"Amount not yet explained: **{unexplained:,.2f}**.", ""]
    if narrative:
        md += ["## Note for the finance manager", "", narrative.strip(), ""]
    attention = [(b, results[b.id]) for b in sorted(bank, key=lambda x: x.date)
                 if results[b.id].status in ("review", "unmatched", "suspense") or results[b.id].source == "ai"]
    if attention:
        md += ["## Needs attention", ""]
        notes = state.get("resolution_notes") or {}
        for b, r in attention:
            md.append(f"- **{b.date} {b.description}** ({b.amount:,.2f}): {STATUS_LABEL[r.status]}, {r.method}. "
                      f"{' '.join(r.why or r.reasons)}" + (f" *{notes[b.id]}.*" if b.id in notes else ""))
        md.append("")
    if state.get("warnings"):
        md += ["## Warnings", ""] + [f"- {w}" for w in state["warnings"]] + [""]
    md += ["## Audit trail", ""] + [f"- {l}" for l in state.get("log") or []]
    return "\n".join(md) + "\n"
