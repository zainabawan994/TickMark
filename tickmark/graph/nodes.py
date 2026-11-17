"""Graph nodes. Each takes the state (and the runtime context) and returns a partial update."""
from __future__ import annotations

from collections import Counter
from pathlib import Path

from langgraph.runtime import Runtime
from langgraph.types import Send, interrupt

from ..agents.extractors import extract_statement_from_images, extract_statement_from_text, map_columns, write_narrative
from ..agents.resolver import resolve_exception as run_resolver
from ..config import RunContext, Settings
from ..ingest.pdf import page_images, parse_pdf_lines, pdf_lines
from ..ingest.tables import missing_fields, normalize_bank, normalize_ledger, read_rows, to_table, detect_columns
from ..ingest.values import r2
from ..journal import UnbalancedEntryError, build_journal as make_journal
from ..matching.engine import MatchConfig, reconcile
from ..memory import AliasMemory
from ..models import Override, ResolverTask
from ..reporting import facts_for_narrative, write_reports
from .state import RecState


def _ctx(runtime: Runtime | None) -> RunContext:
    c = getattr(runtime, "context", None)
    if isinstance(c, RunContext):
        return c
    if isinstance(c, dict):  # e.g. LangGraph Studio passes plain JSON
        return RunContext(settings=Settings(**c.get("settings", c)))
    return RunContext()


def _match_cfg(s: Settings) -> MatchConfig:
    return MatchConfig(name_threshold=s.name_threshold, fee_tolerance=s.fee_tolerance, date_window=s.date_window,
                       ai_auto_approve=s.ai_auto_approve)


# ----------------------------------------------------------------- ingestion
def ingest_bank(state: RecState, runtime: Runtime[RunContext]) -> dict:
    ctx = _ctx(runtime)
    s, model = ctx.settings, ctx.get_model()
    path = Path(state.get("bank_path", ""))
    if not path.is_file():
        return {"errors": [f"Bank statement not found: {path}"]}
    warnings: list[str] = []
    info: dict = {"file": path.name}
    try:
        if path.suffix.lower() == ".pdf":
            lines = pdf_lines(path)
            if not lines:
                if model is None:
                    return {"errors": ["The PDF has no text layer (a scan). Enable the LLM or export the statement as CSV."]}
                rows, st = extract_statement_from_images(model, page_images(path))
                info.update(method="LLM vision (scanned PDF)", opening=st.opening_balance, closing=st.closing_balance)
            else:
                parsed = parse_pdf_lines(lines)
                info.update(opening=parsed.opening, closing=parsed.closing)
                good = len(parsed.rows) >= 2 and parsed.verified is not False and parsed.unsure <= max(1, len(parsed.rows) // 5)
                if good:
                    rows = parsed.rows
                    info["method"] = "PDF layout" + (", balances verified" if parsed.verified else "")
                elif model is not None:
                    rows, st = extract_statement_from_text(model, "\n".join(l.text for l in lines))
                    info.update(method="LLM reading of PDF text", opening=st.opening_balance or parsed.opening,
                                closing=st.closing_balance or parsed.closing)
                else:
                    rows = parsed.rows
                    info["method"] = "PDF layout (unverified)"
                    warnings.append("The PDF layout was unclear and no LLM is available. Check bank line directions.")
        else:
            table = to_table(read_rows(path), "bank")
            mapping = detect_columns(table.headers, "bank")
            info["method"] = "columns detected"
            if missing_fields(mapping, "bank") and model is not None:
                mapping = map_columns(model, "bank", table.headers, table.rows[:5])
                info["method"] = "columns identified by LLM"
            if miss := missing_fields(mapping, "bank"):
                return {"errors": [f"Couldn't find the {', '.join(miss)} column(s) in {path.name}."]}
            norm = normalize_bank(table, mapping, s.date_order)
            rows = norm.rows
            info["columns"] = {k: table.headers[v] for k, v in mapping.items()}
            if norm.skipped:
                warnings.append(f"Skipped {len(norm.skipped)} bank row(s), e.g. row {norm.skipped[0]['row']}: {norm.skipped[0]['reason']}.")
    except Exception as e:
        return {"errors": [f"Couldn't read {path.name}: {type(e).__name__}: {e}"]}
    if info.get("opening") is not None and info.get("closing") is not None:
        info["balance_check"] = abs(r2(info["opening"] + sum(t.amount for t in rows)) - info["closing"]) < 0.01
        if not info["balance_check"]:
            warnings.append("Opening balance + transactions doesn't equal the closing balance: a line may be missing or misread.")
    info["count"] = len(rows)
    return {"bank": rows, "bank_info": info, "warnings": warnings,
            "log": [f"Read {len(rows)} bank lines from {path.name} ({info['method']})."]}


def ingest_ledger(state: RecState, runtime: Runtime[RunContext]) -> dict:
    ctx = _ctx(runtime)
    s, model = ctx.settings, ctx.get_model()
    path = Path(state.get("ledger_path", ""))
    if not path.is_file():
        return {"errors": [f"Ledger not found: {path}"]}
    try:
        table = to_table(read_rows(path), "ledger")
        mapping = detect_columns(table.headers, "ledger")
        method = "columns detected"
        if missing_fields(mapping, "ledger") and model is not None:
            mapping = map_columns(model, "ledger", table.headers, table.rows[:5])
            method = "columns identified by LLM"
        if miss := missing_fields(mapping, "ledger"):
            return {"errors": [f"Couldn't find the {', '.join(miss)} column(s) in {path.name}."]}
        norm = normalize_ledger(table, mapping, s.date_order)
    except Exception as e:
        return {"errors": [f"Couldn't read {path.name}: {type(e).__name__}: {e}"]}
    kinds = Counter(l.type for l in norm.rows)
    warnings = [f"Skipped {len(norm.skipped)} ledger row(s), e.g. row {norm.skipped[0]['row']}: {norm.skipped[0]['reason']}."] if norm.skipped else []
    return {"ledger": norm.rows, "ledger_info": {"file": path.name, "method": method, "count": len(norm.rows), **kinds,
                                                   "columns": {k: table.headers[v] for k, v in mapping.items()}},
            "warnings": warnings,
            "log": [f"Read {kinds['AR']} open invoices, {kinds['AP']} open bills and {kinds['GL']} booked bank entries from {path.name} ({method})."]}


def validate(state: RecState) -> dict:
    errors, warnings = [], []
    if not state.get("errors"):
        if not state.get("bank"):
            errors.append("No bank transactions could be read.")
        if not state.get("ledger"):
            errors.append("No open ledger items could be read.")
    bank = state.get("bank") or []
    dupes = [k for k, n in Counter((b.date, b.amount, b.description) for b in bank).items() if n > 1]
    if dupes:
        warnings.append(f"{len(dupes)} bank line(s) appear twice with the same date, amount and description: check for duplicates.")
    refs = Counter(l.ref for l in state.get("ledger") or [] if l.ref and l.type != "GL")
    if dup_refs := [r for r, n in refs.items() if n > 1]:
        warnings.append(f"Ledger numbers used more than once: {', '.join(dup_refs[:5])}.")
    return {"errors": errors, "warnings": warnings}


def route_after_validate(state: RecState) -> str:
    return "report" if state.get("errors") else "match"


# ----------------------------------------------------------------- matching
def match(state: RecState, runtime: Runtime[RunContext]) -> dict:
    ctx = _ctx(runtime)
    s = ctx.settings
    memory = AliasMemory(ctx.memory_file)
    out = reconcile(state["bank"], state["ledger"], _match_cfg(s), aliases=memory.aliases)
    counts = Counter(r.status for r in out.results.values())
    tasks: list[ResolverTask] = []
    if ctx.get_model() is not None:
        tasks = build_tasks(state, out.results, out.open_after, s)
    msg = ", ".join(f"{n} {k}" for k, n in counts.most_common())
    return {"results": out.results, "outstanding": out.outstanding, "open_after": out.open_after, "tasks": tasks,
            "log": [f"Matcher: {msg}." + (f" Sending {len(tasks)} exception(s) to the AI resolver." if tasks else "")]}


def build_tasks(state: RecState, results, open_after, s: Settings) -> list[ResolverTask]:
    tasks = []
    ledger = state["ledger"]
    for b in state["bank"]:
        r = results[b.id]
        if r.source != "engine" or r.status not in ("unmatched", "review") or r.kind == "recorded":
            continue
        direction = "AR" if b.amount > 0 else "AP"
        own = {a.ledger_id: a.amount for a in r.allocations}
        items = []
        for l in ledger:
            avail = r2(open_after.get(l.id, 0) + own.get(l.id, 0))
            if l.type == direction and avail > 0.005:
                items.append(l.model_copy(update={"open": avail}))
        items.sort(key=lambda l: (abs(l.open - abs(b.amount)), l.date))
        summary = " ".join(r.why or r.reasons)[:600] if r.status == "unmatched" else f"{r.method}, {r.confidence}% sure. " + " ".join(r.reasons)[:600]
        tasks.append(ResolverTask(bank=b, matcher_status=r.status, matcher_summary=summary,
                                  candidate_ids=list(own) or r.candidates, items=items[:300], fee_tolerance=s.fee_tolerance))
    return tasks


def route_after_match(state: RecState):
    tasks = state.get("tasks") or []
    if not tasks:
        return "build_journal"
    return [Send("resolve_exception", {"task": t}) for t in tasks]  # map: one agent per exception, in parallel


def resolve_exception(payload: dict, runtime: Runtime[RunContext]) -> dict:
    ctx = _ctx(runtime)
    task: ResolverTask = payload["task"]
    res = run_resolver(ctx.get_model(), task, ctx.settings.agent_recursion_limit)
    d = res.decision
    line = (f"AI resolver on {task.bank.id}: {d.action} {', '.join(d.ledger_ids)} ({d.confidence}) after {res.tool_calls} lookup(s)."
            if d else f"AI resolver on {task.bank.id} failed: {res.error}")
    return {"resolutions": [res], "log": [line]}


# ----------------------------------------------------------------- guardrails
RANK = {"high": 3, "medium": 2, "low": 1}


def apply_resolutions(state: RecState, runtime: Runtime[RunContext]) -> dict:
    """Reduce step: validate every AI decision before it can touch the books, then re-run the matcher."""
    ctx = _ctx(runtime)
    bank = {b.id: b for b in state["bank"]}
    ledger = {l.id: l for l in state["ledger"]}
    results, open_after = state["results"], state["open_after"]
    overrides: dict[str, Override] = dict(state.get("overrides") or {})
    notes: dict[str, str] = {}
    claimed: dict[str, str] = {}
    log: list[str] = []
    ordered = sorted(state.get("resolutions") or [], key=lambda r: -(RANK[r.decision.confidence] if r.decision else 0))
    for res in ordered:
        d, b = res.decision, bank.get(res.bank_id)
        if b is None:
            continue
        if res.error or d is None:
            notes[b.id] = f"AI could not decide: {res.error}"
            continue
        if d.action == "match":
            if d.confidence == "low":
                notes[b.id] = f"AI's low-confidence guess ({', '.join(d.ledger_ids)}) not applied: {d.explanation}"
                continue
            direction = "AR" if b.amount > 0 else "AP"
            own = {a.ledger_id: a.amount for a in results[b.id].allocations}
            valid, problems = [], []
            for i in dict.fromkeys(d.ledger_ids):
                l = ledger.get(i)
                if l is None:
                    problems.append(f"{i} does not exist")
                elif l.type != direction:
                    problems.append(f"{l.ref or i} is the wrong kind of item for {'money in' if b.amount > 0 else 'money out'}")
                elif r2(open_after.get(i, 0) + own.get(i, 0)) <= 0.005:
                    problems.append(f"{l.ref or i} is already fully paid")
                elif i in claimed:
                    problems.append(f"{l.ref or i} is already claimed by {claimed[i]}")
                else:
                    valid.append(i)
            if not valid:
                notes[b.id] = "AI suggestion rejected by checks: " + "; ".join(problems)
                log.append(f"Guardrail rejected AI match for {b.id}: {'; '.join(problems)}.")
                continue
            for i in valid:
                claimed[i] = b.id
            overrides[b.id] = Override(kind="match", ledger_ids=valid, source="ai", confidence=d.confidence, explanation=d.explanation)
            notes[b.id] = "AI match applied" + (f" (dropped: {'; '.join(problems)})" if problems else "")
        else:
            if d.action == "suspense" and results[b.id].status == "review":
                # The engine had a plausible proposal; an unsure AI shouldn't discard it. A person decides.
                notes[b.id] = f"AI couldn't confirm the suggested match: {d.explanation}"
                continue
            if d.action != "suspense" and d.confidence == "low":
                notes[b.id] = f"AI's low-confidence guess ({d.action.replace('_', ' ')}) not applied: {d.explanation}"
                continue
            kind = {"bank_charge": "fees", "interest": "interest", "suspense": "suspense"}[d.action]
            overrides[b.id] = Override(kind=kind, source="ai", confidence=d.confidence, explanation=d.explanation)
            notes[b.id] = f"AI classified as {d.action.replace('_', ' ')}"
    out = reconcile(state["bank"], state["ledger"], _match_cfg(ctx.settings), overrides=overrides,
                    aliases=AliasMemory(ctx.memory_file).aliases)
    applied = sum(1 for n in notes.values() if n.startswith("AI match applied") or n.startswith("AI classified"))
    log.append(f"Applied {applied} of {len(ordered)} AI decision(s) after validation.")
    return {"results": out.results, "outstanding": out.outstanding, "open_after": out.open_after,
            "overrides": overrides, "resolution_notes": notes, "log": log}


# ----------------------------------------------------------------- journal + approval
def build_journal(state: RecState, runtime: Runtime[RunContext]) -> dict:
    ctx = _ctx(runtime)
    try:
        journal = make_journal(state["bank"], state["results"], state["ledger"], ctx.settings.accounts)
    except UnbalancedEntryError as e:
        return {"errors": [str(e)], "journal": []}
    return {"journal": journal, "log": [f"Drafted {len(journal)} balanced journal entries."]}


def policy_decision(r, s: Settings) -> tuple[str, str]:
    if r.status == "suspense":
        return ("approved", "policy: suspense entries are posted automatically") if s.auto_post_suspense \
            else ("pending", "suspense entry: needs a person")
    if r.source == "user":
        return "approved", "decided by a person"
    if r.source == "ai":
        if s.ai_auto_approve and r.status in ("matched", "bankitem") and r.confidence >= 85:
            return "approved", "policy: high-confidence AI decision"
        return "pending", f"AI suggestion ({r.confidence}% confidence): needs a person"
    if r.status in ("matched", "bankitem") and r.confidence >= s.auto_approve_threshold:
        return "approved", f"policy: {r.confidence}% confidence (threshold {s.auto_approve_threshold}%)"
    return "pending", f"{r.method} at {r.confidence}% confidence: needs a person"


def approval_gate(state: RecState, runtime: Runtime[RunContext]) -> dict:
    ctx = _ctx(runtime)
    s = ctx.settings
    results = state["results"]
    bank = {b.id: b for b in state["bank"]}
    approvals: dict[str, dict] = {}
    pending = []
    for je in state.get("journal") or []:
        status, reason = policy_decision(results[je.bank_id], s)
        approvals[je.id] = {"status": status, "by": "policy" if status == "approved" else "", "reason": reason}
        if status == "pending":
            pending.append(je)
    log = [f"Approval policy: {len(approvals) - len(pending)} approved automatically, {len(pending)} need a person."]
    if pending and s.interactive:
        # Pause the graph. The caller resumes with Command(resume={je_id: "approve" | "reject" | "skip"}).
        answers = interrupt({"type": "approval_request", "items": [
            {"je_id": je.id, "bank_id": je.bank_id, "date": je.date, "description": bank[je.bank_id].description,
             "amount": bank[je.bank_id].amount, "memo": je.memo, "reason": approvals[je.id]["reason"],
             "explanation": results[je.bank_id].reasons,
             "lines": [l.model_dump() for l in je.lines]} for je in pending]}) or {}
        memory = AliasMemory(ctx.memory_file)
        learned = 0
        for je in pending:
            ans = str(answers.get(je.id, "skip")).lower()
            if ans in ("approve", "approved", "a"):
                approvals[je.id] = {"status": "approved", "by": "person", "reason": "approved by a person"}
                r = results[je.bank_id]
                if r.kind == "match" and r.party:  # remember this wording for next month
                    learned += memory.learn(bank[je.bank_id].description, r.party)
            elif ans in ("reject", "rejected", "r"):
                approvals[je.id] = {"status": "rejected", "by": "person", "reason": "rejected by a person"}
        if learned:
            memory.save()
        n_ok = sum(1 for je in pending if approvals[je.id]["status"] == "approved")
        log.append(f"Reviewer approved {n_ok} and rejected "
                   f"{sum(1 for je in pending if approvals[je.id]['status'] == 'rejected')} of {len(pending)}; "
                   f"learned {learned} name variation(s).")
    return {"approvals": approvals, "log": log}


# ----------------------------------------------------------------- report
def report(state: RecState, runtime: Runtime[RunContext]) -> dict:
    ctx = _ctx(runtime)
    narrative = ""
    model = ctx.get_model()
    if model is not None and state.get("results") and not state.get("errors"):
        try:
            narrative = write_narrative(model, facts_for_narrative(state))
        except Exception as e:  # the report must never fail because of the LLM
            narrative = ""
            state = {**state, "warnings": list(state.get("warnings") or []) + [f"Narrative skipped: {type(e).__name__}"]}
    outputs = write_reports(state, ctx.settings, Path(state.get("out_dir") or "tickmark_output"), narrative)
    return {"outputs": outputs, "narrative": narrative, "log": [f"Wrote {len(outputs)} report file(s) to {state.get('out_dir') or 'tickmark_output'}."]}
