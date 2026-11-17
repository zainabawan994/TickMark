"""Local web app: upload files, watch the agent work, make decisions, download results.

Each run is one LangGraph thread. The graph runs in a background thread; when it
reaches the approval gate it pauses (``interrupt``), the browser shows the pending
entries, and the person's answers resume it with ``Command(resume=...)``.
"""
from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from langgraph.types import Command
from pydantic import BaseModel

from ..config import PROVIDERS, RunContext, Settings
from ..graph.build import build_graph, make_checkpointer
from ..reporting import STATUS_LABEL, _matched_to

STATIC = Path(__file__).parent / "static"
SAMPLES = Path(__file__).resolve().parents[2] / "samples"
ALLOWED = {".csv", ".txt", ".xlsx", ".xlsm", ".pdf"}
DOWNLOADS = {"approved_entries": "Approved journal entries", "report": "Full report", "review_queue": "Review list",
             "outstanding": "Outstanding items", "summary": "Summary"}


@dataclass
class Run:
    id: str
    dir: Path
    ctx: RunContext
    files: dict[str, str]
    status: str = "running"  # running | waiting | done | error
    done_nodes: set[str] = field(default_factory=set)
    ai_total: int = 0
    ai_done: int = 0
    pending: list[dict] = field(default_factory=list)
    error: str = ""
    started: float = field(default_factory=time.time)


class Runner:
    def __init__(self, model_factory: Callable[[Settings], Any] | None = None, work_dir: str | Path = ".tickmark/runs"):
        self.graph = build_graph(checkpointer=make_checkpointer())
        self.runs: dict[str, Run] = {}
        self.lock = threading.Lock()
        self.model_factory = model_factory  # lets tests and demos inject a model
        self.work_dir = Path(work_dir)

    def _config(self, run: Run) -> dict:
        return {"configurable": {"thread_id": run.id}, "max_concurrency": run.ctx.settings.max_concurrency}

    def start(self, bank: Path, ledger: Path, settings: Settings, run_id: str | None = None) -> Run:
        run_id = run_id or uuid.uuid4().hex[:10]
        run_dir = self.work_dir / run_id
        model = self.model_factory(settings) if (self.model_factory and settings.use_llm) else None
        run = Run(run_id, run_dir, RunContext(settings=settings, model=model), {"bank": bank.name, "ledger": ledger.name})
        with self.lock:
            self.runs[run_id] = run
        payload = {"bank_path": str(bank), "ledger_path": str(ledger), "out_dir": str(run_dir / "out")}
        threading.Thread(target=self._drive, args=(run, payload), daemon=True).start()
        return run

    def decide(self, run: Run, answers: dict[str, str]) -> None:
        with self.lock:
            run.status, run.pending = "running", []
        threading.Thread(target=self._drive, args=(run, Command(resume=answers)), daemon=True).start()

    def _drive(self, run: Run, payload: Any) -> None:
        try:
            interrupted = None
            for chunk in self.graph.stream(payload, self._config(run), context=run.ctx, stream_mode="updates"):
                for node, update in chunk.items():
                    if node == "__interrupt__":
                        interrupted = update[0].value
                        continue
                    with self.lock:
                        run.done_nodes.add(node)
                        if node == "match":
                            run.ai_total = len((update or {}).get("tasks") or [])
                        elif node == "resolve_exception":
                            run.ai_done += 1
            values = self.graph.get_state(self._config(run)).values
            with self.lock:
                if interrupted:
                    run.status, run.pending = "waiting", interrupted.get("items", [])
                elif values.get("errors"):
                    run.status, run.error = "error", " ".join(values["errors"])
                else:
                    run.status = "done"
        except Exception as e:  # keep the server alive and tell the browser what happened
            with self.lock:
                run.status, run.error = "error", f"{type(e).__name__}: {e}"

    # ------------------------------------------------------------ views
    def snapshot(self, run: Run) -> dict:
        values = self.graph.get_state(self._config(run)).values
        ai_on = run.ctx.get_model() is not None
        steps = self._steps(run, values, ai_on)
        out: dict[str, Any] = {"id": run.id, "status": run.status, "error": run.error, "files": run.files,
                               "steps": steps, "warnings": values.get("warnings") or [], "ai_on": ai_on}
        results = values.get("results")
        if results:
            out.update(self._results(values))
        if run.status == "waiting":
            lines = {l["id"]: l for l in out.get("lines", [])}
            out["pending"] = [{**p, "line": lines.get(p["bank_id"])} for p in run.pending]
        if run.status == "done":
            outputs = values.get("outputs") or {}
            out["downloads"] = [{"key": k, "label": label} for k, label in DOWNLOADS.items() if k in outputs]
            out["narrative"] = values.get("narrative") or ""
        return out

    def _steps(self, run: Run, values: dict, ai_on: bool) -> list[dict]:
        d = run.done_nodes
        bank_n, ledger_n = len(values.get("bank") or []), len(values.get("ledger") or [])
        results = values.get("results") or {}
        open_n = sum(1 for r in results.values() if r.status in ("unmatched", "review"))
        approvals = values.get("approvals") or {}
        steps = [
            {"key": "read", "label": "Read your files", "done": "validate" in d,
             "detail": f"{bank_n} bank lines, {ledger_n} ledger items" if "validate" in d else ""},
            {"key": "match", "label": "Match bank lines to the ledger", "done": "match" in d,
             "detail": f"{len(results) - open_n} matched, {open_n} to look at" if "match" in d else ""},
            {"key": "ai", "label": "AI checks the hard cases", "done": "apply_resolutions" in d,
             "skipped": ("match" in d and run.ai_total == 0) or not ai_on,
             "detail": (f"{run.ai_done} of {run.ai_total} checked" if run.ai_total else "") if ai_on else "AI is off"},
            {"key": "journal", "label": "Draft journal entries", "done": "build_journal" in d,
             "detail": f"{len(values.get('journal') or [])} entries" if "build_journal" in d else ""},
            {"key": "decide", "label": "Your decisions", "done": "approval_gate" in d, "waiting": run.status == "waiting",
             "detail": f"{len(run.pending)} waiting for you" if run.status == "waiting" else
                       (f"{sum(a['status'] == 'approved' for a in approvals.values())} approved" if "approval_gate" in d else "")},
            {"key": "report", "label": "Save reports", "done": "report" in d, "detail": ""},
        ]
        if run.status == "running":
            for s in steps:
                if not s["done"] and not s.get("skipped"):
                    s["active"] = True
                    break
        return steps

    def _results(self, values: dict) -> dict:
        bank = sorted(values.get("bank") or [], key=lambda b: b.date)
        ledger = {l.id: l for l in values.get("ledger") or []}
        results = values["results"]
        approvals = values.get("approvals") or {}
        notes = values.get("resolution_notes") or {}
        je_by_bank = {je.bank_id: je for je in values.get("journal") or []}
        lines = []
        for b in bank:
            r = results[b.id]
            je = je_by_bank.get(b.id)
            approval = approvals.get(je.id, {}) if je else {}
            lines.append({"id": b.id, "date": b.date, "description": b.description, "amount": b.amount, "status": r.status,
                          "outcome": outcome(r.status, approval.get("status")), "approved_by": approval.get("by", ""),
                          "label": STATUS_LABEL[r.status], "method": r.method, "confidence": r.confidence, "source": r.source,
                          "matched_to": _matched_to(r, ledger) if r.status != "unmatched" else "",
                          "reasons": r.why or r.reasons, "suggestion": r.suggestion, "note": notes.get(b.id, ""),
                          "je_id": je.id if je else None, "je_status": approvals.get(je.id, {}).get("status") if je else None})
        oc = [l["outcome"] for l in lines]
        return {
            "lines": lines,
            "period": [bank[0].date, bank[-1].date] if bank else None,
            "summary": {  # every line lands in exactly one group, so these add up to "lines"
                "lines": len(lines), "matched": oc.count("matched"), "suspense": oc.count("suspense"),
                "waiting": oc.count("waiting"), "unresolved": oc.count("unresolved"),
                "approved": sum(v["status"] == "approved" for v in approvals.values()),
            },
        }


def outcome(status: str, je_status: str | None) -> str:
    """Where a bank line ended up, after the matcher, the AI and the person."""
    if status == "recorded":
        return "matched"          # already in the books
    if je_status == "approved":
        return "suspense" if status == "suspense" else "matched"
    if je_status == "rejected" or status == "unmatched":
        return "unresolved"
    return "waiting"


class Decisions(BaseModel):
    answers: dict[str, str]


def create_app(model_factory: Callable[[Settings], Any] | None = None, work_dir: str | Path = ".tickmark/runs") -> FastAPI:
    runner = Runner(model_factory, work_dir)
    app = FastAPI(title="Tickmark", docs_url=None, redoc_url=None)
    app.state.runner = runner

    def get_run(run_id: str) -> Run:
        run = runner.runs.get(run_id)
        if run is None:
            raise HTTPException(404, "This run no longer exists. The app may have been restarted.")
        return run

    def settings_for(use_ai: bool, auto: bool) -> Settings:
        return Settings(use_llm=use_ai, interactive=True, ai_auto_approve=auto, auto_post_suspense=auto)

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return (STATIC / "index.html").read_text(encoding="utf-8")

    @app.get("/api/info")
    def info() -> dict:
        s = Settings()
        ctx = RunContext(settings=s, model=model_factory(s) if model_factory else None)
        ready = ctx.get_model() is not None
        spec = PROVIDERS.get(s.provider, {"label": s.provider})
        return {"model": s.model.split(":", 1)[-1], "provider": spec["label"], "ai_ready": ready,
                "reason": "" if ready else ctx.unavailable_reason, "samples": (SAMPLES / "bank_statement.pdf").exists()}

    @app.post("/api/runs")
    async def start_run(bank: UploadFile = File(...), ledger: UploadFile = File(...),
                        use_ai: bool = Form(True), auto: bool = Form(False)) -> dict:
        run_id = uuid.uuid4().hex[:10]
        inbox = runner.work_dir / run_id / "in"
        inbox.mkdir(parents=True, exist_ok=True)
        paths = {}
        for kind, up in (("bank", bank), ("ledger", ledger)):
            ext = Path(up.filename or "").suffix.lower()
            if ext not in ALLOWED or (kind == "ledger" and ext == ".pdf"):
                raise HTTPException(400, f"The {kind} file must be {'CSV, Excel or PDF' if kind == 'bank' else 'CSV or Excel'}.")
            dest = inbox / f"{kind}{ext}"
            dest.write_bytes(await up.read())
            paths[kind] = dest
        run = runner.start(paths["bank"], paths["ledger"], settings_for(use_ai, auto), run_id)
        run.files = {"bank": bank.filename or "bank", "ledger": ledger.filename or "ledger"}
        return {"id": run.id}

    @app.post("/api/runs/sample")
    def start_sample(use_ai: bool = Form(True), auto: bool = Form(False)) -> dict:
        if not (SAMPLES / "bank_statement.pdf").exists():
            raise HTTPException(404, "Sample files aren't available in this installation.")
        run = runner.start(SAMPLES / "bank_statement.pdf", SAMPLES / "ledger.xlsx", settings_for(use_ai, auto))
        return {"id": run.id}

    @app.get("/api/runs/{run_id}")
    def status(run_id: str) -> dict:
        return runner.snapshot(get_run(run_id))

    @app.post("/api/runs/{run_id}/decisions")
    def decide(run_id: str, body: Decisions) -> dict:
        run = get_run(run_id)
        if run.status != "waiting":
            raise HTTPException(409, "This run isn't waiting for decisions.")
        clean = {k: v for k, v in body.answers.items() if v in ("approve", "reject", "skip")}
        runner.decide(run, clean)
        return {"ok": True}

    @app.get("/api/runs/{run_id}/files/{key}")
    def download(run_id: str, key: str):
        run = get_run(run_id)
        outputs = runner.graph.get_state(runner._config(run)).values.get("outputs") or {}
        if key not in DOWNLOADS or key not in outputs:
            raise HTTPException(404, "File not found.")
        path = Path(outputs[key])
        return FileResponse(path, filename=path.name)

    return app
