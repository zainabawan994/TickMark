"""Command line: `tickmark run --bank statement.pdf --ledger ledger.xlsx`."""
from __future__ import annotations

import argparse
import os
import sys
import uuid

from langgraph.types import Command

from .config import PROVIDERS, RunContext, Settings, key_for
from .graph.build import build_graph, make_checkpointer


def _load_dotenv() -> None:
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="tickmark", description="Automatic bank reconciliation agent (LangGraph).")
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="Reconcile a bank statement against a ledger")
    r.add_argument("--bank", required=True, help="Bank statement: CSV, XLSX or PDF")
    r.add_argument("--ledger", required=True, help="Ledger / open items: CSV or XLSX")
    r.add_argument("--out", default="tickmark_output", help="Output folder (default: tickmark_output)")
    r.add_argument("--model", help="LangChain model id, e.g. google_genai:gemini-flash-latest or anthropic:claude-sonnet-5-5")
    r.add_argument("--no-llm", action="store_true", help="Deterministic matching only; no AI calls")
    r.add_argument("--interactive", action="store_true", help="Pause for a person to approve or reject what policy can't")
    r.add_argument("--ai-auto-approve", action="store_true", help="Approve the AI's high-confidence matches automatically")
    r.add_argument("--auto-post-suspense", action="store_true", help="Approve suspense entries automatically")
    r.add_argument("--threshold", type=int, default=85, help="Engine confidence needed for automatic approval (default 85)")
    r.add_argument("--fee-tolerance", type=float, default=50.0, help="Bank fee shortfall tolerance (default 50)")
    r.add_argument("--date-window", type=int, default=90, help="Days an invoice may precede its payment (default 90)")
    r.add_argument("--name-threshold", type=float, default=0.72, help="Name similarity needed, 0-1 (default 0.72)")
    r.add_argument("--date-order", choices=["auto", "dmy", "mdy"], default="auto")
    r.add_argument("--memory", default=".tickmark/aliases.json", help="Where learned name variations are kept")
    r.add_argument("--max-concurrency", type=int, default=4, help="Parallel AI resolver agents (default 4)")
    r.add_argument("--rpm", type=float, help="Cap LLM requests per minute (use on free tiers, e.g. 8)")
    c = sub.add_parser("check", help="Check your API key and model before a real run")
    c.add_argument("--model", help="Model to check (default: TICKMARK_MODEL or the provider default)")
    c.add_argument("--list", action="store_true", help="List the Gemini models your key can use")
    u = sub.add_parser("ui", help="Open the web app in your browser")
    u.add_argument("--host", default="127.0.0.1", help="Address to listen on (default: this computer only)")
    u.add_argument("--port", type=int, default=8765, help="Port (default 8765)")
    u.add_argument("--no-browser", action="store_true", help="Don't open a browser window")
    return p.parse_args(argv)


def settings_from(a: argparse.Namespace) -> Settings:
    s = Settings(use_llm=not a.no_llm, interactive=a.interactive, ai_auto_approve=a.ai_auto_approve,
                 auto_post_suspense=a.auto_post_suspense, auto_approve_threshold=a.threshold, fee_tolerance=a.fee_tolerance,
                 date_window=a.date_window, name_threshold=a.name_threshold, date_order=a.date_order,
                 memory_path=a.memory, max_concurrency=a.max_concurrency)
    if a.model:
        s.model = a.model
    if a.rpm:
        s.requests_per_minute = a.rpm
    return s


def ask_reviewer(request: dict) -> dict:
    """Terminal UI for the approval interrupt."""
    items = request["items"]
    print(f"\n{len(items)} journal entr{'y needs' if len(items) == 1 else 'ies need'} your decision. "
          "[a]pprove, [r]eject, [s]kip (leave pending), [A] approve all remaining, [q] skip all remaining.\n")
    answers, bulk = {}, None
    for i, it in enumerate(items, 1):
        if bulk:
            answers[it["je_id"]] = bulk
            continue
        print(f"── {i}/{len(items)}  {it['je_id']}  {it['date']}  {it['description']}  {it['amount']:,.2f}")
        print(f"   {it['memo']}  ({it['reason']})")
        for reason in it["explanation"][:4]:
            print(f"   • {reason}")
        for l in it["lines"]:
            side = f"Dr {l['debit']:>12,.2f}" if l["debit"] else f"Cr {l['credit']:>12,.2f}"
            print(f"     {side}  {l['account']}{'  ' + l['party'] if l['party'] else ''}{'  ' + l['ref'] if l['ref'] else ''}")
        while True:
            choice = input("   Decision [a/r/s/A/q]: ").strip()
            if choice in ("a", "r", "s", "A", "q"):
                break
        if choice in ("A", "q"):
            bulk = "approve" if choice == "A" else "skip"
            answers[it["je_id"]] = bulk
        else:
            answers[it["je_id"]] = {"a": "approve", "r": "reject", "s": "skip"}[choice]
    return answers


def run(a: argparse.Namespace) -> int:
    settings = settings_from(a)
    ctx = RunContext(settings=settings)
    if settings.use_llm:
        if ctx.get_model() is None:
            print(f"Note: AI steps are off ({ctx.unavailable_reason}). Running deterministic matching only.", file=sys.stderr)
        else:
            print(f"Using {settings.model}" + (f", max {settings.requests_per_minute:g} requests/minute" if settings.requests_per_minute else ""))
    graph = build_graph(checkpointer=make_checkpointer())
    config = {"configurable": {"thread_id": str(uuid.uuid4())}, "max_concurrency": settings.max_concurrency}
    payload = {"bank_path": a.bank, "ledger_path": a.ledger, "out_dir": a.out}
    seen = 0
    while True:
        interrupted = None
        for chunk in graph.stream(payload, config, context=ctx, stream_mode="updates"):
            for node, update in chunk.items():
                if node == "__interrupt__":
                    interrupted = update[0].value
                    continue
                for line in (update or {}).get("log", []):
                    print(f"[{node}] {line}")
        state = graph.get_state(config).values
        for w in (state.get("warnings") or [])[seen:]:
            print(f"Warning: {w}")
        seen = len(state.get("warnings") or [])
        if interrupted is None:
            break
        payload = Command(resume=ask_reviewer(interrupted))
    for e in state.get("errors") or []:
        print(f"Error: {e}", file=sys.stderr)
    outputs = state.get("outputs") or {}
    if outputs:
        print("\nOutputs:")
        for name, path in outputs.items():
            print(f"  {name:18} {path}")
    return 1 if state.get("errors") else 0


def explain_error(e: Exception) -> str:
    msg = f"{type(e).__name__}: {e}"
    low = msg.lower()
    if any(t in low for t in ("api key not valid", "api_key_invalid", "permission_denied", "unauthenticated", " 401", " 403")):
        return f"The API key was rejected. Check it in .env (Gemini keys come from https://aistudio.google.com/apikey).\n  {msg[:300]}"
    if any(t in low for t in (" 404", "not_found", "not found", "is not supported")):
        return f"This model isn't available to your key. Run `tickmark check --list` and set TICKMARK_MODEL.\n  {msg[:300]}"
    if any(t in low for t in (" 429", "resource_exhausted", "quota", "rate limit")):
        return ("Rate limit or quota reached. On a free tier, run with `--rpm 8 --max-concurrency 1`, "
                f"or pick a model with free-tier quota.\n  {msg[:300]}")
    return msg[:500]


def list_gemini_models() -> list[str]:
    from google import genai
    var = key_for("google_genai")
    client = genai.Client(api_key=os.environ[var])
    names = []
    for m in client.models.list():
        actions = getattr(m, "supported_actions", None) or []
        name = (m.name or "").removeprefix("models/")
        if "generateContent" in actions and name.startswith("gemini"):
            names.append(name)
    return sorted(names)


def check(a: argparse.Namespace) -> int:
    from .agents.extractors import map_columns
    from .agents.resolver import resolve_exception
    from .models import BankTxn, LedgerItem, ResolverTask

    settings = Settings()
    if a.model:
        settings.model = a.model
    provider = settings.provider
    spec = PROVIDERS.get(provider, {"label": provider, "keys": []})
    print(f"Model:     {settings.model}")
    print(f"Provider:  {spec['label']}")
    var = key_for(provider)
    if spec["keys"] and not var:
        print(f"✗ No API key found. Add {spec['keys'][0]}=... to .env (or export it).")
        return 1
    print(f"API key:   found in {var}")
    if provider == "google_genai" and a.list:
        try:
            names = list_gemini_models()
            print(f"\nGemini models your key can use ({len(names)}):")
            for n in names:
                print(f"  google_genai:{n}")
            print()
        except Exception as e:
            print(f"✗ Couldn't list models: {explain_error(e)}")
            return 1
    ctx = RunContext(settings=settings)
    model = ctx.get_model()
    if model is None:
        print(f"✗ {ctx.unavailable_reason}")
        return 1
    try:
        mapping = map_columns(model, "bank", ["Posting Date", "Narrative", "Paid out", "Paid in"],
                              [["02/10/2026", "TRF FROM ACME TRADING", "", "250.00"], ["03/10/2026", "POS COFFEE", "4.50", ""]])
    except Exception as e:
        print(f"✗ Structured output failed. {explain_error(e)}")
        return 1
    ok = mapping.get("date") == 0 and mapping.get("description") == 1 and mapping.get("debit") == 2 and mapping.get("credit") == 3
    print(f"{'✓' if ok else '!'} Structured output: column mapping {'correct' if ok else 'returned ' + str(mapping)}")
    task = ResolverTask(
        bank=BankTxn(id="B1", date="2026-10-05", description="TRF FROM ACME TRADNG", amount=250.0),
        matcher_status="unmatched", matcher_summary="Name spelled differently; no reference quoted.",
        items=[LedgerItem(id="L1", date="2026-10-01", type="AR", ref="INV-1001", party="Acme Trading Ltd", amount=250, open=250),
               LedgerItem(id="L2", date="2026-10-02", type="AR", ref="INV-1002", party="Beta Foods", amount=900, open=900)])
    res = resolve_exception(model, task, recursion_limit=12)
    if res.error or res.decision is None:
        print(f"✗ Agent with tools failed. {explain_error(Exception(res.error or 'no decision'))}")
        return 1
    d = res.decision
    right = d.action == "match" and d.ledger_ids == ["L1"]
    print(f"{'✓' if right else '!'} Agent with tools: {d.action} {d.ledger_ids} ({d.confidence}) after {res.tool_calls} lookup(s)"
          + ("" if right else " (expected a match to L1; the model may need a stronger tier)"))
    print("\nReady. Run: tickmark run --bank <statement> --ledger <ledger>")
    return 0


def ui(a: argparse.Namespace) -> int:
    import threading
    import webbrowser

    import uvicorn

    from .ui.server import create_app
    url = f"http://{'localhost' if a.host in ('127.0.0.1', '0.0.0.0') else a.host}:{a.port}"
    print(f"Tickmark is running at {url}  (press Ctrl+C to stop)")
    if not a.no_browser:
        threading.Timer(1.2, lambda: webbrowser.open(url)).start()
    uvicorn.run(create_app(), host=a.host, port=a.port, log_level="warning")
    return 0


def main(argv=None) -> int:
    _load_dotenv()
    a = parse_args(argv)
    return {"check": check, "ui": ui}.get(a.cmd, run)(a)


if __name__ == "__main__":
    raise SystemExit(main())
