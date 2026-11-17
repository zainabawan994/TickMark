from pathlib import Path

import pytest

from tickmark.config import Accounts
from tickmark.ingest.pdf import parse_pdf_lines, pdf_lines
from tickmark.ingest.tables import detect_columns, normalize_bank, normalize_ledger, read_rows, to_table
from tickmark.ingest.values import parse_amount, parse_date
from tickmark.journal import build_journal
from tickmark.matching.engine import MatchConfig, reconcile
from tickmark.matching.names import build_party_index, name_score, tokens_of
from tickmark.models import LedgerItem

SAMPLES = Path(__file__).parent.parent / "samples"


def load(bank="bank_statement.csv", ledger="ledger.csv"):
    if bank.endswith(".pdf"):
        bank_rows = parse_pdf_lines(pdf_lines(SAMPLES / bank)).rows
    else:
        bt = to_table(read_rows(SAMPLES / bank), "bank")
        bank_rows = normalize_bank(bt, detect_columns(bt.headers, "bank")).rows
    lt = to_table(read_rows(SAMPLES / ledger), "ledger")
    return bank_rows, normalize_ledger(lt, detect_columns(lt.headers, "ledger")).rows


@pytest.mark.parametrize("raw,expected", [("1,234.56", 1234.56), ("(842.17)", -842.17), ("25.00 DR", -25.0),
                                          ("1.234,56", 1234.56), ("-1,200", -1200.0), ("", None), ("abc", None)])
def test_parse_amount(raw, expected):
    assert parse_amount(raw) == expected


@pytest.mark.parametrize("raw,order,expected", [("31/10/2026", "dmy", "2026-10-31"), ("10/31/2026", "dmy", "2026-10-31"),
                                                ("03/04/2026", "mdy", "2026-03-04"), ("2026-10-03", "dmy", "2026-10-03"),
                                                ("Oct 3, 2026", "dmy", "2026-10-03"), ("3-Oct-26", "dmy", "2026-10-03"),
                                                (46300, "dmy", "2026-10-05")])
def test_parse_date(raw, order, expected):
    assert parse_date(raw, order) == expected


def test_csv_xlsx_and_pdf_read_identically():
    csv_bank, csv_ledger = load()
    pdf_bank, xlsx_ledger = load("bank_statement.pdf", "ledger.xlsx")
    assert len(csv_bank) == len(pdf_bank) == 16
    assert [(b.date, b.amount) for b in csv_bank] == [(b.date, b.amount) for b in pdf_bank]
    assert [(l.ref, l.type, l.open) for l in csv_ledger] == [(l.ref, l.type, l.open) for l in xlsx_ledger]


def test_pdf_balance_verified():
    parsed = parse_pdf_lines(pdf_lines(SAMPLES / "bank_statement.pdf"))
    assert parsed.verified is True and parsed.opening == 48210.55 and parsed.closing == 74236.81


def test_matcher_handles_every_tricky_case():
    bank, ledger = load()
    out = reconcile(bank, ledger, MatchConfig())
    ref = {l.id: l.ref for l in ledger}
    got = {b.id: (out.results[b.id].status, out.results[b.id].method, sorted(ref[a.ledger_id] for a in out.results[b.id].allocations))
           for b in bank}
    assert got["B1"] == ("matched", "Reference match", ["INV-2041"])
    assert got["B3"] == ("matched", "Combined payment", ["INV-2033", "INV-2038", "INV-2044"])   # one payment, three invoices
    assert got["B4"][0] == "recorded"                                                          # cheque already booked
    assert got["B5"] == ("matched", "Amount and name", ["INV-2036"])                           # McKenzie vs MacKenzie
    assert got["B6"] == ("matched", "Partial payment", ["INV-2047"])                           # partial, by reference
    assert out.results["B7"].fee == 15.0                                                       # fee deducted in transit
    assert got["B8"][:2] == ("review", "Amount only")
    assert out.results["B10"].excess == 150.0                                                  # overpayment
    assert got["B11"] == ("review", "Partial payment (oldest first)", ["INV-2035", "INV-2043"])
    assert got["B12"] == ("matched", "Amount and name", ["BILL-5530"])                         # ADOBE *SUBSCRIPTION
    assert got["B13"][0] == got["B14"][0] == "unmatched"
    assert "INV-2045" in out.results["B13"].why[1] and "INV-2046" in out.results["B13"].why[1]  # explains the ambiguity
    assert got["B15"][0] == got["B16"][0] == "bankitem"
    assert {o["ref"]: o["remaining"] for o in out.outstanding}["INV-2047"] == 7400.0


def test_every_journal_entry_balances():
    bank, ledger = load()
    out = reconcile(bank, ledger, MatchConfig())
    journal = build_journal(bank, out.results, ledger, Accounts())
    assert len(journal) == 13 and all(je.balanced for je in journal)
    nordic = next(je for je in journal if "Nordic" in je.memo)
    assert {(l.account, l.debit, l.credit) for l in nordic.lines} == {
        ("Bank - operating account", 3185.0, 0.0), ("Bank charges", 15.0, 0.0), ("Accounts receivable", 0.0, 3200.0)}


@pytest.mark.parametrize("desc,party,should_match", [
    ("HASAN TEXTILE", "Hassan Textiles (Pvt) Ltd", True), ("PACIFIC FRT LINES", "Pacific Freight Lines", True),
    ("ABC", "Alpha Beta Corp", True), ("NORTHWIND TRADING", "Northwind Traders", True),
    ("GREEN ENERGY CO", "Green Valley Farms", False), ("BLUE SKY MEDIA", "Blue Ocean Logistics", False),
])
def test_name_scoring(desc, party, should_match):
    parties = ["Hassan Textiles (Pvt) Ltd", "Pacific Freight Lines", "Alpha Beta Corp", "Northwind Traders",
               "Green Valley Farms", "Blue Ocean Logistics", "Smith & Jones Partners"]
    idx = build_party_index([LedgerItem(id=f"L{i}", date="2026-01-01", type="AR", party=p, amount=1, open=1) for i, p in enumerate(parties)])
    dt = tokens_of(desc, True)
    assert (name_score(dt, "".join(dt), idx[party.lower()]) >= 0.72) is should_match
