"""Fuzzy matching of customer/supplier names against bank descriptions.

A bank line like "FASTER PMT QUANTUM RETAIL GRP" must match "Quantum Retail Ltd",
while "GREEN ENERGY CO" must not match "Green Valley Farms". The score combines:
* token similarity (Jaro-Winkler, prefixes, consonant abbreviations like FRT/freight),
* IDF weights, so words shared by many parties ("group", "services") count less,
* acronyms ("ABC" for Alpha Beta Corp) and run-together names ("OFFICEMAX").
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

from rapidfuzz.distance import JaroWinkler

from ..models import LedgerItem

LEGAL = set("ltd limited inc incorporated llc llp lp plc co company corp corporation pvt private pty gmbh ag sa sarl "
            "bv nv srl spa oy ab as kk the and of".split())
ABBR = {"grp": "group", "intl": "international", "svcs": "services", "svc": "services", "serv": "services",
        "mfg": "manufacturing", "assoc": "associates", "assocs": "associates", "bros": "brothers", "sys": "systems",
        "mgmt": "management", "mgt": "management", "natl": "national", "univ": "university", "hosp": "hospital",
        "engg": "engineering", "ent": "enterprises", "ents": "enterprises", "inds": "industries", "dist": "distribution",
        "distrib": "distribution", "trdg": "trading", "mktg": "marketing", "props": "properties", "ctr": "center",
        "centre": "center", "labs": "laboratories", "lab": "laboratories", "pharma": "pharmaceuticals",
        "elec": "electrical", "rtl": "retail", "frt": "freight", "whse": "warehouse", "mkt": "market",
        "hldgs": "holdings", "hldg": "holdings", "solns": "solutions", "tech": "technologies", "techs": "technologies",
        "org": "organization", "dev": "development", "constr": "construction", "cons": "consulting"}
NOISE = set("payment payments pmt pymt paymt transfer trf tfr xfer transf ach wire wires ref reference credit debit deposit "
            "dep from to inv invoice invoices bill bills pos online internet mobile ibft neft rtgs imps upi chq cheque check "
            "fpx sepa bacs fp dd so standing order direct inward outward incoming outgoing in out funds fund trn txn tx trx "
            "bank branch acct account ac no num number via by for remittance remit received recd rcvd paid pay domestic "
            "local mob web net banking ft cr dr memo id faster instant swift mt103 iban beneficiary ben ordering customer "
            "cust settlement batch eft giro bpay interac etransfer zelle venmo paypal stripe card visa mastercard "
            "purchase".split())
GENERIC = set("green blue red black white global first united national american general city north south east west new "
              "metro pacific atlantic royal star sun best prime smart digital alpha apex golden silver one central "
              "international group services solutions holdings trading enterprises industries systems design studio "
              "clinic consulting partners associates capital home office".split())


def tokens_of(s: str, is_desc: bool = False) -> list[str]:
    s = str(s or "").lower().replace("&", " and ")
    s = re.sub(r"['’`]", "", s)
    words = [ABBR.get(w, w) for w in re.sub(r"[^a-z0-9]+", " ", s).split()]
    words = [w for w in words if w not in LEGAL and not re.search(r"\d", w) and len(w) > 1]
    return [w for w in words if w not in NOISE] if is_desc else words


def _is_subseq(d: str, p: str) -> bool:
    it = iter(p)
    return all(ch in it for ch in d)


def tok_sim(p: str, d: str) -> float:
    if p == d:
        return 1.0
    if min(len(p), len(d)) >= 4:
        s = JaroWinkler.normalized_similarity(p, d)
        if s >= 0.88:
            return s
    if len(d) >= 4 and p.startswith(d):
        return 0.88
    if len(p) >= 4 and d.startswith(p):
        return 0.9
    if len(d) == 3 and len(p) >= 5 and p.startswith(d):
        return 0.85
    if len(d) >= 3 and len(p) >= 5 and d[0] == p[0] and len(d) <= len(p) * 0.75 and _is_subseq(d, p):
        return 0.85
    return 0.0


@dataclass
class PartyInfo:
    key: str
    name: str
    tokens: list[str]
    acro: str = ""
    w: list[float] = field(default_factory=list)
    compact: str = ""
    initials: str = ""
    first_bonus: bool = False


def party_key(name: str) -> str:
    return str(name or "").lower().strip()


def build_party_index(ledger: list[LedgerItem]) -> dict[str, PartyInfo]:
    parties: dict[str, PartyInfo] = {}
    for item in ledger:
        if item.type == "GL" or not item.party:
            continue
        key = party_key(item.party)
        item.party_key = key
        if key not in parties:
            raw = [w for w in re.sub(r"[^a-z0-9]+", " ", item.party.lower().replace("&", " ")).split()
                   if w not in ("the", "and", "of", "pvt", "private")]
            parties[key] = PartyInfo(key, item.party, tokens_of(item.party), acro="".join(w[0] for w in raw) if len(raw) >= 2 else "")
    df: dict[str, int] = {}
    for p in parties.values():
        for t in set(p.tokens):
            df[t] = df.get(t, 0) + 1
    n = max(len(parties), 1)
    for p in parties.values():
        p.w = [math.log(1 + n / df[t]) for t in p.tokens]
        p.compact = "".join(p.tokens)
        p.initials = "".join(t[0] for t in p.tokens) if len(p.tokens) >= 2 else ""
        p.first_bonus = bool(p.tokens and len(p.tokens[0]) >= 4 and df[p.tokens[0]] == 1 and p.tokens[0] not in GENERIC)
    return parties


def name_score(dt: list[str], d_compact: str, p: PartyInfo) -> float:
    """Similarity (0-1) between bank-description tokens and one party."""
    if not p.tokens or not dt:
        return 0.0
    num = den = 0.0
    for t, w in zip(p.tokens, p.w):
        num += max(tok_sim(t, d) for d in dt) * w
        den += w
    pc = num / den if den else 0.0
    dc = sum(max(tok_sim(t, d) for t in p.tokens) for d in dt) / len(dt)
    score = max(pc, 0.5 * pc + 0.5 * dc)
    if len(p.compact) >= 6 and p.compact in d_compact:
        score = max(score, 0.95)
    if (len(p.initials) >= 2 and p.initials in dt) or (len(p.acro) >= 3 and p.acro in dt):
        score = max(score, 0.8)
    if p.first_bonus and max(tok_sim(p.tokens[0], d) for d in dt) >= 0.93:
        score = max(score, 0.74 + 0.22 * pc)
    return min(1.0, score)
