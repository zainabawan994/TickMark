"""Long-term memory of name variations (bank wording -> ledger party).

When a person approves a match the matcher was unsure about, the bank wording is
remembered, so next month "MOBILE DEP CEDAR PT" scores 100% for Cedar Point Dental.
"""
from __future__ import annotations

import json
from pathlib import Path

from .matching.engine import description_key


class AliasMemory:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.aliases: dict[str, str] = {}
        if self.path.exists():
            try:
                self.aliases = json.loads(self.path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                self.aliases = {}

    def learn(self, bank_description: str, party: str) -> bool:
        key = description_key(bank_description)
        if not key or not party or self.aliases.get(key) == party:
            return False
        self.aliases[key] = party
        return True

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.aliases, indent=2, sort_keys=True), encoding="utf-8")
