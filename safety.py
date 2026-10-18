"""Data-driven, layered safety helpers. No LLM calls here (those live in app.py).

Layers (strongest wins):
  1. normalize()      - Unicode/Urdu/Arabic variants, diacritics, stretched letters, punctuation
  2. pattern_level()  - regex *stems* loaded from data/crisis_patterns.json (editable, no code change)
  3. history_level()  - earlier user turns keep the conversation at >= "elevated"
The AI classifier (app.classify_risk) is a separate semantic layer on top of these.
"""
from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path
from typing import Iterable, Literal

DATA_DIR = Path(__file__).resolve().parent / "data"
Level = Literal["low", "elevated", "immediate"]
_RANK = {"low": 0, "elevated": 1, "immediate": 2}

# Urdu / Arabic / Persian letter variants -> one canonical form (applied to text AND patterns).
_CHAR_MAP = str.maketrans({
    "ك": "ک", "ي": "ی", "ى": "ی", "ې": "ی", "ے": "ی", "ئ": "ی",
    "ه": "ہ", "ۃ": "ہ", "ة": "ہ", "ھ": "ہ", "ۂ": "ہ",
    "أ": "ا", "إ": "ا", "آ": "ا", "ٱ": "ا", "ؤ": "و",
    "’": "'", "‘": "'", "`": "'",
})
_INVISIBLE = dict.fromkeys(map(ord, "​‌‍⁠﻿ـ"), None)


def normalize(text: str) -> str:
    t = unicodedata.normalize("NFKC", text or "")
    t = "".join(ch for ch in t if unicodedata.category(ch) != "Mn")  # harakat/accents
    t = t.translate(_INVISIBLE).translate(_CHAR_MAP).lower()
    t = re.sub(r"(.)\1{2,}", r"\1", t)          # "diiiie" -> "die"
    t = re.sub(r"[^\w\s']", " ", t)             # punctuation/emoji -> space (keep apostrophe)
    return re.sub(r"\s+", " ", t).strip()


def _load_patterns() -> dict[str, list[re.Pattern]]:
    raw = json.loads((DATA_DIR / "crisis_patterns.json").read_text(encoding="utf-8"))
    compiled: dict[str, list[re.Pattern]] = {"strong": [], "weak": [], "distress": []}
    for level in compiled:
        for items in (raw.get(level) or {}).values():
            for pattern in items:
                compiled[level].append(re.compile(pattern.translate(_CHAR_MAP), re.I))
    return compiled


PATTERNS = _load_patterns()


def pattern_level(text: str) -> Level:
    """strong -> immediate; weak/distress -> elevated; nothing -> low."""
    t = normalize(text)
    if any(p.search(t) for p in PATTERNS["strong"]):
        return "immediate"
    if any(p.search(t) for p in PATTERNS["weak"]) or any(p.search(t) for p in PATTERNS["distress"]):
        return "elevated"
    return "low"


def has_explicit_crisis_signal(text: str) -> bool:
    """Stable public name: True only for explicit first-person danger phrases."""
    return pattern_level(text) == "immediate"


def has_distress_signal(text: str) -> bool:
    return pattern_level(text) != "low"


def max_level(*levels: str | None) -> Level:
    best: Level = "low"
    for lv in levels:
        if lv in _RANK and _RANK[lv] > _RANK[best]:
            best = lv  # type: ignore[assignment]
    return best


def history_level(user_texts: Iterable[str]) -> Level:
    """Earlier risk keeps the conversation cautious: capped at 'elevated'."""
    for text in user_texts:
        if pattern_level(text) != "low":
            return "elevated"
    return "low"
