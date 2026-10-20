"""Check that an answer stays inside the passages it was given.

Two cheap, offline checks:

* every number in the answer must appear somewhere in the passages -- an exact
  token match, so an invented score or date is caught;
* every capitalised multi-word name is looked up the same way -- a heuristic,
  because Wikipedia prose also capitalises ordinary phrases.

Used by ``ask.py --check`` and by the eval's ``--with-llm`` check, so both
report the same thing. Advisory: a missing token is a reason to look, not proof
of a hallucination (the model may rephrase a number, e.g. "2-1" -> "2–1").
"""

from __future__ import annotations

import re
from typing import Iterable

# [1], [12] -- citation markers are not facts.
CITATION = re.compile(r"\[\d+\]")
# 2, 2.1, 1,000 -- keep the token exactly as written.
NUMBER = re.compile(r"\d+(?:[.,]\d+)*")
# Two or more capitalised words: "Bayer Leverkusen", "Emirates Stadium".
PROPER_NAME = re.compile(r"\b([A-Z][a-z]+(?: [A-Z][a-z]+)+)\b")
# A sentence may start with a capitalised word that is not a name: "In November
# Arsenal won..." must not be reported as an unsupported name "In November".
SENTENCE_STARTERS = frozenset({
    "a", "after", "an", "and", "at", "before", "but", "by", "despite",
    "during", "following", "for", "he", "her", "his", "however", "in", "it",
    "its", "later", "meanwhile", "of", "on", "she", "that", "the", "their",
    "then", "there", "these", "they", "this", "those", "to", "with",
})


def strip_citations(text: str) -> str:
    """Remove [n] markers so they are never treated as numbers."""
    return CITATION.sub(" ", text or "")


def numbers_in(text: str) -> set[str]:
    """Every numeric token in the text, citations ignored."""
    return {
        match.group(0).rstrip(".,")
        for match in NUMBER.finditer(strip_citations(text))
    }


def capitalised_terms(text: str) -> set[str]:
    """Every multi-word capitalised phrase in the text, citations ignored.

    Leading sentence starters are dropped ("In November" -> "November"), so a
    normal sentence opening is not mistaken for a name.
    """
    terms: set[str] = set()
    for match in PROPER_NAME.finditer(strip_citations(text)):
        words = match.group(1).split()
        while words and words[0].casefold() in SENTENCE_STARTERS:
            words.pop(0)
        if words:
            terms.add(" ".join(words))
    return terms


def _ground(passages: Iterable[str] | str) -> str:
    """Lowercased passage text, whether given as one string or many."""
    if isinstance(passages, str):
        return passages.casefold()
    return " ".join(passages).casefold()


def unsupported_numbers(
    answer: str, passages: Iterable[str] | str
) -> list[str]:
    """Numbers in the answer that do not appear in the passages, sorted."""
    ground = _ground(passages)
    return sorted(
        number for number in numbers_in(answer) if number.casefold() not in ground
    )


def unsupported_terms(answer: str, passages: Iterable[str] | str) -> list[str]:
    """Capitalised names in the answer that do not appear in the passages."""
    ground = _ground(passages)
    return sorted(
        term for term in capitalised_terms(answer) if term.casefold() not in ground
    )


def check_answer(
    answer: str, passages: Iterable[str] | str
) -> dict[str, list[str]]:
    """Run both checks and return ``{"numbers": [...], "names": [...]}``."""
    return {
        "numbers": unsupported_numbers(answer, passages),
        "names": unsupported_terms(answer, passages),
    }
