"""Tests for the grounding checks and `ask.py --check` (no database, no LLM)."""

from __future__ import annotations

from src.generation import grounding
from src.generation.answer_chain import GenerationResult
from src.retrieval.retriever import RetrievedChunk

import ask

PASSAGES = (
    "Arsenal beat Chelsea 2-1 at home on 1 March [1]. "
    "They lost 2-0 to Manchester City in the EFL Cup final [2]."
)


# ---------------------------------------------------------------------------
# grounding.py
# ---------------------------------------------------------------------------
def test_citations_are_not_treated_as_numbers():
    # [3][12] are markers; a score is checked as its two numbers.
    assert grounding.numbers_in("They won 2-1 [3][12].") == {"2", "1"}


def test_supported_numbers_pass():
    answer = "Arsenal beat Chelsea 2-1 [1]."
    assert grounding.unsupported_numbers(answer, PASSAGES) == []


def test_invented_number_is_flagged():
    answer = "Arsenal beat Chelsea 4-3 [1]."
    assert grounding.unsupported_numbers(answer, PASSAGES) == ["3", "4"]


def test_multiple_passages_are_joined():
    answer = "They lost the final 2-0 [2]."
    assert grounding.unsupported_numbers(answer, [PASSAGES]) == []


def test_invented_name_is_flagged():
    answer = "Arsenal beat Real Madrid 2-1 [1]."
    assert grounding.unsupported_terms(answer, PASSAGES) == ["Real Madrid"]


def test_sentence_opening_is_not_reported_as_a_name():
    # "In November" is capitalised because it starts a sentence, not a name.
    assert grounding.capitalised_terms("In November, Arsenal won 3-0.") == {
        "November"
    }
    assert grounding.unsupported_terms(
        "In November, Arsenal won 3-0 [1].", ["Arsenal won 3-0 in November."]
    ) == []


def test_supported_name_passes():
    answer = "They beat Chelsea 2-1, then lost to Manchester City [1][2]."
    assert grounding.unsupported_terms(answer, PASSAGES) == []


def test_check_answer_returns_both_lists():
    report = grounding.check_answer("Arsenal beat Chelsea 4-3.", PASSAGES)
    assert report == {"numbers": ["3", "4"], "names": []}


# ---------------------------------------------------------------------------
# ask.py --check
# ---------------------------------------------------------------------------
def _chunk() -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id="id-1",
        title="2025–26 Arsenal F.C. season",
        section="March",
        url="https://example.org/season",
        content_type="prose",
        text="Review > March\n\n" + PASSAGES,
        score=0.2,
        header_chars=len("Review > March\n\n"),
        fetched_at="2026-10-19T20:17:32+00:00",
    )


def _result(answer: str) -> GenerationResult:
    return GenerationResult(answer=answer, sources=[_chunk()], question="q")


def test_ask_check_passes_when_numbers_are_grounded(monkeypatch, capsys):
    seen = {}

    def fake_answer(question, history=None):
        seen["question"] = question
        return _result("Arsenal beat Chelsea 2-1 [1].")

    monkeypatch.setattr(ask, "answer_question", fake_answer)
    code = ask.main(["ask.py", "--check", "who", "did", "they", "beat?"])

    out = capsys.readouterr().out
    assert code == 0
    # --check is a flag, not part of the question.
    assert seen["question"] == "who did they beat?"
    assert "unsupported numbers: none" in out


def test_ask_check_fails_on_an_invented_number(monkeypatch):
    monkeypatch.setattr(
        ask, "answer_question", lambda q, history=None: _result("They won 4-3 [1].")
    )
    assert ask.main(["ask.py", "score?", "--check"]) == 3


def test_ask_prints_grouped_sources(monkeypatch, capsys):
    monkeypatch.setattr(
        ask, "answer_question", lambda q, history=None: _result("They won 2-1 [1].")
    )
    assert ask.main(["ask.py", "score?"]) == 0
    out = capsys.readouterr().out
    assert "Sources:" in out
    assert "section: March" in out
