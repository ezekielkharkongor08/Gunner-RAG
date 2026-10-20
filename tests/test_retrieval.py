"""Unit tests for retrieval/retriever.py (prose-only, no table logic)."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, replace

import pytest

from langchain_core.documents import Document

import src.retrieval.retriever as retr
from src.retrieval.retriever import RetrievedChunk, _doc_to_chunk, rewrite_queries


class _FakeLLMResponse:
    def __init__(self, content):
        self.content = content


class _FakeLLM:
    def __init__(self, *args, **kwargs):
        pass

    def invoke(self, prompt):
        return _FakeLLMResponse("\n".join(self.variants))


# ---------------------------------------------------------------------------
# _doc_to_chunk / RetrievedChunk / citation
# ---------------------------------------------------------------------------
def test_doc_to_chunk_maps_metadata_including_url():
    doc = Document(
        page_content="Arsenal moved to Emirates in 2006.",
        metadata={
            "chunk_id": "https://en.wikipedia.org/wiki/Arsenal_F.C.#History#0",
            "title": "Arsenal F.C.",
            "section": "History",
            "section_path": "History",
            "url": "https://en.wikipedia.org/wiki/Arsenal_F.C.",
            "language": "en",
            "content_type": "prose",
            "chunk_index": 0,
            "chunk_count": 2,
        },
    )
    chunk = _doc_to_chunk(doc, 0.25)
    assert chunk.chunk_id == "https://en.wikipedia.org/wiki/Arsenal_F.C.#History#0"
    assert chunk.title == "Arsenal F.C."
    assert chunk.section == "History"
    assert chunk.url == "https://en.wikipedia.org/wiki/Arsenal_F.C."
    assert chunk.content_type == "prose"
    assert chunk.text == "Arsenal moved to Emirates in 2006."
    assert chunk.score == 0.25


def test_citation_includes_title_section_and_url():
    chunk = RetrievedChunk(
        chunk_id="id",
        title="Arsenal F.C.",
        section="History > Early years",
        url="https://en.wikipedia.org/wiki/Arsenal_F.C.",
        content_type="prose",
        text="...",
        score=0.1,
    )
    cited = chunk.citation()
    assert "Arsenal F.C." in cited
    assert "section: History > Early years" in cited
    assert "source: https://en.wikipedia.org/wiki/Arsenal_F.C." in cited


def test_retrieved_chunk_has_no_table_specific_fields():
    # RetrievedChunk should not carry table_name / row_index / table_caption /
    # is_table() -- those concepts were removed from the prose-only pipeline.
    chunk = RetrievedChunk(
        chunk_id="id", title="T", section="S", url="U", content_type="prose",
        text="...", score=0.1,
    )
    assert not hasattr(chunk, "table_name")
    assert not hasattr(chunk, "row_index")
    assert not hasattr(chunk, "table_caption")
    assert not hasattr(chunk, "is_table")
    # Frozen dataclass: cannot add stray fields.
    with pytest.raises(FrozenInstanceError):
        chunk.table_name = "x"


# ---------------------------------------------------------------------------
# rewrite_queries
# ---------------------------------------------------------------------------
def _enable_rewrite(monkeypatch) -> None:
    """Turn rewriting on for this test (default mode is fallback)."""
    patched = replace(
        retr.settings,
        retrieval=replace(
            retr.settings.retrieval,
            query_rewrite_mode="always",
            query_rewrite_enabled=True,
        ),
    )
    monkeypatch.setattr(retr, "settings", patched)


def test_rewrite_queries_includes_original_first(monkeypatch):
    _enable_rewrite(monkeypatch)

    class _FakeLLM:
        def invoke(self, prompt):
            return _FakeLLMResponse(
                "Who won the league in 1998?\nArsenal 1998 manager"
            )

    monkeypatch.setattr(retr, "ChatOllama", lambda *a, **k: _FakeLLM())

    queries = rewrite_queries("Who managed Arsenal in 1998?")
    assert queries[0] == "Who managed Arsenal in 1998?"
    assert "Who won the league in 1998?" in queries
    assert len(queries) <= 1 + retr.settings.retrieval.query_rewrite_count


def test_rewrite_queries_falls_back_to_original_on_llm_failure(monkeypatch):
    _enable_rewrite(monkeypatch)

    class _BrokenLLM:
        def invoke(self, prompt):
            raise RuntimeError("LLM down")

    monkeypatch.setattr(retr, "ChatOllama", lambda *a, **k: _BrokenLLM())
    queries = rewrite_queries("Any question?")
    assert queries == ["Any question?"]


def test_rewrite_prompt_names_the_indexed_pages(monkeypatch):
    """The prompt must carry the real page titles and the keep-the-subject rule.

    Without them the model drifted from "November" to "November weather".
    """
    _enable_rewrite(monkeypatch)

    captured = {}

    class _CaptureLLM:
        def invoke(self, prompt):
            captured["prompt"] = prompt
            return _FakeLLMResponse("")

    monkeypatch.setattr(retr, "ChatOllama", lambda *a, **k: _CaptureLLM())
    rewrite_queries("Who was Arsenal's top scorer?")
    prompt = captured["prompt"]

    # Domain comes from settings.yaml, not from a literal in the code.
    for title in retr.settings.pages:
        assert title in prompt
    assert "Keep the subject of the question" in prompt
    assert "Keep any season, year, month or competition" in prompt
    assert "Who was Arsenal's top scorer?" in prompt


# ---------------------------------------------------------------------------
# Phase 2: rewrite switch, season normalisation, merge
# ---------------------------------------------------------------------------
def test_rewrite_queries_off_uses_only_the_normalised_original(monkeypatch):
    patched = replace(
        retr.settings,
        retrieval=replace(
            retr.settings.retrieval,
            query_rewrite_mode="off",
            query_rewrite_enabled=False,
        ),
    )
    monkeypatch.setattr(retr, "settings", patched)

    calls = {"count": 0}

    def _fail(*args, **kwargs):  # the LLM must not be built when the switch is off
        calls["count"] += 1
        return _FakeLLM()

    monkeypatch.setattr(retr, "ChatOllama", _fail)
    queries = rewrite_queries("What happened in 2025-2026?")
    assert queries == ["What happened in 2025\u201326?"]
    assert calls["count"] == 0


def test_rewrite_queries_normalises_seasons_everywhere(monkeypatch):
    _enable_rewrite(monkeypatch)

    class _SeasonLLM:
        def invoke(self, prompt):
            return _FakeLLMResponse(
                "Arsenal 2025-2026 season\nPremier League 2025/26 winners"
            )

    monkeypatch.setattr(retr, "ChatOllama", lambda *a, **k: _SeasonLLM())
    queries = rewrite_queries("What did Arsenal win in 2025-2026?")

    assert queries[0] == "What did Arsenal win in 2025\u201326?"
    assert "Arsenal 2025\u201326 season" in queries
    for query in queries:
        for raw in ("2025-26", "2025-2026", "2025/26"):
            assert raw not in query


def test_fuse_rankings_keeps_the_best_distance_per_chunk():
    def doc(chunk_id):
        return Document(page_content="x", metadata={"chunk_id": chunk_id})

    fused = retr.fuse_rankings(
        [[(doc("a"), 0.5), (doc("b"), 0.4)], [(doc("a"), 0.2)]]
    )
    by_id = {d.metadata["chunk_id"]: score for d, score in fused}
    assert set(by_id) == {"a", "b"}
    assert by_id["a"] == 0.2
    assert by_id["b"] == 0.4


def test_fuse_rankings_rewards_chunks_found_by_several_queries():
    def doc(chunk_id):
        return Document(page_content="x", metadata={"chunk_id": chunk_id})

    # "shared" is first in both lists; "single" is first in one list only.
    fused = retr.fuse_rankings(
        [[(doc("shared"), 0.30), (doc("a"), 0.11)],
         [(doc("shared"), 0.30), (doc("b"), 0.11)]]
    )
    assert fused[0][0].metadata["chunk_id"] == "shared"


def test_fuse_rankings_of_nothing_is_empty():
    assert retr.fuse_rankings([]) == []


def test_threshold_constant_is_documented_default():
    assert retr.settings.retrieval.score_threshold == 0.7
    assert retr.settings.retrieval.final_top_n == 12


def _patch_retrieve(monkeypatch, **overrides) -> None:
    """Patch settings, the section router and the gate out of retrieve()."""
    retrieval = replace(
        retr.settings.retrieval, retrieval_gate="none", **overrides
    )
    monkeypatch.setattr(
        retr, "settings", replace(retr.settings, retrieval=retrieval)
    )
    monkeypatch.setattr(retr, "route_section", lambda question: None)


def _result(chunk_id: str, score: float):
    return (
        Document(
            page_content="body",
            metadata={
                "chunk_id": chunk_id,
                "title": "T",
                "section": "S",
                "url": "U",
                "content_type": "prose",
            },
        ),
        score,
    )


def test_fallback_mode_does_not_rewrite_when_the_first_search_is_strong(monkeypatch):
    _patch_retrieve(
        monkeypatch, query_rewrite_mode="fallback", query_rewrite_weak_score=0.65
    )
    calls = []

    def fake_search(queries, **kwargs):
        calls.append(list(queries))
        return [_result("a", 0.30)]

    def no_llm(*args, **kwargs):
        raise AssertionError("rewriting must not run for a strong first search")

    monkeypatch.setattr(retr, "search_queries", fake_search)
    monkeypatch.setattr(retr, "ChatOllama", no_llm)

    chunks = retr.retrieve("who won?")
    assert [c.chunk_id for c in chunks] == ["a"]
    assert calls == [["who won?"]]


def test_fallback_mode_adds_variants_after_a_weak_first_search(monkeypatch):
    _patch_retrieve(
        monkeypatch, query_rewrite_mode="fallback", query_rewrite_weak_score=0.65
    )

    def fake_search(queries, **kwargs):
        queries = list(queries)
        if queries == ["weak question"]:
            return [_result("original", 0.80)]
        return [_result("variant", 0.40), _result("original", 0.82)]

    class _LLM:
        def invoke(self, prompt):
            return _FakeLLMResponse("better query")

    monkeypatch.setattr(retr, "search_queries", fake_search)
    monkeypatch.setattr(retr, "ChatOllama", lambda *a, **k: _LLM())

    chunks = retr.retrieve("weak question")
    # Original ranking stays at the head; the new chunk is appended.
    assert [c.chunk_id for c in chunks] == ["original", "variant"]


def test_off_mode_never_rewrites(monkeypatch):
    _patch_retrieve(
        monkeypatch, query_rewrite_mode="off", query_rewrite_weak_score=0.10
    )

    def no_llm(*args, **kwargs):
        raise AssertionError("rewriting must not run when mode is off")

    monkeypatch.setattr(
        retr, "search_queries", lambda queries, **kwargs: [_result("a", 0.9)]
    )
    monkeypatch.setattr(retr, "ChatOllama", no_llm)

    chunks = retr.retrieve("weak question")
    assert [c.chunk_id for c in chunks] == ["a"]


def test_always_mode_rewrites_even_when_the_first_search_is_strong(monkeypatch):
    _patch_retrieve(
        monkeypatch, query_rewrite_mode="always", query_rewrite_enabled=True
    )
    searches = []

    def fake_search(queries, **kwargs):
        searches.append(list(queries))
        return [_result("x", 0.20)]

    class _LLM:
        def invoke(self, prompt):
            return _FakeLLMResponse("variant one")

    monkeypatch.setattr(retr, "search_queries", fake_search)
    monkeypatch.setattr(retr, "ChatOllama", lambda *a, **k: _LLM())

    retr.retrieve("question")
    assert searches[-1] == ["question", "variant one"]


# ---------------------------------------------------------------------------
# P1: the retrieval gate
# ---------------------------------------------------------------------------
def _scored(score: float, chunk_id: str) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id, title="T", section="S", url="U",
        content_type="prose", text="text", score=score,
    )


def test_relative_gate_keeps_chunks_near_the_best():
    kept = retr.apply_gate(
        [_scored(0.70, "a"), _scored(0.75, "b"), _scored(0.85, "c")],
        mode="relative", margin=0.10, k=8, fallback_on_empty=False,
    )
    assert [c.chunk_id for c in kept] == ["a", "b"]


def test_relative_gate_never_returns_nothing_when_results_exist():
    # November's real score was 0.712: the old absolute gate rejected it.
    kept = retr.apply_gate(
        [_scored(0.712, "nov")], mode="relative", margin=0.10, k=8,
        fallback_on_empty=False,
    )
    assert [c.chunk_id for c in kept] == ["nov"]


def test_relative_gate_respects_the_cap():
    chunks = [_scored(0.10 + i / 100, str(i)) for i in range(5)]
    kept = retr.apply_gate(chunks, mode="relative", margin=1.0, k=3)
    assert len(kept) == 3


def test_absolute_gate_keeps_only_chunks_below_the_threshold():
    kept = retr.apply_gate(
        [_scored(0.60, "a"), _scored(0.80, "b")],
        mode="absolute", threshold=0.70, k=8, fallback_on_empty=False,
    )
    assert [c.chunk_id for c in kept] == ["a"]


def test_absolute_gate_can_return_nothing():
    kept = retr.apply_gate(
        [_scored(0.90, "a")],
        mode="absolute", threshold=0.70, k=8, fallback_on_empty=False,
    )
    assert kept == []


def test_absolute_gate_fallback_returns_the_best_chunk():
    kept = retr.apply_gate(
        [_scored(0.90, "a"), _scored(0.95, "b")],
        mode="absolute", threshold=0.70, k=8, fallback_on_empty=True,
    )
    assert [c.chunk_id for c in kept] == ["a"]


def test_no_gate_keeps_the_first_k_in_the_given_order():
    # "none" trusts the caller's ranking (RRF order), it does not re-sort.
    kept = retr.apply_gate(
        [_scored(0.90, "a"), _scored(0.10, "b")], mode="none", k=1,
    )
    assert [c.chunk_id for c in kept] == ["a"]


def test_gate_preserves_the_input_order_and_handles_empty():
    # RRF order matters: the gate filters, it does not re-rank.
    kept = retr.apply_gate(
        [_scored(0.80, "slow"), _scored(0.20, "fast")],
        mode="relative", margin=1.0, k=8,
    )
    assert [c.chunk_id for c in kept] == ["slow", "fast"]
    assert retr.apply_gate([]) == []


# ---------------------------------------------------------------------------
# P3: follow-up detection and condensation
# ---------------------------------------------------------------------------
PAIRS = [("How did Arsenal's 2025-2026 season end?", "They reached the final.")]


def test_followup_is_false_without_history():
    assert not retr.is_followup("who did they beat?", None)
    assert not retr.is_followup("who did they beat?", [])


def test_short_question_with_history_is_a_followup():
    assert retr.is_followup("who did they beat?", PAIRS)


def test_pronoun_makes_a_long_question_a_followup():
    assert retr.is_followup(
        "what was the score when they played in the final?", PAIRS
    )


def test_standalone_long_question_is_not_a_followup():
    assert not retr.is_followup(
        "what was the score of the 2006 champions league final", PAIRS
    )


def test_format_history_includes_questions_and_answers():
    text = retr._format_history(PAIRS)
    assert "Turn 1 user: How did Arsenal's 2025-2026 season end?" in text
    assert "Turn 1 assistant: They reached the final." in text


def test_condense_question_uses_the_llm_and_normalises_seasons(monkeypatch):
    captured = {}

    class _LLM:
        def invoke(self, prompt):
            captured["prompt"] = prompt
            return _FakeLLMResponse(
                "Arsenal 2025-2026 Champions League final score"
            )

    monkeypatch.setattr(retr, "ChatOllama", lambda *a, **k: _LLM())
    result = retr.condense_question("what was the score?", PAIRS)

    assert result == "Arsenal 2025\u201326 Champions League final score"
    assert "They reached the final." in captured["prompt"]
    assert "what was the score?" in captured["prompt"]


def test_condense_question_falls_back_to_the_original_on_failure(monkeypatch):
    class _Broken:
        def invoke(self, prompt):
            raise RuntimeError("LLM down")

    monkeypatch.setattr(retr, "ChatOllama", lambda *a, **k: _Broken())
    assert retr.condense_question("what was the score?", PAIRS) == (
        "what was the score?"
    )


def test_condense_question_can_be_disabled(monkeypatch):
    patched = replace(
        retr.settings,
        retrieval=replace(retr.settings.retrieval, condense_followups=False),
    )
    monkeypatch.setattr(retr, "settings", patched)

    def no_llm(*args, **kwargs):
        raise AssertionError("the LLM must not run when condensation is off")

    monkeypatch.setattr(retr, "ChatOllama", no_llm)
    assert retr.condense_question("what was the score?", PAIRS) == (
        "what was the score?"
    )


def test_condense_question_reads_only_the_most_recent_turn(monkeypatch):
    # Two months of history made the model answer "November and March".
    patched = replace(
        retr.settings,
        retrieval=replace(retr.settings.retrieval, condense_history_turns=1),
    )
    monkeypatch.setattr(retr, "settings", patched)
    captured = {}

    class _LLM:
        def invoke(self, prompt):
            captured["prompt"] = prompt
            return _FakeLLMResponse("Who did Arsenal beat at home in March?")

    monkeypatch.setattr(retr, "ChatOllama", lambda *a, **k: _LLM())
    history = [
        ("describe the things that happen in November", "Arsenal won in November."),
        ("give a brief review of March", "Arsenal beat Chelsea in March."),
    ]
    retr.condense_question("who did they beat at home?", history)

    assert "March" in captured["prompt"]
    assert "November" not in captured["prompt"]
