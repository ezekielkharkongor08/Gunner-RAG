"""Offline tests for the answer chain: context format, sources, num_ctx, prompt.

No Ollama and no database: the chat model is replaced with a fake.
"""

from __future__ import annotations

import logging
from dataclasses import replace

import src.generation.answer_chain as ac
from src.config import settings
from src.retrieval.retriever import RetrievedChunk

HEADER = "Arsenal F.C.\nSection: History\n\n"
BODY = "Arsenal moved from Highbury to the Emirates Stadium in 2006."


def _chunk(index: int = 1, **overrides) -> RetrievedChunk:
    values = dict(
        chunk_id=f"id-{index}",
        title="Arsenal F.C.",
        section="History",
        url="https://en.wikipedia.org/wiki/Arsenal_F.C.",
        content_type="prose",
        text=HEADER + BODY,
        score=0.2,
        header_chars=len(HEADER),
        fetched_at="2026-10-19T20:17:32+00:00",
    )
    values.update(overrides)
    return RetrievedChunk(**values)


class _FakeResponse:
    def __init__(self, metadata):
        self.response_metadata = metadata


# ---------------------------------------------------------------------------
# 2.4 Context format and sources
# ---------------------------------------------------------------------------
def test_context_is_numbered_without_repeated_citation_line():
    context = ac.format_context([_chunk(1), _chunk(2)])
    assert context.startswith("[1]\n" + BODY)
    assert "[2]\n" in context
    # The title/section header is not repeated as a citation line.
    assert "section:" not in context
    assert "source:" not in context
    assert "https://" not in context
    assert "Section: History" not in context


def test_context_keeps_header_when_stripping_is_disabled(monkeypatch):
    patched = replace(
        ac.settings,
        retrieval=replace(ac.settings.retrieval, context_strip_header=False),
    )
    monkeypatch.setattr(ac, "settings", patched)
    context = ac.format_context([_chunk(1)])
    assert context.startswith("[1]\n" + HEADER)


def test_empty_context_message():
    assert ac.format_context([]) == "No relevant context was found."


def test_sources_are_built_in_code_with_fetched_at():
    text = ac.format_sources([_chunk(1), _chunk(2)])
    # Same section -> one grouped line with both passage numbers.
    assert "[1][2] Arsenal F.C." in text
    assert "section: History" in text
    assert "source: https://en.wikipedia.org/wiki/Arsenal_F.C." in text
    assert "fetched: 2026-10-19T20:17:32+00:00" in text


def test_sources_group_chunks_from_the_same_section():
    chunks = [
        _chunk(1),
        _chunk(2),
        _chunk(3, section="Honours", url="https://example.org/honours"),
    ]
    text = ac.format_sources(chunks)
    # Two chunks from History share one line; Honours gets its own.
    assert text.count("section: History") == 1
    assert "[1][2] Arsenal F.C." in text
    assert "[3] Arsenal F.C." in text
    assert text.count("\n") == 1


def test_group_sources_keeps_passage_numbers_in_order():
    groups = ac.group_sources([_chunk(1), _chunk(2, section="Honours")])
    assert [numbers for numbers, _ in groups] == [[1], [2]]


def test_body_text_strips_only_the_header():
    chunk = _chunk(1)
    assert chunk.body_text() == BODY


# ---------------------------------------------------------------------------
# 2.5 num_ctx and token logging
# ---------------------------------------------------------------------------
def test_build_answer_chain_passes_num_ctx_and_answer_temperature(monkeypatch):
    seen: dict = {}

    class _FakeChat:
        def __init__(self, **kwargs):
            seen.update(kwargs)

    monkeypatch.setattr(ac, "ChatOllama", _FakeChat)
    ac.build_answer_chain()

    assert seen["num_ctx"] == settings.ollama.num_ctx
    assert seen["temperature"] == settings.ollama.answer_temperature
    assert seen["model"] == settings.ollama.chat_model


def test_token_usage_logs_prompt_eval_count(caplog):
    with caplog.at_level(logging.INFO, logger=ac.__name__):
        ac._log_token_usage(_FakeResponse({"prompt_eval_count": 321}), "answer")
    assert "prompt tokens: 321" in caplog.text


def test_token_usage_warns_above_90_percent(caplog):
    limit = settings.ollama.num_ctx
    with caplog.at_level(logging.WARNING, logger=ac.__name__):
        ac._log_token_usage(_FakeResponse({"prompt_eval_count": limit}), "answer")
    assert "above 90%" in caplog.text


def test_token_usage_is_quiet_without_metadata(caplog):
    with caplog.at_level(logging.WARNING, logger=ac.__name__):
        ac._log_token_usage(_FakeResponse({}), "answer")
    assert "above 90%" not in caplog.text


# ---------------------------------------------------------------------------
# 2.6 Prompt rules
# ---------------------------------------------------------------------------
def test_system_prompt_has_the_answer_rules():
    prompt = ac.SYSTEM_PROMPT
    assert "Use only facts" in prompt
    assert "Copy names, scores and dates exactly" in prompt
    assert "Add no details" in prompt
    assert "say which part is missing" in prompt
    assert "indexed pages do not contain it" in prompt
    assert ac._ANSWER_LENGTH_RULES[settings.answer_style] in prompt
    assert "[1]" in prompt
    # The old URL citation style is gone.
    assert "](URL)" not in prompt


def test_answer_style_selects_the_length_rule():
    assert "Keep answers short" in ac._ANSWER_LENGTH_RULES["short"]
    assert "normal-length answer" in ac._ANSWER_LENGTH_RULES["normal"]
    assert "detailed answer" in ac._ANSWER_LENGTH_RULES["long"]
    assert settings.answer_style in ac._ANSWER_LENGTH_RULES
    assert ac._ANSWER_LENGTH_RULES[settings.answer_style] in ac.SYSTEM_PROMPT


def test_answer_style_is_loaded_from_settings_yaml():
    assert settings.answer_style == "normal"
    assert settings.redacted()["answer_style"] == "normal"


def test_summarize_prompt_uses_number_markers():
    assert "[1]" in ac.SUMMARIZE_SYSTEM_PROMPT
    assert "](URL)" not in ac.SUMMARIZE_SYSTEM_PROMPT


# ---------------------------------------------------------------------------
# P3: conversation history
# ---------------------------------------------------------------------------
def test_history_pairs_become_alternating_messages():
    messages = ac.build_history_messages(
        [("Who did they beat?", "They beat Chelsea [1].")]
    )
    assert [type(message).__name__ for message in messages] == [
        "HumanMessage",
        "AIMessage",
    ]
    assert messages[0].content == "Who did they beat?"
    assert messages[1].content == "They beat Chelsea [1]."


def test_history_strings_stay_user_messages():
    # Older callers (and tests) pass bare questions: keep working.
    messages = ac.build_history_messages(["first question", "second question"])
    assert [type(message).__name__ for message in messages] == [
        "HumanMessage",
        "HumanMessage",
    ]


def test_history_is_empty_when_there_is_none():
    assert ac.build_history_messages(None) == []
    assert ac.build_history_messages([]) == []


def test_system_prompt_answers_only_the_latest_question():
    assert "Answer only the latest question" in ac.SYSTEM_PROMPT
    assert "do not repeat or merge earlier answers" in ac.SYSTEM_PROMPT


def test_system_prompt_separates_home_and_away():
    assert "Never mix home and away" in ac.SYSTEM_PROMPT
    assert "named opponent's ground is an away match" in ac.SYSTEM_PROMPT


def test_system_prompt_states_the_current_season_from_settings():
    assert settings.current_season
    assert f"The current season is {settings.current_season}" in ac.SYSTEM_PROMPT


def test_season_note_spells_out_both_years_for_a_named_season():
    note = ac.season_note("who did Arsenal beat in the 2024-25 quarter-final?")
    assert "Season asked: 2024-25" in note
    assert "mentions 2024 or 2025" in note
    assert "Do not use a result from another season" in note


def test_season_note_accepts_an_en_dash():
    note = ac.season_note("who won the 2023\u201324 league?")
    assert "Season asked: 2023-24" in note
    assert "mentions 2023 or 2024" in note


def test_season_note_is_empty_when_no_season_is_named():
    assert ac.season_note("who won the league?") == ""
    assert ac.season_note("") == ""


def test_filter_to_season_keeps_only_the_named_season():
    old = _chunk(
        1,
        text=(
            "2025\u201326 Arsenal F.C. season\nSection: Background\n\n"
            "In September 2024 they beat Real Madrid 5-1."
        ),
    )
    new = _chunk(
        2,
        text=(
            "2025\u201326 Arsenal F.C. season\nSection: Quarter-final\n\n"
            "Arsenal played Sporting CP in 2026."
        ),
    )
    kept = ac.filter_to_season(
        [new, old], "who did they beat in the 2024-25 quarter-final?"
    )
    assert kept == [old]


def test_filter_to_season_returns_all_when_nothing_matches():
    chunks = [_chunk(1), _chunk(2)]
    assert ac.filter_to_season(chunks, "what happened in 2022-23?") == chunks


def test_filter_to_season_is_a_noop_without_a_season():
    chunks = [_chunk(1), _chunk(2)]
    assert ac.filter_to_season(chunks, "who won the league?") == chunks


def test_system_prompt_keeps_previous_season_facts_with_their_year():
    prompt = ac.SYSTEM_PROMPT
    assert "Background sections recap the previous season" in prompt
    assert "Never move a previous-season result into the current season" in prompt
    assert "If the question names a season, answer for that season only" in prompt
    assert "Do not answer with another season's result." in prompt
    assert "Do not expand a surname into a full name." in prompt


def test_system_prompt_forbids_inventing_dates():
    assert "If a passage gives no date for an event, do not add one" in (
        ac.SYSTEM_PROMPT
    )
