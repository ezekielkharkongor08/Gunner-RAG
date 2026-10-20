"""Unit tests for the section router (pure matching; no database)."""

from __future__ import annotations

import pytest

from src.config import settings
from src.retrieval.section_router import (
    match_section,
    normalise,
    segment_matches,
    token_in_question,
)

PATHS = [
    "Review > November",
    "Review > March",
    "Review > October",
    "Review > Pre-season > First team transfers (summer transfer window)",
    "Player records",
    "Club records",
]


def test_normalise_drops_case_and_punctuation():
    assert normalise("Pre-season") == "preseason"
    assert normalise("  Summer  transfer-window ") == "summertransferwindow"


def test_single_word_matches_whole_words_only():
    assert token_in_question("March", "what happened in March")
    assert not token_in_question("March", "the marching band")


def test_phrase_matches_with_spaces_or_hyphens():
    assert token_in_question("summer transfer window", "summer transfer window")
    assert token_in_question("Pre-season", "tell me about preseason")


def test_segment_matches_exact_and_tail():
    assert segment_matches("Review > November", "November")
    assert segment_matches(
        "Review > Pre-season > First team transfers (summer transfer window)",
        "Summer transfer window",
    )
    assert not segment_matches("Review > October", "November")


def test_segment_matches_a_parent_section():
    # The stored path for the kit section is "Kits > Kit information".
    assert segment_matches("Kits > Kit information", "Kits")


def test_match_section_returns_the_unique_path():
    assert match_section("describe November", PATHS, ["November", "March"]) == (
        "Review > November"
    )


def test_match_section_none_when_no_token():
    assert match_section("who won the league?", PATHS, ["November", "March"]) is None


def test_match_section_none_when_ambiguous():
    # "Records" matches two paths, so the router must decline.
    assert match_section("list the club records", PATHS, ["Records"]) is None


def test_match_section_uses_the_longest_token():
    paths = ["Review > September", "Review > Pre-season"]
    tokens = ["Season", "Pre-season"]
    assert match_section("what happened in pre-season", paths, tokens) == (
        "Review > Pre-season"
    )


def test_tokens_come_from_settings_yaml():
    # The list is configuration, not code: if settings.yaml loses it, routing
    # silently stops working, so this test guards the wiring.
    assert settings.section_tokens
    assert "November" in settings.section_tokens


@pytest.mark.parametrize(
    "question",
    [
        "describe the things that happen in the month of November",
        "give more details about november",
        "November",
    ],
)
def test_all_november_phrasings_resolve(question):
    assert match_section(question, PATHS, settings.section_tokens) == (
        "Review > November"
    )
