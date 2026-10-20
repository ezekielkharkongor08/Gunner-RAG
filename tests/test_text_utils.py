"""Unit tests for text_utils.py."""

from __future__ import annotations

from src.text_utils import (
    find_seasons,
    normalize_season_text,
    page_type_for_title,
)


def test_normalize_season_text_four_digit_end():
    assert normalize_season_text("2025-2026 season") == "2025–26 season"
    assert normalize_season_text("2025–2026") == "2025–26"


def test_normalize_season_text_short_end():
    assert normalize_season_text("2025-26") == "2025–26"
    assert normalize_season_text("2025/26") == "2025–26"


def test_normalize_season_text_other_century_keeps_full_year():
    assert normalize_season_text("1999-2000") == "1999–2000"


def test_normalize_season_text_leaves_plain_text_alone():
    assert normalize_season_text("Arsenal won the league.") == (
        "Arsenal won the league."
    )


def test_find_seasons_dedupes_and_keeps_order():
    text = "2025-26 season, then 2026-27, then 2025-26 again."
    assert find_seasons(text) == ["2025–26", "2026–27"]


def test_find_seasons_respects_limit():
    text = "1999-2000 2000-01 2001-02 2002-03"
    assert len(find_seasons(text, limit=2)) == 2


def test_page_type_for_title():
    assert page_type_for_title("2025–26 Arsenal F.C. season") == "season"
    assert page_type_for_title("List of Arsenal F.C. records and statistics") == "list"
    assert page_type_for_title("Arsenal F.C.") == "club"
    assert page_type_for_title("Some other page") == "other"
