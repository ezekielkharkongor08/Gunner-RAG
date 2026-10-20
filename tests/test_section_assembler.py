"""Unit tests for section_assembler.py (offline, no DB)."""

from __future__ import annotations

from langchain_core.documents import Document

from src.config import settings
from src.ingestion.section_assembler import assemble_sections, summarize_report

TITLE = "2025–26 Arsenal F.C. season"
URL = "https://en.wikipedia.org/wiki/2025%E2%80%9326_Arsenal_F.C._season"


def _doc(path: str, text: str) -> Document:
    return Document(
        page_content=text,
        metadata={
            "source": "wikipedia",
            "title": TITLE,
            "url": URL,
            "section": path.split(" > ")[-1],
            "section_path": path,
            "language": "en",
            "content_type": "prose",
        },
    )


def _long(prefix: str, length: int) -> str:
    return prefix + ("x" * max(0, length - len(prefix)))


def _fixture() -> list[Document]:
    return [
        _doc("Introduction", _long("The season summary. ", 900)),
        _doc("Kits", "Supplier: Adidas. Manufacturer: Adidas."),
        _doc("Kits > Kit information", _long("Home kit. ", 400)),
        _doc("Review", _long("Review overview. ", 100)),
        _doc("Review > May", _long("On 19 May Arsenal won. ", 400)),
        _doc(
            "Honours and achievements > Regional honours > London FA",
            _long("London Senior Cup. ", 300),
        ),
        _doc(
            "Honours and achievements > Regional honours > Kent County FA",
            "Kent Senior Cup winners.",
        ),
        _doc("Player records > Appearances", _long("David O'Leary. ", 590)),
        _doc("Player records > Appearances > Most appearances", "104 matches."),
        _doc(
            "Competitions > Premier League > Matches",
            "51 characters of match data here, no sibling at all.",
        ),
        _doc(
            "Statistics > Appearances",
            "The following 31 players made appearances for the club.",
        ),
        _doc("Statistics > Goals", _long("Arsenal scored 121 goals. ", 300)),
        _doc(
            "First team > First team coaching staff",
            "Note: Age as of 30 June 2026.",
        ),
    ]


def test_boilerplate_is_dropped():
    docs, report = assemble_sections(_fixture())
    paths = {doc.metadata["section_path"] for doc in docs}

    assert "Statistics > Appearances" not in paths
    assert "First team > First team coaching staff" not in paths
    reasons = {item["path"]: item["reason"] for item in report["dropped"]}
    assert "boilerplate" in reasons["Statistics > Appearances"]
    assert "boilerplate" in reasons["First team > First team coaching staff"]


def test_the_following_section_with_a_money_total_is_kept():
    """The transfer spend/income totals live in short "The following ..." sections."""
    money = _doc(
        "Transfers > Transfers in",
        "The following players joined Arsenal permanently and signed "
        "professional contracts with the club.\n\n"
        "Total expenditure: \u00a3248.2 million (excluding potential add-ons).",
    )
    docs, report = assemble_sections(_fixture() + [money])
    by_path = {doc.metadata["section_path"]: doc for doc in docs}

    assert "Transfers > Transfers in" in by_path
    assert "\u00a3248.2 million" in by_path["Transfers > Transfers in"].page_content
    assert all(
        item["path"] != "Transfers > Transfers in" for item in report["dropped"]
    )


def test_the_following_stub_without_money_is_still_dropped():
    stub = _doc(
        "Transfers > New contracts",
        "The following Arsenal players signed their first or new professional "
        "contracts with the club.",
    )
    docs, _report = assemble_sections(_fixture() + [stub])
    paths = {doc.metadata["section_path"] for doc in docs}
    assert "Transfers > New contracts" not in paths


def test_goals_section_is_kept():
    docs, _ = assemble_sections(_fixture())
    goals = [d for d in docs if d.metadata["section_path"] == "Statistics > Goals"]
    assert goals
    assert "scored 121 goals" in goals[0].page_content


def test_short_parent_merges_into_first_child():
    docs, report = assemble_sections(_fixture())
    by_path = {doc.metadata["section_path"]: doc for doc in docs}

    kit = by_path["Kits > Kit information"]
    assert kit.page_content.startswith("Kits: Supplier: Adidas.")
    assert kit.metadata["merged_from"] == ["Kits"]

    may = by_path["Review > May"]
    assert may.page_content.startswith("Review: Review overview.")
    assert "Review" in may.metadata["merged_from"]
    assert any(
        item["from"] == "Kits" and item["into"] == "Kits > Kit information"
        for item in report["merged"]
    )


def test_short_sibling_merges_into_previous_sibling():
    docs, _ = assemble_sections(_fixture())
    london = [
        doc for doc in docs
        if doc.metadata["section_path"].endswith("London FA")
    ][0]

    assert london.page_content.rstrip().endswith(
        "Kent County FA: Kent Senior Cup winners."
    )
    assert "Kent County FA" in london.metadata["merged_from"]


def test_short_child_merges_into_parent():
    docs, _ = assemble_sections(_fixture())
    appearances = [
        doc for doc in docs
        if doc.metadata["section_path"] == "Player records > Appearances"
    ][0]

    assert appearances.page_content.rstrip().endswith(
        "Most appearances: 104 matches."
    )
    assert "Most appearances" in appearances.metadata["merged_from"]


def test_unmergeable_tiny_section_is_dropped():
    docs, report = assemble_sections(_fixture())
    paths = {doc.metadata["section_path"] for doc in docs}

    assert "Competitions > Premier League > Matches" not in paths
    reasons = {item["path"]: item["reason"] for item in report["dropped"]}
    assert "too short" in reasons["Competitions > Premier League > Matches"]


def test_no_section_is_shorter_than_keep_limit():
    docs, _ = assemble_sections(_fixture())
    assert docs
    for doc in docs:
        assert doc.metadata["body_chars"] >= settings.chunk.min_section_keep


def test_metadata_is_added():
    docs, _ = assemble_sections(_fixture())
    by_path = {doc.metadata["section_path"]: doc for doc in docs}

    introduction = by_path["Introduction"]
    assert introduction.metadata["is_lead"] is True
    assert introduction.metadata["heading_level"] == 0
    assert introduction.metadata["page_type"] == "season"
    assert introduction.metadata["seasons"] == ["2025–26"]
    assert introduction.metadata["parent_section"] == ""
    assert len(introduction.metadata["section_hash"]) == 64

    may = by_path["Review > May"]
    assert may.metadata["is_lead"] is False
    assert may.metadata["heading_level"] == 3
    assert may.metadata["parent_section"] == "Review"


def test_summary_counts_match():
    docs, report = assemble_sections(_fixture())
    assert summarize_report(report) == (
        f"sections in={len(_fixture())} out={len(docs)} "
        f"dropped={len(report['dropped'])} merged={len(report['merged'])}"
    )
    assert report["sections_out"] == len(docs)


def test_empty_input():
    docs, report = assemble_sections([])
    assert docs == []
    assert report["sections_in"] == 0
    assert report["sections_out"] == 0
